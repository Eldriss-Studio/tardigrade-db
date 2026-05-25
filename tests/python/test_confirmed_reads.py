"""Acceptance tests for confirmed-vs-unconfirmed read semantics.

Closes the rule-vs-reality gap surfaced in the SpacetimeDB research
report §8.1: CLAUDE.md's "Reliability & Consistency Rules" mandates
that every externally visible read declare ``confirmed`` vs
``unconfirmed`` semantics, but until this work no read API exposed
the choice.

Contract (from ``~/.claude/plans/spacetimedb-confirmed-reads-contract.md``):

- Default mode (``unconfirmed``) returns immediately. Backward-compat
  for every existing caller — no behaviour change.
- ``mode="confirmed"`` blocks until ``durable_offset()`` reaches the
  offset that was current at the moment the read was issued (i.e.,
  every concurrent in-flight write is now durable). Requires an
  explicit ``timeout_ms``; raises ``ValueError`` otherwise — confirmed
  reads without a deadline are a foot-gun.
- Timeout exhaustion raises a typed error (``tdb::durability::read_timeout``
  in the miette diagnostic vocabulary).
- Multiple confirmed waiters on the same offset all unblock on a
  single flush (Condvar coalescing).
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import tardigrade_db


KEY_DIM = 64


def _key(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(KEY_DIM, dtype=np.float32)


def _layer_payload(seed: int) -> list[tuple[int, np.ndarray]]:
    return [(0, _key(seed + 1000))]


# ─────────────────────────────────────────────────────────────────────────
# AT #1 — Unconfirmed mode never blocks on pending writes
# ─────────────────────────────────────────────────────────────────────────


class TestUnconfirmedReadDoesNotBlockOnPendingWrites:
    """Default mode returns immediately even with un-fsynced buffered writes."""

    def test_unconfirmed_returns_in_under_10ms_with_buffered_writes(
        self, tmp_path
    ) -> None:
        engine = tardigrade_db.Engine.open_with_write_buffer(
            str(tmp_path / "engine"),
            max_batch_size=100,
            max_idle_ms=60000,  # disable idle-flush so writes stay buffered
        )
        # Five buffered writes — none fsynced yet.
        for i in range(5):
            engine.mem_write_pack(
                owner=1,
                retrieval_key=_key(i),
                layer_payloads=_layer_payload(i),
                salience=50.0,
            )
        # Default-mode read should not wait on flush.
        start = time.monotonic()
        engine.mem_read_pack(_key(0), k=5, owner=1)
        elapsed = time.monotonic() - start
        assert elapsed < 0.10, (
            f"Unconfirmed read should return in < 100ms even with pending "
            f"buffered writes; took {elapsed*1000:.1f}ms. Likely the read "
            f"path is silently waiting on durability."
        )


# ─────────────────────────────────────────────────────────────────────────
# AT #2 — Confirmed mode blocks until durability catches up
# ─────────────────────────────────────────────────────────────────────────


class TestConfirmedReadBlocksUntilFlush:
    """Confirmed read returns only after a flush makes pending writes durable."""

    def test_confirmed_read_unblocks_within_50ms_of_flush(self, tmp_path) -> None:
        engine = tardigrade_db.Engine.open_with_write_buffer(
            str(tmp_path / "engine"),
            max_batch_size=100,
            max_idle_ms=60000,
        )
        # Buffered write — not fsynced.
        engine.mem_write_pack(
            owner=1,
            retrieval_key=_key(0),
            layer_payloads=_layer_payload(0),
            salience=50.0,
        )

        results_holder: list = []
        reader_started = threading.Event()
        reader_done = threading.Event()

        def confirmed_reader() -> None:
            reader_started.set()
            results_holder.append(
                engine.mem_read_pack(
                    _key(0),
                    k=5,
                    owner=1,
                    mode="confirmed",
                    timeout_ms=5000,
                )
            )
            reader_done.set()

        thread = threading.Thread(target=confirmed_reader)
        thread.start()
        reader_started.wait(timeout=1.0)
        # Give the reader a moment to enter its wait — without this the
        # main thread could flush before the reader registers.
        time.sleep(0.05)
        assert not reader_done.is_set(), (
            "Confirmed reader returned before flush. Either the contract "
            "isn't enforced or the buffered write was already durable."
        )

        flush_time = time.monotonic()
        engine.flush_buffer()
        reader_done.wait(timeout=2.0)
        unblock_latency = time.monotonic() - flush_time
        thread.join(timeout=1.0)

        assert reader_done.is_set(), "Confirmed reader did not unblock on flush."
        assert unblock_latency < 0.50, (
            f"Confirmed reader took {unblock_latency*1000:.1f}ms after "
            "flush. Should unblock within 50ms — Condvar wakeup latency."
        )
        assert results_holder, "Confirmed reader produced no results."


# ─────────────────────────────────────────────────────────────────────────
# AT #3 — Confirmed mode honours its timeout
# ─────────────────────────────────────────────────────────────────────────


class TestConfirmedReadTimeout:
    """Confirmed read raises a typed error when timeout exceeds without flush."""

    def test_confirmed_read_with_short_timeout_raises_read_timeout(
        self, tmp_path
    ) -> None:
        engine = tardigrade_db.Engine.open_with_write_buffer(
            str(tmp_path / "engine"),
            max_batch_size=100,
            max_idle_ms=60000,
        )
        # Buffered write, never flushed.
        engine.mem_write_pack(
            owner=1,
            retrieval_key=_key(0),
            layer_payloads=_layer_payload(0),
            salience=50.0,
        )

        start = time.monotonic()
        with pytest.raises(Exception) as excinfo:
            engine.mem_read_pack(
                _key(0),
                k=5,
                owner=1,
                mode="confirmed",
                timeout_ms=50,
            )
        elapsed = time.monotonic() - start
        assert 0.04 < elapsed < 0.30, (
            f"Timeout fired after {elapsed*1000:.1f}ms; expected ~50ms. "
            "Either the wait isn't honouring the deadline or it's spinning."
        )
        # The error message must mention durability + timeout so a
        # consumer sees what happened. Exact exception type may be
        # PyRuntimeError (PyO3) wrapping the miette diagnostic.
        message = str(excinfo.value).lower()
        assert "durability" in message or "timeout" in message, (
            f"ReadTimeout error message should mention 'durability' or "
            f"'timeout' so consumers can diagnose; got: {excinfo.value!r}"
        )

    def test_confirmed_mode_without_timeout_ms_raises_value_error(
        self, tmp_path
    ) -> None:
        engine = tardigrade_db.Engine.open_with_write_buffer(
            str(tmp_path / "engine"),
            max_batch_size=100,
            max_idle_ms=60000,
        )
        # Confirmed mode + missing timeout_ms = programmer error,
        # not a runtime condition. Reject loudly.
        with pytest.raises((ValueError, TypeError)):
            engine.mem_read_pack(
                _key(0),
                k=5,
                owner=1,
                mode="confirmed",
            )


# ─────────────────────────────────────────────────────────────────────────
# AT #4 — Multiple confirmed waiters all unblock on one flush
# ─────────────────────────────────────────────────────────────────────────


class TestConcurrentConfirmedReadersCoalesce:
    """Three confirmed readers waiting on the same offset all unblock on one flush."""

    def test_three_readers_unblock_on_single_flush(self, tmp_path) -> None:
        engine = tardigrade_db.Engine.open_with_write_buffer(
            str(tmp_path / "engine"),
            max_batch_size=100,
            max_idle_ms=60000,
        )
        engine.mem_write_pack(
            owner=1,
            retrieval_key=_key(0),
            layer_payloads=_layer_payload(0),
            salience=50.0,
        )

        n_readers = 3
        results: list = [None] * n_readers
        errors: list = [None] * n_readers
        ready = threading.Barrier(n_readers + 1)  # +1 for the flusher

        def reader(idx: int) -> None:
            ready.wait(timeout=2.0)
            try:
                results[idx] = engine.mem_read_pack(
                    _key(0),
                    k=5,
                    owner=1,
                    mode="confirmed",
                    timeout_ms=5000,
                )
            except Exception as exc:  # noqa: BLE001
                errors[idx] = exc

        threads = [
            threading.Thread(target=reader, args=(i,)) for i in range(n_readers)
        ]
        for t in threads:
            t.start()
        # Wait for every reader to be queued at the barrier.
        ready.wait(timeout=2.0)
        # Give all three a moment to actually enter the wait.
        time.sleep(0.05)
        engine.flush_buffer()
        for t in threads:
            t.join(timeout=2.0)

        assert all(e is None for e in errors), (
            f"Confirmed readers raised: {errors}. Either Condvar didn't "
            "wake all waiters or a flush-induced error leaked."
        )
        assert all(r is not None for r in results), (
            "Some confirmed readers never returned. Likely a missed wake."
        )


# ─────────────────────────────────────────────────────────────────────────
# AT (foundation) — Engine.durable_offset is monotonically non-decreasing
# ─────────────────────────────────────────────────────────────────────────


class TestDurableOffsetMonotonicity:
    """``Engine.durable_offset()`` is publicly accessible and only advances."""

    def test_durable_offset_does_not_decrease_across_writes(self, tmp_path) -> None:
        engine = tardigrade_db.Engine(str(tmp_path / "engine"))
        prev = engine.durable_offset()
        for i in range(5):
            engine.mem_write_pack(
                owner=1,
                retrieval_key=_key(i),
                layer_payloads=_layer_payload(i),
                salience=50.0,
            )
            engine.flush()
            current = engine.durable_offset()
            assert current >= prev, (
                f"durable_offset regressed: {prev} → {current} after "
                f"write {i}. Must be monotonically non-decreasing."
            )
            prev = current
