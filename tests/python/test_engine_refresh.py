# ATDD Layer 2 for Step 5 (Gap 7) — Python binding for Engine.refresh().
#
# Pattern under test: PyO3 wrapper. Exposes the Rust Engine::refresh()
# (Memento re-application) via the same `tardigrade_db.Engine` Python
# class. Errors map to PyRuntimeError.
#
# These tests MUST fail with AttributeError until Step 5b lands the
# PyO3 binding. They are CPU-only and need no vLLM / GPU.

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db


def _make_pack_payload(seed: float, kv_dim: int = 8, seq_len: int = 4):
    """Tiny synthetic pack — numeric data, not prose. Sufficient for
    testing API mechanics; semantic correctness is the GPU test's job."""
    half = seq_len * kv_dim
    return np.full(2 * half, seed, dtype=np.float32)


def test_engine_refresh_sees_writes_from_other_handle(tmp_path):
    """GIVEN two Python Engine handles at the same path,
    WHEN handle A writes a pack and handle B calls refresh(),
    THEN B.pack_count() reflects the write.
    """
    db = str(tmp_path / "shared")
    a = tardigrade_db.Engine(db)
    b = tardigrade_db.Engine(db)
    assert a.pack_count() == 0
    assert b.pack_count() == 0

    key = np.ones(8, dtype=np.float32)
    payload = _make_pack_payload(0.5)
    pid = a.mem_write_pack(1, key, [(0, payload)], 80.0)

    assert b.pack_count() == 0, "B should be stale before refresh"
    b.refresh()
    assert b.pack_count() == 1, "B should see A's write after refresh"
    assert b.pack_exists(pid)


def test_engine_refresh_is_no_op_when_nothing_changed(tmp_path):
    """No external writes → refresh is cheap and a no-op.

    Calling refresh() on a freshly-opened engine that nobody else has
    touched must not raise and must leave pack_count unchanged.
    """
    db = str(tmp_path / "noop")
    a = tardigrade_db.Engine(db)
    a.refresh()  # must not raise
    assert a.pack_count() == 0

    # Still works after a self-write
    key = np.ones(8, dtype=np.float32)
    payload = _make_pack_payload(0.7)
    a.mem_write_pack(1, key, [(0, payload)], 80.0)
    count_before = a.pack_count()
    a.refresh()
    assert a.pack_count() == count_before, "second no-op refresh must be safe"


def test_engine_refresh_after_local_writes_keeps_local_state(tmp_path):
    """Refreshing your own handle must not erase your own in-flight state.

    Edge case: refresh implementation must merge new disk state INTO the
    existing in-memory state, not replace it wholesale. Otherwise calling
    refresh() right after a self-write would be a footgun.
    """
    db = str(tmp_path / "self")
    a = tardigrade_db.Engine(db)
    key = np.ones(8, dtype=np.float32)
    payload = _make_pack_payload(0.9)
    pid = a.mem_write_pack(1, key, [(0, payload)], 80.0)

    a.refresh()
    assert a.pack_exists(pid), "self-refresh must not erase own writes"
    assert a.pack_count() == 1
