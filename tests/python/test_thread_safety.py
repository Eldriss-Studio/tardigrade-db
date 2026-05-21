# ATDD tests for thread safety and GIL release.
#
# Validates that the Arc<Mutex<Engine>> wrapper allows concurrent Python
# threads to run while the engine computes, and that concurrent access
# produces no data corruption.

import tempfile
import threading
import time

import numpy as np
import pytest

import tardigrade_db


@pytest.fixture
def engine(tmp_path):
    return tardigrade_db.Engine(str(tmp_path))


def _random_key(dim=64):
    return np.random.randn(dim).astype(np.float32)


def _random_value(dim=64):
    return np.random.randn(dim).astype(np.float32)


class TestGILRelease:
    """ATDD: Engine operations release the GIL during computation."""

    def test_gil_released_during_mem_read(self, engine):
        """GIVEN an engine with 100 memories
        WHEN thread A calls mem_read (blocking in Rust)
        AND thread B increments a counter during that time
        THEN thread B's counter is > 0 (proving GIL was released)"""
        for _ in range(100):
            engine.mem_write(1, 0, _random_key(), _random_value(), 50.0, None)

        counter = {"value": 0}
        done = threading.Event()

        def count_loop():
            while not done.is_set():
                counter["value"] += 1
                time.sleep(0.001)

        t = threading.Thread(target=count_loop)
        t.start()

        engine.mem_read(_random_key(), 5, None)

        done.set()
        t.join()
        assert counter["value"] > 0, (
            "GIL was not released during mem_read — "
            "background thread could not run"
        )

    def test_gil_released_during_mem_write_pack(self, engine):
        """GIVEN an empty engine
        WHEN thread A calls mem_write_pack (blocking in Rust for fsync)
        AND thread B increments a counter during that time
        THEN thread B's counter is > 0"""
        counter = {"value": 0}
        done = threading.Event()

        def count_loop():
            while not done.is_set():
                counter["value"] += 1
                time.sleep(0.001)

        t = threading.Thread(target=count_loop)
        t.start()

        for _ in range(10):
            engine.mem_write_pack(
                1, _random_key(), [(0, _random_value(128))], 50.0
            )

        done.set()
        t.join()
        assert counter["value"] > 0, "GIL was not released during mem_write_pack"


class TestConcurrentAccess:
    """ATDD: Thread-safe engine access from multiple Python threads."""

    def test_concurrent_writes_no_data_loss(self, engine):
        """GIVEN a thread-safe engine
        WHEN thread A writes 25 packs with owner=1
        AND thread B writes 25 packs with owner=2 concurrently
        THEN engine.pack_count() == 50 after both threads complete"""

        def write_packs(owner, count):
            for _ in range(count):
                engine.mem_write_pack(
                    owner, _random_key(), [(0, _random_value(128))], 50.0,
                )

        t1 = threading.Thread(target=write_packs, args=(1, 25))
        t2 = threading.Thread(target=write_packs, args=(2, 25))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert engine.pack_count() == 50, (
            f"Expected 50 packs from concurrent writes, got {engine.pack_count()}"
        )

    def test_concurrent_read_and_write(self, engine):
        """GIVEN an engine with 50 pre-loaded packs
        WHEN thread A writes 25 more packs
        AND thread B reads continuously during writes
        THEN thread B never crashes and final pack count is 75"""
        for _ in range(50):
            engine.mem_write_pack(
                1, _random_key(), [(0, _random_value(128))], 50.0,
            )

        read_errors = []
        read_count = {"value": 0}

        def reader():
            while not done.is_set():
                try:
                    engine.mem_read(_random_key(), 3, None)
                    read_count["value"] += 1
                except Exception as e:
                    read_errors.append(str(e))

        def writer():
            for _ in range(25):
                engine.mem_write_pack(
                    2, _random_key(), [(0, _random_value(128))], 50.0,
                )

        done = threading.Event()
        reader_thread = threading.Thread(target=reader)
        writer_thread = threading.Thread(target=writer)

        reader_thread.start()
        writer_thread.start()
        writer_thread.join()
        done.set()
        reader_thread.join()

        assert not read_errors, f"Reader thread hit errors: {read_errors}"
        assert read_count["value"] > 0, "Reader thread never completed a read"
        assert engine.pack_count() == 75, (
            f"Expected 75 packs, got {engine.pack_count()}"
        )


class TestPackTextLockFree:
    """pack_text serves from a cached Arc<TextStore> handle outside the
    engine mutex; concurrent writers holding the mutex must not block it."""

    def test_pack_text_returns_while_writer_holds_engine(self, engine):
        """A writer thread hammering mem_write_pack (taking the engine mutex
        for each call) must not stall pack_text in another thread. With the
        lock-free path, pack_text serves in microseconds regardless of the
        writer's contention. Without it, pack_text would queue behind every
        mem_write_pack call and the latency would be in the milliseconds."""
        seeded_pack_id = engine.mem_write_pack(
            1, _random_key(), [(0, _random_value(128))], 50.0, text="seed",
        )
        assert engine.pack_text(seeded_pack_id) == "seed"

        writer_done = threading.Event()
        writer_started = threading.Event()

        def writer():
            writer_started.set()
            # Drive enough writes that the writer is contending for the
            # engine mutex throughout the reader's measurement window.
            for _ in range(50):
                engine.mem_write_pack(
                    2, _random_key(), [(0, _random_value(128))], 30.0,
                )
            writer_done.set()

        t = threading.Thread(target=writer)
        t.start()
        writer_started.wait()

        # Measure: pack_text under contention. Each call should be
        # essentially instant because it never touches the engine mutex.
        # A 5 ms ceiling per call is generous — the lock-free path is
        # sub-microsecond; the locked path on a hot engine sits in the
        # low milliseconds per call once queued.
        slowest = 0.0
        for _ in range(100):
            start = time.perf_counter()
            text = engine.pack_text(seeded_pack_id)
            elapsed = time.perf_counter() - start
            assert text == "seed"
            slowest = max(slowest, elapsed)

        t.join()
        assert writer_done.is_set()

        # Even a single observed call above 5 ms would suggest pack_text
        # had to queue behind a write. The whole point of caching the
        # text-store handle outside the mutex is to make that impossible.
        assert slowest < 0.005, (
            f"pack_text observed {slowest * 1000:.2f} ms worst-case while a "
            "writer held the engine — should be sub-millisecond if the "
            "lock-free path is wired correctly"
        )

    def test_pack_text_observes_concurrent_writes(self, engine):
        """The cached Arc<TextStore> handle observes every write the engine
        performs because both share the same ArcSwap-backed store. A reader
        that polls pack_text while a writer adds packs must eventually see
        each new text."""
        ids: list[int] = []
        ids_lock = threading.Lock()

        def writer():
            for i in range(20):
                pid = engine.mem_write_pack(
                    1, _random_key(), [(0, _random_value(128))], 50.0,
                    text=f"text-{i}",
                )
                with ids_lock:
                    ids.append(pid)

        t = threading.Thread(target=writer)
        t.start()

        observed: set[int] = set()
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            with ids_lock:
                snapshot = list(ids)
            for pid in snapshot:
                if pid in observed:
                    continue
                text = engine.pack_text(pid)
                if text is not None:
                    observed.add(pid)
            if not t.is_alive() and len(observed) == 20:
                break

        t.join()
        assert len(observed) == 20, (
            f"pack_text observed {len(observed)}/20 packs that the writer "
            "claims to have stored — the cached handle should see every "
            "engine write"
        )
