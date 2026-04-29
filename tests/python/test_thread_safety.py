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
