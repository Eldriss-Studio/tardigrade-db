"""Sweep thread counts and batch sizes to map the parallelism ceiling.

Diagnostic. Reports throughput at N ∈ {1, 2, 4, 8, 16} threads, both
with single-query calls and with the batch API.
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

import numpy as np

import tardigrade_db


CORPUS_OWNERS = (1, 2, 3)
CORPUS_PACKS_PER_OWNER = 64
KEY_DIM = 64
VALUE_DIM = 128
QUERIES_PER_THREAD = 50
TOP_K = 5
THREAD_COUNTS = [1, 2, 4, 8, 16]


def seeded_engine() -> tardigrade_db.Engine:
    root = Path(tempfile.mkdtemp(prefix="tardigrade-scaling-"))
    engine = tardigrade_db.Engine(str(root / "engine"))
    rng = np.random.default_rng(seed=0)
    for owner in CORPUS_OWNERS:
        for _ in range(CORPUS_PACKS_PER_OWNER):
            key = rng.standard_normal(KEY_DIM).astype(np.float32)
            value = rng.standard_normal(VALUE_DIM).astype(np.float32)
            engine.mem_write_pack(owner, key, [(0, value)], 50.0)
    return engine


def measure_threaded(engine, n_threads: int, batch_size: int) -> float:
    """Returns total qps across n_threads, each doing QUERIES_PER_THREAD
    queries in batches of batch_size."""
    rng = np.random.default_rng(seed=99)
    queries = [
        rng.standard_normal(KEY_DIM).astype(np.float32)
        for _ in range(n_threads)
    ]

    def reader(q: np.ndarray) -> None:
        n_batches = QUERIES_PER_THREAD // batch_size
        for _ in range(n_batches):
            if batch_size == 1:
                engine.mem_read_pack(q, TOP_K, None)
            else:
                engine.mem_read_pack_batch([q] * batch_size, TOP_K, None)

    threads = [threading.Thread(target=reader, args=(q,)) for q in queries]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start
    total_queries = n_threads * QUERIES_PER_THREAD
    return total_queries / elapsed


def main() -> int:
    engine = seeded_engine()

    # Warmup so tier transitions settle.
    rng = np.random.default_rng(seed=99)
    warmup_q = rng.standard_normal(KEY_DIM).astype(np.float32)
    for _ in range(200):
        engine.mem_read_pack(warmup_q, TOP_K, None)

    print(f"{'threads':>8} | {'b=1 qps':>12} | {'b=10 qps':>12} | {'b=50 qps':>12}")
    print("-" * 56)
    baseline_single_b1 = None
    for n in THREAD_COUNTS:
        b1 = measure_threaded(engine, n, 1)
        b10 = measure_threaded(engine, n, 10)
        b50 = measure_threaded(engine, n, 50)
        if n == 1:
            baseline_single_b1 = b1
        print(f"{n:>8} | {b1:>12,.0f} | {b10:>12,.0f} | {b50:>12,.0f}")

    print()
    print("Speedup vs 1-thread b=1 baseline:")
    for n in THREAD_COUNTS:
        b1 = measure_threaded(engine, n, 1)
        b50 = measure_threaded(engine, n, 50)
        print(
            f"  {n:>2} threads — b=1: {b1/baseline_single_b1:.2f}x | "
            f"b=50: {b50/baseline_single_b1:.2f}x"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
