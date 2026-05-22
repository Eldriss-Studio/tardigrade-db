"""Concurrent reads with disjoint queries per thread.

Variant of concurrent_reads_demo.py that gives each thread a different
query. If governance Mutex contention on shared hot cells is the
binding constraint, this should scale better than the shared-query
demo. If 8-thread speedup here is still ~2.5x, then per-cell
contention isn't the bottleneck — something else is.

Diagnostic, not a release-shipped demo.
"""

from __future__ import annotations

import sys
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
READER_THREADS = 8
QUERIES_PER_READER = 25
TOP_K = 5


def seeded_engine(engine_dir: Path) -> tardigrade_db.Engine:
    rng = np.random.default_rng(seed=0)
    engine = tardigrade_db.Engine(str(engine_dir))
    for owner in CORPUS_OWNERS:
        for _ in range(CORPUS_PACKS_PER_OWNER):
            key = rng.standard_normal(KEY_DIM).astype(np.float32)
            value = rng.standard_normal(VALUE_DIM).astype(np.float32)
            engine.mem_write_pack(owner, key, [(0, value)], 50.0)
    return engine


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="tardigrade-disjoint-"))
    engine = seeded_engine(root / "engine")

    # Each thread gets a distinct query — different RNG seed per thread —
    # so the retrieved top-5 cells differ across threads.
    rng = np.random.default_rng(seed=99)
    thread_queries = [
        rng.standard_normal(KEY_DIM).astype(np.float32) for _ in range(READER_THREADS)
    ]

    # Single-thread baseline: run all 8 query sets sequentially.
    start = time.perf_counter()
    for q in thread_queries:
        for _ in range(QUERIES_PER_READER):
            engine.mem_read_pack(q, TOP_K, None)
    single_elapsed = time.perf_counter() - start
    single_total = READER_THREADS * QUERIES_PER_READER
    single_qps = single_total / single_elapsed
    print(
        f"single-thread (disjoint queries): {single_qps:,.0f} qps "
        f"({single_total} calls)"
    )

    # Multi-thread: each thread runs its own query.
    def reader(thread_query: np.ndarray) -> None:
        for _ in range(QUERIES_PER_READER):
            engine.mem_read_pack(thread_query, TOP_K, None)

    threads = [
        threading.Thread(target=reader, args=(q,)) for q in thread_queries
    ]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    multi_elapsed = time.perf_counter() - start
    multi_total = READER_THREADS * QUERIES_PER_READER
    multi_qps = multi_total / multi_elapsed
    print(
        f"{READER_THREADS}-thread (disjoint queries): {multi_qps:,.0f} qps "
        f"({multi_total} calls)"
    )

    speedup = multi_qps / single_qps
    print(f"speedup vs single-thread: {speedup:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
