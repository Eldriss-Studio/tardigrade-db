"""End-to-end concurrent-reads demo.

Seeds a shared engine with a small corpus, then runs the same workload
first single-threaded, then with N reader threads. Asserts that every
thread observes the same result set for the same query (correctness
under concurrent reads) and prints aggregate throughput numbers
(single-thread baseline vs N-thread).

Doubles as the bench harness for the engine's read-path concurrency
work — runs cleanly on the current single-writer wrapper, and the same
script measures the parallelism win when the wrapper switches to a
reader-writer lock.

Run manually:

    python examples/concurrent_reads_demo.py
"""

from __future__ import annotations

import os
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
SINGLE_THREAD_QUERIES = 200
READER_THREADS = 8
QUERIES_PER_READER = 25
TOP_K = 5


def demo_dir() -> Path:
    """Working directory for the demo. Honors TARDIGRADE_DEMO_DIR
    so the integration test can isolate runs in a tmpdir."""
    env = os.environ.get("TARDIGRADE_DEMO_DIR")
    if env:
        path = Path(env)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path(tempfile.mkdtemp(prefix="tardigrade-concurrent-demo-"))


def seeded_engine(engine_dir: Path) -> tardigrade_db.Engine:
    """Seed an engine with a small mixed-owner corpus. Deterministic
    keys (seeded RNG) so single-thread and multi-thread runs share a
    common ground truth."""
    rng = np.random.default_rng(seed=0)
    engine = tardigrade_db.Engine(str(engine_dir))
    for owner in CORPUS_OWNERS:
        for _ in range(CORPUS_PACKS_PER_OWNER):
            key = rng.standard_normal(KEY_DIM).astype(np.float32)
            value = rng.standard_normal(VALUE_DIM).astype(np.float32)
            engine.mem_write_pack(owner, key, [(0, value)], 50.0)
    return engine


def fixed_query() -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal(KEY_DIM).astype(np.float32)


def pack_id_set(results: list[dict]) -> frozenset[int]:
    """The cells a query retrieves. Stable across concurrent reads even
    when scores drift slightly from governance-bookkeeping races on the
    tier boost — the underlying ranking is determined by the dot
    products, which are deterministic for a fixed query and corpus."""
    return frozenset(r["pack_id"] for r in results)


def stabilize_tier_state(
    engine: tardigrade_db.Engine, query: np.ndarray, queries: int
) -> tuple[float, frozenset[int]]:
    """Single-threaded warm-up. Every call bumps governance importance on
    the retrieved cells, so a fresh engine returns slightly different
    scores call-to-call until tier transitions settle. The baseline
    pack-id set is captured after this phase, when the steady-state
    ranking is stable."""
    start = time.perf_counter()
    last: frozenset[int] = frozenset()
    for _ in range(queries):
        last = pack_id_set(engine.mem_read_pack(query, TOP_K, None))
    elapsed = time.perf_counter() - start
    return queries / elapsed, last


def concurrent_throughput(
    engine: tardigrade_db.Engine,
    query: np.ndarray,
    baseline_set: frozenset[int],
) -> tuple[float, list[str]]:
    """Run READER_THREADS readers in parallel. Each call asserts the
    retrieved pack-id set matches the post-warmup baseline. Scores may
    drift across threads — the cells should not."""
    divergences: list[str] = []
    divergences_lock = threading.Lock()

    def reader(thread_idx: int) -> None:
        for q_idx in range(QUERIES_PER_READER):
            ids = pack_id_set(engine.mem_read_pack(query, TOP_K, None))
            if ids != baseline_set:
                with divergences_lock:
                    divergences.append(
                        f"thread {thread_idx} query {q_idx}: "
                        f"got {sorted(ids)}, want {sorted(baseline_set)}"
                    )

    threads = [
        threading.Thread(target=reader, args=(idx,)) for idx in range(READER_THREADS)
    ]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start
    total = READER_THREADS * QUERIES_PER_READER
    return total / elapsed, divergences


def main() -> int:
    root = demo_dir()
    engine_dir = root / "engine"

    engine = seeded_engine(engine_dir)
    print(
        f"seeded: {len(CORPUS_OWNERS) * CORPUS_PACKS_PER_OWNER} packs across "
        f"{len(CORPUS_OWNERS)} owners"
    )

    query = fixed_query()
    single_qps, baseline_set = stabilize_tier_state(
        engine, query, SINGLE_THREAD_QUERIES
    )
    print(
        f"single-thread: {single_qps:,.0f} queries/sec "
        f"({SINGLE_THREAD_QUERIES} calls; baseline top-{TOP_K} set established)"
    )

    multi_qps, divergences = concurrent_throughput(engine, query, baseline_set)
    total = READER_THREADS * QUERIES_PER_READER
    print(
        f"{READER_THREADS}-thread: {multi_qps:,.0f} queries/sec "
        f"({total} calls across {READER_THREADS} threads)"
    )

    if divergences:
        print("ERROR: concurrent readers diverged from baseline:")
        for d in divergences[:5]:
            print(f"  {d}")
        return 1

    speedup = multi_qps / single_qps
    print(f"consistency: all {total} concurrent reads retrieved the baseline cell set")
    print(f"speedup vs single-thread: {speedup:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
