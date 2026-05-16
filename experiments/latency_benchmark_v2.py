#!/usr/bin/env python3
"""Latency benchmark v2 — GPU runner over the v2 primitives.

Wires the Strategy + Template Method primitives in
``tdb_bench.latency.bench_v2`` to real Qwen3-0.6B + the engine, then
writes the report JSON consumed by
``docs/positioning/latency_first.md`` (Track A slice A3).

Usage::

    PYTHONPATH=python python experiments/latency_benchmark_v2.py \\
        --output target/latency-bench-v2.json

Prerequisites: Qwen3-0.6B (or the env-var override), CUDA optional.

The harness:

1. Ingests N synthetic memory cells into a fresh engine (timed).
2. Runs the warm-up workload (cached, not measured).
3. Runs the measured workload — N queries, per-query latency captured.
4. Computes p50/p95/p99 + recall@5 per scale.
5. Writes the report JSON.

Compare to the legacy ``experiments/latency_benchmark.py`` which
hardcoded a single Vamana-vs-brute-force diff at fixed scale; v2 is
extensible (add new corpus sizes in :data:`CORPUS_SIZES` and rerun).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "python"))
sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402

import tardigrade_db  # noqa: E402
from tdb_bench.latency.bench_v2 import (  # noqa: E402
    CORPUS_SIZES,
    LatencyBenchRunner,
    WorkloadResult,
)


_DEFAULT_HIDDEN_DIM = 1024  # Qwen3-0.6B; override via env if swapping models.
_DEFAULT_OUTPUT_PATH = "target/latency-bench-v2.json"


class _SyntheticEngineIngestProbe:
    """Ingests random keys/values into a fresh engine each call.

    Tears down the previous engine before each call so each corpus
    size starts cold — the harness measures full ingest time, not
    incremental.
    """

    def __init__(self, hidden_dim: int = _DEFAULT_HIDDEN_DIM) -> None:
        self._hidden_dim = hidden_dim
        self.engine: object | None = None
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self._keys: list[np.ndarray] = []

    def ingest(self, count: int) -> float:
        # Fresh engine per call.
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="tdb_latbench_")
        self.engine = tardigrade_db.Engine(self._tmpdir.name)
        self._keys = []

        rng = np.random.default_rng(seed=42)
        start = time.perf_counter()
        for _ in range(count):
            k = rng.standard_normal(self._hidden_dim).astype(np.float32)
            v = rng.standard_normal(16).astype(np.float32)
            self._keys.append(k)
            self.engine.mem_write(
                owner=1, layer=0, key=k, value=v, salience=1.0, parent_cell_id=None
            )
        self.engine.flush()
        return time.perf_counter() - start


class _SyntheticEngineQueryWorkload:
    """Replays ingested keys as queries; measures latency + recall@5."""

    def __init__(self, engine, keys: list[np.ndarray], num_queries: int) -> None:
        self._engine = engine
        self._keys = keys
        self._n = num_queries

    def run(self) -> WorkloadResult:
        if not self._keys or self._n == 0:
            return WorkloadResult(latencies_ms=[], recall_at_k=0.0)
        rng = np.random.default_rng(seed=0)
        hits = 0
        latencies_ms: list[float] = []
        for _ in range(self._n):
            idx = int(rng.integers(0, len(self._keys)))
            q = self._keys[idx]
            t0 = time.perf_counter()
            results = self._engine.mem_read(query_key=q, k=5, owner=1)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            # Recall@5: original cell id is `idx` (0-indexed in ingest order).
            cell_ids = [r.cell_id if hasattr(r, "cell_id") else r[0] for r in results]
            if idx in cell_ids:
                hits += 1
        return WorkloadResult(latencies_ms=latencies_ms, recall_at_k=hits / self._n)


def _make_workload_factory(probe: _SyntheticEngineIngestProbe):
    from tdb_bench.latency.bench_v2 import MEASURED_QUERIES, WARMUP_QUERIES

    def _factory(corpus_size: int, *, phase: str = "measured") -> _SyntheticEngineQueryWorkload:
        n = WARMUP_QUERIES if phase == "warmup" else MEASURED_QUERIES
        assert probe.engine is not None, "ingest() must run before queries"
        return _SyntheticEngineQueryWorkload(
            engine=probe.engine, keys=probe._keys, num_queries=n  # noqa: SLF001
        )

    return _factory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=_DEFAULT_OUTPUT_PATH, help="JSON output path"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=int(os.getenv("TDB_LATBENCH_HIDDEN_DIM", str(_DEFAULT_HIDDEN_DIM))),
        help="Synthetic key dimension (defaults to Qwen3-0.6B's 1024).",
    )
    parser.add_argument(
        "--scales",
        nargs="*",
        type=int,
        default=None,
        help=f"Corpus sizes to sweep (default: {CORPUS_SIZES}).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    probe = _SyntheticEngineIngestProbe(hidden_dim=args.hidden_dim)
    factory = _make_workload_factory(probe)
    runner = LatencyBenchRunner(
        ingest_probe=probe,
        workload_factory=factory,
        corpus_sizes=args.scales if args.scales else CORPUS_SIZES,
    )

    print(f"Running latency-bench-v2 across {runner._sizes}...")  # noqa: SLF001
    report = runner.run()
    out_path = Path(args.output).resolve()
    report.write_json(out_path)

    # Print a summary table to stdout.
    print("\n┌─────────┬─────────────┬───────────┬─────────┬─────────┬─────────┐")
    print("│ scale   │ ingest_secs │ recall@5  │ p50_ms  │ p95_ms  │ p99_ms  │")
    print("├─────────┼─────────────┼───────────┼─────────┼─────────┼─────────┤")
    for s in report.scales:
        pct = s.latency_percentiles_ms
        print(
            f"│ {s.corpus_size:>7} │ {s.ingest_seconds:>11.2f} │ "
            f"{s.recall_at_k:>9.4f} │ "
            f"{pct['p50']:>7.2f} │ {pct['p95']:>7.2f} │ {pct['p99']:>7.2f} │"
        )
    print("└─────────┴─────────────┴───────────┴─────────┴─────────┴─────────┘")
    print(f"\nReport written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
