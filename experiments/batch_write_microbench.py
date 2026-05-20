"""Microbenchmark for `Engine.mem_write_batch_packs` vs sequential writes.

Both paths persist the same N packs. The sequential path issues one fsync
per pack via `mem_write_pack`. The batch path issues one fsync for all N
via `mem_write_batch_packs`. The wall-time ratio is dominated by fsync
cost, so the speedup is a direct measure of "how many fsyncs we collapsed".

Run from the repo root:
    source .venv/bin/activate
    python experiments/batch_write_microbench.py

Writes a JSON record to docs/perf/batch-write.json on success.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np

import tardigrade_db

DIM = 64
N_PACKS = 50
N_RUNS = 5
OWNER = 1


def _make_packs(seed_base: int):
    rng = np.random.default_rng(seed_base)
    packs = []
    for i in range(N_PACKS):
        key = rng.standard_normal(DIM).astype(np.float32)
        layer = rng.standard_normal(128).astype(np.float32)
        packs.append((OWNER, key, [(0, layer)], 70.0, f"fact {i}"))
    return packs


def _time_sequential() -> float:
    packs = _make_packs(seed_base=0)
    with tempfile.TemporaryDirectory() as tmp:
        engine = tardigrade_db.Engine(tmp, vamana_threshold=99_999)
        start = time.perf_counter()
        for owner, key, layers, salience, text in packs:
            engine.mem_write_pack(owner, key, layers, salience, text)
        return time.perf_counter() - start


def _time_batch() -> float:
    packs = _make_packs(seed_base=0)
    with tempfile.TemporaryDirectory() as tmp:
        engine = tardigrade_db.Engine(tmp, vamana_threshold=99_999)
        start = time.perf_counter()
        engine.mem_write_batch_packs(packs)
        return time.perf_counter() - start


def main() -> None:
    print(f"timing {N_PACKS} writes × {N_RUNS} runs (sequential vs batch)...", flush=True)
    sequential_runs = []
    batch_runs = []
    for r in range(N_RUNS):
        s = _time_sequential()
        b = _time_batch()
        sequential_runs.append(s)
        batch_runs.append(b)
        print(
            f"  run {r + 1}: sequential={s * 1000:.1f} ms  batch={b * 1000:.1f} ms  "
            f"({s / b if b > 0 else float('inf'):.2f}× faster)",
            flush=True,
        )

    sequential_runs.sort()
    batch_runs.sort()
    s_med = sequential_runs[len(sequential_runs) // 2]
    b_med = batch_runs[len(batch_runs) // 2]
    speedup = s_med / b_med if b_med > 0 else float("inf")

    result = {
        "topic": "batch vs sequential write wall time",
        "n_packs": N_PACKS,
        "n_runs": N_RUNS,
        "dim": DIM,
        "sequential_seconds_median": s_med,
        "batch_seconds_median": b_med,
        "speedup": speedup,
        "pass_criterion_speedup": 2.0,
        "passed": speedup >= 2.0,
    }

    out_path = Path("docs/perf/batch-write.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")

    print()
    print(f"  sequential median:  {s_med * 1000:.1f} ms")
    print(f"  batch median:       {b_med * 1000:.1f} ms")
    print(f"  speedup:            {speedup:.2f}×")
    print(f"  pass (≥ 2×):        {'YES' if result['passed'] else 'NO'}")
    print(f"  wrote: {out_path}")

    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
