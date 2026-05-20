"""Phase 1 bench gate — `list_packs_metadata` vs `list_packs(fetch_text=True)`.

The first attempt at this phase (text-fetch skip only) returned a 1.27× speedup
at 10K packs, well below the original 3× target. The bottleneck turned out to
be per-row Python dict construction at the PyO3 boundary, not the text fetch.
The current implementation uses a columnar metadata path (four parallel numpy
arrays via `PyArray1::from_slice`), which avoids the dict allocations entirely.
The revised target is ≥ 8× speedup at 10K packs.

Run from the repo root:
    source .venv/bin/activate
    python experiments/phase1-list-packs-microbench.py

Writes a JSON record to docs/perf/phase1-list-packs.json on success.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np

import tardigrade_db

N_PACKS = 10_000
DIM = 8
OWNER = 1
N_RUNS = 5  # take the median to suppress jitter


def _setup_engine(path: Path) -> tardigrade_db.Engine:
    engine = tardigrade_db.Engine(str(path), vamana_threshold=99_999)
    rng = np.random.default_rng(0)
    for i in range(N_PACKS):
        key = rng.standard_normal(DIM).astype(np.float32)
        value = rng.standard_normal(DIM).astype(np.float32)
        engine.mem_write_pack(
            OWNER,
            key,
            [(0, value)],
            70.0,
            text=f"fact number {i} — Sonia translated a German pharmaceutical patent",
        )
    return engine


def _time(fn) -> float:
    runs = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        fn()
        runs.append(time.perf_counter() - start)
    runs.sort()
    return runs[len(runs) // 2]  # median


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        print(f"setting up {N_PACKS} packs (this takes ~60-90s on SSD)...")
        engine = _setup_engine(Path(tmp))

        print(f"timing list_packs(fetch_text=True) × {N_RUNS} runs...")
        legacy_secs = _time(lambda: engine.list_packs(OWNER, fetch_text=True))

        print(f"timing list_packs_metadata() × {N_RUNS} runs...")
        metadata_secs = _time(lambda: engine.list_packs_metadata(OWNER))

        speedup = legacy_secs / metadata_secs if metadata_secs > 0 else float("inf")

        result = {
            "phase": 1,
            "topic": "list_packs metadata vs full",
            "n_packs": N_PACKS,
            "n_runs": N_RUNS,
            "legacy_seconds_median": legacy_secs,
            "metadata_seconds_median": metadata_secs,
            "speedup": speedup,
            "pass_criterion_speedup": 8.0,
            "passed": speedup >= 8.0,
        }

        out_path = Path("docs/perf/phase1-list-packs.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")

        print()
        print(f"  legacy   (fetch_text=True): {legacy_secs * 1000:.2f} ms")
        print(f"  metadata (no text):         {metadata_secs * 1000:.2f} ms")
        print(f"  speedup:                    {speedup:.2f}×")
        print(f"  pass (≥ 8×):                {'YES' if result['passed'] else 'NO'}")
        print(f"  wrote: {out_path}")

        if not result["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
