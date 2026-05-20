"""Microbenchmark for `Engine.list_packs_metadata` at 10K packs.

Asserts an absolute latency budget on the columnar metadata path. The
budget reflects that `list_packs_metadata` answers from `PackDirectory`'s
in-memory indices (`pack_id -> owner`, `pack_id -> cell_ids`) plus
`governance`, with zero `BlockPool::get` calls. At this scale a healthy
implementation completes in single-digit milliseconds.

For context the legacy `list_packs(fetch_text=True)` path is timed too —
its extra cost over the metadata path is per-pack `pack_text()` plus the
per-row Python dict allocation.

Run from the repo root:
    source .venv/bin/activate
    python experiments/list_packs_microbench.py

Writes a JSON record to docs/perf/list-packs.json on success.
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

        metadata_ms = metadata_secs * 1000
        result = {
            "topic": "list_packs metadata absolute latency",
            "n_packs": N_PACKS,
            "n_runs": N_RUNS,
            "legacy_seconds_median": legacy_secs,
            "metadata_seconds_median": metadata_secs,
            "metadata_ms_median": metadata_ms,
            "metadata_to_legacy_ratio": speedup,
            "pass_criterion_metadata_ms": 10.0,
            "passed": metadata_ms < 10.0,
        }

        out_path = Path("docs/perf/list-packs.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")

        print()
        print(f"  legacy   (fetch_text=True): {legacy_secs * 1000:.2f} ms")
        print(f"  metadata (no text):         {metadata_ms:.2f} ms")
        print(f"  ratio (legacy / metadata):  {speedup:.2f}×")
        print(f"  pass (metadata < 10ms):     {'YES' if result['passed'] else 'NO'}")
        print(f"  wrote: {out_path}")

        if not result["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
