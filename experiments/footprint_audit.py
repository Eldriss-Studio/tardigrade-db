#!/usr/bin/env python3
"""Footprint growth audit — Track A slice A2 GPU runner.

Ingests synthetic memory cells at the sample points in
:data:`FOOTPRINT_SAMPLE_POINTS` and snapshots engine arena bytes +
process RSS at each point. Writes the JSON the positioning doc
(slice A3) consumes.

Usage::

    PYTHONPATH=python python experiments/footprint_audit.py \\
        --output target/footprint-audit.json

The synthetic key uses 1024-dim float32 (matches Qwen3-0.6B's
hidden size) with a 16-dim value tensor. ``--hidden-dim`` lets you
sweep different capture models.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "python"))

import numpy as np  # noqa: E402

import tardigrade_db  # noqa: E402
from tdb_bench.footprint import (  # noqa: E402
    FOOTPRINT_SAMPLE_POINTS,
    FootprintReporter,
    GrowthCurve,
)


_DEFAULT_HIDDEN_DIM = 1024
_DEFAULT_VALUE_DIM = 16
_DEFAULT_OUTPUT_PATH = "target/footprint-audit.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=_DEFAULT_OUTPUT_PATH, help="JSON output path"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=int(os.getenv("TDB_FOOTPRINT_HIDDEN_DIM", str(_DEFAULT_HIDDEN_DIM))),
        help="Synthetic key dimension (defaults to Qwen3-0.6B's 1024).",
    )
    parser.add_argument(
        "--sample-points",
        nargs="*",
        type=int,
        default=None,
        help=f"Corpus sizes to sample (default: {FOOTPRINT_SAMPLE_POINTS}).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    sample_points = args.sample_points if args.sample_points else FOOTPRINT_SAMPLE_POINTS
    hidden_dim = args.hidden_dim

    with tempfile.TemporaryDirectory(prefix="tdb_footprint_audit_") as tmpdir:
        engine = tardigrade_db.Engine(tmpdir)
        reporter = FootprintReporter(engine)
        rng = np.random.default_rng(seed=42)

        def ingest_one(i: int) -> None:
            key = rng.standard_normal(hidden_dim).astype(np.float32)
            value = rng.standard_normal(_DEFAULT_VALUE_DIM).astype(np.float32)
            engine.mem_write(
                owner=1,
                layer=0,
                key=key,
                value=value,
                salience=1.0,
                parent_cell_id=None,
            )
            # Flush every 100 writes to materialise segment bytes for
            # accurate arena measurement; final flush also at the end.
            if (i + 1) % 100 == 0:
                engine.flush()

        curve = GrowthCurve(reporter=reporter, sample_points=sample_points)
        print(f"Running footprint audit at scales {sample_points}...")
        curve.run(ingest_fn=ingest_one)
        engine.flush()

        out_path = Path(args.output).resolve()
        curve.write_json(out_path)

        # Pretty table.
        print(
            "\n┌────────────┬──────────────┬───────────┬──────────────┬────────────────┐"
        )
        print(
            "│ cells      │ arena_bytes  │ /cell     │ segments     │ process_rss    │"
        )
        print(
            "├────────────┼──────────────┼───────────┼──────────────┼────────────────┤"
        )
        for s in curve.snapshots:
            print(
                f"│ {s.cell_count:>10} │ {s.arena_bytes:>12} │ "
                f"{s.arena_bytes_per_cell:>9} │ {s.segment_count:>12} │ "
                f"{s.process_rss_bytes:>14} │"
            )
        print(
            "└────────────┴──────────────┴───────────┴──────────────┴────────────────┘"
        )
        print(f"\nReport written to: {out_path}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
