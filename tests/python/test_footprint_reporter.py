"""ATDD: FootprintReporter.

`FootprintReporter` snapshots process-level RSS via psutil *and*
engine-level arena bytes via `engine.status()` so the positioning
doc can quote both axes from a single source. The growth-curve
audit (`experiments/footprint_audit.py`) ingests at sample points
and writes a Repository-shaped JSON output.

Slices covered: A2.1 (snapshot primitive), A2.2 (growth curve).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import tardigrade_db
from tdb_bench.footprint.reporter import (
    FOOTPRINT_SAMPLE_POINTS,
    FootprintReporter,
    FootprintSnapshot,
    GrowthCurve,
)


def _empty_engine() -> tuple[object, str]:
    tmp = tempfile.mkdtemp(prefix="tdb_footprint_reporter_test_")
    return tardigrade_db.Engine(tmp), tmp


def _write_cell(engine, key_seed: float = 1.0) -> None:
    key = np.array([key_seed] + [0.0] * 7, dtype=np.float32)
    value = np.ones(16, dtype=np.float32)
    engine.mem_write(
        owner=1, layer=0, key=key, value=value, salience=1.0, parent_cell_id=None
    )


# ─── A2.1 — snapshot primitive ────────────────────────────────────────────


class TestSnapshot:
    def test_snapshot_carries_engine_and_process_fields(self):
        engine, _ = _empty_engine()
        reporter = FootprintReporter(engine)
        snap = reporter.snapshot()

        assert isinstance(snap, FootprintSnapshot)
        # Engine-level fields (from EngineStatus, see A2.3 commit).
        assert snap.cell_count == 0
        assert snap.segment_count >= 0
        assert snap.arena_bytes >= 0
        assert snap.arena_bytes_per_cell == 0  # empty engine
        # Process-level RSS — measured at snapshot time, must be > 0
        # because the Python process exists.
        assert snap.process_rss_bytes > 0

    def test_arena_bytes_grows_between_snapshots(self):
        engine, _ = _empty_engine()
        reporter = FootprintReporter(engine)

        before = reporter.snapshot()
        for _ in range(5):
            _write_cell(engine)
        engine.flush()
        after = reporter.snapshot()

        assert after.cell_count > before.cell_count
        assert after.arena_bytes > before.arena_bytes

    def test_snapshot_to_dict_round_trip(self):
        engine, _ = _empty_engine()
        snap = FootprintReporter(engine).snapshot()
        d = snap.to_dict()
        for key in (
            "cell_count",
            "segment_count",
            "arena_bytes",
            "arena_bytes_per_cell",
            "process_rss_bytes",
        ):
            assert key in d


# ─── A2.2 — growth curve ──────────────────────────────────────────────────


class TestGrowthCurve:
    def test_records_snapshot_at_each_sample_point(self):
        engine, _ = _empty_engine()
        reporter = FootprintReporter(engine)

        # Use small sample points so the test stays under a second.
        curve = GrowthCurve(reporter=reporter, sample_points=[0, 5, 15])
        curve.run(ingest_fn=lambda i: _write_cell(engine, key_seed=float(i) + 1.0))

        # Three snapshots, in order, cell-counts monotonically increasing.
        assert len(curve.snapshots) == 3
        cell_counts = [s.cell_count for s in curve.snapshots]
        assert cell_counts == sorted(cell_counts)  # monotonic non-decreasing
        # First snapshot at 0 — engine still empty.
        assert curve.snapshots[0].cell_count == 0
        # Last at 15 — at least 15 cells (could be more if chunks expand).
        assert curve.snapshots[-1].cell_count >= 15

    def test_arena_bytes_monotonically_nondecreasing(self):
        engine, _ = _empty_engine()
        reporter = FootprintReporter(engine)

        curve = GrowthCurve(reporter=reporter, sample_points=[0, 10, 20])
        curve.run(ingest_fn=lambda i: _write_cell(engine, key_seed=float(i) + 1.0))

        arenas = [s.arena_bytes for s in curve.snapshots]
        # Arena must grow or stay flat; never shrink (no compaction in the path).
        assert all(a <= b for a, b in zip(arenas, arenas[1:]))

    def test_default_sample_points_pinned_to_constant(self):
        # Constant in the module is reused in the growth-audit script so
        # docs and bench stay aligned. Pin so a drift breaks the test.
        assert FOOTPRINT_SAMPLE_POINTS[0] == 0
        assert FOOTPRINT_SAMPLE_POINTS[-1] >= 100  # ends at least at small-corpus scale


# ─── Repository: GrowthCurve.write_json ───────────────────────────────────


class TestGrowthCurveSerialization:
    def test_json_round_trip(self, tmp_path: Path):
        engine, _ = _empty_engine()
        reporter = FootprintReporter(engine)
        curve = GrowthCurve(reporter=reporter, sample_points=[0, 5])
        curve.run(ingest_fn=lambda i: _write_cell(engine, key_seed=float(i) + 1.0))

        out = tmp_path / "footprint.json"
        curve.write_json(out)

        d = json.loads(out.read_text(encoding="utf-8"))
        assert "snapshots" in d
        assert len(d["snapshots"]) == 2
        for snap in d["snapshots"]:
            assert "cell_count" in snap
            assert "arena_bytes" in snap
            assert "process_rss_bytes" in snap
