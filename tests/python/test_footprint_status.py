"""ATDD (Track A — A2.3, Python side):

`Engine.status()` surfaces `arena_bytes` and `arena_bytes_per_cell`
to the Python bench harness so the footprint reporter
(`experiments/footprint_audit.py`) can record growth curves.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import tardigrade_db


def _empty_engine() -> tuple[object, str]:
    tmp = tempfile.mkdtemp(prefix="tdb_footprint_test_")
    return tardigrade_db.Engine(tmp), tmp


class TestStatusReportsArenaFootprint:
    def test_status_dict_has_arena_keys(self):
        engine, _ = _empty_engine()
        s = engine.status()
        assert "arena_bytes" in s
        assert "arena_bytes_per_cell" in s

    def test_empty_engine_has_zero_per_cell(self):
        # cell_count == 0 -> per-cell average must be zero (guards
        # against div-by-zero leaking into bench-side calculations).
        engine, _ = _empty_engine()
        s = engine.status()
        assert s["cell_count"] == 0
        assert s["arena_bytes_per_cell"] == 0

    def test_arena_bytes_grows_after_write(self):
        engine, _ = _empty_engine()
        before = engine.status()["arena_bytes"]

        # Use the lowest-level write path the test infra has:
        # mem_write_pack accepts a KVPack-shaped argument. The Python
        # binding for it is exposed via the engine instance.
        # Use raw mem_write because it's the standard test vector.
        key = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        value = np.ones(16, dtype=np.float32)
        engine.mem_write(owner=1, layer=0, key=key, value=value, salience=1.0, parent_cell_id=None)
        engine.flush()

        after = engine.status()["arena_bytes"]
        assert after > before, f"arena did not grow: before={before} after={after}"

    def test_arena_bytes_per_cell_matches_division(self):
        engine, _ = _empty_engine()
        key = np.array([1.0] + [0.0] * 7, dtype=np.float32)
        value = np.ones(16, dtype=np.float32)
        engine.mem_write(owner=1, layer=0, key=key, value=value, salience=1.0, parent_cell_id=None)
        engine.mem_write(owner=1, layer=0, key=key, value=value, salience=1.0, parent_cell_id=None)
        engine.flush()

        s = engine.status()
        assert s["cell_count"] > 0
        # Exact division pinned in Rust; Python sees the same value.
        assert s["arena_bytes_per_cell"] == s["arena_bytes"] // s["cell_count"]
