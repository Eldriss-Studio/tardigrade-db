"""ATDD: Python binding parity for the durable action scheduler.

The Rust acceptance suite at
``crates/tdb-engine/tests/scheduler_acceptance.rs`` covers the
semantics end-to-end. This file pins that the PyO3 surface
matches: schedule / cancel / list / fire / persistence across
reopen.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

LOW_SALIENCE = 5.0
EVICTION_THRESHOLD = 15.0


def _write_low_pack(engine, owner: int, marker: float = 1.0) -> int:
    key = np.array([marker, 0.0, 0.0, 0.0], dtype=np.float32)
    layers = [
        (0, np.array([marker] * 64, dtype=np.float32)),
        (1, np.array([marker] * 64, dtype=np.float32)),
    ]
    return engine.mem_write_pack(owner, key, layers, LOW_SALIENCE, None)


@pytest.fixture
def engine_path(tmp_path):
    return tmp_path / "engine"


class TestSchedulerBinding:
    def test_empty_schedule_lists_nothing(self, engine_path):
        import tardigrade_db
        engine = tardigrade_db.Engine(str(engine_path))
        assert engine.list_scheduled() == []

    def test_schedule_returns_id_and_appears_in_list(self, engine_path):
        import tardigrade_db
        engine = tardigrade_db.Engine(str(engine_path))
        future = time.time() + 60
        sid = engine.schedule_evict_draft(future, owner=7, threshold=EVICTION_THRESHOLD)
        assert isinstance(sid, int)
        entries = engine.list_scheduled()
        assert len(entries) == 1
        assert entries[0]["id"] == sid
        assert entries[0]["action"]["type"] == "evict_draft"
        assert entries[0]["action"]["owner"] == 7

    def test_fire_due_runs_eviction_and_removes_entry(self, engine_path):
        import tardigrade_db
        engine = tardigrade_db.Engine(str(engine_path))
        _write_low_pack(engine, 1, 1.0)
        _write_low_pack(engine, 1, 1.1)

        past = time.time() - 1
        engine.schedule_evict_draft(past, owner=1, threshold=EVICTION_THRESHOLD)

        fired = engine.fire_due_scheduled()
        assert fired == 1
        assert engine.pack_count() == 0
        assert engine.list_scheduled() == []

    def test_cancel_removes_and_lists_shrink(self, engine_path):
        import tardigrade_db
        engine = tardigrade_db.Engine(str(engine_path))
        future = time.time() + 60
        sid_a = engine.schedule_evict_draft(future, owner=1, threshold=EVICTION_THRESHOLD)
        engine.schedule_evict_draft(future, owner=2, threshold=EVICTION_THRESHOLD)
        assert engine.cancel_scheduled(sid_a) is True
        assert engine.cancel_scheduled(99999) is False
        remaining = engine.list_scheduled()
        assert len(remaining) == 1
        assert remaining[0]["action"]["owner"] == 2

    def test_schedule_survives_engine_reopen(self, engine_path):
        import tardigrade_db
        future = time.time() + 60
        engine = tardigrade_db.Engine(str(engine_path))
        engine.schedule_evict_draft(future, owner=42, threshold=EVICTION_THRESHOLD)
        del engine

        reopened = tardigrade_db.Engine(str(engine_path))
        entries = reopened.list_scheduled()
        assert len(entries) == 1
        assert entries[0]["action"]["owner"] == 42
        assert entries[0]["action"]["type"] == "evict_draft"
