"""ATDD: Python binding parity for ``Engine.sweep_now``.

Mirrors the Rust acceptance suite at
``crates/tdb-engine/tests/sweep_now_acceptance.rs``.
"""

from __future__ import annotations

import numpy as np
import pytest

LOW_SALIENCE = 5.0
HIGH_SALIENCE = 95.0
EVICTION_THRESHOLD = 15.0


def _write_pack(engine, owner: int, marker: float, salience: float) -> int:
    key = np.array([marker, 0.0, 0.0, 0.0], dtype=np.float32)
    layers = [
        (0, np.array([marker] * 64, dtype=np.float32)),
        (1, np.array([marker] * 64, dtype=np.float32)),
    ]
    return engine.mem_write_pack(owner, key, layers, salience, None)


@pytest.fixture
def engine(tmp_path):
    import tardigrade_db
    return tardigrade_db.Engine(str(tmp_path / "engine"))


class TestSweepNow:
    def test_empty_engine_returns_zero(self, engine):
        assert engine.sweep_now(0.0, EVICTION_THRESHOLD) == 0

    def test_evicts_low_importance_draft_packs(self, engine):
        low = _write_pack(engine, 1, 1.0, LOW_SALIENCE)
        high = _write_pack(engine, 1, 1.1, HIGH_SALIENCE)

        evicted = engine.sweep_now(0.0, EVICTION_THRESHOLD)
        assert evicted == 1
        assert not engine.pack_exists(low)
        assert engine.pack_exists(high)

    def test_idempotent_after_first_call(self, engine):
        _write_pack(engine, 1, 1.0, LOW_SALIENCE)
        _write_pack(engine, 1, 1.1, HIGH_SALIENCE)

        first = engine.sweep_now(0.0, EVICTION_THRESHOLD)
        second = engine.sweep_now(0.0, EVICTION_THRESHOLD)
        assert (first, second) == (1, 0)

    def test_default_threshold_matches_maintenance_default(self, engine):
        """No-argument call uses the same defaults as the
        :class:`MaintenanceWorker` background sweep, so the
        synchronous and asynchronous paths behave the same way."""
        _write_pack(engine, 1, 1.0, LOW_SALIENCE)
        assert engine.sweep_now() == 1
