"""ATDD tests for background maintenance worker."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db
from tardigrade_hooks.encoding import encode_per_token

SWEEP_INTERVAL_FAST = 0.05
COMPACTION_INTERVAL_SLOW = 9999.0
AGGRESSIVE_DECAY_HOURS = 720.0
EVICTION_THRESHOLD = 14.0


def _store_pack(engine, owner, seed, salience=80.0):
    key = encode_per_token(
        np.array([[seed, 0.0, 0.0, 0.0]], dtype=np.float32), dim=4
    )
    return engine.mem_write_pack(
        owner, key, [(0, np.array([seed] * 16, dtype=np.float32))], salience
    )


def test_maintenance_enabled(tmp_path):
    """GIVEN an engine with maintenance started,
    WHEN we wait for a short interval,
    THEN sweep_count > 0."""
    engine = tardigrade_db.Engine(str(tmp_path))
    _store_pack(engine, 1, 1.0, salience=50.0)

    engine.start_maintenance(
        sweep_interval_secs=SWEEP_INTERVAL_FAST,
        compaction_interval_secs=COMPACTION_INTERVAL_SLOW,
        hours_per_tick=1.0,
    )
    time.sleep(0.3)
    engine.stop_maintenance()

    status = engine.maintenance_status()
    assert status["sweep_count"] >= 2, f"expected >= 2 sweeps, got {status['sweep_count']}"


def test_maintenance_status_dict(tmp_path):
    """GIVEN maintenance running,
    WHEN querying status,
    THEN all expected keys are present."""
    engine = tardigrade_db.Engine(str(tmp_path))
    engine.start_maintenance(sweep_interval_secs=SWEEP_INTERVAL_FAST)
    time.sleep(0.15)
    engine.stop_maintenance()

    status = engine.maintenance_status()
    expected_keys = {
        "sweep_count", "compaction_count", "total_packs_evicted",
        "total_bytes_reclaimed", "last_sweep_epoch_secs", "last_compaction_epoch_secs",
    }
    assert expected_keys == set(status.keys())


def test_maintenance_start_stop(tmp_path):
    """GIVEN an engine,
    WHEN start then stop maintenance,
    THEN is_maintenance_running transitions correctly."""
    engine = tardigrade_db.Engine(str(tmp_path))

    assert not engine.is_maintenance_running()

    engine.start_maintenance(sweep_interval_secs=1.0)
    assert engine.is_maintenance_running()

    engine.stop_maintenance()
    assert not engine.is_maintenance_running()


def test_maintenance_disabled_by_default(tmp_path):
    """GIVEN a default Engine,
    THEN maintenance is not running."""
    engine = tardigrade_db.Engine(str(tmp_path))
    assert not engine.is_maintenance_running()
    status = engine.maintenance_status()
    assert status["sweep_count"] == 0
