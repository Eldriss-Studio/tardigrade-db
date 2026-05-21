"""ATDD acceptance tests for the Rust-native background maintenance worker.

The Python ``GovernanceSweepThread`` Active Object was removed in favour of
``Engine.start_maintenance()`` / ``stop_maintenance()`` — a single Rust
thread that runs governance sweep (decay + eviction) and segment compaction
with no GIL contention.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db


@pytest.fixture
def engine(tmp_path):
    return tardigrade_db.Engine(str(tmp_path))


def write_cell(engine, cell_id_hint=0, salience=50.0):
    """Write a cell and return its ID."""
    key = np.full(32, float(cell_id_hint), dtype=np.float32)
    return engine.mem_write(1, 0, key, np.zeros(32, dtype=np.float32), salience, None)


# ── ATDD Test 1: Maintenance runs automatically ──────────────────────────


def test_maintenance_decays_importance(engine):
    """Start maintenance with a short interval and an aggressive
    hours_per_tick. After waiting, the worker should have run at least
    one sweep and the cell's importance should have decayed."""
    cell_id = write_cell(engine, salience=10.0)
    importance_before = engine.cell_importance(cell_id)

    engine.start_maintenance(
        sweep_interval_secs=0.1,
        compaction_interval_secs=3600.0,
        eviction_threshold=0.0,
        hours_per_tick=24.0,
    )
    try:
        time.sleep(0.5)
    finally:
        engine.stop_maintenance()

    status = engine.maintenance_status()
    assert status["sweep_count"] >= 2, (
        f"Expected >=2 sweep cycles, got {status['sweep_count']}"
    )

    importance_after = engine.cell_importance(cell_id)
    assert importance_after < importance_before, (
        f"Importance should have decayed: "
        f"before={importance_before}, after={importance_after}"
    )


# ── ATDD Test 2: Promoted cells stay promoted ────────────────────────────


def test_maintenance_preserves_promoted_cells(engine):
    """Promoted cells should stay promoted while sweep runs."""
    key = np.ones(32, dtype=np.float32)
    cell_id = engine.mem_write(1, 0, key, np.zeros(32, dtype=np.float32), 50.0, None)

    # Boost to >=65 via reads: 55 + 4*3 = 67 -> Validated.
    for _ in range(4):
        engine.mem_read(key, 1, None)
    assert engine.cell_tier(cell_id) == 1  # Validated

    engine.start_maintenance(
        sweep_interval_secs=0.1,
        compaction_interval_secs=3600.0,
        eviction_threshold=0.0,
        hours_per_tick=0.01,
    )
    try:
        time.sleep(0.3)
    finally:
        engine.stop_maintenance()

    assert engine.cell_tier(cell_id) == 1  # Validated


# ── ATDD Test 3: Stops cleanly ───────────────────────────────────────────


def test_maintenance_stops_cleanly(engine):
    """Start maintenance, stop it. The worker should report not running."""
    engine.start_maintenance(
        sweep_interval_secs=0.1,
        compaction_interval_secs=3600.0,
    )
    assert engine.is_maintenance_running()

    engine.stop_maintenance()
    # stop_maintenance is best-effort synchronous; give the worker thread
    # a brief moment to wind down its current iteration.
    for _ in range(20):
        if not engine.is_maintenance_running():
            break
        time.sleep(0.05)
    assert not engine.is_maintenance_running()


# ── ATDD Test 4: Sweep does not corrupt state under concurrent writes ──


def test_maintenance_concurrent_safety(engine):
    """Run maintenance concurrently with writes/reads. No exceptions, no
    lost data — the engine's Arc<Mutex<>> serialises access.
    """
    engine.start_maintenance(
        sweep_interval_secs=0.05,
        compaction_interval_secs=3600.0,
        hours_per_tick=1.0,
    )

    errors = []
    try:
        for i in range(50):
            try:
                key = np.full(32, float(i), dtype=np.float32)
                engine.mem_write(1, 0, key, np.zeros(32, dtype=np.float32), 50.0, None)
                engine.mem_read(key, 3, None)
            except Exception as exc:  # noqa: BLE001 — collected for assertion
                errors.append(str(exc))
    finally:
        engine.stop_maintenance()

    assert errors == [], f"Errors during concurrent maintenance+writes: {errors}"
    assert engine.cell_count() == 50
