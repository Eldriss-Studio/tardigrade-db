"""ATDD acceptance tests for background governance sweep (Active Object pattern)."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db
from tardigrade_hooks.sweep import GovernanceSweepThread


@pytest.fixture
def engine(tmp_path):
    return tardigrade_db.Engine(str(tmp_path))


def write_cell(engine, cell_id_hint=0, salience=50.0):
    """Write a cell and return its ID."""
    key = np.full(32, float(cell_id_hint), dtype=np.float32)
    return engine.mem_write(1, 0, key, np.zeros(32, dtype=np.float32), salience, None)


# ── ATDD Test 1: Sweep runs automatically ─────────────────────────────────


def test_sweep_runs_automatically(engine):
    """Start sweep with short interval. After waiting, tick_count > 0."""
    cell_id = write_cell(engine, salience=10.0)
    importance_before = engine.cell_importance(cell_id)

    # 0.1s interval, 24 hours per tick (= 1 full day of decay per tick).
    sweep = GovernanceSweepThread(engine, interval_secs=0.1, hours_per_tick=24.0)
    sweep.start()

    time.sleep(0.5)  # Let 4-5 ticks run.
    sweep.stop()

    assert sweep.tick_count >= 2, f"Expected ≥2 ticks, got {sweep.tick_count}"

    importance_after = engine.cell_importance(cell_id)
    assert importance_after < importance_before, (
        f"Importance should have decayed: before={importance_before}, after={importance_after}"
    )


# ── ATDD Test 2: Sweep promotes active cells ──────────────────────────────


def test_sweep_promotes_active_cells(engine):
    """Write cell, boost via reads, let sweep evaluate. Tier should promote."""
    key = np.ones(32, dtype=np.float32)
    cell_id = engine.mem_write(1, 0, key, np.zeros(32, dtype=np.float32), 50.0, None)

    # Initial: 50 + 5 (write) = 55 → Draft.
    assert engine.cell_tier(cell_id) == 0  # Draft

    # Boost to ≥65 via reads: 55 + 4×3 = 67 → Validated.
    for _ in range(4):
        engine.mem_read(key, 1, None)

    # Tier is already updated by mem_read's governance boost.
    assert engine.cell_tier(cell_id) == 1  # Validated

    # Start sweep — it should maintain the promoted tier (not revert).
    sweep = GovernanceSweepThread(engine, interval_secs=0.1, hours_per_tick=0.01)
    sweep.start()
    time.sleep(0.3)
    sweep.stop()

    # Tier should still be Validated (sweep doesn't undo promotion with tiny decay).
    assert engine.cell_tier(cell_id) == 1  # Validated


# ── ATDD Test 3: Sweep evicts stale cells ─────────────────────────────────


def test_sweep_evicts_stale_cells(engine):
    """Write cell with low salience. Many sweep cycles decay it below threshold."""
    cell_id = write_cell(engine, salience=10.0)

    # Run sweep with aggressive decay: each tick = 100 days.
    sweep = GovernanceSweepThread(engine, interval_secs=0.05, hours_per_tick=2400.0)
    sweep.start()
    time.sleep(0.3)  # ~6 ticks × 100 days = 600 days of decay
    sweep.stop()

    importance = engine.cell_importance(cell_id)
    # 15.0 (10 + 5 write boost) × 0.995^600 ≈ 0.75 — well below eviction threshold.
    assert importance < 5.0, f"Importance {importance:.2f} should be <5.0 after heavy decay"


# ── ATDD Test 4: Sweep stops cleanly on close ────────────────────────────


def test_sweep_stops_on_close(engine):
    """Start sweep, stop it. Thread should terminate."""
    sweep = GovernanceSweepThread(engine, interval_secs=0.1)
    sweep.start()
    assert sweep.is_running

    sweep.stop(timeout=2.0)
    assert not sweep.is_running, "Sweep thread should be stopped"


# ── ATDD Test 5: Sweep does not corrupt state ────────────────────────────


def test_sweep_does_not_corrupt(engine):
    """Run sweep concurrently with writes/reads for 0.5s. No exceptions."""
    sweep = GovernanceSweepThread(engine, interval_secs=0.05, hours_per_tick=1.0)
    sweep.start()

    # Concurrent writes and reads.
    errors = []
    for i in range(50):
        try:
            key = np.full(32, float(i), dtype=np.float32)
            engine.mem_write(1, 0, key, np.zeros(32, dtype=np.float32), 50.0, None)
            engine.mem_read(key, 3, None)
        except Exception as e:
            errors.append(str(e))

    sweep.stop()

    assert len(errors) == 0, f"Errors during concurrent sweep+write: {errors}"
    assert engine.cell_count() == 50
