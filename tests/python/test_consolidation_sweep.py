"""Acceptance tests for ConsolidationSweepThread — offline background consolidation."""

import time

import numpy as np
import pytest

import tardigrade_db

from tardigrade_hooks.consolidation_sweep import ConsolidationSweepThread

DIM = 8
OWNER = 1


def _make_engine(path):
    return tardigrade_db.Engine(str(path), vamana_threshold=9999)


def _write_validated_pack(engine, text):
    rng = np.random.default_rng(42)
    key = rng.standard_normal(DIM).astype(np.float32)
    val = rng.standard_normal(DIM).astype(np.float32)
    return engine.mem_write_pack(OWNER, key, [(0, val)], 70.0, text=text)


class TestSweepLifecycle:
    def test_start_and_stop(self, tmp_path):
        engine = _make_engine(tmp_path)
        sweep = ConsolidationSweepThread(engine, owner=OWNER, interval_secs=0.05)
        sweep.start()
        assert sweep.is_running
        sweep.stop(timeout=2.0)
        assert not sweep.is_running

    def test_stops_on_close(self, tmp_path):
        engine = _make_engine(tmp_path)
        sweep = ConsolidationSweepThread(engine, owner=OWNER, interval_secs=0.05)
        sweep.start()
        sweep.stop(timeout=2.0)
        assert not sweep.is_running


class TestSweepConsolidation:
    def test_sweep_consolidates_eligible_packs(self, tmp_path):
        engine = _make_engine(tmp_path)
        pid = _write_validated_pack(engine, "Alice moved to Berlin in 2023.")

        sweep = ConsolidationSweepThread(engine, owner=OWNER, interval_secs=0.05)
        sweep.start()
        time.sleep(0.3)
        sweep.stop(timeout=2.0)

        assert engine.view_count(pid) > 0

    def test_sweep_is_idempotent_across_cycles(self, tmp_path):
        engine = _make_engine(tmp_path)
        pid = _write_validated_pack(engine, "Bob built a house in Munich.")

        sweep = ConsolidationSweepThread(engine, owner=OWNER, interval_secs=0.05)
        sweep.start()
        time.sleep(0.4)
        sweep.stop(timeout=2.0)

        vc = engine.view_count(pid)
        assert vc > 0


class TestSweepStatus:
    def test_status_reports_counts(self, tmp_path):
        engine = _make_engine(tmp_path)
        _write_validated_pack(engine, "Claire studies astrophysics.")

        sweep = ConsolidationSweepThread(engine, owner=OWNER, interval_secs=0.05)
        sweep.start()
        time.sleep(0.3)
        sweep.stop(timeout=2.0)

        status = sweep.status
        assert "packs_consolidated" in status
        assert "views_attached" in status
        assert status["packs_consolidated"] >= 1
        assert status["views_attached"] >= 1
