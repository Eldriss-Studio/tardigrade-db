"""Acceptance tests for MemoryConsolidator — multi-view consolidation engine.

Stores a canonical memory, consolidates it into views, and verifies
that view keys are attached to the canonical pack (parent-document
pattern: views are retrieval surfaces, not separate packs).

Uses a minimal synthetic engine (no real LLM) by pre-storing packs
with known retrieval keys and text, then running consolidation.
"""

import numpy as np
import pytest

import tardigrade_db

from tardigrade_hooks.constants import (
    CONSOLIDATION_MIN_TIER,
    DEFAULT_VIEW_FRAMINGS,
)
from tardigrade_hooks.consolidator import (
    ConsolidationPolicy,
    DefaultConsolidationPolicy,
    MemoryConsolidator,
)


# -- Fixtures ----------------------------------------------------------------

DIM = 8
OWNER = 1
FACT_TEXT = "Sonia translated a pharmaceutical patent from German to English for a Berlin-based biotech startup in March 2024."


def _make_engine(path):
    engine = tardigrade_db.Engine(str(path), vamana_threshold=9999)
    return engine


SALIENCE_DRAFT = 10.0
SALIENCE_VALIDATED = 70.0  # ι≥65 → Validated tier


def _write_pack(engine, text, salience=SALIENCE_VALIDATED):
    """Write a minimal pack with random retrieval key."""
    rng = np.random.default_rng(42)
    key = rng.standard_normal(DIM).astype(np.float32)
    value = rng.standard_normal(DIM).astype(np.float32)
    pack_id = engine.mem_write_pack(OWNER, key, [(0, value)], salience, text=text)
    return pack_id


@pytest.fixture
def env(tmp_path):
    """Engine with one canonical pack at Validated tier."""
    engine = _make_engine(tmp_path)
    pack_id = _write_pack(engine, FACT_TEXT)
    return engine, pack_id


# -- Tests -------------------------------------------------------------------

class TestConsolidationBasics:
    """Core consolidation behavior — views attach to the canonical pack."""

    def test_consolidate_attaches_views(self, env):
        engine, pack_id = env
        consolidator = MemoryConsolidator(engine, owner=OWNER)
        count = consolidator.consolidate(pack_id)
        assert count >= 1
        assert engine.view_count(pack_id) >= 1

    def test_views_discoverable_via_engine(self, env):
        engine, pack_id = env
        consolidator = MemoryConsolidator(engine, owner=OWNER)
        consolidator.consolidate(pack_id)
        assert engine.view_count(pack_id) > 0

    def test_canonical_text_unchanged(self, env):
        engine, pack_id = env
        text_before = engine.pack_text(pack_id)
        consolidator = MemoryConsolidator(engine, owner=OWNER)
        consolidator.consolidate(pack_id)
        assert engine.pack_text(pack_id) == text_before

    def test_canonical_importance_unchanged(self, env):
        engine, pack_id = env
        packs_before = {p["pack_id"]: p["importance"] for p in engine.list_packs(OWNER)}
        importance_before = packs_before[pack_id]
        consolidator = MemoryConsolidator(engine, owner=OWNER)
        consolidator.consolidate(pack_id)
        packs_after = {p["pack_id"]: p["importance"] for p in engine.list_packs(OWNER)}
        assert packs_after[pack_id] == importance_before


class TestTierGating:
    """Only Validated+ packs get consolidated."""

    def test_draft_pack_skipped(self, tmp_path):
        engine = _make_engine(tmp_path)
        pack_id = _write_pack(engine, FACT_TEXT, salience=SALIENCE_DRAFT)
        consolidator = MemoryConsolidator(engine, owner=OWNER)
        count = consolidator.consolidate(pack_id)
        assert count == 0

    def test_validated_pack_consolidated(self, env):
        engine, pack_id = env
        consolidator = MemoryConsolidator(engine, owner=OWNER)
        count = consolidator.consolidate(pack_id)
        assert count > 0


class TestIdempotency:
    """Consolidation must not attach duplicate views."""

    def test_second_consolidation_adds_no_views(self, env):
        engine, pack_id = env
        consolidator = MemoryConsolidator(engine, owner=OWNER)
        first = consolidator.consolidate(pack_id)
        assert first > 0
        second = consolidator.consolidate(pack_id)
        assert second == 0

    def test_view_count_stable_after_double_consolidation(self, env):
        engine, pack_id = env
        consolidator = MemoryConsolidator(engine, owner=OWNER)
        consolidator.consolidate(pack_id)
        vc1 = engine.view_count(pack_id)
        consolidator.consolidate(pack_id)
        vc2 = engine.view_count(pack_id)
        assert vc1 == vc2


class TestConsolidateAll:
    """Batch consolidation across all packs for an owner."""

    def test_consolidate_all_processes_eligible_packs(self, tmp_path):
        engine = _make_engine(tmp_path)
        p1 = _write_pack(engine, "Fact one about Alice.")
        p2 = _write_pack(engine, "Fact two about Bob.")
        consolidator = MemoryConsolidator(engine, owner=OWNER)
        result = consolidator.consolidate_all()
        assert p1 in result
        assert p2 in result
        assert all(count > 0 for count in result.values())


class TestConsolidationPolicy:
    """Policy controls which packs get consolidated."""

    def test_default_policy_skips_draft(self):
        policy = DefaultConsolidationPolicy()
        assert not policy.should_consolidate(tier=0, importance=10.0)

    def test_default_policy_allows_validated(self):
        policy = DefaultConsolidationPolicy()
        assert policy.should_consolidate(tier=1, importance=70.0)

    def test_default_policy_allows_core(self):
        policy = DefaultConsolidationPolicy()
        assert policy.should_consolidate(tier=2, importance=90.0)

    def test_default_policy_view_count(self):
        policy = DefaultConsolidationPolicy()
        assert policy.view_count(tier=0) == 0
        assert policy.view_count(tier=1) == len(DEFAULT_VIEW_FRAMINGS)
        assert policy.view_count(tier=2) == len(DEFAULT_VIEW_FRAMINGS)


class TestEngineViewKeys:
    def test_add_view_keys_returns_count(self, tmp_path):
        engine = tardigrade_db.Engine(str(tmp_path), vamana_threshold=9999)
        key = np.random.randn(8).astype(np.float32)
        val = np.random.randn(8).astype(np.float32)
        pid = engine.mem_write_pack(OWNER, key, [(0, val)], 80.0, text="Test")
        v1 = np.random.randn(8).astype(np.float32)
        v2 = np.random.randn(8).astype(np.float32)
        assert engine.add_view_keys(pid, [v1, v2]) == 2

    def test_view_count(self, tmp_path):
        engine = tardigrade_db.Engine(str(tmp_path), vamana_threshold=9999)
        key = np.random.randn(8).astype(np.float32)
        val = np.random.randn(8).astype(np.float32)
        pid = engine.mem_write_pack(OWNER, key, [(0, val)], 80.0, text="Test")
        assert engine.view_count(pid) == 0
        engine.add_view_keys(pid, [np.random.randn(8).astype(np.float32)])
        assert engine.view_count(pid) == 1

    def test_view_count_errors_on_missing(self, tmp_path):
        engine = tardigrade_db.Engine(str(tmp_path), vamana_threshold=9999)
        with pytest.raises(RuntimeError):
            engine.view_count(9999)

    def test_view_match_returns_canonical(self, tmp_path):
        engine = tardigrade_db.Engine(str(tmp_path), vamana_threshold=9999)
        ckey = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        pid = engine.mem_write_pack(OWNER, ckey, [(0, np.zeros(8, dtype=np.float32))], 80.0, text="Fact")
        vkey = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        engine.add_view_keys(pid, [vkey])
        results = engine.mem_read_pack(vkey, 5, OWNER)
        pids = [r["pack_id"] for r in results]
        assert pid in pids
        assert pids.count(pid) == 1
