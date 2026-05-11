"""Acceptance tests for MemoryConsolidator — multi-view consolidation engine.

Stores a canonical memory, consolidates it into views, and verifies
the resulting pack topology (edges, text, tier gating, idempotency).

Uses a minimal synthetic engine (no real LLM) by pre-storing packs
with known retrieval keys and text, then running consolidation.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import tardigrade_db

from tardigrade_hooks.constants import (
    CONSOLIDATION_MIN_TIER,
    DEFAULT_VIEW_FRAMINGS,
    EDGE_SUPPORTS,
)
from tardigrade_hooks.consolidator import (
    ConsolidationPolicy,
    DefaultConsolidationPolicy,
    MemoryConsolidator,
)
from tardigrade_hooks.view_generator import ViewGenerator


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


# -- Stub model for view storage (no real LLM needed) -----------------------

class _StubModel:
    """Minimal stand-in: consolidator only needs the model for
    KnowledgePackStore.store(), which we bypass in tests by directly
    calling _store_view_pack on the consolidator."""
    pass


class _StubTokenizer:
    pass


# -- Tests -------------------------------------------------------------------

class TestConsolidationBasics:
    """Core consolidation behavior."""

    def test_consolidate_creates_view_packs(self, env):
        engine, pack_id = env
        consolidator = MemoryConsolidator(engine, owner=OWNER)
        view_ids = consolidator.consolidate(pack_id)
        assert len(view_ids) == len(DEFAULT_VIEW_FRAMINGS)

    def test_view_packs_linked_via_supports_edge(self, env):
        engine, pack_id = env
        consolidator = MemoryConsolidator(engine, owner=OWNER)
        view_ids = consolidator.consolidate(pack_id)
        for vid in view_ids:
            supports = engine.pack_supports(vid)
            assert pack_id in supports, (
                f"View pack {vid} not linked to canonical {pack_id}"
            )

    def test_view_packs_have_text(self, env):
        engine, pack_id = env
        consolidator = MemoryConsolidator(engine, owner=OWNER)
        view_ids = consolidator.consolidate(pack_id)
        for vid in view_ids:
            text = engine.pack_text(vid)
            assert text is not None and len(text.strip()) > 0

    def test_view_texts_differ_from_canonical(self, env):
        engine, pack_id = env
        consolidator = MemoryConsolidator(engine, owner=OWNER)
        view_ids = consolidator.consolidate(pack_id)
        canonical_text = engine.pack_text(pack_id)
        for vid in view_ids:
            assert engine.pack_text(vid) != canonical_text

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
        view_ids = consolidator.consolidate(pack_id)
        assert view_ids == []

    def test_validated_pack_consolidated(self, env):
        engine, pack_id = env
        consolidator = MemoryConsolidator(engine, owner=OWNER)
        view_ids = consolidator.consolidate(pack_id)
        assert len(view_ids) > 0


class TestIdempotency:
    """Consolidation must not create duplicate views."""

    def test_second_consolidation_creates_no_new_views(self, env):
        engine, pack_id = env
        consolidator = MemoryConsolidator(engine, owner=OWNER)
        first_views = consolidator.consolidate(pack_id)
        second_views = consolidator.consolidate(pack_id)
        assert second_views == [], (
            f"Expected no new views on second call, got {second_views}"
        )

    def test_total_pack_count_stable_after_double_consolidation(self, env):
        engine, pack_id = env
        consolidator = MemoryConsolidator(engine, owner=OWNER)
        consolidator.consolidate(pack_id)
        count_after_first = len(engine.list_packs(OWNER))
        consolidator.consolidate(pack_id)
        count_after_second = len(engine.list_packs(OWNER))
        assert count_after_first == count_after_second


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
        assert all(len(views) == len(DEFAULT_VIEW_FRAMINGS) for views in result.values())


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
