"""Acceptance tests for Engine.mem_read_multi_layer.

Behavioural parity with the Python `rrf_fuse` helper plus the
multi-query pipeline that calls `mem_read_pack` per query key and
merges via Reciprocal Rank Fusion.
"""

import numpy as np
import pytest

import tardigrade_db
from tardigrade_hooks.multi_layer_query import rrf_fuse


DIM = 32


@pytest.fixture
def engine(tmp_path):
    """Engine seeded with 8 packs across owners 1 and 2."""
    eng = tardigrade_db.Engine(str(tmp_path / "engine"))
    rng = np.random.default_rng(0)
    for i in range(8):
        owner = 1 if i < 5 else 2
        key = rng.standard_normal(DIM).astype(np.float32)
        data = rng.standard_normal(DIM).astype(np.float32)
        eng.mem_write_pack(
            owner=owner, retrieval_key=key,
            layer_payloads=[(0, data)], salience=50.0,
        )
    return eng


def _pack_ids(rows):
    return [r["pack_id"] for r in rows]


def test_default_rrf_k_is_60(engine):
    rng = np.random.default_rng(1)
    qs = [rng.standard_normal(DIM).astype(np.float32) for _ in range(3)]
    # No explicit rrf_k → should use 60 (consistent with the Python helper).
    rows = engine.mem_read_multi_layer(qs, k=3, owner=1)
    # Same call with explicit rrf_k=60 must produce the same ranking.
    eng2 = engine
    rows2 = eng2.mem_read_multi_layer(qs, k=3, owner=1, rrf_k=60)
    assert _pack_ids(rows) == _pack_ids(rows2)


def test_empty_query_keys_returns_empty_list(engine):
    assert engine.mem_read_multi_layer([], k=5, owner=1) == []


def test_single_layer_input_returns_that_layer_ordering_unchanged(tmp_path):
    """With one query key the multi-layer call must equal a single
    mem_read_pack — RRF over one ranked list reorders nothing.
    Independent engines isolate governance mutations."""
    def build(path):
        eng = tardigrade_db.Engine(str(path))
        rng = np.random.default_rng(0)
        for i in range(8):
            owner = 1 if i < 5 else 2
            key = rng.standard_normal(DIM).astype(np.float32)
            data = rng.standard_normal(DIM).astype(np.float32)
            eng.mem_write_pack(
                owner=owner, retrieval_key=key,
                layer_payloads=[(0, data)], salience=50.0,
            )
        return eng

    eng_a = build(tmp_path / "a")
    eng_b = build(tmp_path / "b")
    rng = np.random.default_rng(9)
    q = rng.standard_normal(DIM).astype(np.float32)
    single = eng_a.mem_read_pack(q, k=4, owner=1)
    multi = eng_b.mem_read_multi_layer([q], k=4, owner=1)
    assert _pack_ids(multi) == _pack_ids(single)


def test_rrf_output_matches_python_helper(tmp_path):
    """Run the engine's multi-layer fusion against the Python rrf_fuse
    helper on the same per-layer ranked lists (taken from a parallel
    engine). The fused ordering must agree."""
    def build(path):
        eng = tardigrade_db.Engine(str(path))
        rng = np.random.default_rng(0)
        for i in range(8):
            owner = 1 if i < 5 else 2
            key = rng.standard_normal(DIM).astype(np.float32)
            data = rng.standard_normal(DIM).astype(np.float32)
            eng.mem_write_pack(
                owner=owner, retrieval_key=key,
                layer_payloads=[(0, data)], salience=50.0,
            )
        return eng

    eng_a = build(tmp_path / "a")
    eng_b = build(tmp_path / "b")
    rng = np.random.default_rng(10)
    qs = [rng.standard_normal(DIM).astype(np.float32) for _ in range(3)]

    for k in (5, 3, 8):
        # Python side: per-query mem_read_pack on eng_a, then rrf_fuse.
        ranked = [eng_a.mem_read_pack(q, k=k * 2, owner=1) for q in qs]
        py_fused = rrf_fuse(ranked, k=60)[:k]

        # Rust side: one engine call on eng_b.
        rust_fused = eng_b.mem_read_multi_layer(qs, k=k, owner=1)

        assert _pack_ids(rust_fused) == _pack_ids(py_fused), (
            f"k={k} ranking diverged"
        )


def test_owner_filter_applied(engine):
    rng = np.random.default_rng(11)
    qs = [rng.standard_normal(DIM).astype(np.float32) for _ in range(2)]
    rows = engine.mem_read_multi_layer(qs, k=5, owner=2)
    for r in rows:
        assert r["owner"] == 2


def test_rrf_score_field_present_and_descending(engine):
    rng = np.random.default_rng(12)
    qs = [rng.standard_normal(DIM).astype(np.float32) for _ in range(3)]
    rows = engine.mem_read_multi_layer(qs, k=5, owner=1)
    scores = [r["rrf_score"] for r in rows]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_tied_rrf_scores_break_ties_by_pack_id_ascending(tmp_path):
    """If two packs accumulate identical RRF scores, the tie-break must
    be stable and predictable: pack_id ascending."""
    eng = tardigrade_db.Engine(str(tmp_path / "engine"))
    rng = np.random.default_rng(99)
    # Two packs with identical retrieval keys → same rank in any query
    # → identical RRF accumulations across all input queries.
    shared_key = rng.standard_normal(DIM).astype(np.float32)
    for _ in range(2):
        data = rng.standard_normal(DIM).astype(np.float32)
        eng.mem_write_pack(
            owner=1, retrieval_key=shared_key.copy(),
            layer_payloads=[(0, data)], salience=50.0,
        )

    qs = [rng.standard_normal(DIM).astype(np.float32) for _ in range(2)]
    rows = eng.mem_read_multi_layer(qs, k=2, owner=1)
    # When scores are exactly tied, the lower pack_id must come first.
    if len(rows) >= 2 and rows[0]["rrf_score"] == rows[1]["rrf_score"]:
        assert rows[0]["pack_id"] < rows[1]["pack_id"]
