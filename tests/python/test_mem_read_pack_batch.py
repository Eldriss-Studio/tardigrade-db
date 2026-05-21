"""Acceptance tests for Engine.mem_read_pack_batch.

Batch read API: one PyO3 crossing for N queries instead of N crossings.
Per-query behaviour must match `mem_read_pack` one-for-one across the
allowed argument shapes for `k` and `owner`.
"""

import numpy as np
import pytest

import tardigrade_db


DIM = 32


@pytest.fixture
def engine(tmp_path):
    eng = tardigrade_db.Engine(str(tmp_path / "engine"))
    rng = np.random.default_rng(0)
    # Five packs across two owners — enough to exercise filtering + ranking.
    for i in range(5):
        owner = 1 if i < 3 else 2
        key = rng.standard_normal(DIM).astype(np.float32)
        data = rng.standard_normal(DIM).astype(np.float32)
        eng.mem_write_pack(
            owner=owner,
            retrieval_key=key,
            layer_payloads=[(0, data)],
            salience=50.0,
        )
    return eng


def _pack_ids(rows):
    """Project each row down to pack_id — the behavioural ordering contract.

    Equality on full row dicts is fragile because every retrieval mutates
    governance (importance bump, tier transition) and the per-row `score`
    folds those mutations in. Comparing pack_id sequences keeps the test
    focused on the actual ranking output.
    """
    return [r["pack_id"] for r in rows]


def test_batch_results_match_per_query_calls_one_for_one(tmp_path):
    """Build two engines with identical state; run batch on one, per-query
    on the other. Engine reads mutate governance (importance/tier), so
    a single fixture serving both modes would see one pass interfere
    with the other — separate engines isolate the comparison."""
    def build_engine(path):
        eng = tardigrade_db.Engine(str(path))
        rng = np.random.default_rng(0)
        for i in range(5):
            owner = 1 if i < 3 else 2
            key = rng.standard_normal(DIM).astype(np.float32)
            data = rng.standard_normal(DIM).astype(np.float32)
            eng.mem_write_pack(
                owner=owner, retrieval_key=key,
                layer_payloads=[(0, data)], salience=50.0,
            )
        return eng

    eng_a = build_engine(tmp_path / "a")
    eng_b = build_engine(tmp_path / "b")
    rng = np.random.default_rng(1)
    queries = [rng.standard_normal(DIM).astype(np.float32) for _ in range(4)]

    per_query = [eng_a.mem_read_pack(q, k=2, owner=1) for q in queries]
    batched = eng_b.mem_read_pack_batch(queries, k=2, owner=1)

    assert len(batched) == len(per_query)
    for b, p in zip(batched, per_query):
        assert _pack_ids(b) == _pack_ids(p)


def test_empty_batch_returns_empty_list(engine):
    assert engine.mem_read_pack_batch([], k=3, owner=1) == []


def test_heterogeneous_k_per_query(engine):
    rng = np.random.default_rng(2)
    queries = [rng.standard_normal(DIM).astype(np.float32) for _ in range(3)]
    ks = [1, 2, 3]
    batched = engine.mem_read_pack_batch(queries, k=ks, owner=1)
    assert len(batched) == 3
    for rows, expected_k in zip(batched, ks):
        assert len(rows) == expected_k


def test_owner_filter_applies_independently_per_query(engine):
    rng = np.random.default_rng(3)
    queries = [rng.standard_normal(DIM).astype(np.float32) for _ in range(3)]
    # Mixed owners: q0 -> owner=1, q1 -> owner=2, q2 -> no filter (None).
    owners = [1, 2, None]
    batched = engine.mem_read_pack_batch(queries, k=2, owner=owners)
    for rows, owner in zip(batched, owners):
        if owner is not None:
            for r in rows:
                assert r["owner"] == owner


def test_scalar_k_and_owner_broadcast_across_queries(engine):
    rng = np.random.default_rng(4)
    queries = [rng.standard_normal(DIM).astype(np.float32) for _ in range(3)]
    batched = engine.mem_read_pack_batch(queries, k=2, owner=2)
    for rows in batched:
        for r in rows:
            assert r["owner"] == 2


def test_k_list_length_mismatch_raises_value_error(engine):
    rng = np.random.default_rng(5)
    queries = [rng.standard_normal(DIM).astype(np.float32) for _ in range(3)]
    with pytest.raises(ValueError, match="k"):
        engine.mem_read_pack_batch(queries, k=[1, 2], owner=1)


def test_owner_list_length_mismatch_raises_value_error(engine):
    rng = np.random.default_rng(6)
    queries = [rng.standard_normal(DIM).astype(np.float32) for _ in range(3)]
    with pytest.raises(ValueError, match="owner"):
        engine.mem_read_pack_batch(queries, k=2, owner=[1, 2])


def test_batch_with_centered_refinement_matches_per_query(tmp_path):
    """Same isolation pattern as the one-for-one parity test, with the
    centered refinement strategy installed before the read pass."""
    def build_engine(path):
        eng = tardigrade_db.Engine(str(path))
        rng = np.random.default_rng(0)
        for i in range(5):
            owner = 1 if i < 3 else 2
            key = rng.standard_normal(DIM).astype(np.float32)
            data = rng.standard_normal(DIM).astype(np.float32)
            eng.mem_write_pack(
                owner=owner, retrieval_key=key,
                layer_payloads=[(0, data)], salience=50.0,
            )
        eng.set_refinement_mode("centered")
        return eng

    eng_a = build_engine(tmp_path / "ra")
    eng_b = build_engine(tmp_path / "rb")
    rng = np.random.default_rng(7)
    queries = [rng.standard_normal(DIM).astype(np.float32) for _ in range(3)]
    per_query = [eng_a.mem_read_pack(q, k=2, owner=1) for q in queries]
    batched = eng_b.mem_read_pack_batch(queries, k=2, owner=1)
    for b, p in zip(batched, per_query):
        assert _pack_ids(b) == _pack_ids(p)
