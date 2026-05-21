"""Acceptance tests for Engine.compute_retrieval_key (Rust-side retrieval-key compute).

Behavioural parity with python/tardigrade_vllm/retrieval_key.py — the Rust path
must produce the same vector the Python strategy produced, within numerical
tolerance, and behave identically on edge cases (empty input, all-out-of-bounds
IDs, unloaded embedding table).
"""

import numpy as np
import pytest

import tardigrade_db
from tardigrade_vllm.retrieval_key import (
    LastTokenEmbeddingStrategy,
    MeanPoolEmbeddingStrategy,
)

VOCAB = 64
HIDDEN = 16
MAX_ABS_DELTA = 1e-6


@pytest.fixture
def embed_table():
    rng = np.random.default_rng(1234)
    return rng.standard_normal((VOCAB, HIDDEN)).astype(np.float32)


@pytest.fixture
def engine(tmp_path, embed_table):
    eng = tardigrade_db.Engine(str(tmp_path / "engine"))
    eng.load_embedding_table(embed_table)
    return eng


def test_last_token_strategy_matches_python_within_tolerance(engine, embed_table):
    token_ids = [3, 7, 11, 15, 19]
    rust = engine.compute_retrieval_key(token_ids, "last_token")
    py = LastTokenEmbeddingStrategy().compute(token_ids, embed_table)
    assert rust is not None
    assert np.max(np.abs(rust - py)) < MAX_ABS_DELTA


def test_mean_pool_strategy_matches_python_within_tolerance(engine, embed_table):
    token_ids = [0, 4, 9, 21, 33]
    rust = engine.compute_retrieval_key(token_ids, "mean_pool")
    py = MeanPoolEmbeddingStrategy().compute(token_ids, embed_table)
    assert rust is not None
    assert np.max(np.abs(rust - py)) < MAX_ABS_DELTA


def test_filters_out_of_bounds_token_ids_matching_python_filter(engine, embed_table):
    token_ids = [5, 99999, -1, 17, 8, 50000]  # only 5, 17, 8 are in-range (VOCAB=64)
    rust = engine.compute_retrieval_key(token_ids, "last_token")
    py = LastTokenEmbeddingStrategy().compute(token_ids, embed_table)
    assert rust is not None
    assert np.max(np.abs(rust - py)) < MAX_ABS_DELTA


def test_empty_token_id_list_returns_none(engine):
    assert engine.compute_retrieval_key([], "last_token") is None
    assert engine.compute_retrieval_key([], "mean_pool") is None


def test_all_out_of_bounds_token_ids_returns_none(engine):
    assert engine.compute_retrieval_key([VOCAB + 10, -5], "last_token") is None


def test_compute_retrieval_key_returns_none_when_embedding_table_unloaded(tmp_path):
    eng = tardigrade_db.Engine(str(tmp_path / "engine"))
    assert eng.compute_retrieval_key([1, 2, 3], "last_token") is None


def test_unknown_strategy_raises_value_error(engine):
    with pytest.raises(ValueError, match="strategy"):
        engine.compute_retrieval_key([1, 2, 3], "not_a_real_strategy")


def test_projected_strategy_with_identity_matrix_equals_last_token(engine, embed_table):
    identity = np.eye(HIDDEN, dtype=np.float32)
    token_ids = [2, 8, 13, 22]
    engine.set_projection_matrix(identity)
    rust = engine.compute_retrieval_key(token_ids, "projected")
    py = LastTokenEmbeddingStrategy().compute(token_ids, embed_table)
    assert rust is not None
    assert np.max(np.abs(rust - py)) < MAX_ABS_DELTA


def test_load_embedding_table_overwrites_previous(tmp_path):
    eng = tardigrade_db.Engine(str(tmp_path / "engine"))
    table_a = np.full((4, 8), 1.0, dtype=np.float32)
    table_b = np.full((4, 8), 7.5, dtype=np.float32)
    eng.load_embedding_table(table_a)
    key_a = eng.compute_retrieval_key([0], "last_token")
    eng.load_embedding_table(table_b)
    key_b = eng.compute_retrieval_key([0], "last_token")
    assert np.allclose(key_a, 1.0)
    assert np.allclose(key_b, 7.5)


def test_load_embedding_table_counter_records_one_load_after_repeated_compute(tmp_path):
    eng = tardigrade_db.Engine(str(tmp_path / "engine"))
    rng = np.random.default_rng(0)
    table = rng.standard_normal((VOCAB, HIDDEN)).astype(np.float32)
    eng.load_embedding_table(table)
    for _ in range(50):
        eng.compute_retrieval_key([1, 2, 3], "last_token")
    assert eng.embedding_table_load_count() == 1
