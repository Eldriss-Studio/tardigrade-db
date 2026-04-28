# ATDD tests for retrieval key strategies and alignment diagnostics.
#
# Tests standalone functions — no vLLM dependency required.

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tardigrade_vllm.retrieval_key import (
    LAST_TOKEN_EMBEDDING,
    MEAN_POOL_EMBEDDING,
    LastTokenEmbeddingStrategy,
    MeanPoolEmbeddingStrategy,
    ProjectedEmbeddingStrategy,
    check_key_alignment,
    get_strategy,
)


def test_last_token_embedding_strategy_returns_last_token():
    """GIVEN an embedding table [vocab_size x dim] and token_ids [10, 20, 2],
    WHEN computing retrieval key via LastTokenEmbeddingStrategy,
    THEN result equals embed[2] (last valid token)."""
    embed = np.random.randn(100, 16).astype(np.float32)
    strategy = LastTokenEmbeddingStrategy()

    result = strategy.compute([10, 20, 2], embed)

    assert result is not None
    np.testing.assert_array_equal(result, embed[2])


def test_last_token_embedding_strategy_skips_out_of_range():
    """GIVEN token_ids where some are out of range,
    WHEN computing retrieval key,
    THEN only valid tokens are considered."""
    embed = np.random.randn(50, 8).astype(np.float32)
    strategy = LastTokenEmbeddingStrategy()

    result = strategy.compute([5, 999, 3], embed)

    assert result is not None
    np.testing.assert_array_equal(result, embed[3])


def test_last_token_embedding_strategy_empty_inputs():
    """GIVEN empty token_ids or empty embed,
    WHEN computing retrieval key,
    THEN returns None."""
    strategy = LastTokenEmbeddingStrategy()

    assert strategy.compute([], np.ones((10, 4), dtype=np.float32)) is None
    assert strategy.compute([1, 2], np.array([])) is None


def test_get_strategy_returns_default():
    """GIVEN the LAST_TOKEN_EMBEDDING constant,
    WHEN calling get_strategy(),
    THEN returns LastTokenEmbeddingStrategy instance."""
    strategy = get_strategy(LAST_TOKEN_EMBEDDING)
    assert isinstance(strategy, LastTokenEmbeddingStrategy)


def test_get_strategy_raises_on_unknown_name():
    """GIVEN an unknown strategy name 'foobar',
    WHEN calling get_strategy(),
    THEN ValueError is raised with available strategies listed."""
    with pytest.raises(ValueError, match="Unknown retrieval key strategy"):
        get_strategy("foobar")


def test_alignment_check_passes_on_matching_dimensions():
    """GIVEN hidden_size=1024 and kv_dim=1024 (matched),
    WHEN check_key_alignment is called,
    THEN it returns True."""
    assert check_key_alignment(1024, 1024) is True


def test_alignment_check_warns_on_dimension_mismatch(caplog):
    """GIVEN hidden_size=2048 and kv_dim=1024 (mismatched),
    WHEN check_key_alignment is called,
    THEN it returns False and logs a warning."""
    with caplog.at_level(logging.WARNING, logger="tardigrade_vllm"):
        result = check_key_alignment(2048, 1024)

    assert result is False
    assert "Key space mismatch" in caplog.text


# ── MeanPoolEmbeddingStrategy ─────────────────────────────────────────────


def test_mean_pool_strategy_averages_all_tokens():
    """GIVEN an embedding table and 3 valid token IDs,
    WHEN computing retrieval key via MeanPoolEmbeddingStrategy,
    THEN result equals the mean of all 3 embeddings."""
    embed = np.eye(4, dtype=np.float32)
    strategy = MeanPoolEmbeddingStrategy()

    result = strategy.compute([0, 1, 2], embed)

    expected = embed[[0, 1, 2]].mean(axis=0)
    np.testing.assert_allclose(result, expected)


def test_mean_pool_strategy_filters_invalid_tokens():
    """GIVEN token IDs with out-of-range values,
    WHEN computing,
    THEN only valid IDs are included in the mean."""
    embed = np.random.randn(10, 4).astype(np.float32)
    strategy = MeanPoolEmbeddingStrategy()

    result = strategy.compute([3, 999, 7], embed)

    expected = embed[[3, 7]].mean(axis=0)
    np.testing.assert_allclose(result, expected)


def test_mean_pool_strategy_empty_returns_none():
    strategy = MeanPoolEmbeddingStrategy()
    assert strategy.compute([], np.ones((10, 4), dtype=np.float32)) is None


def test_mean_pool_strategy_available_via_factory():
    strategy = get_strategy(MEAN_POOL_EMBEDDING)
    assert isinstance(strategy, MeanPoolEmbeddingStrategy)


# ── ProjectedEmbeddingStrategy ────────────────────────────────────────────


def test_projected_strategy_maps_hidden_to_kv_dim():
    """GIVEN hidden_size=8 and kv_dim=4,
    WHEN computing retrieval key via ProjectedEmbeddingStrategy,
    THEN result has kv_dim dimensions."""
    embed = np.random.randn(100, 8).astype(np.float32)
    strategy = ProjectedEmbeddingStrategy(kv_dim=4, hidden_size=8)

    result = strategy.compute([5, 10, 15], embed)

    assert result is not None
    assert result.shape == (4,)
    assert result.dtype == np.float32


def test_projected_strategy_uses_custom_projection():
    """GIVEN a known projection matrix,
    WHEN computing,
    THEN result equals projection @ embedding."""
    embed = np.eye(4, dtype=np.float32)
    projection = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    strategy = ProjectedEmbeddingStrategy(kv_dim=2, hidden_size=4, projection=projection)

    result = strategy.compute([2], embed)

    expected = projection @ embed[2]
    np.testing.assert_allclose(result, expected)


def test_projected_strategy_deterministic_default_projection():
    """GIVEN two instances with same dimensions and no explicit projection,
    WHEN computing the same input,
    THEN results are identical (seeded RNG)."""
    embed = np.random.randn(50, 8).astype(np.float32)
    s1 = ProjectedEmbeddingStrategy(kv_dim=4, hidden_size=8)
    s2 = ProjectedEmbeddingStrategy(kv_dim=4, hidden_size=8)

    r1 = s1.compute([10], embed)
    r2 = s2.compute([10], embed)

    np.testing.assert_array_equal(r1, r2)
