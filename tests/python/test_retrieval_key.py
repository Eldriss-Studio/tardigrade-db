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
    LastTokenEmbeddingStrategy,
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
    """GIVEN strategy name 'last_token_embedding',
    WHEN calling get_strategy(),
    THEN returns LastTokenEmbeddingStrategy instance."""
    strategy = get_strategy("last_token_embedding")
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
