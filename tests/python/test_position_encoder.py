"""ATDD tests for position encoding strategies (RoPE remapping)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tardigrade_hooks.position import (
    AbsolutePositionEncoder,
    RoPEPositionEncoder,
)

HEAD_DIM = 64
BATCH_SIZE = 1
NUM_HEADS = 4
SEQ_LEN = 8
ROPE_BASE = 10000.0


# ── AbsolutePositionEncoder ──────────────────────────────────────────────


def test_absolute_remap_is_noop():
    """GIVEN keys with absolute position encoding,
    WHEN remap_keys is called,
    THEN keys are returned unchanged."""
    encoder = AbsolutePositionEncoder()
    keys = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    old_positions = torch.arange(SEQ_LEN)

    result = encoder.remap_keys(keys, old_positions, new_start=10)

    torch.testing.assert_close(result, keys)


def test_absolute_position_ids_offset():
    """GIVEN cache_len=5 and seq_len=3,
    WHEN build_position_ids is called,
    THEN position IDs are [5, 6, 7]."""
    encoder = AbsolutePositionEncoder()
    ids = encoder.build_position_ids(cache_len=5, seq_len=3)

    assert ids.shape == (1, 3)
    assert ids.tolist() == [[5, 6, 7]]


# ── RoPEPositionEncoder ──────────────────────────────────────────────────


def test_rope_remap_identity_when_positions_unchanged():
    """GIVEN keys rotated at positions [0..N],
    WHEN remapped to the same positions (new_start=0),
    THEN output matches input within float tolerance."""
    encoder = RoPEPositionEncoder(head_dim=HEAD_DIM, base=ROPE_BASE)
    keys = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    positions = torch.arange(SEQ_LEN)

    result = encoder.remap_keys(keys, positions, new_start=0)

    torch.testing.assert_close(result, keys, atol=1e-5, rtol=1e-5)


def test_rope_remap_shifts_position():
    """GIVEN keys rotated at position 0,
    WHEN remapped to position 10,
    THEN the output differs from input (rotation changed)."""
    encoder = RoPEPositionEncoder(head_dim=HEAD_DIM, base=ROPE_BASE)
    keys = torch.randn(BATCH_SIZE, NUM_HEADS, 1, HEAD_DIM)
    old_positions = torch.tensor([0])

    result = encoder.remap_keys(keys, old_positions, new_start=10)

    assert not torch.allclose(result, keys, atol=1e-4)


def test_rope_remap_roundtrip():
    """GIVEN keys at position 5, remapped to position 20, then back to 5,
    THEN the double-remapped keys match the original."""
    encoder = RoPEPositionEncoder(head_dim=HEAD_DIM, base=ROPE_BASE)
    keys = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    positions_5 = torch.arange(5, 5 + SEQ_LEN)

    shifted = encoder.remap_keys(keys, positions_5, new_start=20)
    positions_20 = torch.arange(20, 20 + SEQ_LEN)
    restored = encoder.remap_keys(shifted, positions_20, new_start=5)

    torch.testing.assert_close(restored, keys, atol=1e-4, rtol=1e-4)


def test_rope_position_ids_offset():
    """GIVEN cache_len=10 and seq_len=5,
    WHEN build_position_ids is called,
    THEN position IDs are [10, 11, 12, 13, 14]."""
    encoder = RoPEPositionEncoder(head_dim=HEAD_DIM)
    ids = encoder.build_position_ids(cache_len=10, seq_len=5)

    assert ids.shape == (1, 5)
    assert ids.tolist() == [[10, 11, 12, 13, 14]]


def test_rope_preserves_tensor_shape():
    """GIVEN a (batch, heads, seq, head_dim) tensor,
    WHEN remapped,
    THEN output shape is identical."""
    encoder = RoPEPositionEncoder(head_dim=HEAD_DIM)
    keys = torch.randn(2, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    positions = torch.arange(SEQ_LEN)

    result = encoder.remap_keys(keys, positions, new_start=100)

    assert result.shape == keys.shape


def test_rope_different_base_produces_different_result():
    """GIVEN two RoPE encoders with different bases,
    WHEN remapping the same keys to the same position,
    THEN results differ."""
    keys = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    positions = torch.arange(SEQ_LEN)

    enc_a = RoPEPositionEncoder(head_dim=HEAD_DIM, base=10000.0)
    enc_b = RoPEPositionEncoder(head_dim=HEAD_DIM, base=500000.0)

    result_a = enc_a.remap_keys(keys, positions, new_start=50)
    result_b = enc_b.remap_keys(keys, positions, new_start=50)

    assert not torch.allclose(result_a, result_b, atol=1e-4)
