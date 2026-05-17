# ATDD tests for vLLM block format conversion.
#
# Tests the format conversion between TardigradeDB flat KV arrays
# and vLLM paged attention blocks. These work without vLLM installed.

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tardigrade_vllm.format import blocks_to_flat, flat_to_blocks


def test_flat_to_blocks_correct_shape():
    """GIVEN a flat KV array (seq=8, kv_dim=4, block_size=4),
    THEN flat_to_blocks returns (2, 4, 2, 2) shaped K and V blocks."""
    num_kv_heads = 2
    head_dim = 2
    kv_dim = num_kv_heads * head_dim
    seq_len = 8
    block_size = 4

    flat = np.random.randn(2 * seq_len * kv_dim).astype(np.float32)
    k_blocks, v_blocks = flat_to_blocks(flat, num_kv_heads, head_dim, block_size)

    assert k_blocks.shape == (2, block_size, num_kv_heads, head_dim)
    assert v_blocks.shape == (2, block_size, num_kv_heads, head_dim)


def test_flat_to_blocks_pads_partial():
    """GIVEN seq_len=5 and block_size=4 (not evenly divisible),
    THEN flat_to_blocks pads to 2 blocks (8 slots)."""
    num_kv_heads = 2
    head_dim = 2
    kv_dim = num_kv_heads * head_dim
    seq_len = 5
    block_size = 4

    flat = np.random.randn(2 * seq_len * kv_dim).astype(np.float32)
    k_blocks, v_blocks = flat_to_blocks(flat, num_kv_heads, head_dim, block_size)

    assert k_blocks.shape == (2, block_size, num_kv_heads, head_dim)


def test_round_trip_preserves_data():
    """GIVEN a flat KV array,
    WHEN converted to blocks and back,
    THEN the data matches (within padding)."""
    num_kv_heads = 4
    head_dim = 8
    kv_dim = num_kv_heads * head_dim
    seq_len = 10
    block_size = 4

    original_flat = np.random.randn(2 * seq_len * kv_dim).astype(np.float32)
    k_blocks, v_blocks = flat_to_blocks(original_flat, num_kv_heads, head_dim, block_size)
    recovered_flat = blocks_to_flat(k_blocks, v_blocks, seq_len, num_kv_heads, head_dim)

    np.testing.assert_allclose(original_flat, recovered_flat, atol=1e-6)


def test_blocks_to_flat_trims_padding():
    """GIVEN blocks with padding (12 slots but seq_len=10),
    THEN blocks_to_flat trims to seq_len."""
    num_kv_heads = 2
    head_dim = 4
    kv_dim = num_kv_heads * head_dim
    seq_len = 10
    block_size = 4
    num_blocks = 3  # 12 slots, 2 padded

    k_blocks = np.random.randn(num_blocks, block_size, num_kv_heads, head_dim).astype(np.float32)
    v_blocks = np.random.randn(num_blocks, block_size, num_kv_heads, head_dim).astype(np.float32)

    flat = blocks_to_flat(k_blocks, v_blocks, seq_len, num_kv_heads, head_dim)

    assert flat.shape == (2 * seq_len * kv_dim,)
