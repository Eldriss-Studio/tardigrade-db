"""Acceptance tests for Engine.flat_to_paged / paged_to_flat (Rust KV reshape).

These must produce byte-identical output to the Python `format.py` baseline
(flat_to_blocks / blocks_to_flat), across edge cases and Qwen3-0.6B dims.
"""

import numpy as np
import pytest

import tardigrade_db
from tardigrade_vllm.format import blocks_to_flat, flat_to_blocks


NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 16
KV_DIM = NUM_KV_HEADS * HEAD_DIM


def _make_flat(seq_len, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(2 * seq_len * KV_DIM).astype(np.float32)


@pytest.mark.parametrize("seq_len", [1, 15, 16, 17, 100])
def test_flat_to_paged_round_trip_is_lossless(seq_len):
    flat = _make_flat(seq_len, seed=seq_len)
    k_blocks, v_blocks = tardigrade_db.flat_to_paged(flat, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE)
    restored = tardigrade_db.paged_to_flat(k_blocks, v_blocks, seq_len, NUM_KV_HEADS, HEAD_DIM)
    assert np.array_equal(restored, flat)


@pytest.mark.parametrize("seq_len", [1, 15, 16, 17, 100])
def test_flat_to_paged_matches_python_baseline_byte_for_byte(seq_len):
    flat = _make_flat(seq_len, seed=seq_len + 1000)
    k_rust, v_rust = tardigrade_db.flat_to_paged(flat, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE)
    k_py, v_py = flat_to_blocks(flat, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE)
    # numpy ravels K/V blocks the same way we flatten in Rust (row-major)
    assert np.array_equal(k_rust, k_py.ravel())
    assert np.array_equal(v_rust, v_py.ravel())


@pytest.mark.parametrize("seq_len", [1, 15, 16, 17, 100])
def test_paged_to_flat_matches_python_baseline_byte_for_byte(seq_len):
    flat = _make_flat(seq_len, seed=seq_len + 2000)
    k_blocks, v_blocks = tardigrade_db.flat_to_paged(flat, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE)
    restored_rust = tardigrade_db.paged_to_flat(k_blocks, v_blocks, seq_len, NUM_KV_HEADS, HEAD_DIM)
    # The Python baseline takes (num_blocks, block_size, kv_heads, head_dim)-
    # shaped arrays; reshape ours to match.
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    k_shape = k_blocks.reshape(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    v_shape = v_blocks.reshape(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    restored_py = blocks_to_flat(k_shape, v_shape, seq_len, NUM_KV_HEADS, HEAD_DIM)
    assert np.array_equal(restored_rust, restored_py)


def test_seq_len_not_multiple_of_block_size_zero_pads_unused_slots():
    seq_len = 17  # one partially-filled block
    flat = _make_flat(seq_len, seed=42)
    k_blocks, _ = tardigrade_db.flat_to_paged(flat, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE)
    num_blocks = 2
    k_shape = k_blocks.reshape(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    # First 17 token slots have data; remaining 15 must be zero.
    assert np.all(k_shape.reshape(-1, NUM_KV_HEADS, HEAD_DIM)[17:32] == 0)


def test_block_size_one_is_legal():
    flat = _make_flat(seq_len=5, seed=7)
    k_blocks, v_blocks = tardigrade_db.flat_to_paged(flat, NUM_KV_HEADS, HEAD_DIM, 1)
    restored = tardigrade_db.paged_to_flat(k_blocks, v_blocks, 5, NUM_KV_HEADS, HEAD_DIM)
    assert np.array_equal(restored, flat)


def test_seq_len_zero_returns_empty_blocks():
    flat = np.zeros(0, dtype=np.float32)
    k_blocks, v_blocks = tardigrade_db.flat_to_paged(flat, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE)
    assert k_blocks.size == 0
    assert v_blocks.size == 0
    restored = tardigrade_db.paged_to_flat(k_blocks, v_blocks, 0, NUM_KV_HEADS, HEAD_DIM)
    assert restored.size == 0


def test_paged_to_flat_rejects_mismatched_seq_len():
    flat = _make_flat(seq_len=5, seed=11)
    k_blocks, v_blocks = tardigrade_db.flat_to_paged(flat, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE)
    # Asking for more tokens than the blocks can possibly contain.
    with pytest.raises(ValueError, match="seq_len"):
        tardigrade_db.paged_to_flat(k_blocks, v_blocks, 9999, NUM_KV_HEADS, HEAD_DIM)


def test_flat_to_paged_rejects_buffer_size_not_matching_seq_len():
    # Buffer must be a multiple of 2 * kv_dim; otherwise it can't be a valid
    # [K_flat | V_flat] structure.
    bad = np.zeros(2 * KV_DIM + 1, dtype=np.float32)  # 1 extra element
    with pytest.raises(ValueError, match="flat_kv"):
        tardigrade_db.flat_to_paged(bad, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE)
