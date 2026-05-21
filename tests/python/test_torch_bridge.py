"""Acceptance tests for the torch-tensor bridge functions.

These accept a torch.Tensor and read its raw buffer via data_ptr() so
the caller doesn't have to round-trip through numpy. No libtorch is
linked into the Rust binary — the consumer's torch handles all the
C++ side; Rust just reads the memory.

Tests are gated on torch being importable. If torch isn't installed,
the bridge functions either don't exist or raise a clear error.
"""

import numpy as np
import pytest

import tardigrade_db


torch = pytest.importorskip("torch")


KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 16
KV_DIM = KV_HEADS * HEAD_DIM


def test_torch_path_pack_contents_equal_numpy_path(tmp_path):
    """Same buffer, two paths, byte-identical pack contents."""
    seq_len = 100
    rng = np.random.default_rng(0)
    flat_np = rng.standard_normal(2 * seq_len * KV_DIM).astype(np.float32)
    flat_torch = torch.from_numpy(flat_np.copy())

    k_np, v_np = tardigrade_db.flat_to_paged(flat_np, KV_HEADS, HEAD_DIM, BLOCK_SIZE)
    k_t, v_t = tardigrade_db.flat_to_paged_torch(flat_torch, KV_HEADS, HEAD_DIM, BLOCK_SIZE)

    assert np.array_equal(k_t, k_np)
    assert np.array_equal(v_t, v_np)


def test_torch_path_round_trips_losslessly():
    seq_len = 17
    rng = np.random.default_rng(1)
    flat = torch.from_numpy(rng.standard_normal(2 * seq_len * KV_DIM).astype(np.float32))
    k_blocks, v_blocks = tardigrade_db.flat_to_paged_torch(flat, KV_HEADS, HEAD_DIM, BLOCK_SIZE)
    restored = tardigrade_db.paged_to_flat(k_blocks, v_blocks, seq_len, KV_HEADS, HEAD_DIM)
    assert np.array_equal(restored, flat.numpy())


def test_torch_path_rejects_cuda_tensor_with_clear_message():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(2)
    flat = torch.from_numpy(rng.standard_normal(2 * KV_DIM).astype(np.float32)).cuda()
    with pytest.raises(ValueError, match="cpu"):
        tardigrade_db.flat_to_paged_torch(flat, KV_HEADS, HEAD_DIM, BLOCK_SIZE)


def test_torch_path_rejects_non_float32_dtype_with_clear_message():
    rng = np.random.default_rng(3)
    flat = torch.from_numpy(rng.standard_normal(2 * KV_DIM).astype(np.float32)).to(torch.float16)
    with pytest.raises(ValueError, match="float32"):
        tardigrade_db.flat_to_paged_torch(flat, KV_HEADS, HEAD_DIM, BLOCK_SIZE)


def test_torch_path_rejects_non_contiguous_tensor():
    """A transposed/sliced tensor's data_ptr() doesn't point at a
    contiguous f32 buffer; reject loudly rather than silently corrupt."""
    rng = np.random.default_rng(4)
    base = torch.from_numpy(rng.standard_normal(4 * KV_DIM).astype(np.float32))
    sliced = base[::2]  # non-contiguous view
    with pytest.raises(ValueError, match="contiguous"):
        tardigrade_db.flat_to_paged_torch(sliced, KV_HEADS, HEAD_DIM, BLOCK_SIZE)


def test_torch_path_detaches_safely_from_autograd():
    """A grad-tracking tensor must not retain a reference to the engine
    write path — Rust must read the data and let the autograd graph go."""
    flat = torch.randn(2 * KV_DIM, requires_grad=True)
    # No exception, behaviour identical to a non-grad tensor.
    k_blocks, v_blocks = tardigrade_db.flat_to_paged_torch(
        flat, KV_HEADS, HEAD_DIM, BLOCK_SIZE,
    )
    expected_k, expected_v = tardigrade_db.flat_to_paged(
        flat.detach().numpy(), KV_HEADS, HEAD_DIM, BLOCK_SIZE,
    )
    assert np.array_equal(k_blocks, expected_k)
    assert np.array_equal(v_blocks, expected_v)


def test_legacy_numpy_path_still_works_alongside_torch_path():
    rng = np.random.default_rng(5)
    flat_np = rng.standard_normal(2 * KV_DIM).astype(np.float32)
    flat_t = torch.from_numpy(flat_np.copy())
    assert tardigrade_db.flat_to_paged(flat_np, KV_HEADS, HEAD_DIM, BLOCK_SIZE) is not None
    assert tardigrade_db.flat_to_paged_torch(flat_t, KV_HEADS, HEAD_DIM, BLOCK_SIZE) is not None
