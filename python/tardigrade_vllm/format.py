# Block format conversion between TardigradeDB and vLLM paged attention.
#
# TardigradeDB stores KV as flat arrays per layer: [K_flat | V_flat]
#   where each half is (seq_len, kv_heads * head_dim)
#
# vLLM stores KV in paged blocks: (num_blocks, block_size, num_heads, head_dim)
#   with separate K and V block tensors per layer

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def flat_to_blocks(flat_kv, num_kv_heads, head_dim, block_size):
    """Convert TardigradeDB flat KV array to vLLM paged blocks.

    Args:
        flat_kv: numpy array of shape (2 * seq_len * kv_dim,) — [K_flat | V_flat]
        num_kv_heads: number of KV attention heads
        head_dim: dimension per head
        block_size: vLLM block size (typically 16)

    Returns:
        (k_blocks, v_blocks) each of shape (num_blocks, block_size, num_kv_heads, head_dim)
    """
    kv_dim = num_kv_heads * head_dim
    half = len(flat_kv) // 2
    seq_len = half // kv_dim

    k_flat = flat_kv[:half].reshape(seq_len, num_kv_heads, head_dim)
    v_flat = flat_kv[half:].reshape(seq_len, num_kv_heads, head_dim)

    # Pad to full blocks
    num_blocks = (seq_len + block_size - 1) // block_size
    padded_len = num_blocks * block_size
    pad_len = padded_len - seq_len

    if pad_len > 0:
        k_pad = np.zeros((pad_len, num_kv_heads, head_dim), dtype=k_flat.dtype)
        v_pad = np.zeros((pad_len, num_kv_heads, head_dim), dtype=v_flat.dtype)
        k_flat = np.concatenate([k_flat, k_pad], axis=0)
        v_flat = np.concatenate([v_flat, v_pad], axis=0)

    k_blocks = k_flat.reshape(num_blocks, block_size, num_kv_heads, head_dim)
    v_blocks = v_flat.reshape(num_blocks, block_size, num_kv_heads, head_dim)

    return k_blocks, v_blocks


def blocks_to_flat(k_blocks, v_blocks, seq_len, num_kv_heads, head_dim):
    """Convert vLLM paged blocks to TardigradeDB flat KV array.

    Args:
        k_blocks: shape (num_blocks, block_size, num_kv_heads, head_dim)
        v_blocks: shape (num_blocks, block_size, num_kv_heads, head_dim)
        seq_len: actual sequence length (blocks may be padded)
        num_kv_heads: number of KV attention heads
        head_dim: dimension per head

    Returns:
        numpy array of shape (2 * seq_len * kv_dim,) — [K_flat | V_flat]
    """
    kv_dim = num_kv_heads * head_dim

    # Handle torch tensors
    if torch is not None and isinstance(k_blocks, torch.Tensor):
        k_blocks = k_blocks.detach().cpu().numpy()
        v_blocks = v_blocks.detach().cpu().numpy()

    # Flatten and trim to actual seq_len
    k_flat = k_blocks.reshape(-1, num_kv_heads, head_dim)[:seq_len]
    v_flat = v_blocks.reshape(-1, num_kv_heads, head_dim)[:seq_len]

    k_flat = k_flat.reshape(seq_len, kv_dim)
    v_flat = v_flat.reshape(seq_len, kv_dim)

    return np.concatenate([k_flat.ravel(), v_flat.ravel()]).astype(np.float32)
