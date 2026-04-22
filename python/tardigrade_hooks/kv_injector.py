"""KV cache injection — Adapter pattern for TardigradeDB → PyTorch attention cache.

Reshapes TardigradeDB's flat f32 vectors into PyTorch's `past_key_values`
format and injects them into a HuggingFace DynamicCache.

The core transformation:
    flat vector (d_model,) → reshaped (1, num_heads, 1, head_dim)

where d_model = num_heads × head_dim.
"""

import numpy as np
import torch
from transformers import DynamicCache

from .hook import MemoryCellHandle


def reshape_to_kv(
    flat_vector: np.ndarray,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Reshape a flat f32 vector to PyTorch KV cache format.

    Args:
        flat_vector: Shape (d_model,) where d_model = num_heads * head_dim.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.

    Returns:
        Tensor of shape (1, num_heads, 1, head_dim) — one cache entry for one token.
    """
    d_model = num_heads * head_dim
    if len(flat_vector) != d_model:
        msg = f"Vector length {len(flat_vector)} != num_heads({num_heads}) * head_dim({head_dim}) = {d_model}"
        raise ValueError(msg)

    tensor = torch.tensor(flat_vector, dtype=torch.float32)
    return tensor.reshape(1, num_heads, 1, head_dim)


def inject_into_cache(
    cache: DynamicCache,
    layer_idx: int,
    handles: list[MemoryCellHandle],
    num_heads: int,
    head_dim: int,
) -> DynamicCache:
    """Inject retrieved memory cells into a DynamicCache at a specific layer.

    Each handle's key and value vectors are reshaped to (1, num_heads, 1, head_dim)
    and appended to the cache via cache.update().

    Args:
        cache: The DynamicCache to extend (modified in-place and returned).
        layer_idx: Which transformer layer to inject into.
        handles: List of MemoryCellHandle objects from TardigradeDB retrieval.
        num_heads: Number of attention heads in the model.
        head_dim: Dimension per attention head.

    Returns:
        The same cache object, now containing the injected entries.
    """
    for handle in handles:
        key_tensor = reshape_to_kv(handle.key, num_heads, head_dim)
        value_tensor = reshape_to_kv(handle.value, num_heads, head_dim)
        cache.update(key_states=key_tensor, value_states=value_tensor, layer_idx=layer_idx)

    return cache


def prepare_injection(
    cache: DynamicCache,
    input_ids: torch.Tensor,
) -> dict:
    """Prepare model forward kwargs that account for injected KV cache entries.

    When injecting past KV entries, the model needs adjusted position_ids
    and attention_mask to account for the extra cache length.

    Args:
        cache: A DynamicCache with injected entries.
        input_ids: The input token IDs tensor (batch, seq_len).

    Returns:
        Dict of kwargs to pass to model.forward(): past_key_values, position_ids, attention_mask.
    """
    cache_len = cache.get_seq_length()
    seq_len = input_ids.shape[1]
    total_len = cache_len + seq_len

    position_ids = torch.arange(cache_len, total_len, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones(1, total_len, dtype=torch.long)

    return {
        "past_key_values": cache,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }


def build_injection_cache(
    handles_by_layer: dict[int, list[MemoryCellHandle]],
    num_heads: int,
    head_dim: int,
    num_layers: int,
) -> DynamicCache:
    """Build a complete DynamicCache from retrieved memory cells across layers.

    Args:
        handles_by_layer: Mapping of layer_idx → list of handles to inject.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        num_layers: Total number of layers in the model (for empty-layer padding).

    Returns:
        A DynamicCache ready to pass as `past_key_values` to model.forward().
    """
    cache = DynamicCache()

    for layer_idx in range(num_layers):
        handles = handles_by_layer.get(layer_idx, [])
        if handles:
            inject_into_cache(cache, layer_idx, handles, num_heads, head_dim)

    return cache
