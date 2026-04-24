# Multi-pack composition strategies for KV cache injection.
#
# Strategy pattern: different approaches to combining multiple retrieved
# KV packs into a single DynamicCache for injection.
#
# - NaiveConcatComposer: concatenates K/V tensors per layer. Cheap but
#   produces non-monotonic RoPE positions (3/10 on cross-referencing queries).
# - SequentialRecomputeComposer: processes facts sequentially through the
#   model, building contextual KV. Requires stored fact text (1/10).
# - RoPECorrectedConcatComposer: fixes RoPE positions before concatenation
#   using unrotate/re-rotate (CacheBlend approach, EuroSys 2025).

from abc import ABC, abstractmethod

import numpy as np
import torch
from transformers import DynamicCache


class CompositionStrategy(ABC):
    """Protocol for composing multiple KV packs into one DynamicCache."""

    @abstractmethod
    def compose(self, packs, num_kv_heads, head_dim, kv_dim, n_layers):
        """Compose multiple pack dicts into a single DynamicCache.

        Args:
            packs: list of pack dicts from engine.mem_read_pack().
                   Each has {"layers": [{"layer_idx": int, "data": list[float]}, ...]}.
            num_kv_heads: number of KV attention heads.
            head_dim: dimension per head.
            kv_dim: num_kv_heads * head_dim.
            n_layers: total model layers.

        Returns:
            DynamicCache with combined KV entries.
        """


class NaiveConcatComposer(CompositionStrategy):
    """Concatenate K/V tensors from multiple packs per layer.

    For each layer, the K and V tensors from all packs are concatenated
    along the sequence dimension. The resulting cache has seq_len equal
    to the sum of all individual pack seq_lens.

    Warning: packs carry RoPE rotations at their original positions.
    Concatenation produces non-monotonic position sequences which may
    degrade accuracy on RoPE models (Knowledge Packs reports -6%).
    """

    def compose(self, packs, num_kv_heads, head_dim, kv_dim, n_layers):
        cache = DynamicCache()

        for layer_idx in range(n_layers):
            all_k = []
            all_v = []

            for pack in packs:
                layer_info = _find_layer(pack, layer_idx)
                if layer_info is None:
                    continue

                kt, vt, _ = _unpack_layer(layer_info, num_kv_heads, head_dim, kv_dim)
                all_k.append(kt)
                all_v.append(vt)

            if all_k:
                cache.update(
                    torch.cat(all_k, dim=2),
                    torch.cat(all_v, dim=2),
                    layer_idx,
                )

        return cache


class RoPECorrectedConcatComposer(CompositionStrategy):
    """Concatenate K/V tensors with RoPE position correction.

    Based on CacheBlend (EuroSys 2025 Best Paper). Each pack's K vectors
    carry RoPE rotations at their original positions [0..N]. This composer
    unrotates them and re-rotates at contiguous positions so the combined
    cache has monotonically increasing positions.

    V vectors are unchanged — RoPE only affects K.

    Accepts a PositionEncoder via constructor injection:
    - RoPEPositionEncoder: applies unrotate/re-rotate (Qwen, Llama)
    - AbsolutePositionEncoder: no-op, degrades to NaiveConcatComposer (GPT-2)
    """

    def __init__(self, position_encoder):
        self.position_encoder = position_encoder

    def compose(self, packs, num_kv_heads, head_dim, kv_dim, n_layers):
        cache = DynamicCache()

        for layer_idx in range(n_layers):
            all_k = []
            all_v = []
            cumulative_offset = 0

            for pack in packs:
                layer_info = _find_layer(pack, layer_idx)
                if layer_info is None:
                    continue

                kt, vt, seq_len = _unpack_layer(layer_info, num_kv_heads, head_dim, kv_dim)

                # Fix RoPE: remap K vectors to contiguous positions
                old_positions = torch.arange(seq_len)
                kt = self.position_encoder.remap_keys(kt, old_positions, cumulative_offset)

                all_k.append(kt)
                all_v.append(vt)  # V unchanged — RoPE only affects K
                cumulative_offset += seq_len

            if all_k:
                cache.update(
                    torch.cat(all_k, dim=2),
                    torch.cat(all_v, dim=2),
                    layer_idx,
                )

        return cache


class SequentialRecomputeComposer(CompositionStrategy):
    """Recompute KV cache sequentially through the model.

    Processes each fact through the model in order, with previous
    facts' KV already in the cache. This produces a coherent cache
    where each fact is contextualized by all preceding facts —
    correct RoPE positions and cross-fact attention.

    Requires model, tokenizer, and a text_registry mapping pack_id
    to the original fact text.
    """

    def __init__(self, model, tokenizer, text_registry):
        self.model = model
        self.tokenizer = tokenizer
        self.text_registry = text_registry

    def compose(self, packs, num_kv_heads, head_dim, kv_dim, n_layers):
        accumulated_cache = None

        for pack in packs:
            pack_id = pack["pack_id"]
            fact_text = self.text_registry.get(pack_id)
            if fact_text is None:
                continue

            messages = [{"role": "system", "content": fact_text}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            input_ids = self.tokenizer.encode(formatted, return_tensors="pt")

            # Build attention mask covering accumulated cache + new tokens
            if accumulated_cache is not None:
                kv_len = accumulated_cache.get_seq_length()
                q_len = input_ids.shape[1]
                attn_mask = torch.ones(1, kv_len + q_len, dtype=torch.long)
            else:
                attn_mask = None

            with torch.no_grad():
                out = self.model(
                    input_ids,
                    past_key_values=accumulated_cache,
                    attention_mask=attn_mask,
                    use_cache=True,
                )

            accumulated_cache = out.past_key_values

        if accumulated_cache is None:
            return DynamicCache()

        return accumulated_cache


def _unpack_layer(layer_info, num_kv_heads, head_dim, kv_dim):
    """Unpack a layer dict into K and V tensors.

    Returns (kt, vt, seq_len) where kt and vt have shape
    (1, num_kv_heads, seq_len, head_dim).
    """
    val = np.array(layer_info["data"], dtype=np.float32)
    half = len(val) // 2
    seq_len = half // kv_dim

    kt = torch.tensor(val[:half]).reshape(
        1, seq_len, num_kv_heads, head_dim
    ).permute(0, 2, 1, 3)
    vt = torch.tensor(val[half:]).reshape(
        1, seq_len, num_kv_heads, head_dim
    ).permute(0, 2, 1, 3)

    return kt, vt, seq_len


def _find_layer(pack, layer_idx):
    """Find a layer by index in a pack dict."""
    for layer_info in pack["layers"]:
        if layer_info["layer_idx"] == layer_idx:
            return layer_info
    return None
