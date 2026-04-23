"""MemoryInjector — Decorator pattern for transparent KV cache injection.

Wraps a HuggingFace model, intercepting forward() to:
1. Compute a cheap query via the model's embedding layer (no attention)
2. Retrieve relevant KV memories from TardigradeDB
3. Build a DynamicCache with injected entries
4. Run a single forward pass with the injected cache

Callers use model(input_ids) or model.generate() as usual — the injection
is transparent. On empty retrieval, behavior is identical to the unwrapped
model (Decorator contract).

Approach C: hook-based, single forward pass, embedding-based query.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import DynamicCache

from .hook import MemoryCellHandle
from .kv_injector import build_injection_cache, prepare_injection
from .position import AbsolutePositionEncoder, PositionEncoder


class MemoryInjector(nn.Module):
    """Decorator that injects TardigradeDB memories into a HuggingFace model.

    Wraps the model's forward pass: retrieves relevant KV cache entries
    from the engine and prepends them to the attention cache. Transparent
    to callers — when no memories match, output is identical to the
    unwrapped model.

    Args:
        model: A HuggingFace causal LM (e.g., GPT2LMHeadModel).
        engine: A tardigrade_db.Engine instance.
        owner: Agent/owner ID for memory isolation.
        position_encoder: Strategy for position ID remapping.
        k: Number of memory cells to retrieve per layer.
    """

    def __init__(
        self,
        model: nn.Module,
        engine,
        owner: int = 1,
        position_encoder: Optional[PositionEncoder] = None,
        k: int = 5,
    ):
        super().__init__()
        self.model = model
        self.engine = engine
        self.owner = owner
        self.position_encoder = position_encoder or AbsolutePositionEncoder()
        self.k = k

        # Extract model config for cache construction.
        self.config = model.config
        self.num_heads = self.config.n_head
        self.head_dim = self.config.n_embd // self.config.n_head
        self.num_layers = self.config.n_layer

        # Track whether we're in a generate() call to avoid double-injection.
        self._generating = False

    def _get_embedding_layer(self) -> nn.Embedding:
        """Find the token embedding layer in the wrapped model."""
        # HuggingFace convention: model.transformer.wte (GPT-2)
        # or model.model.embed_tokens (Llama, Qwen).
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "wte"):
            return self.model.transformer.wte
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens
        msg = "Cannot find embedding layer. Supported: GPT-2 (.transformer.wte), Llama (.model.embed_tokens)"
        raise AttributeError(msg)

    def _query_from_embeddings(self, input_ids: torch.Tensor) -> np.ndarray:
        """Compute a retrieval query from token embeddings (no attention needed).

        Mean-pools the embedding vectors across the sequence dimension
        to produce a single d_model-dimensional query vector.
        """
        embed_layer = self._get_embedding_layer()
        with torch.no_grad():
            embeddings = embed_layer(input_ids)  # (batch, seq, d_model)
        # Mean across sequence, take first batch element.
        mean_emb = embeddings[0].mean(dim=0).numpy().astype(np.float32)
        return mean_emb

    def _retrieve_by_layer(self, query: np.ndarray) -> dict[int, list[MemoryCellHandle]]:
        """Retrieve memory cells from TardigradeDB, grouped by layer."""
        results = self.engine.mem_read(query, self.k, self.owner)
        if not results:
            return {}

        handles_by_layer: dict[int, list[MemoryCellHandle]] = {}
        for r in results:
            handle = MemoryCellHandle(
                cell_id=r.cell_id,
                owner=r.owner,
                layer=r.layer,
                score=r.score,
                key=np.array(r.key(), dtype=np.float32),
                value=np.array(r.value(), dtype=np.float32),
            )
            handles_by_layer.setdefault(handle.layer, []).append(handle)

        return handles_by_layer

    def build_memory_cache(self, input_ids: torch.Tensor) -> Optional[DynamicCache]:
        """Build a DynamicCache from retrieved memories for the given input.

        Routes each handle to its original layer. For layers without handles,
        pads with zero-valued KV entries to maintain uniform cache depth
        (required by models like GPT-2).

        Supports both legacy single-token handles and dual-store KV payloads.

        Public for testing — allows inspection of cache contents before forward.

        Returns:
            A DynamicCache with injected KV entries, or None if no memories found.
        """
        query = self._query_from_embeddings(input_ids)
        handles_by_layer = self._retrieve_by_layer(query)

        if not handles_by_layer:
            return None

        # Detect if handles are dual-store (full KV payload) or legacy (single token).
        kv_dim = self.num_heads * self.head_dim
        sample_handle = next(iter(next(iter(handles_by_layer.values()))))
        is_dual = len(sample_handle.value) > kv_dim

        if is_dual:
            # Dual-store: each handle contains full per-token K+V.
            # Figure out the seq length from the payload, then pad missing
            # layers with zeros BEFORE building the cache for uniform depth.
            payload_len = len(sample_handle.value)
            tokens_per_handle = payload_len // (2 * kv_dim)

            # Create zero-payload handles for missing layers.
            zero_payload = np.zeros(payload_len, dtype=np.float32)
            zero_key = np.zeros(kv_dim, dtype=np.float32)
            zero_handle = MemoryCellHandle(
                cell_id=0, owner=self.owner, layer=0, score=0.0,
                key=zero_key, value=zero_payload,
            )

            padded: dict[int, list[MemoryCellHandle]] = {}
            for layer_idx in range(self.num_layers):
                padded[layer_idx] = handles_by_layer.get(layer_idx, [zero_handle])

            cache = build_injection_cache(
                handles_by_layer=padded,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                num_layers=self.num_layers,
            )
        else:
            # Legacy: single-token handles. Pad with zero handles for uniform depth.
            max_handles = max(len(h) for h in handles_by_layer.values())
            zero_handle = MemoryCellHandle(
                cell_id=0, owner=self.owner, layer=0, score=0.0,
                key=np.zeros(kv_dim, dtype=np.float32),
                value=np.zeros(kv_dim, dtype=np.float32),
            )

            padded_by_layer: dict[int, list[MemoryCellHandle]] = {}
            for layer_idx in range(self.num_layers):
                layer_handles = handles_by_layer.get(layer_idx, [])
                padding = [zero_handle] * (max_handles - len(layer_handles))
                padded_by_layer[layer_idx] = layer_handles + padding

            cache = build_injection_cache(
                handles_by_layer=padded_by_layer,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                num_layers=self.num_layers,
            )

        return cache

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        **kwargs,
    ):
        """Forward pass with transparent memory injection.

        On the first call (prefill), retrieves memories and injects them
        as past_key_values. On subsequent calls (decode during generate),
        passes through without re-injection.
        """
        # During generate(), HF calls forward() multiple times:
        # - First call: prefill (inject here)
        # - Subsequent calls: decode (pass through, cache already populated)
        if past_key_values is not None:
            # Decode step — cache already contains our injections.
            return self.model(input_ids, past_key_values=past_key_values, **kwargs)

        # Prefill step — attempt memory injection.
        cache = self.build_memory_cache(input_ids)

        if cache is None:
            # No memories found — transparent pass-through.
            return self.model(input_ids, **kwargs)

        # Build adjusted position_ids and attention_mask.
        fwd_kwargs = prepare_injection(cache, input_ids)
        fwd_kwargs.update(kwargs)

        return self.model(input_ids, **fwd_kwargs)

    def generate(self, input_ids: torch.Tensor, **kwargs):
        """Generate tokens with memory injection on the prefill step.

        Builds the memory cache once, then delegates to the wrapped
        model's generate() with the injected past_key_values.
        """
        cache = self.build_memory_cache(input_ids)

        if cache is None:
            return self.model.generate(input_ids, **kwargs)

        fwd_kwargs = prepare_injection(cache, input_ids)
        return self.model.generate(
            input_ids,
            past_key_values=fwd_kwargs["past_key_values"],
            attention_mask=fwd_kwargs["attention_mask"],
            **kwargs,
        )
