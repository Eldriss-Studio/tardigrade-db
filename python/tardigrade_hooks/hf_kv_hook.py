# HuggingFaceKVHook -- captures real KV cache tensors for TardigradeDB.
#
# Adapter pattern: translates HuggingFace model internals into
# TardigradeDB's flat f32 key/value API.
#
# Storage: per-token K projections (without RoPE, skip position 0)
# Retrieval: per-token Q projections (without RoPE, skip position 0)
#
# Q*K scoring matches how the model's attention actually works:
# Q vectors are trained to query, K vectors are trained to be queried.
# Using K*K fails because K vectors share a large common component
# across all sequences regardless of content.

import numpy as np
import torch

from .hook import MemoryCellHandle, TardigradeHook, WriteDecision


class HuggingFaceKVHook(TardigradeHook):
    """Hook that captures K projections for storage and Q projections for queries.

    Stores per-token K vectors (without RoPE) as search keys.
    Queries with per-token Q vectors (without RoPE) for Q*K attention scoring.
    Skips position 0 (attention sink token) which dominates dot products.

    For GQA models: Q heads are expanded to match K heads via repeat_interleave,
    handled automatically based on model config.

    Requires the model object for access to q_proj/k_proj weights.
    """

    def __init__(self, engine, owner=1, k=5, model_config=None, model=None):
        self.engine = engine
        self.owner = owner
        self.k = k
        self.model = model

        if model_config is not None:
            self.num_kv_heads = getattr(model_config, "num_key_value_heads", None)
            if self.num_kv_heads is None:
                self.num_kv_heads = model_config.num_attention_heads
            self.num_q_heads = model_config.num_attention_heads
            # head_dim: prefer explicit config, else derive from hidden_size
            self.head_dim = getattr(model_config, "head_dim", model_config.hidden_size // self.num_q_heads)
            self.gqa_ratio = self.num_q_heads // self.num_kv_heads
            # For per-token encoding: K dim (what we store)
            self.kv_dim = self.num_kv_heads * self.head_dim
            # Q dim after GQA expansion (what we query with)
            self.q_dim = self.num_q_heads * self.head_dim
        else:
            self.num_kv_heads = None
            self.num_q_heads = None
            self.head_dim = None
            self.gqa_ratio = 1
            self.kv_dim = None
            self.q_dim = None

    def _get_attn_layer(self, layer_idx):
        """Get the attention module for a given layer."""
        if self.model is None:
            return None
        # HuggingFace convention: model.model.layers[i].self_attn (Llama/Qwen)
        # or model.transformer.h[i].attn (GPT-2)
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx].self_attn
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[layer_idx].attn
        return None

    def _project_k(self, hidden_states, layer_idx):
        """Apply K projection (without RoPE) to hidden states.

        Returns per-token K vectors as numpy array, shape (seq-1, kv_dim).
        Skips position 0 (attention sink).
        """
        attn = self._get_attn_layer(layer_idx)
        if attn is None or not hasattr(attn, "k_proj"):
            return None

        with torch.no_grad():
            k = attn.k_proj(hidden_states)  # (seq, num_kv_heads * head_dim)
            if hasattr(attn, "k_norm"):
                k = k.view(-1, self.num_kv_heads, self.head_dim)
                k = attn.k_norm(k)
                k = k.view(-1, self.kv_dim)

        return k[1:].numpy().astype(np.float32)  # skip position 0

    def _project_q(self, hidden_states, layer_idx):
        """Apply Q projection (without RoPE) to hidden states.

        Returns per-token Q vectors as numpy array, shape (seq-1, q_dim).
        Skips position 0 (attention sink).
        """
        attn = self._get_attn_layer(layer_idx)
        if attn is None or not hasattr(attn, "q_proj"):
            return None

        with torch.no_grad():
            q = attn.q_proj(hidden_states)  # (seq, num_q_heads * head_dim)
            if hasattr(attn, "q_norm"):
                q = q.view(-1, self.num_q_heads, self.head_dim)
                q = attn.q_norm(q)
                q = q.view(-1, self.q_dim)

        return q[1:].numpy().astype(np.float32)  # skip position 0

    def _encode_per_token(self, token_vecs, dim):
        """Encode per-token vectors with sentinel header."""
        n = len(token_vecs)
        header = np.array([-1.0e9, float(n), float(dim)], dtype=np.float32)
        return np.concatenate([header, token_vecs.ravel()])

    def _extract_kv_payload(self, past_key_values, layer_idx):
        """Extract full K+V payload from past_key_values for injection."""
        layer_cache = past_key_values.layers[layer_idx]
        k = layer_cache.keys[0]   # (heads, seq, head_dim)
        v = layer_cache.values[0]
        h, s, d = k.shape
        kv_dim = h * d
        k_flat = k.permute(1, 0, 2).reshape(s, kv_dim).numpy().astype(np.float32)
        v_flat = v.permute(1, 0, 2).reshape(s, kv_dim).numpy().astype(np.float32)
        return np.concatenate([k_flat.ravel(), v_flat.ravel()])

    def on_generate(self, layer, past_key_values=None, hidden_states=None, model_hidden_states=None):
        """Store per-token K projections (without RoPE, skip position 0).

        Requires either model_hidden_states (from output_hidden_states=True)
        or falls back to past_key_values K (with RoPE, less ideal).
        """
        if past_key_values is None:
            return WriteDecision(should_write=False)

        # Try K projection without RoPE (preferred)
        k_tokens = None
        if model_hidden_states is not None:
            h = model_hidden_states[0] if model_hidden_states.ndim == 3 else model_hidden_states
            k_tokens = self._project_k(h, layer)

        if k_tokens is not None and len(k_tokens) > 0:
            per_token_key = self._encode_per_token(k_tokens, self.kv_dim)
        else:
            # Fallback: use K from cache (has RoPE, skip pos 0)
            lc = past_key_values.layers[layer]
            k = lc.keys[0]
            h, s, d = k.shape
            kv_dim = h * d
            k_flat = k.permute(1, 0, 2).reshape(s, kv_dim)[1:].numpy().astype(np.float32)
            per_token_key = self._encode_per_token(k_flat, kv_dim)

        flat_kv = self._extract_kv_payload(past_key_values, layer)

        norm = float(np.linalg.norm(per_token_key[3:]) / max(len(per_token_key) - 3, 1))
        salience = min(norm * 50.0, 100.0)

        return WriteDecision(
            should_write=True,
            salience=salience,
            key=per_token_key,
            value=flat_kv,
        )

    def on_prefill(self, layer, past_key_values=None, query_states=None, model_hidden_states=None):
        """Query with per-token Q projections (Q*K scoring).

        Uses Q projection (without RoPE) for queries against stored K vectors.
        For GQA: expands K stored keys by repeating heads to match Q dimension.
        """
        if past_key_values is None and model_hidden_states is None:
            return []

        # Get Q projection (preferred) or fallback to K
        q_tokens = None
        if model_hidden_states is not None:
            h = model_hidden_states[0] if model_hidden_states.ndim == 3 else model_hidden_states
            q_tokens = self._project_q(h, layer)

        if q_tokens is not None and len(q_tokens) > 0:
            # For GQA: stored K has kv_dim, Q has q_dim (larger).
            # Use mean-pooled Q as a single query vector (same dim as mean-pooled K).
            # This gives the best Q*K signal through the existing retriever.
            q_mean = q_tokens.mean(axis=0).astype(np.float32)
            # Average Q heads to match K head count for dot product compatibility.
            if self.gqa_ratio > 1:
                q_reshaped = q_mean.reshape(self.num_q_heads, self.head_dim)
                q_grouped = q_reshaped.reshape(self.num_kv_heads, self.gqa_ratio, self.head_dim)
                q_for_search = q_grouped.mean(axis=1).reshape(self.kv_dim)
            else:
                q_for_search = q_mean

            query_key = q_for_search
        elif past_key_values is not None:
            # Fallback: use K from cache
            lc = past_key_values.layers[layer]
            k = lc.keys[0]
            h_heads, s, d = k.shape
            kv_dim = h_heads * d
            k_flat = k.permute(1, 0, 2).reshape(s, kv_dim)[1:].numpy().astype(np.float32)
            query_key = self._encode_per_token(k_flat, kv_dim)
        else:
            return []

        results = self.engine.mem_read(query_key, self.k, self.owner)

        return [
            MemoryCellHandle(
                cell_id=r.cell_id,
                owner=r.owner,
                layer=r.layer,
                score=r.score,
                key=np.array(r.key(), dtype=np.float32),
                value=np.array(r.value(), dtype=np.float32),
            )
            for r in results
        ]
