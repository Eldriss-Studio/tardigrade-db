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

from .encoding import encode_per_token
from .hook import MemoryCellHandle, TardigradeHook, WriteDecision


class HuggingFaceKVHook(TardigradeHook):
    """Hook that captures K projections for storage and Q projections for queries.

    Stores per-token K vectors (without RoPE) as search keys.
    Queries with per-token Q vectors (without RoPE) for Q*K attention scoring.
    Skips position 0 (attention sink token) which dominates dot products.

    For GQA models: K heads are expanded to match Q heads via repeat_interleave
    for retrieval keys. The stored injection payload remains the compact original KV cache.

    Requires the model object for access to q_proj/k_proj weights.
    """

    def __init__(self, engine, owner=1, k=5, model_config=None, model=None, use_hidden_states=False):
        self.engine = engine
        self.owner = owner
        self.k = k
        self.model = model
        self.use_hidden_states = use_hidden_states

        if model_config is not None:
            self.hidden_size = model_config.hidden_size
            self.num_kv_heads = getattr(model_config, "num_key_value_heads", None)
            if self.num_kv_heads is None:
                self.num_kv_heads = model_config.num_attention_heads
            self.num_q_heads = model_config.num_attention_heads
            self.head_dim = getattr(model_config, "head_dim", model_config.hidden_size // self.num_q_heads)
            self.gqa_ratio = self.num_q_heads // self.num_kv_heads
            self.kv_dim = self.num_kv_heads * self.head_dim
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

        k_tokens = k[1:].detach().cpu().numpy().astype(np.float32)  # skip position 0
        return self._expand_k_tokens_for_gqa(k_tokens)

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

        return q[1:].detach().cpu().numpy().astype(np.float32)  # skip position 0

    def _expand_k_tokens_for_gqa(self, k_tokens):
        """Repeat KV heads so stored K retrieval keys match Q dimensions."""
        if (
            k_tokens is None
            or self.gqa_ratio <= 1
            or self.num_kv_heads is None
            or self.head_dim is None
        ):
            return k_tokens

        reshaped = k_tokens.reshape(-1, self.num_kv_heads, self.head_dim)
        expanded = np.repeat(reshaped, self.gqa_ratio, axis=1)
        return expanded.reshape(-1, self.q_dim).astype(np.float32)

    def _encode_per_token(self, token_vecs, dim):
        """Encode per-token vectors with Q4-safe sentinel header."""
        return encode_per_token(token_vecs, dim)

    def _extract_kv_payload(self, past_key_values, layer_idx):
        """Extract full K+V payload from past_key_values for injection."""
        layer_cache = past_key_values.layers[layer_idx]
        k = layer_cache.keys[0]   # (heads, seq, head_dim)
        v = layer_cache.values[0]
        h, s, d = k.shape
        kv_dim = h * d
        k_flat = k.permute(1, 0, 2).reshape(s, kv_dim).detach().cpu().numpy().astype(np.float32)
        v_flat = v.permute(1, 0, 2).reshape(s, kv_dim).detach().cpu().numpy().astype(np.float32)
        return np.concatenate([k_flat.ravel(), v_flat.ravel()])

    def on_generate(self, layer, past_key_values=None, hidden_states=None, model_hidden_states=None):
        """Store per-token vectors (skip position 0).

        Mode depends on use_hidden_states:
          False: stores K projections (without RoPE) for Q*K scoring
          True:  stores raw hidden states for symmetric scoring
        """
        if past_key_values is None and model_hidden_states is None:
            return WriteDecision(should_write=False)

        tokens = None
        if model_hidden_states is not None:
            h = model_hidden_states[0] if model_hidden_states.ndim == 3 else model_hidden_states

            if self.use_hidden_states:
                # Raw hidden states — symmetric, no projection
                tokens = h[1:].detach().cpu().numpy().astype(np.float32)
            else:
                tokens = self._project_k(h, layer)

        if tokens is not None and len(tokens) > 0:
            per_token_key = self._encode_per_token(tokens, tokens.shape[1])
        elif past_key_values is not None:
            # Fallback: use K from cache (has RoPE, skip pos 0)
            lc = past_key_values.layers[layer]
            k = lc.keys[0]
            h, s, d = k.shape
            kv_dim = h * d
            k_flat = k.permute(1, 0, 2).reshape(s, kv_dim)[1:].detach().cpu().numpy().astype(np.float32)
            k_flat = self._expand_k_tokens_for_gqa(k_flat)
            per_token_key = self._encode_per_token(k_flat, k_flat.shape[1])
        else:
            return WriteDecision(should_write=False)

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
        """Query with per-token vectors.

        Mode depends on use_hidden_states:
          False: Q projection for Q*K scoring
          True:  raw hidden states for symmetric scoring
        """
        if past_key_values is None and model_hidden_states is None:
            return []

        tokens = None
        if model_hidden_states is not None:
            h = model_hidden_states[0] if model_hidden_states.ndim == 3 else model_hidden_states

            if self.use_hidden_states:
                tokens = h[1:].detach().cpu().numpy().astype(np.float32)
            else:
                tokens = self._project_q(h, layer)

        if tokens is not None and len(tokens) > 0:
            query_tokens_2d = tokens
        elif past_key_values is not None:
            lc = past_key_values.layers[layer]
            k = lc.keys[0]
            h_heads, s, d = k.shape
            kv_dim = h_heads * d
            k_flat = k.permute(1, 0, 2).reshape(s, kv_dim)[1:].detach().cpu().numpy().astype(np.float32)
            query_tokens_2d = self._expand_k_tokens_for_gqa(k_flat)
        else:
            return []

        # Direct Token Query API: skip Python-side encode_per_token round-trip.
        # Engine builds the per-token encoded key in Rust (one allocation).
        # Falls back to encoded path only if mem_read_tokens is unavailable
        # (older engine builds).
        if hasattr(self.engine, "mem_read_tokens"):
            results = self.engine.mem_read_tokens(query_tokens_2d, self.k, self.owner)
        else:
            query_key = self._encode_per_token(query_tokens_2d, query_tokens_2d.shape[1])
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
