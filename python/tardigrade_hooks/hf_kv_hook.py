# HuggingFaceKVHook -- captures real KV cache tensors from past_key_values.
#
# Adapter pattern: translates HuggingFace DynamicCache format into
# TardigradeDB's flat f32 key/value API.
#
# Dual-store pattern:
#   key   = mean-pooled K projection (search index, one vector per memory)
#   value = flattened per-token K + V (injection payload, full fidelity)
#
# This replaces the hidden-state proxy approach in hf_hook.py for any
# use case that needs real KV injection quality.

import numpy as np

from .hook import MemoryCellHandle, TardigradeHook, WriteDecision


class HuggingFaceKVHook(TardigradeHook):
    """Hook that captures actual K/V projections from past_key_values.

    Unlike HuggingFaceHook (which uses hidden_states), this hook extracts
    the projected K and V tensors that the model's attention mechanism
    was trained to use. These produce dramatically better retrieval
    quality (75% vs 31% recall in validation experiments).

    Requires the model to be called with use_cache=True.
    """

    def __init__(self, engine, owner=1, k=5, model_config=None):
        """Initialize the KV hook.

        Args:
            engine: A tardigrade_db.Engine instance.
            owner: Owner/agent ID for memory isolation.
            k: Number of cells to retrieve on prefill.
            model_config: HuggingFace model config (for head count / dim).
        """
        self.engine = engine
        self.owner = owner
        self.k = k

        # Extract head geometry from model config.
        if model_config is not None:
            self.num_heads = getattr(model_config, "num_key_value_heads", None)
            if self.num_heads is None:
                self.num_heads = model_config.num_attention_heads
            self.head_dim = model_config.hidden_size // model_config.num_attention_heads
        else:
            self.num_heads = None
            self.head_dim = None

    def _extract_layer_keys(self, past_key_values, layer):
        """Extract K tensor from past_key_values at a given layer.

        Returns (mean_k, flat_kv):
            mean_k: Mean-pooled K across tokens, shape (num_heads * head_dim,).
            flat_kv: Flattened K + V for all tokens, shape (2 * seq * num_heads * head_dim,).
        """
        # DynamicCache API: .layers[i].keys / .layers[i].values
        layer_cache = past_key_values.layers[layer]
        k_tensor = layer_cache.keys   # (batch, heads, seq, head_dim)
        v_tensor = layer_cache.values  # (batch, heads, seq, head_dim)

        # Take first batch element.
        k = k_tensor[0]  # (heads, seq, head_dim)
        v = v_tensor[0]

        h, s, d = k.shape

        # Mean-pooled K: (heads, seq, head_dim) -> (seq, heads*head_dim) -> mean -> (heads*head_dim,)
        k_flat_per_token = k.permute(1, 0, 2).reshape(s, h * d)  # (seq, heads*head_dim)
        mean_k = k_flat_per_token.mean(dim=0).numpy().astype(np.float32)

        # Full K+V payload: flatten both and concatenate.
        k_flat = k.permute(1, 0, 2).reshape(s, h * d).numpy().astype(np.float32)
        v_flat = v.permute(1, 0, 2).reshape(s, h * d).numpy().astype(np.float32)
        flat_kv = np.concatenate([k_flat.ravel(), v_flat.ravel()])

        return mean_k, flat_kv

    def on_generate(self, layer, past_key_values=None, hidden_states=None):
        """Capture real K/V projections from past_key_values.

        Args:
            layer: Transformer layer index.
            past_key_values: The model's DynamicCache (from use_cache=True).
            hidden_states: Ignored (kept for ABC compatibility).

        Returns:
            WriteDecision with:
                key = mean-pooled K (search index)
                value = flattened K+V (injection payload)
                salience = L2 norm of mean K
        """
        if past_key_values is None:
            return WriteDecision(should_write=False)

        mean_k, flat_kv = self._extract_layer_keys(past_key_values, layer)

        norm = float(np.linalg.norm(mean_k))
        salience = min(norm * 10.0, 100.0)

        return WriteDecision(
            should_write=True,
            salience=salience,
            key=mean_k,
            value=flat_kv,
        )

    def on_prefill(self, layer, past_key_values=None, query_states=None):
        """Retrieve relevant memories using mean-pooled K as query.

        Args:
            layer: Transformer layer index to query from.
            past_key_values: The query's DynamicCache (from use_cache=True).
            query_states: Ignored (kept for ABC compatibility).

        Returns:
            List of MemoryCellHandle objects.
        """
        if past_key_values is None:
            return []

        mean_k, _ = self._extract_layer_keys(past_key_values, layer)

        results = self.engine.mem_read(mean_k, self.k, self.owner)

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
