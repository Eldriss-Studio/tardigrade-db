"""Pluggable retrieval-key encoding strategies for `KnowledgePackStore`.

Design pattern: **Strategy**.

Different model architectures encode retrieval-discriminative information
in different parts of the residual stream. The calibration sweep in
`tardigrade_hooks.calibrate` showed:

- **Uniform-softmax models** (Qwen3, Llama-3, Mistral, Gemma-2): mean-pooled
  hidden states at a deep semantic layer encode retrieval well (Qwen3 hits
  20/20 top-1 across most layers above the embeddings).
- **Hybrid-attention models** (RecurrentGemma, Qwen3-Next, Jamba, Zamba):
  mean-pooled hidden states *flatline* at every layer (RecurrentGemma best
  was 3/20 = 15% across all 27 layers). Per Michalak & Abreu 2025 the
  retrieval signal in those models lives in attention heads, not in the
  full residual stream. Switching the retrieval-key encoding to the K
  projections at a single softmax layer lifts RecurrentGemma from 3/20
  to **20/20 top-1** — same engine, same corpus, different feature.

This module ships two concrete strategies:

- :class:`HiddenStateKeyStrategy` — encodes from
  ``output.hidden_states[query_layer]``. Pre-Phase-2 behavior. Default for
  uniform-softmax models.
- :class:`KVectorKeyStrategy` — encodes from the K projections at a
  specific softmax attention layer (``cache.layers[softmax_layer_idx]
  .keys[0]``). Designed for hybrid models where hidden-state encoding
  fails. Today's Qwen3 sweep showed it also works on uniform-softmax
  models at specific layers (Qwen3 layer 26 K-vector hits 19/20) — so
  the strategy is not architecture-locked; calibration picks the winner
  empirically.

Both share a single ``compute(hidden_states, kv, hidden_size)`` interface
that takes already-captured forward-pass outputs. This lets the
calibration sweep do ONE forward per fact/query, then iterate cheaply
over many ``(strategy, layer)`` candidates without re-running the model.

# Not to be confused with `tardigrade_vllm.retrieval_key`

The vLLM serving path (in :mod:`tardigrade_vllm.retrieval_key`) has its
own ``RetrievalKeyStrategy`` ABC with strategies that compute keys from
the **token embedding table** (no forward pass — required for serving
latency). That's a different problem. The hooks-side strategies here
DO run a forward pass and use the resulting hidden states / K vectors
directly. Both ABCs may unify in the future; for now they coexist.

# Example

>>> from tardigrade_hooks import HiddenStateKeyStrategy, KVectorKeyStrategy
>>> # Uniform-softmax default: penultimate hidden state.
>>> kps = KnowledgePackStore(
...     engine, model, tok,
...     retrieval_key_strategy=HiddenStateKeyStrategy(query_layer=17),
... )
>>> # Hybrid model: K vectors at a specific softmax layer.
>>> kps = KnowledgePackStore(
...     engine, recurrent_gemma_model, tok,
...     retrieval_key_strategy=KVectorKeyStrategy(softmax_layer_idx=11),
... )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ._hidden_states import _is_softmax_cache_layer
from .encoding import encode_per_token


class RetrievalKeyStrategy(ABC):
    """Encode forward-pass outputs into a retrieval key.

    Subclasses implement ``compute()`` against pre-captured features
    (per-layer hidden states and the populated DynamicCache). This
    keeps the calibration sweep efficient: one forward pass per text,
    many strategies tested against the same features.
    """

    @abstractmethod
    def compute(
        self,
        hidden_states: list[np.ndarray],
        kv: Any,  # DynamicCache from transformers; Any to avoid import cycle
        hidden_size: int,
    ) -> np.ndarray:
        """Return the per-token retrieval key for the text whose forward
        produced ``hidden_states`` and ``kv``.

        Args:
            hidden_states: One ``(seq_len, dim)`` numpy array per index in
                ``output.hidden_states`` — index 0 is embeddings, indices
                1..n are per-layer outputs.
            kv: The populated ``DynamicCache`` from the same forward pass.
                Strategies that read K/V tensors index into ``kv.layers``.
            hidden_size: Model's ``config.hidden_size``. Used by some
                strategies as the encode dim; others compute their own.

        Returns:
            A 1-D ``float32`` numpy array — the retrieval key in the
            format expected by ``encode_per_token`` (header + flattened
            per-token vectors).
        """

    @abstractmethod
    def describe(self) -> str:
        """Short identifier for logging and persistence.

        Examples: ``"hidden_state@17"``, ``"k_vector@11"``. Used by the
        calibration registry and progress logs to identify which
        ``(strategy, layer)`` combination is being evaluated.
        """

    @property
    @abstractmethod
    def layer_index(self) -> int:
        """Strategy-native layer index.

        For :class:`HiddenStateKeyStrategy` this is the ``query_layer``
        (index into ``output.hidden_states``). For
        :class:`KVectorKeyStrategy` it's the ``softmax_layer_idx`` (index
        into ``cache.layers``). Same conceptual role (which layer the
        strategy reads from) but different indexing space — callers that
        care should branch on the strategy type.
        """


class HiddenStateKeyStrategy(RetrievalKeyStrategy):
    """Encodes from ``output.hidden_states[query_layer]``.

    Pre-Phase-2 default. Works well on uniform-softmax models
    (Qwen3, Llama-3, Mistral, Gemma-2). Flatlines on hybrid-attention
    models where mean-pooled hidden states wash out the ~15% of heads
    that carry the retrieval signal.

    Args:
        query_layer: Index into ``output.hidden_states``. 0 is the
            embedding output; 1..n are per-layer outputs. The
            calibration sweep in ``tardigrade_hooks.calibrate`` picks
            this empirically; the library default
            (in ``constants.DEFAULT_CAPTURE_LAYER_RATIO``) is
            ``int(num_hidden_layers * 0.67)``.
    """

    def __init__(self, query_layer: int):
        self._query_layer = int(query_layer)

    @property
    def query_layer(self) -> int:
        return self._query_layer

    @property
    def layer_index(self) -> int:
        return self._query_layer

    def describe(self) -> str:
        return f"hidden_state@{self._query_layer}"

    def compute(self, hidden_states, kv, hidden_size):
        if self._query_layer >= len(hidden_states):
            raise ValueError(
                f"query_layer={self._query_layer} out of range "
                f"(model returned {len(hidden_states)} hidden_states indices)"
            )
        h = hidden_states[self._query_layer]
        # h[1:] skips the BOS/pos-0 token. For seq_len <= 1 the slice is
        # empty and the resulting key is header-only — a degenerate
        # retrieval signal but a valid encoding. This matches the
        # pre-Phase-2 inline-extraction behavior in KnowledgePackStore;
        # raising here would break consumers who store single-token
        # facts (e.g. test fixtures with one-word stubs).
        return encode_per_token(h[1:], hidden_size)


class KVectorKeyStrategy(RetrievalKeyStrategy):
    """Encodes from the K projections of a specific softmax attention layer.

    Designed for hybrid-attention models (RecurrentGemma, Qwen3-Next,
    Jamba, Zamba, Falcon-Mamba, Granite-4, MiniMax, Hunyuan-T1, IBM Bamba,
    Nemotron-H) where :class:`HiddenStateKeyStrategy` produces
    near-random retrieval. The K vectors at softmax layers ARE the
    model's own retrieval primitive — attention is literally
    ``softmax(QK^T / √d_k) · V``. Per Michalak & Abreu 2025 (arXiv:
    2510.19861) retrieval lives in attention heads, not in the
    mean-pooled residual stream.

    On uniform-softmax models, K-vector encoding also works at specific
    layers (Qwen3 layer 26: 19/20 top-1) — calibration is encouraged to
    treat both strategies as candidates and pick the winner empirically.

    # Known limitations

    - **Requires a softmax layer at ``softmax_layer_idx``.** On hybrid
      models most layers are linear/recurrent and lack ``.keys`` on the
      cache layer object. ``compute()`` raises ``ValueError`` if the
      requested layer isn't softmax-attention. The Factory consults
      ``model.config.layers_block_type`` to enumerate valid indices.
    - **RoPE positional sensitivity.** K vectors have RoPE rotation
      baked in. The fact's K-at-position-N and the query's K-at-
      position-M dot-product less well than RoPE-free hidden states.
      Today's empirical results: K-vector encoding tolerates this on
      RecurrentGemma (partial-RoPE: 64 of 256 head dims rotated, 192
      position-invariant) and on specific Qwen3 layers; broadly fails
      on shallow Qwen3 layers where RoPE dominates.

    Args:
        softmax_layer_idx: Index into ``cache.layers`` that must be a
            softmax-attention layer (not linear / recurrent / SSM).
    """

    def __init__(self, softmax_layer_idx: int):
        self._softmax_layer_idx = int(softmax_layer_idx)

    @property
    def softmax_layer_idx(self) -> int:
        return self._softmax_layer_idx

    @property
    def layer_index(self) -> int:
        return self._softmax_layer_idx

    def describe(self) -> str:
        return f"k_vector@{self._softmax_layer_idx}"

    def compute(self, hidden_states, kv, hidden_size):
        # hidden_size is unused for this strategy — encode dim is derived
        # from K shape (kv_heads × head_dim) — but kept in the signature
        # for interface compatibility with HiddenStateKeyStrategy.
        del hidden_states, hidden_size

        if self._softmax_layer_idx >= len(kv.layers):
            raise ValueError(
                f"softmax_layer_idx={self._softmax_layer_idx} out of range "
                f"(cache has {len(kv.layers)} layers)"
            )
        layer = kv.layers[self._softmax_layer_idx]
        if not _is_softmax_cache_layer(layer):
            raise ValueError(
                f"layer {self._softmax_layer_idx} is not a softmax-attention "
                f"layer (cache layer has no .keys tensor — likely a "
                f"linear/recurrent/SSM layer). KVectorKeyStrategy requires "
                f"a softmax layer index; consult model.config.layers_block_type."
            )
        k = layer.keys[0]  # (num_kv_heads, seq_len, head_dim)
        kv_heads, seq_len, head_dim = k.shape
        k_flat = k.permute(1, 0, 2).reshape(seq_len, kv_heads * head_dim)
        k_np = k_flat.detach().float().cpu().numpy().astype(np.float32)
        # Same degenerate-but-valid policy as HiddenStateKeyStrategy: a
        # seq_len of 1 produces an empty post-skip array and a
        # header-only key. Don't raise — the engine accepts the
        # degenerate key and downstream callers (test fixtures storing
        # one-token stubs) keep working.
        encode_dim = kv_heads * head_dim
        return encode_per_token(k_np[1:], encode_dim)
