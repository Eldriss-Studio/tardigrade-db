"""Per-layer hidden-state extraction shared between kp_injector and calibrate.

The chat-template wrap + forward-pass + per-layer hidden-state cache pattern
is the same shape in two places: ``KnowledgePackStore.store()`` (where the
hidden state at ``query_layer`` is the retrieval key) and the calibration
sweep (where every layer's hidden state is a *candidate* retrieval key).
Extracted here so the two stay in sync.

Two model behaviors are accommodated:

- Uniform-softmax models (Qwen3, Llama-3, Mistral) return the populated
  cache in ``output.past_key_values``. A pre-instantiated cache passed in
  via ``past_key_values=`` is *replaced*, not mutated.
- Hybrid models (RecurrentGemma) return ``CausalLMOutput`` rather than
  ``CausalLMOutputWithPast`` — no ``.past_key_values`` attribute — but the
  pre-instantiated cache *is* mutated in place during the forward.

The helper passes a pre-instantiated cache for the hybrid case, then
prefers the returned cache when populated, falling back otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Lazy: `torch` and `transformers` are imported at function-call time
# rather than module load. CI lint jobs (e.g. bench-smoke-gate) install
# numpy but not torch / transformers; eager-importing them here would
# break unrelated lint paths that touch the `tardigrade_hooks` package.
# Same rationale as the `tardigrade_db` lazy-import in calibrate.py.


def _is_softmax_cache_layer(layer) -> bool:
    """Softmax-attention cache layers expose ``.keys`` as a tensor;
    linear/recurrent layers do not."""
    import torch  # local: see module-level comment

    k = getattr(layer, "keys", None)
    return isinstance(k, torch.Tensor)


def _softmax_layer_payloads(
    kv, seq_len: int, kv_dim: int
) -> list[tuple[int, np.ndarray]]:
    """Build the per-layer K/V payload list for ``mem_write_pack``, including
    only softmax-attention layers. Linear/recurrent layers are skipped (they
    have no ``.keys``/``.values`` and per the v11 spike + Michalak & Abreu
    2025 contribute nothing observable to fact retrieval)."""
    out: list[tuple[int, np.ndarray]] = []
    for li in range(len(kv.layers)):
        layer = kv.layers[li]
        if not _is_softmax_cache_layer(layer):
            continue
        k = layer.keys[0]
        v = layer.values[0]
        k_np = (
            k.permute(1, 0, 2)
            .reshape(seq_len, kv_dim)
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        v_np = (
            v.permute(1, 0, 2)
            .reshape(seq_len, kv_dim)
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        payload = np.concatenate([k_np.ravel(), v_np.ravel()])
        out.append((li, payload))
    return out


def _compute_per_layer_hidden_states(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    wrap_chat: bool,
    adapter: Any,
    return_cache: bool = False,
) -> tuple:
    """Forward ``text`` through ``model`` once; return per-layer hidden
    states (CPU float32 numpy), softmax-layer K/V payloads, and seq_len.

    Hidden-state list length is ``model.config.num_hidden_layers + 1`` —
    index 0 is the embedding output, indices 1..n are the per-layer outputs.

    If ``return_cache=True``, the populated ``DynamicCache`` is appended to
    the returned tuple. Strategies that read K vectors directly (e.g.
    :class:`tardigrade_hooks.retrieval_key_strategy.KVectorKeyStrategy`)
    need raw cache access — per-layer hidden states alone aren't enough.

    Return shape:

    - ``return_cache=False`` (default, backwards-compat):
      ``(hidden_per_layer, payloads, seq_len)``
    - ``return_cache=True``:
      ``(hidden_per_layer, payloads, seq_len, kv)``

    If ``wrap_chat`` is True, ``text`` is wrapped via ``adapter.store_messages``
    + ``tokenizer.apply_chat_template`` before encoding. Otherwise ``text``
    is encoded as-is.
    """
    device = model.device
    if wrap_chat:
        messages = adapter.store_messages(text)
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_ids = tokenizer.encode(formatted, return_tensors="pt").to(device)
    else:
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    seq_len = int(input_ids.shape[1])

    # Local imports — see module-level comment.
    import torch
    from transformers import DynamicCache

    cache_in = DynamicCache(config=model.config)
    with torch.no_grad():
        out = model(
            input_ids,
            past_key_values=cache_in,
            use_cache=True,
            output_hidden_states=True,
        )
    returned = getattr(out, "past_key_values", None)
    if (
        returned is not None
        and hasattr(returned, "get_seq_length")
        and returned.get_seq_length() > 0
    ):
        kv = returned
    else:
        kv = cache_in

    cfg = model.config
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    kv_dim = num_kv_heads * head_dim
    payloads = _softmax_layer_payloads(kv, seq_len, kv_dim)

    hidden_per_layer = [
        h[0].float().cpu().numpy().astype(np.float32) for h in out.hidden_states
    ]
    if return_cache:
        return hidden_per_layer, payloads, seq_len, kv
    return hidden_per_layer, payloads, seq_len


def layer_kind_labels(cfg: Any, n_hidden_states: int) -> list[str]:
    """Return a label per ``hidden_states`` index.

    Index 0 is the embedding output; indices 1..n_layers are the per-layer
    outputs. Returns one of: ``"embedding"``, ``"attention"``, ``"recurrent"``,
    or the raw config string truncated to 8 chars if neither.
    """
    block_types = getattr(cfg, "layers_block_type", None) or getattr(
        cfg, "layer_types", None
    )
    if not block_types:
        # Uniform-softmax model with no per-layer typing in config.
        return ["embedding"] + ["attention"] * (n_hidden_states - 1)
    labels = ["embedding"]
    for t in block_types:
        if t in ("attention", "full_attention"):
            labels.append("attention")
        elif t in ("recurrent", "linear_attention"):
            labels.append("recurrent")
        else:
            labels.append(str(t)[:8])
    while len(labels) < n_hidden_states:
        labels.append("unknown")
    return labels
