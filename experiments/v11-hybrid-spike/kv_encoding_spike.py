"""Spike: K-vector encoding as retrieval key for hybrid models.

Hidden-state-based calibration flatlined on RecurrentGemma at
every layer (best ~3/20 top-1). Per Michalak & Abreu 2025 retrieval
lives in attention heads, not in any layer's mean-pooled residual
stream. The natural alternative: use the K projections of softmax
attention layers directly as the retrieval encoding.

This spike answers: **does K-vector encoding produce retrievable
keys on a hybrid model?** Acceptance: at least one softmax layer
achieves ≥ 50% top-1 on the 20-fact bundled corpus. If yes →
graduate the strategy into `RetrievalKeyStrategy`. If no → publish
negative result, pivot to per-head selection or a learned adapter.

Also runs on Qwen3 as a sanity check — K-vector encoding should
match or beat hidden-state baseline (~90% top-1) on a uniform-
softmax model. Worse-than-baseline points at a RoPE positional
artifact that needs fixing before the library refactor.

Plan: ~/.claude/plans/keen-extracting-mongoose.md
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import tardigrade_db
from tardigrade_hooks.chat_template_adapter import select_chat_template_adapter
from tardigrade_hooks.constants import CALIBRATION_SALIENCE
from tardigrade_hooks.encoding import encode_per_token

MODEL_ID = os.environ.get("HYBRID_SPIKE_MODEL", "google/recurrentgemma-2b-it")
DEVICE = os.environ.get(
    "HYBRID_SPIKE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
)
LIMIT = int(os.environ.get("HYBRID_SPIKE_LIMIT", "20"))
OWNER = 1
TOP_K = 5
FACTS_PATH = Path(__file__).parent / "facts.json"


def load_model():
    print(f"Loading {MODEL_ID} on {DEVICE}…", flush=True)
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    model = model.to(DEVICE)
    model.train(False)
    return model, tok


def is_softmax_cache_layer(layer) -> bool:
    """Softmax-attention cache layers expose .keys as a tensor."""
    k = getattr(layer, "keys", None)
    return isinstance(k, torch.Tensor)


def softmax_layer_indices(cache) -> list[int]:
    """Indices into cache.layers that are softmax-attention layers."""
    return [i for i, layer in enumerate(cache.layers) if is_softmax_cache_layer(layer)]


def forward_with_cache(model, tok, text: str, *, wrap_chat: bool, adapter):
    """Forward `text` through model; return (cache, seq_len). Cache has
    K/V populated for softmax layers."""
    device = model.device
    if wrap_chat:
        messages = adapter.store_messages(text)
        formatted = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_ids = tok.encode(formatted, return_tensors="pt").to(device)
    else:
        input_ids = tok.encode(text, return_tensors="pt").to(device)
    seq_len = int(input_ids.shape[1])
    cache_in = DynamicCache(config=model.config)
    with torch.no_grad():
        out = model(input_ids, past_key_values=cache_in, use_cache=True)
    returned = getattr(out, "past_key_values", None)
    if (
        returned is not None
        and hasattr(returned, "get_seq_length")
        and returned.get_seq_length() > 0
    ):
        kv = returned
    else:
        kv = cache_in
    return kv, seq_len


def kvector_retrieval_key(cache, layer_idx: int) -> tuple[np.ndarray, int]:
    """Extract K vectors from a softmax attention layer and encode as
    a retrieval key.

    cache.layers[layer_idx].keys[0] has shape (num_kv_heads, seq_len,
    head_dim). Reshape to (seq_len, num_kv_heads × head_dim) and pass
    to encode_per_token.
    """
    layer = cache.layers[layer_idx]
    if not is_softmax_cache_layer(layer):
        raise ValueError(f"layer {layer_idx} is not a softmax attention layer")
    k = layer.keys[0]                                  # (kv_heads, seq, head_dim)
    kv_heads, seq_len, head_dim = k.shape
    k_flat = k.permute(1, 0, 2).reshape(seq_len, kv_heads * head_dim)
    k_np = k_flat.detach().float().cpu().numpy().astype(np.float32)
    dim = kv_heads * head_dim
    return encode_per_token(k_np[1:], dim), dim  # skip pos 0 to match library convention


def evaluate_layer(model, tok, adapter, facts, layer_idx: int) -> tuple[int, int]:
    """Run engine round-trip with K-vector encoding from this softmax
    layer. Return (top1_hits, topk_hits)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = tardigrade_db.Engine(tmpdir)
        fact_to_pack: dict[int, int] = {}
        for i, item in enumerate(facts):
            cache, _ = forward_with_cache(model, tok, item["fact"], wrap_chat=True, adapter=adapter)
            ret_key, _ = kvector_retrieval_key(cache, layer_idx)
            pack_id = engine.mem_write_pack(
                OWNER, ret_key, [], CALIBRATION_SALIENCE, text=item["fact"]
            )
            fact_to_pack[i] = pack_id

        top1 = 0
        topk = 0
        for i, item in enumerate(facts):
            cache, _ = forward_with_cache(model, tok, item["query"], wrap_chat=False, adapter=adapter)
            qkey, _ = kvector_retrieval_key(cache, layer_idx)
            packs = engine.mem_read_pack(qkey, TOP_K, OWNER)
            ids = [p["pack_id"] for p in packs]
            if ids[:1] == [fact_to_pack[i]]:
                top1 += 1
            if fact_to_pack[i] in ids:
                topk += 1
        return top1, topk


def main():
    facts = json.loads(FACTS_PATH.read_text())[:LIMIT]
    model, tok = load_model()
    adapter = select_chat_template_adapter(tok)

    # Discover softmax layer indices by capturing one cache.
    print(f"\nProbing model for softmax attention layers…", flush=True)
    probe_cache, _ = forward_with_cache(model, tok, facts[0]["fact"], wrap_chat=True, adapter=adapter)
    softmax_layers = softmax_layer_indices(probe_cache)
    n_total = len(probe_cache.layers)
    print(f"  total cache layers: {n_total}")
    print(f"  softmax layers:     {softmax_layers}  ({len(softmax_layers)} of {n_total})")
    print(f"  adapter:            {type(adapter).__name__}")
    del probe_cache

    print(f"\nK-vector encoding sweep on {len(softmax_layers)} softmax layers ({len(facts)} facts)…\n", flush=True)
    print(f"{'Layer':<8}{'Top-1':<12}{'Top-5':<12}{'Notes'}", flush=True)
    print("-" * 60, flush=True)

    results = []
    running_best = (-1, -1)
    for layer_idx in softmax_layers:
        try:
            top1, topk = evaluate_layer(model, tok, adapter, facts, layer_idx)
            is_best = (top1, topk) > running_best
            running_best = max(running_best, (top1, topk))
            note = "★ new best" if is_best else ""
            print(f"{layer_idx:<8}{top1:>2}/{LIMIT:<9}{topk:>2}/{LIMIT:<9}{note}", flush=True)
            results.append({"layer": layer_idx, "top1": top1, "top5": topk})
        except Exception as exc:
            print(f"{layer_idx:<8}ERROR: {type(exc).__name__}: {exc}", flush=True)
            results.append({"layer": layer_idx, "error": f"{type(exc).__name__}: {exc}"})

    print()
    print("=" * 70)
    print(f"K-vector encoding on {MODEL_ID}")
    print("-" * 70)
    valid = [r for r in results if "error" not in r]
    if valid:
        best = max(valid, key=lambda r: (r["top1"], r["top5"], r["layer"]))
        print(f"Best softmax layer: {best['layer']}  top-1 {best['top1']}/{LIMIT}  top-5 {best['top5']}/{LIMIT}")
        bar = 0.5 * LIMIT
        verdict = "PASS — graduate the strategy into RetrievalKeyStrategy" if best["top1"] >= bar else "FAIL — pivot to per-head or learned-adapter approach"
        print(f"Acceptance (≥{int(bar)} top-1): {verdict}")
    else:
        print("All layers errored — investigate before drawing conclusions.")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
