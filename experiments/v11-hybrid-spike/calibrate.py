"""Per-layer engine-retrieval sweep — calibration prototype.

For each candidate retrieval-key layer in the model, runs the engine
storage-and-retrieval round-trip on the 20-fact harness and reports
top-1 / top-5 hit rate. The best layer is the one the library's
`DEFAULT_CAPTURE_LAYER_RATIO = 0.75` heuristic *should* be picking
but doesn't for hybrid models.

Why this exists: the engine-isolation test on RecurrentGemma showed
0/20 top-1 and 6/20 top-5 at the default layer 19 (recurrent), and
1/20 top-1 / 3/20 top-5 at the next attention layer (20). Manual
layer-hunting is the wrong UX; the library needs a calibration step
that does this sweep automatically when a new model is plugged in.
This script prototypes that calibration.

Performance trick: the model forward pass is the expensive part. We
do it ONCE per fact and ONCE per query with `output_hidden_states=True`,
cache all layers' hidden states to CPU memory, then sweep the layer
index by re-indexing the cached arrays. ~30s total runtime including
load on RTX 3070 Ti.

Output: a layer-by-layer table plus a JSON dump for downstream use.
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
from tardigrade_hooks.encoding import encode_per_token

MODEL_ID = os.environ.get("HYBRID_SPIKE_MODEL", "google/recurrentgemma-2b-it")
DEVICE = os.environ.get(
    "HYBRID_SPIKE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
)
LIMIT = int(os.environ.get("HYBRID_SPIKE_LIMIT", "20"))
OWNER = 1
TOP_K = 5
FACTS_PATH = Path(__file__).parent / "facts.json"
OUT_JSON = Path(__file__).parent / f"calibrate_{MODEL_ID.replace('/', '_')}.json"


def load_model():
    print(f"Loading {MODEL_ID} on {DEVICE}…", flush=True)
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    model = model.to(DEVICE)
    model.train(False)
    return model, tok


def is_softmax_layer(layer) -> bool:
    k = getattr(layer, "keys", None)
    return isinstance(k, torch.Tensor)


def cache_layer_payloads(kv, seq_len, kv_dim):
    """Compute softmax-only K/V payloads for storage. Same for all retrieval
    layers — the payload doesn't change when we change which layer we use
    for the retrieval key."""
    out = []
    for li in range(len(kv.layers)):
        layer = kv.layers[li]
        if not is_softmax_layer(layer):
            continue
        k = layer.keys[0]
        v = layer.values[0]
        k_np = k.permute(1, 0, 2).reshape(seq_len, kv_dim).detach().float().cpu().numpy().astype(np.float32)
        v_np = v.permute(1, 0, 2).reshape(seq_len, kv_dim).detach().float().cpu().numpy().astype(np.float32)
        payload = np.concatenate([k_np.ravel(), v_np.ravel()])
        out.append((li, payload))
    return out


def forward_collect(model, tok, text: str, wrap_chat: bool, adapter) -> tuple[list[np.ndarray], list, int]:
    """Run a single forward; return (per-layer hidden states on CPU,
    softmax K/V payloads, seq_len). Hidden states list has length
    n_layers + 1 (index 0 = embeddings)."""
    device = model.device
    if wrap_chat:
        messages = adapter.store_messages(text)
        formatted = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_ids = tok.encode(formatted, return_tensors="pt").to(device)
    else:
        input_ids = tok.encode(text, return_tensors="pt").to(device)
    seq_len = input_ids.shape[1]

    cache_in = DynamicCache(config=model.config)
    with torch.no_grad():
        out = model(input_ids, past_key_values=cache_in, use_cache=True,
                    output_hidden_states=True)
    returned = getattr(out, "past_key_values", None)
    if returned is not None and hasattr(returned, "get_seq_length") and returned.get_seq_length() > 0:
        kv = returned
    else:
        kv = cache_in

    cfg = model.config
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    kv_dim = num_kv_heads * head_dim
    payloads = cache_layer_payloads(kv, seq_len, kv_dim)

    hidden_per_layer = [
        h[0].float().cpu().numpy().astype(np.float32) for h in out.hidden_states
    ]
    return hidden_per_layer, payloads, seq_len


def layer_kinds_from_config(cfg, n_hidden_states: int) -> list[str]:
    """Return a label per hidden_states index. Index 0 = embeddings;
    indices 1..n_layers = output of each model layer."""
    block_types = getattr(cfg, "layers_block_type", None) or getattr(cfg, "layer_types", None)
    if not block_types:
        return ["embedding"] + ["attention"] * (n_hidden_states - 1)
    labels = ["embedding"]
    for t in block_types:
        if t in ("attention", "full_attention"):
            labels.append("attn")
        elif t in ("recurrent", "linear_attention"):
            labels.append("rec")
        else:
            labels.append(t[:5])
    while len(labels) < n_hidden_states:
        labels.append("?")
    return labels


def main():
    facts = json.loads(FACTS_PATH.read_text())[:LIMIT]
    model, tok = load_model()
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    adapter = select_chat_template_adapter(tok)

    print(f"\nModel: {MODEL_ID}")
    print(f"Layers: {n_layers} | hidden_size: {hidden_size}")
    print(f"Adapter: {type(adapter).__name__}\n")

    # Phase 1: forward all facts (with chat-template wrap), cache hidden states + K/V.
    print(f"Forward pass {len(facts)} facts (caching all layers' hidden states)…")
    fact_hidden = []
    fact_payloads = []
    for i, item in enumerate(facts):
        hs, payloads, _ = forward_collect(model, tok, item["fact"], wrap_chat=True, adapter=adapter)
        fact_hidden.append(hs)
        fact_payloads.append(payloads)
        if (i + 1) % 5 == 0:
            print(f"  facts {i+1}/{len(facts)}", flush=True)
    n_hidden_states = len(fact_hidden[0])
    kinds = layer_kinds_from_config(cfg, n_hidden_states)
    print(f"  hidden_states indices: 0..{n_hidden_states-1}\n")

    # Phase 2: forward all queries (no chat-template wrap — matches library).
    print(f"Forward pass {len(facts)} queries…")
    query_hidden = []
    for i, item in enumerate(facts):
        hs, _, _ = forward_collect(model, tok, item["query"], wrap_chat=False, adapter=adapter)
        query_hidden.append(hs)
    print()

    # Phase 3: sweep every layer index, score top-1/top-5 via engine round-trip.
    print(f"Sweeping {n_hidden_states} layer indices via engine round-trip…\n", flush=True)
    print(f"{'Layer':<6}{'Kind':<10}{'Top-1':<10}{'Top-5':<10}{'Notes'}", flush=True)
    print("-" * 60, flush=True)
    results = []
    running_best = (-1, -1)
    for li in range(n_hidden_states):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = tardigrade_db.Engine(tmpdir)
            fact_to_pack: dict[int, int] = {}
            try:
                for i, item in enumerate(facts):
                    h = fact_hidden[i][li]
                    if h.shape[0] < 2:
                        raise ValueError(f"layer {li} hidden too short: {h.shape}")
                    ret_key = encode_per_token(h[1:], hidden_size)
                    pack_id = engine.mem_write_pack(
                        OWNER, ret_key, fact_payloads[i], 50.0, text=item["fact"]
                    )
                    fact_to_pack[i] = pack_id
                top1 = 0
                topk = 0
                for i, item in enumerate(facts):
                    h = query_hidden[i][li]
                    qkey = encode_per_token(h[1:], hidden_size)
                    packs = engine.mem_read_pack(qkey, TOP_K, OWNER)
                    ids = [p["pack_id"] for p in packs]
                    if ids[:1] == [fact_to_pack[i]]:
                        top1 += 1
                    if fact_to_pack[i] in ids:
                        topk += 1
                is_best = (top1, topk) > running_best
                running_best = max(running_best, (top1, topk))
                note = "★ new best" if is_best else ""
                print(f"{li:<6}{kinds[li]:<10}{top1:>2}/{LIMIT:<7}{topk:>2}/{LIMIT:<7}{note}", flush=True)
                results.append({"layer": li, "kind": kinds[li], "top1": top1, "top5": topk})
            except Exception as exc:
                print(f"{li:<6}{kinds[li]:<10}ERROR: {type(exc).__name__}: {exc}", flush=True)
                results.append({"layer": li, "kind": kinds[li], "error": f"{type(exc).__name__}: {exc}"})

    # Phase 4: report.
    # Tiebreak prefers the deepest layer when multiple tie at ceiling —
    # matches the library's LinearSweepStrategy logic in
    # python/tardigrade_hooks/calibrate.py. Shallow layers (especially
    # the embedding at index 0) can ace small synthetic corpora purely
    # on surface-token discrimination; deeper layers encode semantic
    # meaning that survives paraphrasing.
    best = max(
        (r for r in results if "error" not in r),
        key=lambda r: (r["top1"], r["top5"], r["layer"]),
        default=None,
    )

    print("=" * 70)
    print(f"{'Layer':<6}{'Kind':<10}{'Top-1':<10}{'Top-5':<10}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['layer']:<6}{r['kind']:<10}{'ERROR: ' + r['error'][:40]}")
        else:
            mark = " ★" if (best and r["layer"] == best["layer"]) else "  "
            print(f"{r['layer']:<6}{r['kind']:<10}{r['top1']:>2d}/{len(facts)}{'':<5}{r['top5']:>2d}/{len(facts)}{mark}")
    print("=" * 70)
    if best:
        print(f"Best: layer {best['layer']} ({best['kind']}) — top-1 {best['top1']}/{len(facts)}, top-5 {best['top5']}/{len(facts)}")
    else:
        print("No layer produced retrievable results.")

    OUT_JSON.write_text(json.dumps({
        "model_id": MODEL_ID,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "adapter": type(adapter).__name__,
        "results": results,
        "best": best,
    }, indent=2))
    print(f"\nResults saved to: {OUT_JSON.name}")


if __name__ == "__main__":
    sys.exit(main())
