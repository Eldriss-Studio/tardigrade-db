"""Engine-isolation test: does tardigrade-db's retrieval find the right fact?

This script tests *only* the engine's retrieval primitive — given query Q,
does engine.mem_read_pack return the pack containing fact F? No
model.generate(), no cache injection, no continuation-quality concerns.
The metric is top-K hit rate of the stored pack_id.

Why: the v11 spike (spike.py) measured downstream output quality — did
the model produce text containing the fact's answer string? That metric
conflates engine retrieval quality with the model's ability to USE the
retrieved cache to generate coherent continuations. Two different layers.
This script isolates the engine layer.

Run for both hybrid and non-hybrid models, compare:

    HYBRID_SPIKE_MODEL=google/recurrentgemma-2b-it python engine_recall.py
    HYBRID_SPIKE_MODEL=Qwen/Qwen3-1.7B python engine_recall.py

If both produce high top-1 / top-5, the engine is fine on hybrid models;
the 55% in spike.py was about model output quality, not retrieval. If
hybrid is substantially worse, the engine has a real problem with
hybrid-derived retrieval keys.
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

# When HYBRID_SPIKE_PARAPHRASE=1, use the library's bundled paraphrased
# corpus instead of facts.json. The paraphrased queries don't share
# surface-form vocabulary with the facts, so embedding-layer matching
# can't cheat on shared substrings — useful for testing whether a
# layer that looks "tied" on the original corpus actually generalises.
USE_PARAPHRASE: bool = os.environ.get("HYBRID_SPIKE_PARAPHRASE", "0") == "1"


def load_model():
    print(f"Loading {MODEL_ID} on {DEVICE}…", flush=True)
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    model = model.to(DEVICE)
    model.train(False)
    return model, tok


def is_softmax_layer(layer) -> bool:
    """Softmax-attention cache layers expose .keys as a tensor."""
    k = getattr(layer, "keys", None)
    return isinstance(k, torch.Tensor)


def store_fact(engine, model, tok, adapter, fact_text, query_layer, hidden_size,
               num_kv_heads, head_dim, kv_dim) -> int:
    """Write a pack for this fact via the engine's pack API.

    Captures K/V from softmax layers only — recurrent / linear layers are
    skipped, because (a) they don't expose .keys/.values, (b) the v11 spike
    showed they contribute nothing observable to recall anyway. This is
    the codec design under test in this spike.
    """
    device = model.device
    messages = adapter.store_messages(fact_text)
    formatted = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    input_ids = tok.encode(formatted, return_tensors="pt").to(device)
    seq_len = input_ids.shape[1]

    # Pre-instantiate the cache: RecurrentGemma returns CausalLMOutput which
    # has no .past_key_values; Qwen3 returns CausalLMOutputWithPast which
    # does. Pass our own cache in; prefer the returned one if populated,
    # else use ours (which RecurrentGemma mutates in place).
    cache_in = DynamicCache(config=model.config)
    with torch.no_grad():
        out = model(input_ids, past_key_values=cache_in, use_cache=True,
                    output_hidden_states=True)
    returned = getattr(out, "past_key_values", None)
    if (returned is not None and hasattr(returned, "get_seq_length")
            and returned.get_seq_length() > 0):
        kv = returned
    else:
        kv = cache_in

    # Retrieval key: hidden states at query_layer, per-token, skip pos 0.
    hidden = out.hidden_states[query_layer][0]
    h_tokens = hidden[1:].float().cpu().numpy().astype(np.float32)
    retrieval_key = encode_per_token(h_tokens, hidden_size)

    # Per-layer payloads — softmax layers only.
    layer_payloads = []
    for li in range(len(kv.layers)):
        layer = kv.layers[li]
        if not is_softmax_layer(layer):
            continue
        k = layer.keys[0]
        v = layer.values[0]
        k_np = k.permute(1, 0, 2).reshape(seq_len, kv_dim).detach().float().cpu().numpy().astype(np.float32)
        v_np = v.permute(1, 0, 2).reshape(seq_len, kv_dim).detach().float().cpu().numpy().astype(np.float32)
        payload = np.concatenate([k_np.ravel(), v_np.ravel()])
        layer_payloads.append((li, payload))

    return engine.mem_write_pack(
        OWNER, retrieval_key, layer_payloads, CALIBRATION_SALIENCE, text=fact_text
    )


def query_key(model, tok, query_text, query_layer, hidden_size) -> np.ndarray:
    """Compute the retrieval key for a query — same logic as kp_injector."""
    device = model.device
    query_input = tok.encode(query_text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(query_input, output_hidden_states=True)
    hidden = out.hidden_states[query_layer][0]
    h_tokens = hidden[1:].float().cpu().numpy().astype(np.float32)
    return encode_per_token(h_tokens, hidden_size)


def main():
    if USE_PARAPHRASE:
        from tardigrade_hooks._calibration_corpus import DEFAULT_CORPUS
        facts = [
            {"fact": f, "query": q, "answer": ""}
            for (f, q) in DEFAULT_CORPUS[:LIMIT]
        ]
        corpus_label = "paraphrased (DEFAULT_CORPUS)"
    else:
        facts = json.loads(FACTS_PATH.read_text())[:LIMIT]
        corpus_label = "original (facts.json)"
    print(f"\nCorpus: {corpus_label}")

    model, tok = load_model()
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    kv_dim = num_kv_heads * head_dim
    hidden_size = cfg.hidden_size
    # Default: matches DEFAULT_CAPTURE_LAYER_RATIO in the library (0.75).
    # Override via HYBRID_SPIKE_QUERY_LAYER to test the "attention-aware
    # layer selection" hypothesis on hybrid models.
    qlayer_env = os.environ.get("HYBRID_SPIKE_QUERY_LAYER")
    if qlayer_env is not None:
        query_layer = int(qlayer_env)
    else:
        query_layer = int(n_layers * 0.75)

    block_types = getattr(cfg, "layers_block_type", None) or getattr(cfg, "layer_types", None)
    layer_kind = (block_types[query_layer] if block_types and query_layer < len(block_types) else "?")
    print(f"\nModel: {MODEL_ID}")
    print(f"Layers: {n_layers} | query_layer={query_layer} ({layer_kind}) | hidden_size={hidden_size}\n")

    adapter = select_chat_template_adapter(tok)
    print(f"Adapter: {type(adapter).__name__}\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = tardigrade_db.Engine(tmpdir)

        # Store all facts, record pack_id per fact index.
        print(f"Storing {len(facts)} facts...", flush=True)
        fact_to_pack: dict[int, int] = {}
        for i, item in enumerate(facts):
            pack_id = store_fact(
                engine, model, tok, adapter, item["fact"],
                query_layer, hidden_size, num_kv_heads, head_dim, kv_dim
            )
            fact_to_pack[i] = pack_id
            print(f"  [{i+1:>2}/{len(facts)}] {item['answer']!r:<30s} → pack_id={pack_id}", flush=True)

        # Query each fact's question, check if right pack is returned.
        print(f"\nQuerying (top-{TOP_K})...", flush=True)
        top1_hits = 0
        topk_hits = 0
        for i, item in enumerate(facts):
            expected = fact_to_pack[i]
            qk = query_key(model, tok, item["query"], query_layer, hidden_size)
            packs = engine.mem_read_pack(qk, TOP_K, OWNER)
            ids = [p["pack_id"] for p in packs]
            top1 = ids[:1] == [expected] if ids else False
            topk = expected in ids
            top1_hits += int(top1)
            topk_hits += int(topk)
            mark = "✓✓" if top1 else ("✓·" if topk else "··")
            print(f"  [{i+1:>2}] expected={expected:<5d} got={ids} {mark}  ({item['answer']!r})", flush=True)

        print()
        print("=" * 70)
        print(f"Engine-only retrieval on {MODEL_ID}")
        print("-" * 70)
        print(f"Top-1: {top1_hits:>3d}/{len(facts):<3d}  ({100*top1_hits/len(facts):>3.0f}%)")
        print(f"Top-{TOP_K}: {topk_hits:>3d}/{len(facts):<3d}  ({100*topk_hits/len(facts):>3.0f}%)")
        print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
