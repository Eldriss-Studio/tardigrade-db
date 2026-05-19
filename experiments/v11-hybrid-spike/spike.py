"""Hybrid-attention KV-persistence spike — does the soft-prior hypothesis hold?

Cheapest possible test of whether tardigrade-db can support hybrid-attention
models by capturing and reinjecting the cache state from non-softmax layers.

Plan: ~/.claude/plans/prickly-charting-narwhal.md
Research: ../../docs/research/2026-05-19-qwen3-next-hybrid-attention.md
Context: ./README.md
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_ID = os.environ.get("HYBRID_SPIKE_MODEL", "google/recurrentgemma-2b-it")
DEVICE = os.environ.get(
    "HYBRID_SPIKE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
)
LIMIT = int(os.environ.get("HYBRID_SPIKE_LIMIT", "20"))
MAX_NEW_TOKENS = 40
FACTS_PATH = Path(__file__).parent / "facts.json"


def load_model():
    print(f"Loading {MODEL_ID} on {DEVICE}…", flush=True)
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    model = model.to(DEVICE)
    model.train(False)  # inference mode (model.eval equivalent)
    return model, tok


def tensor_attrs(layer) -> dict[str, torch.Tensor]:
    """Discover tensor-valued attributes of a cache layer dynamically.

    RecurrentGemma's recurrent layers expose (conv_states, rg_lru_state);
    Qwen3-Next's LinearAttentionLayer exposes (conv_states, recurrent_states);
    softmax layers expose (keys, values). We don't hardcode — we inspect.
    Keeps the spike portable across hybrid families.
    """
    out: dict[str, torch.Tensor] = {}
    for k in dir(layer):
        if k.startswith("_"):
            continue
        try:
            v = getattr(layer, k)
        except Exception:
            continue
        if isinstance(v, torch.Tensor):
            out[k] = v
    return out


def is_softmax_layer(layer) -> bool:
    """Softmax-attention layers expose .keys as a tensor; recurrent ones don't."""
    k = getattr(layer, "keys", None)
    return isinstance(k, torch.Tensor)


def tokenize_for_prefill(tok, text: str) -> torch.Tensor:
    return tok(text, return_tensors="pt").input_ids.to(DEVICE)


def tokenize_continuation(tok, text: str) -> torch.Tensor:
    """No BOS — the cached prefix already carries it."""
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)


def capture_cache(model, tok, fact_text: str):
    """Forward fact text alone; return the populated cache.

    Some models (RecurrentGemma) return `CausalLMOutput` rather than
    `CausalLMOutputWithPast`, so the cache is *not* echoed back on the
    output object — we must pre-instantiate one and pass it as
    `past_key_values`; HF mutates it in place during the forward pass.
    """
    ids = tokenize_for_prefill(tok, fact_text)
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        model(ids, past_key_values=cache, use_cache=True)
    return cache


def zero_recurrent_layers(cache):
    """In-place: zero non-softmax layer state. Used by H2 (softmax-only inject)."""
    for layer in cache.layers:
        if is_softmax_layer(layer):
            continue
        for attr in tensor_attrs(layer).values():
            attr.zero_()
    return cache


def zero_all_layers(cache):
    """In-place: zero every tensor on every layer. Used by H3 (control).

    Cache shape (seq_len etc.) preserved; only contents zeroed. If H3
    drops to floor recall, then layer *content* is what's biasing
    the model — i.e. the soft-prior hypothesis is real. If H3 stays
    near H1/H2, the model is ignoring cache contents and our recall
    signal is coming from somewhere else (positional state, seq_len
    side effects, etc.).
    """
    for layer in cache.layers:
        for attr in tensor_attrs(layer).values():
            attr.zero_()
    return cache


def zero_softmax_layers(cache):
    """In-place: zero ONLY the softmax K/V. Recurrent layers untouched.

    Used by H4: triangulates H1 ≡ H2. If H4 recall ≈ floor, the
    softmax slice is what's doing the work and "recurrent state
    contributes nothing observable" is the right reading of H1 ≡ H2.
    If H4 recall ≈ H1/H2, the recurrent state alone is sufficient
    and the spike's earlier interpretation was wrong.
    """
    for layer in cache.layers:
        if not is_softmax_layer(layer):
            continue
        for attr in tensor_attrs(layer).values():
            attr.zero_()
    return cache


def decode_continuation(tok, out_ids: torch.Tensor, input_len: int) -> str:
    return tok.decode(out_ids[0, input_len:], skip_special_tokens=True)


def generate(model, tok, input_ids: torch.Tensor, past_key_values=None) -> str:
    kwargs: dict = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    with torch.no_grad():
        out = model.generate(input_ids, **kwargs)
    return decode_continuation(tok, out, input_ids.shape[1])


# -------- the four paths --------

def path_floor(model, tok, fact, query):
    """Query alone, no context. Establishes the no-fact-knowledge floor."""
    ids = tokenize_for_prefill(tok, query)
    return generate(model, tok, ids)


def path_ceiling(model, tok, fact, query):
    """fact + query in one prompt. Upper bound — no caching benefit."""
    ids = tokenize_for_prefill(tok, f"{fact} {query}")
    return generate(model, tok, ids)


def path_h1_full_inject(model, tok, fact, query):
    """Capture full cache (softmax + recurrent); inject both."""
    cache = capture_cache(model, tok, fact)
    ids = tokenize_continuation(tok, query)
    return generate(model, tok, ids, past_key_values=cache)


def path_h2_softmax_only(model, tok, fact, query):
    """Capture full cache, zero recurrent state, inject only softmax."""
    cache = capture_cache(model, tok, fact)
    cache = zero_recurrent_layers(cache)
    ids = tokenize_continuation(tok, query)
    return generate(model, tok, ids, past_key_values=cache)


def path_h3_all_zeroed(model, tok, fact, query):
    """Control: cache passed (preserves seq_len) but all contents zeroed.

    Discriminates "cache content drives recall" (drops to floor) from
    "something other than cache content drives recall" (stays high).
    """
    cache = capture_cache(model, tok, fact)
    cache = zero_all_layers(cache)
    ids = tokenize_continuation(tok, query)
    return generate(model, tok, ids, past_key_values=cache)


def path_h4_recurrent_only(model, tok, fact, query):
    """Capture full cache, zero ONLY softmax K/V, inject what remains.

    Triangulates H1 ≡ H2. If recall drops to floor, softmax slice
    drives recall (the published finding). If recall stays near
    H1/H2, the recurrent state alone is sufficient.
    """
    cache = capture_cache(model, tok, fact)
    cache = zero_softmax_layers(cache)
    ids = tokenize_continuation(tok, query)
    return generate(model, tok, ids, past_key_values=cache)


PATHS = [
    ("Floor (no inject)", path_floor),
    ("H3 (all zeroed)", path_h3_all_zeroed),
    ("H4 (recurrent only)", path_h4_recurrent_only),
    ("H2 (softmax only)", path_h2_softmax_only),
    ("H1 (full inject)", path_h1_full_inject),
    ("Ceiling (re-prefill)", path_ceiling),
]


def hits(answer: str, generated: str) -> bool:
    return answer.lower() in generated.lower()


def print_arch_summary(model):
    cfg = model.config
    n = getattr(cfg, "num_hidden_layers", "?")
    block_types: list = []
    for attr in ("layers_block_type", "layer_types", "block_types"):
        if hasattr(cfg, attr):
            v = getattr(cfg, attr)
            if isinstance(v, (list, tuple)) and v:
                block_types = list(v)
                break
    softmax_keys = {"attention", "full_attention"}
    n_soft = sum(1 for t in block_types if t in softmax_keys)
    n_rec = len(block_types) - n_soft
    print(f"\nModel: {MODEL_ID}")
    print(f"Layers: {n} total | {n_soft} softmax | {n_rec} recurrent")
    if block_types:
        print(f"Pattern (first 8): {block_types[:8]}\n")
    else:
        print("(could not discover layer types from config; "
              "layer-type detection will fall back to runtime introspection)\n")


def main() -> int:
    facts = json.loads(FACTS_PATH.read_text())[:LIMIT]
    model, tok = load_model()
    print_arch_summary(model)

    results = {name: 0 for name, _ in PATHS}
    error_paths: set[str] = set()

    for i, item in enumerate(facts, 1):
        fact, query, expected = item["fact"], item["query"], item["answer"]
        print(f"[{i:>2}/{len(facts)}] expected={expected!r}", flush=True)
        for name, fn in PATHS:
            try:
                gen = fn(model, tok, fact, query)
                ok = hits(expected, gen)
            except Exception as exc:
                gen = f"<ERROR {type(exc).__name__}: {exc}>"
                ok = False
                if name not in error_paths:
                    error_paths.add(name)
                    print(f"      ! {name} first error — traceback:")
                    traceback.print_exc(limit=4)
            results[name] += int(ok)
            mark = "✓" if ok else "·"
            short = gen.replace("\n", " ")[:80]
            print(f"      {mark} {name:<22s} → {short!r}")
        print()

    print("=" * 70)
    print(f"{'Path':<28s} {'R@1':>8s}")
    print("-" * 70)
    for name, _ in PATHS:
        print(f"{name:<28s} {results[name]:>4d}/{len(facts):<4d}")
    print("=" * 70)
    if error_paths:
        print(f"Paths that hit at least one runtime error: "
              f"{sorted(error_paths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
