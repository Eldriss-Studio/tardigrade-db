#!/usr/bin/env python3
"""KV Injection Validation Test.

Tests whether cross-context KV injection actually helps model predictions.
Compares six conditions using experiential memories that GPT-2 cannot infer.

Usage:
    python examples/kv_injection_validation.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import tardigrade_db
from tardigrade_hooks.hf_hook import HuggingFaceHook
from tardigrade_hooks.kv_injector import build_injection_cache, prepare_injection


# ── Prompt pairs: experiential memories GPT-2 cannot infer ──────────

IRRELEVANT_MEMORY = (
    "The quarterly budget review showed infrastructure costs increased "
    "by 23 percent due to the migration from AWS to Google Cloud"
)

PROMPT_PAIRS = [
    {
        "name": "Standup (Marcus)",
        "memory": (
            "During standup Marcus flagged 502 errors in the auth service "
            "and Priya said her PR was blocked on my code review"
        ),
        "query": "At the morning meeting, the person who reported server errors was",
        "target": " Marcus",
    },
    {
        "name": "Lunch (banh mi)",
        "memory": (
            "I ate a banh mi from the food cart downstairs and sat alone "
            "at the window table watching pigeons"
        ),
        "query": "For lunch today I had a",
        "target": " ban",  # GPT-2 tokenizes "banh" starting with " ban"
    },
    {
        "name": "Slack (Lena)",
        "memory": (
            "Lena sent me a Slack message at 4:12pm saying can we sync "
            "tomorrow morning about your onboarding trajectory"
        ),
        "query": "The message from my manager said we need to",
        "target": " sync",
    },
    {
        "name": "Bug (data_pipeline)",
        "memory": (
            "I found a TypeError on line 84 of data_pipeline/ingest.py "
            "because someone's refactor left a NoneType where split was called"
        ),
        "query": "The bug I spent two hours debugging was a",
        "target": " Type",  # "TypeError"
    },
    {
        "name": "Embarrassment (Daniel)",
        "memory": (
            "I accidentally called the tech lead Daniel by the wrong name "
            "David in the backend channel and he replied with a single period"
        ),
        "query": "The embarrassing thing that happened was when I called",
        "target": " Daniel",
    },
]


def load_model():
    print("Loading GPT-2...", end=" ", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2", output_hidden_states=True
    )
    model.eval()
    print(f"OK ({model.config.n_layer} layers, d={model.config.n_embd})")
    return model, tokenizer


def get_outputs(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        return model(**inputs, use_cache=True)


def get_top_tokens(logits, tokenizer, k=10):
    probs = F.softmax(logits, dim=-1)
    topk = torch.topk(probs, k)
    return [
        (tokenizer.decode(idx.item()), topk.values[i].item())
        for i, idx in enumerate(topk.indices)
    ]


def run_pair(model, tokenizer, pair):
    memory_text = pair["memory"]
    query_text = pair["query"]
    target_text = pair["target"]
    name = pair["name"]

    target_ids = tokenizer.encode(target_text)
    target_token = target_ids[0]

    num_heads = model.config.n_head       # 12
    head_dim = model.config.n_embd // model.config.n_head  # 64
    n_layers = model.config.n_layer       # 12

    print(f"\n{'='*70}")
    print(f"  PAIR: {name}")
    print(f"  Memory: \"{memory_text[:70]}...\"")
    print(f"  Query:  \"{query_text}\"")
    print(f"  Target: \"{target_text}\" (token {target_token})")
    print(f"{'='*70}")

    # ── Capture: get hidden states and full KV from memory ────
    mem_outputs = get_outputs(model, tokenizer, memory_text)
    irr_outputs = get_outputs(model, tokenizer, IRRELEVANT_MEMORY)

    # Store mean-pooled hidden states in TardigradeDB
    db_dir = tempfile.mkdtemp(prefix="tdb_inject_")
    engine = tardigrade_db.Engine(db_dir)
    hook = HuggingFaceHook(engine, owner=1, k=1, norm_threshold=0.0)

    for layer_idx in range(n_layers):
        h = mem_outputs.hidden_states[layer_idx + 1].numpy()
        decision = hook.on_generate(layer=layer_idx, hidden_states=h)
        if decision.should_write and decision.key is not None:
            engine.mem_write(
                1, layer_idx, decision.key, decision.value,
                decision.salience, None
            )

    # Store irrelevant memory separately (owner=2)
    engine_irr = tardigrade_db.Engine(tempfile.mkdtemp(prefix="tdb_inject_irr_"))
    hook_irr = HuggingFaceHook(engine_irr, owner=1, k=1, norm_threshold=0.0)
    for layer_idx in range(n_layers):
        h = irr_outputs.hidden_states[layer_idx + 1].numpy()
        decision = hook_irr.on_generate(layer=layer_idx, hidden_states=h)
        if decision.should_write and decision.key is not None:
            engine_irr.mem_write(
                1, layer_idx, decision.key, decision.value,
                decision.salience, None
            )

    results = {}

    # ── Condition 1: Baseline ─────────────────────────────────
    baseline_out = get_outputs(model, tokenizer, query_text)
    baseline_logits = baseline_out.logits[0, -1]
    p_baseline = F.softmax(baseline_logits, dim=-1)[target_token].item()
    results["1. Baseline"] = p_baseline

    # ── Condition 2: Text RAG ─────────────────────────────────
    rag_text = f"{memory_text}. {query_text}"
    rag_out = get_outputs(model, tokenizer, rag_text)
    rag_logits = rag_out.logits[0, -1]
    p_rag = F.softmax(rag_logits, dim=-1)[target_token].item()
    results["2. Text RAG"] = p_rag

    # ── Condition 3: Mean-pooled inject (relevant) ────────────
    try:
        query_out = get_outputs(model, tokenizer, query_text)
        handles_by_layer = {}
        for layer_idx in range(n_layers):
            h = query_out.hidden_states[layer_idx + 1].numpy()
            handles = hook.on_prefill(layer=layer_idx, query_states=h)
            if handles:
                handles_by_layer[layer_idx] = handles

        cache = build_injection_cache(handles_by_layer, num_heads, head_dim, n_layers)
        query_inputs = tokenizer(query_text, return_tensors="pt")
        inject_kwargs = prepare_injection(cache, query_inputs["input_ids"])
        with torch.no_grad():
            mean_out = model(query_inputs["input_ids"], **inject_kwargs)
        p_mean_rel = F.softmax(mean_out.logits[0, -1], dim=-1)[target_token].item()
        results["3. Mean-pool inject (relevant)"] = p_mean_rel
    except Exception as e:
        results["3. Mean-pool inject (relevant)"] = f"ERROR: {e}"

    # ── Condition 4: Mean-pooled inject (irrelevant) ──────────
    try:
        handles_by_layer_irr = {}
        for layer_idx in range(n_layers):
            h = query_out.hidden_states[layer_idx + 1].numpy()
            handles = hook_irr.on_prefill(layer=layer_idx, query_states=h)
            if handles:
                handles_by_layer_irr[layer_idx] = handles

        cache_irr = build_injection_cache(handles_by_layer_irr, num_heads, head_dim, n_layers)
        inject_kwargs_irr = prepare_injection(cache_irr, query_inputs["input_ids"])
        with torch.no_grad():
            mean_irr_out = model(query_inputs["input_ids"], **inject_kwargs_irr)
        p_mean_irr = F.softmax(mean_irr_out.logits[0, -1], dim=-1)[target_token].item()
        results["4. Mean-pool inject (irrelevant)"] = p_mean_irr
    except Exception as e:
        results["4. Mean-pool inject (irrelevant)"] = f"ERROR: {e}"

    # ── Condition 5: Full KV inject (relevant) ────────────────
    try:
        full_kv = mem_outputs.past_key_values
        # DynamicCache: access seq_len via layers[0].keys.shape[2]
        mem_len = full_kv.layers[0].keys.shape[2]
        query_ids = tokenizer(query_text, return_tensors="pt")["input_ids"]
        seq_len = query_ids.shape[1]
        total_len = mem_len + seq_len

        position_ids = torch.arange(mem_len, total_len, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones(1, total_len, dtype=torch.long)

        with torch.no_grad():
            full_out = model(
                query_ids,
                past_key_values=full_kv,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
        p_full_rel = F.softmax(full_out.logits[0, -1], dim=-1)[target_token].item()
        results["5. Full KV inject (relevant)"] = p_full_rel
    except Exception as e:
        results["5. Full KV inject (relevant)"] = f"ERROR: {e}"

    # ── Condition 6: Full KV inject (irrelevant) ──────────────
    try:
        full_kv_irr = irr_outputs.past_key_values
        mem_len_irr = full_kv_irr.layers[0].keys.shape[2]
        total_len_irr = mem_len_irr + seq_len

        position_ids_irr = torch.arange(mem_len_irr, total_len_irr, dtype=torch.long).unsqueeze(0)
        attention_mask_irr = torch.ones(1, total_len_irr, dtype=torch.long)

        with torch.no_grad():
            full_irr_out = model(
                query_ids,
                past_key_values=full_kv_irr,
                position_ids=position_ids_irr,
                attention_mask=attention_mask_irr,
            )
        p_full_irr = F.softmax(full_irr_out.logits[0, -1], dim=-1)[target_token].item()
        results["6. Full KV inject (irrelevant)"] = p_full_irr
    except Exception as e:
        results["6. Full KV inject (irrelevant)"] = f"ERROR: {e}"

    # ── Report ────────────────────────────────────────────────
    print(f"\n  P(\"{target_text.strip()}\") across conditions:\n")
    for condition, value in results.items():
        if isinstance(value, float):
            bar = "█" * int(value * 200)
            print(f"    {condition:<38} {value:.6f}  {bar}")
        else:
            print(f"    {condition:<38} {value}")

    # Top-5 tokens for each condition
    print(f"\n  Top-5 predicted tokens:\n")
    all_logits = {
        "Baseline": baseline_logits,
        "Text RAG": rag_logits,
    }
    try:
        all_logits["Mean-pool (rel)"] = mean_out.logits[0, -1]
    except Exception:
        pass
    try:
        all_logits["Full KV (rel)"] = full_out.logits[0, -1]
    except Exception:
        pass

    for cond_name, logits in all_logits.items():
        top = get_top_tokens(logits, tokenizer, k=5)
        tokens_str = ", ".join(f"\"{t}\"({p:.4f})" for t, p in top)
        print(f"    {cond_name:<20} {tokens_str}")

    return results


def main():
    model, tokenizer = load_model()

    print("\n" + "=" * 70)
    print("  KV INJECTION VALIDATION TEST")
    print("  Testing whether cross-context KV injection helps model predictions")
    print("  Using experiential memories that GPT-2 cannot infer")
    print("=" * 70)

    all_results = {}
    for pair in PROMPT_PAIRS:
        all_results[pair["name"]] = run_pair(model, tokenizer, pair)

    # ── Summary table ─────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}\n")

    conditions = [
        "1. Baseline",
        "2. Text RAG",
        "3. Mean-pool inject (relevant)",
        "4. Mean-pool inject (irrelevant)",
        "5. Full KV inject (relevant)",
        "6. Full KV inject (irrelevant)",
    ]

    header = f"  {'Pair':<25}"
    for c in conditions:
        short = c.split(". ")[1][:12]
        header += f" {short:>12}"
    print(header)
    print("  " + "-" * (25 + 12 * len(conditions)))

    for pair_name, results in all_results.items():
        row = f"  {pair_name:<25}"
        for c in conditions:
            v = results.get(c, "?")
            if isinstance(v, float):
                row += f" {v:>12.6f}"
            else:
                row += f" {'ERR':>12}"
        print(row)

    # ── Verdict ───────────────────────────────────────────────
    print(f"\n  INTERPRETATION:")
    print(f"  Compare each row: if column 3 > column 1, mean-pooled injection helped.")
    print(f"  Compare columns 5 vs 1: if full KV helped, Breno's critique needs nuance.")
    print(f"  Column 2 (Text RAG) is the upper bound — it always has the answer in text.")
    print(f"  Columns 4 and 6 (irrelevant) should be ≈ baseline or worse.\n")


if __name__ == "__main__":
    main()
