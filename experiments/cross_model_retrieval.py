#!/usr/bin/env python3
"""Cross-model retrieval experiment.

Scientific method: test hypotheses, report results, don't assume.

Hypotheses:
  H1: Same-model retrieval via HuggingFaceKVHook achieves high recall (baseline)
  H2: Same-model retrieval via mean-pooled hidden states achieves comparable recall
  H3: Cross-model retrieval (store model A, query model B) has non-zero recall
  H4: Projection method affects cross-model recall

Conditions (all use 100 memories, 30 queries):
  A: Qwen3→Qwen3 via HuggingFaceKVHook (proven path, baseline)
  B: Qwen3→Qwen3 via mean-pool hidden states (isolates hook vs raw)
  C: GPT-2→GPT-2 via mean-pool hidden states (second model baseline)
  D: Qwen3→GPT-2 via mean-pool + truncation projection (cross-model)
  E: GPT-2→Qwen3 via mean-pool + zero-pad projection (cross-model)

Models: Qwen3-0.6B (hidden=1024, 28 layers) + GPT-2 (hidden=768, 12 layers).
Sequential model loading to avoid OOM.
"""

import gc
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(".").resolve() / "python"))
sys.path.insert(0, str(Path(".").resolve() / "experiments"))

import tardigrade_db
from corpus_100 import ALL_QUERIES, MEMORIES
from tardigrade_hooks.hf_kv_hook import HuggingFaceKVHook


def extract_hidden_state(model, tokenizer, text, query_layer):
    """Extract mean-pooled hidden state from a specific layer."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states[query_layer]
    return hs[0].mean(dim=0).numpy().astype(np.float32)


def project_to_dim(vec, target_dim):
    """Project vector to target_dim via truncation or zero-padding."""
    if len(vec) == target_dim:
        return vec
    if len(vec) > target_dim:
        return vec[:target_dim].copy()
    padded = np.zeros(target_dim, dtype=np.float32)
    padded[:len(vec)] = vec
    return padded


def query_recall(engine, query_fn, num_memories):
    """Run 30-query workload, return recall@5 and hit count."""
    hits, total = 0, 0
    for query_text, expected, qtype in ALL_QUERIES:
        if qtype == "negative":
            continue
        if any(e >= num_memories for e in expected):
            continue
        total += 1
        results = query_fn(query_text)
        top5 = [r.cell_id for r in results[:5]]
        if any(m in expected for m in top5):
            hits += 1
    recall = 100 * hits / total if total else 0
    return recall, hits, total


def load_model(name):
    """Load model + tokenizer, return (model, tokenizer, num_layers, query_layer, hidden_dim)."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float32, attn_implementation="eager",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    ql = int(n_layers * 0.67)
    hidden = model.config.hidden_size
    return model, tokenizer, n_layers, ql, hidden


def free_model(*objs):
    for o in objs:
        del o
    gc.collect()


def run_condition_hook(model_name):
    """Condition A: Full HuggingFaceKVHook pipeline (proven 100% recall path)."""
    model, tokenizer, n_layers, ql, hidden = load_model(model_name)
    db_dir = tempfile.mkdtemp(prefix="tdb_xm_hook_")
    engine = tardigrade_db.Engine(db_dir)
    hook = HuggingFaceKVHook(engine, owner=1, model_config=model.config,
                              model=model, use_hidden_states=True)

    for memory in MEMORIES:
        inputs = tokenizer(memory, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = model(**inputs, use_cache=True, output_hidden_states=True)
        d = hook.on_generate(layer=ql, past_key_values=out.past_key_values,
                             model_hidden_states=out.hidden_states[ql])
        if d.should_write:
            engine.mem_write(1, ql, d.key, d.value, d.salience, None)

    def query_fn(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = model(**inputs, use_cache=True, output_hidden_states=True)
        return hook.on_prefill(layer=ql, past_key_values=out.past_key_values,
                               model_hidden_states=out.hidden_states[ql])

    recall, hits, total = query_recall(engine, query_fn, len(MEMORIES))
    shutil.rmtree(db_dir)
    free_model(model, tokenizer, hook)
    return recall, hits, total


def run_condition_meanpool(store_name, query_name, common_dim=None):
    """Conditions B-E: Mean-pool hidden states, optional dimension projection."""
    same_model = store_name == query_name

    # Phase 1: Store
    model, tokenizer, _, ql, hidden = load_model(store_name)
    store_dim = hidden

    if common_dim is None:
        if same_model:
            common_dim = store_dim
        else:
            from transformers import AutoConfig
            query_cfg = AutoConfig.from_pretrained(query_name)
            common_dim = min(store_dim, query_cfg.hidden_size)

    db_dir = tempfile.mkdtemp(prefix="tdb_xm_mp_")
    engine = tardigrade_db.Engine(db_dir)

    for memory in MEMORIES:
        hs = extract_hidden_state(model, tokenizer, memory, ql)
        key = project_to_dim(hs, common_dim)
        engine.mem_write(1, 0, key, np.zeros_like(key), 50.0, None)

    if not same_model:
        free_model(model, tokenizer)

    # Phase 2: Query
    if same_model:
        q_model, q_tok, q_ql = model, tokenizer, ql
    else:
        q_model, q_tok, _, q_ql, _ = load_model(query_name)

    def query_fn(text):
        hs = extract_hidden_state(q_model, q_tok, text, q_ql)
        qk = project_to_dim(hs, common_dim)
        return engine.mem_read(qk, 5, None)

    recall, hits, total = query_recall(engine, query_fn, len(MEMORIES))
    shutil.rmtree(db_dir)
    free_model(q_model, q_tok)
    return recall, hits, total


def main():
    qwen = "Qwen/Qwen3-0.6B"
    gpt2 = "openai-community/gpt2"

    print("=" * 70)
    print("CROSS-MODEL RETRIEVAL EXPERIMENT")
    print(f"Memories: {len(MEMORIES)}, Queries: 30")
    print(f"Models: Qwen3-0.6B (1024-dim) + GPT-2 (768-dim)")
    print("=" * 70)

    results = {}

    # Condition A: Qwen3→Qwen3 via hook (proven path)
    print("\n--- A: Qwen3→Qwen3 via HuggingFaceKVHook (baseline) ---")
    r, h, t = run_condition_hook(qwen)
    results["A"] = r
    print(f"  Recall@5: {r:.1f}% ({h}/{t})")

    # Condition B: Qwen3→Qwen3 via mean-pool (isolate hook effect)
    print("\n--- B: Qwen3→Qwen3 via mean-pool (raw hidden states) ---")
    r, h, t = run_condition_meanpool(qwen, qwen)
    results["B"] = r
    print(f"  Recall@5: {r:.1f}% ({h}/{t})")

    # Condition C: GPT-2→GPT-2 via mean-pool (second model baseline)
    print("\n--- C: GPT-2→GPT-2 via mean-pool ---")
    r, h, t = run_condition_meanpool(gpt2, gpt2)
    results["C"] = r
    print(f"  Recall@5: {r:.1f}% ({h}/{t})")

    # Condition D: Qwen3→GPT-2 via mean-pool + truncation (cross-model)
    print("\n--- D: Qwen3→GPT-2 (cross, truncation to 768) ---")
    r, h, t = run_condition_meanpool(qwen, gpt2, common_dim=768)
    results["D"] = r
    print(f"  Recall@5: {r:.1f}% ({h}/{t})")

    # Condition E: GPT-2→Qwen3 via mean-pool + zero-pad (cross-model)
    print("\n--- E: GPT-2→Qwen3 (cross, zero-pad to 768) ---")
    r, h, t = run_condition_meanpool(gpt2, qwen, common_dim=768)
    results["E"] = r
    print(f"  Recall@5: {r:.1f}% ({h}/{t})")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    print(f"  {'Condition':<50} {'R@5':>8}")
    print(f"  {'─' * 60}")
    labels = {
        "A": "Qwen3→Qwen3 hook (per-token, baseline)",
        "B": "Qwen3→Qwen3 mean-pool (raw hidden states)",
        "C": "GPT-2→GPT-2 mean-pool",
        "D": "Qwen3→GPT-2 cross (truncation)",
        "E": "GPT-2→Qwen3 cross (zero-pad)",
    }
    for k in "ABCDE":
        print(f"  {labels[k]:<50} {results[k]:>7.1f}%")

    # Analysis
    print(f"\n  ANALYSIS:")
    print(f"  Hook vs mean-pool (A vs B): {results['A'] - results['B']:+.1f}% "
          f"({'hook wins' if results['A'] > results['B'] else 'mean-pool competitive'})")

    same_model_avg = (results["B"] + results["C"]) / 2
    cross_model_avg = (results["D"] + results["E"]) / 2
    print(f"  Same-model mean-pool avg: {same_model_avg:.1f}%")
    print(f"  Cross-model avg:          {cross_model_avg:.1f}%")
    print(f"  Cross-model drop:         {same_model_avg - cross_model_avg:+.1f}%")

    if cross_model_avg >= same_model_avg * 0.7:
        verdict = "VIABLE — cross-model within 30% of same-model baseline"
    elif cross_model_avg > 0:
        verdict = "WEAK — non-zero cross-model signal but significant degradation"
    else:
        verdict = "NONE — no cross-model retrieval, model-specific only"

    print(f"\n  VERDICT: {verdict}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
