#!/usr/bin/env python3
"""Cross-model retrieval experiment.

A/B Test pattern:
  A (control): Store with Model X, retrieve with Model X (same-model baseline)
  B (cross):   Store with Model X, retrieve with Model Y (cross-model hypothesis)

Tests whether TardigradeDB retrieval is model-agnostic (database) or
model-specific (cache). Uses real hidden states from two different models.

Models: Qwen3-0.6B (1024-dim hidden) and GPT-2 (768-dim hidden).
Since dimensions differ, we also test a projected variant where both
models' hidden states are projected to a common dimension.

Requires: torch, transformers (CPU inference sufficient for both models).
"""

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

# Use first 20 memories and their matching queries for speed
NUM_MEMORIES = 20


def extract_hidden_state(model, tokenizer, text, query_layer):
    """Extract the mean-pooled hidden state from the specified layer."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states[query_layer]  # (1, seq_len, hidden_dim)
    return hs[0].mean(dim=0).numpy().astype(np.float32)


def project_to_common_dim(vec, target_dim):
    """Project a vector to target_dim via truncation or zero-padding."""
    if len(vec) == target_dim:
        return vec
    if len(vec) > target_dim:
        return vec[:target_dim]
    padded = np.zeros(target_dim, dtype=np.float32)
    padded[: len(vec)] = vec
    return padded


def run_experiment(store_model_name, query_model_name, common_dim=None):
    """Run one store→query experiment. Returns (recall@5, details).

    For cross-model conditions, loads models sequentially (not
    simultaneously) to avoid OOM on memory-constrained systems.
    """
    import gc

    # Phase 1: Store — load store model, extract hidden states, free model
    print(f"\n  Loading {store_model_name} (store)...", end=" ", flush=True)
    store_tokenizer = AutoTokenizer.from_pretrained(store_model_name)
    store_model = AutoModelForCausalLM.from_pretrained(
        store_model_name, dtype=torch.float32, attn_implementation="eager",
    )
    store_model.eval()
    store_layers = store_model.config.num_hidden_layers
    store_ql = int(store_layers * 0.67)
    store_dim = store_model.config.hidden_size
    print(f"OK ({store_layers}L, ql={store_ql}, dim={store_dim})")

    same_model = store_model_name == query_model_name

    if common_dim is None:
        if same_model:
            common_dim = store_dim
        else:
            # Will be determined after loading query model config
            # For now, use store_dim; will adjust if needed
            common_dim = store_dim

    db_dir = tempfile.mkdtemp(prefix="tdb_xmodel_")
    engine = tardigrade_db.Engine(db_dir)

    memories = MEMORIES[:NUM_MEMORIES]
    stored_keys = []
    for memory in memories:
        hs = extract_hidden_state(store_model, store_tokenizer, memory, store_ql)
        stored_keys.append(hs)

    # Free store model before loading query model
    if not same_model:
        del store_model, store_tokenizer
        gc.collect()

    # Determine common_dim from both models if cross-model
    if not same_model:
        from transformers import AutoConfig
        query_config = AutoConfig.from_pretrained(query_model_name)
        query_dim = query_config.hidden_size
        common_dim = min(store_dim, query_dim)

    # Write stored keys to engine with common_dim projection
    for hs in stored_keys:
        key = project_to_common_dim(hs, common_dim)
        value = np.zeros_like(key)
        engine.mem_write(1, 0, key, value, 50.0, None)

    print(f"  Stored {engine.cell_count()} memories (dim={common_dim})")

    # Phase 2: Query — load query model (or reuse store model)
    if same_model:
        query_tokenizer = store_tokenizer
        query_model = store_model
        query_ql = store_ql
    else:
        print(f"  Loading {query_model_name} (query)...", end=" ", flush=True)
        query_tokenizer = AutoTokenizer.from_pretrained(query_model_name)
        query_model = AutoModelForCausalLM.from_pretrained(
            query_model_name, dtype=torch.float32, attn_implementation="eager",
        )
        query_model.eval()
        query_layers = query_model.config.num_hidden_layers
        query_ql = int(query_layers * 0.67)
        print(f"OK ({query_layers}L, ql={query_ql})")

    hits = 0
    total = 0
    for query_text, expected, qtype in ALL_QUERIES:
        if qtype == "negative":
            continue
        if any(e >= NUM_MEMORIES for e in expected):
            continue

        total += 1
        hs = extract_hidden_state(query_model, query_tokenizer, query_text, query_ql)
        query_key = project_to_common_dim(hs, common_dim)

        results = engine.mem_read(query_key, 5, None)
        top5_ids = [r.cell_id for r in results]

        if any(m in expected for m in top5_ids):
            hits += 1

    recall = 100 * hits / total if total else 0
    shutil.rmtree(db_dir)

    del query_model, query_tokenizer
    gc.collect()

    return recall, total


def main():
    print("=" * 70)
    print("CROSS-MODEL RETRIEVAL EXPERIMENT")
    print(f"Memories: {NUM_MEMORIES}, Models: Qwen3-0.6B + GPT-2")
    print("=" * 70)

    qwen = "Qwen/Qwen3-0.6B"
    gpt2 = "openai-community/gpt2"

    # Condition A: Same-model baseline (Qwen→Qwen)
    print("\n--- Condition A: Qwen3→Qwen3 (same-model baseline) ---")
    recall_a, total_a = run_experiment(qwen, qwen)
    print(f"  Recall@5: {recall_a:.1f}% ({int(recall_a * total_a / 100)}/{total_a})")

    # Condition B: Same-model baseline (GPT-2→GPT-2)
    print("\n--- Condition B: GPT-2→GPT-2 (same-model baseline) ---")
    recall_b, total_b = run_experiment(gpt2, gpt2)
    print(f"  Recall@5: {recall_b:.1f}% ({int(recall_b * total_b / 100)}/{total_b})")

    # Condition C: Cross-model (Qwen3→GPT-2, projected to common dim)
    print("\n--- Condition C: Qwen3→GPT-2 (cross-model, projected) ---")
    recall_c, total_c = run_experiment(qwen, gpt2, common_dim=768)
    print(f"  Recall@5: {recall_c:.1f}% ({int(recall_c * total_c / 100)}/{total_c})")

    # Condition D: Cross-model (GPT-2→Qwen3, projected to common dim)
    print("\n--- Condition D: GPT-2→Qwen3 (cross-model, projected) ---")
    recall_d, total_d = run_experiment(gpt2, qwen, common_dim=768)
    print(f"  Recall@5: {recall_d:.1f}% ({int(recall_d * total_d / 100)}/{total_d})")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Condition':<35} {'Recall@5':>10}")
    print(f"  {'─' * 48}")
    print(f"  {'A: Qwen3→Qwen3 (baseline)':<35} {recall_a:>9.1f}%")
    print(f"  {'B: GPT-2→GPT-2 (baseline)':<35} {recall_b:>9.1f}%")
    print(f"  {'C: Qwen3→GPT-2 (cross)':<35} {recall_c:>9.1f}%")
    print(f"  {'D: GPT-2→Qwen3 (cross)':<35} {recall_d:>9.1f}%")

    baseline_avg = (recall_a + recall_b) / 2
    cross_avg = (recall_c + recall_d) / 2
    drop = baseline_avg - cross_avg

    print(f"\n  Same-model average:  {baseline_avg:.1f}%")
    print(f"  Cross-model average: {cross_avg:.1f}%")
    print(f"  Drop:                {drop:+.1f}%")

    if cross_avg >= baseline_avg * 0.7:
        verdict = "PASS — cross-model retrieval viable (within 30% of baseline)"
    elif cross_avg >= baseline_avg * 0.3:
        verdict = "PARTIAL — significant degradation but non-zero cross-model signal"
    elif cross_avg > 0:
        verdict = "WEAK — minimal cross-model signal, effectively model-specific"
    else:
        verdict = "FAIL — no cross-model retrieval, TardigradeDB is model-specific"

    print(f"\n  VERDICT: {verdict}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
