#!/usr/bin/env python3
"""Cross-model retrieval experiment — per-token encoding with projection.

Controlled Experiment (Isolation of Variables):
Each condition isolates one variable while holding the per-token
encoding path constant. We only vary projection and model source.

Hypotheses:
  H1: Per-token encoding still works after dimension projection (same-model)
  H2: Per-token encoding works cross-model with projected hidden states
  H3: Projection method matters (truncation vs random orthogonal)

Conditions (100 memories, 30 queries, per-token encoding throughout):
  A: Qwen3→Qwen3 native dim=1024 (baseline, proven 100%)
  B: Qwen3→Qwen3 truncated to dim=768 (H1: does projection hurt?)
  C: Qwen3→GPT-2 truncation, dim=768 (H2a: cross-model)
  D: GPT-2→Qwen3 zero-pad+truncate, dim=768 (H2b: reverse direction)
  E: Qwen3→Qwen3 random orthogonal to dim=768 (H3: projection method)
  F: GPT-2→GPT-2 native dim=768 (second model baseline)

Models loaded sequentially to avoid OOM.
"""

import gc
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(".").resolve() / "python"))
sys.path.insert(0, str(Path(".").resolve() / "experiments"))

import tardigrade_db
from corpus_100 import ALL_QUERIES, MEMORIES
from tardigrade_hooks.encoding import encode_per_token

COMMON_DIM = 768


def load_model(name):
    """Load model, return (model, tokenizer, query_layer, hidden_dim). Caller must free."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float32, attn_implementation="eager",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    ql = int(n_layers * 0.67)
    hidden = model.config.hidden_size
    print(f"  Loaded {name} ({n_layers}L, ql={ql}, dim={hidden})")
    return model, tokenizer, ql, hidden


def free_model(model, tokenizer):
    del model, tokenizer
    gc.collect()


def extract_per_token_hidden(model, tokenizer, text, query_layer):
    """Extract per-token hidden states from a layer, skipping position 0.

    Position 0 is the attention sink token — it dominates dot products
    and creates gravity wells. The HuggingFaceKVHook skips it (h[1:]).
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states[query_layer][0]  # (seq_len, hidden_dim)
    return hs[1:].numpy().astype(np.float32)  # skip position 0


def make_random_orthogonal_projection(dim_out, dim_in, seed=42):
    """Random orthogonal projection matrix (dim_out, dim_in). Preserves distances approximately."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((dim_out, dim_in)).astype(np.float32)
    u, _, vt = np.linalg.svd(raw, full_matrices=False)
    return u @ vt  # (dim_out, dim_in), orthonormal rows


def project(token_vecs, projection):
    """Project (seq_len, dim_in) → (seq_len, dim_out) via matrix multiply."""
    if projection is None:
        return token_vecs
    return (token_vecs @ projection.T).astype(np.float32)


def truncate_to_dim(token_vecs, target_dim):
    """Truncate or zero-pad (seq_len, dim) to (seq_len, target_dim)."""
    _, dim = token_vecs.shape
    if dim == target_dim:
        return token_vecs
    if dim > target_dim:
        return token_vecs[:, :target_dim].copy()
    padded = np.zeros((token_vecs.shape[0], target_dim), dtype=np.float32)
    padded[:, :dim] = token_vecs
    return padded


def store_memories(model, tokenizer, ql, engine, dim_transform):
    """Store 100 memories with per-token encoding. dim_transform: fn(token_vecs) → token_vecs."""
    for memory in MEMORIES:
        tokens = extract_per_token_hidden(model, tokenizer, memory, ql)
        tokens = dim_transform(tokens)
        key = encode_per_token(tokens, tokens.shape[1])
        value = np.zeros(tokens.shape[1], dtype=np.float32)
        engine.mem_write(1, 0, key, value, 50.0, None)


def query_recall(model, tokenizer, ql, engine, dim_transform):
    """Run 30 queries with per-token encoding, return (R@5, hits, total)."""
    hits, total = 0, 0
    for query_text, expected, qtype in ALL_QUERIES:
        if qtype == "negative":
            continue
        if any(e >= len(MEMORIES) for e in expected):
            continue
        total += 1
        tokens = extract_per_token_hidden(model, tokenizer, query_text, ql)
        tokens = dim_transform(tokens)
        key = encode_per_token(tokens, tokens.shape[1])
        results = engine.mem_read(key, 5, None)
        top5 = [r.cell_id for r in results[:5]]
        if any(m in expected for m in top5):
            hits += 1
    recall = 100 * hits / total if total else 0
    return recall, hits, total


def run_condition(label, store_model_name, query_model_name,
                  store_transform, query_transform):
    """Run one condition: load models sequentially, store, query, report."""
    print(f"\n--- {label} ---")

    # Store phase
    model, tokenizer, ql, _ = load_model(store_model_name)
    db_dir = tempfile.mkdtemp(prefix="tdb_xm_pt_")
    engine = tardigrade_db.Engine(db_dir)
    store_memories(model, tokenizer, ql, engine, store_transform)
    print(f"  Stored {engine.cell_count()} memories")

    same = store_model_name == query_model_name
    if not same:
        free_model(model, tokenizer)
        model, tokenizer, ql, _ = load_model(query_model_name)

    recall, hits, total = query_recall(model, tokenizer, ql, engine, query_transform)
    print(f"  Recall@5: {recall:.1f}% ({hits}/{total})")

    free_model(model, tokenizer)
    shutil.rmtree(db_dir)
    return recall, hits, total


def main():
    qwen = "Qwen/Qwen3-0.6B"
    gpt2 = "openai-community/gpt2"

    ortho_proj = make_random_orthogonal_projection(COMMON_DIM, 1024)

    identity = lambda t: t
    trunc_768 = lambda t: truncate_to_dim(t, COMMON_DIM)
    ortho_768 = lambda t: project(t, ortho_proj)

    print("=" * 70)
    print("CROSS-MODEL RETRIEVAL: Per-Token Encoding + Projection")
    print(f"Memories: {len(MEMORIES)}, Queries: 30, Common dim: {COMMON_DIM}")
    print("=" * 70)

    results = {}

    # A: Qwen3→Qwen3 native (baseline)
    r, _, _ = run_condition("A: Qwen3→Qwen3 native dim=1024 (baseline)",
                            qwen, qwen, identity, identity)
    results["A"] = r

    # B: Qwen3→Qwen3 truncated to 768 (H1: projection effect)
    r, _, _ = run_condition("B: Qwen3→Qwen3 truncated to 768 (H1)",
                            qwen, qwen, trunc_768, trunc_768)
    results["B"] = r

    # C: Qwen3→GPT-2, Qwen truncated to 768, GPT-2 native 768 (H2a)
    r, _, _ = run_condition("C: Qwen3→GPT-2 (H2a: cross-model, truncation)",
                            qwen, gpt2, trunc_768, identity)
    results["C"] = r

    # D: GPT-2→Qwen3, GPT-2 native 768, Qwen truncated to 768 (H2b)
    r, _, _ = run_condition("D: GPT-2→Qwen3 (H2b: cross-model, reverse)",
                            gpt2, qwen, identity, trunc_768)
    results["D"] = r

    # E: Qwen3→Qwen3 random orthogonal to 768 (H3: projection method)
    r, _, _ = run_condition("E: Qwen3→Qwen3 ortho projection to 768 (H3)",
                            qwen, qwen, ortho_768, ortho_768)
    results["E"] = r

    # F: GPT-2→GPT-2 native (second model baseline)
    r, _, _ = run_condition("F: GPT-2→GPT-2 native dim=768 (baseline)",
                            gpt2, gpt2, identity, identity)
    results["F"] = r

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    labels = {
        "A": "Qwen3→Qwen3 native 1024 (baseline)",
        "B": "Qwen3→Qwen3 truncated 768 (H1: projection)",
        "C": "Qwen3→GPT-2 truncation (H2a: cross-model)",
        "D": "GPT-2→Qwen3 truncation (H2b: reverse)",
        "E": "Qwen3→Qwen3 ortho 768 (H3: projection method)",
        "F": "GPT-2→GPT-2 native 768 (baseline)",
    }
    print(f"  {'Condition':<50} {'R@5':>8}")
    print(f"  {'─' * 60}")
    for k in "ABCDEF":
        print(f"  {labels[k]:<50} {results[k]:>7.1f}%")

    # Analysis
    print(f"\n  HYPOTHESIS TESTS:")
    print(f"  H1 (projection hurts?): A={results['A']:.0f}% → B={results['B']:.0f}% "
          f"(Δ={results['B']-results['A']:+.0f}%)")
    print(f"  H2a (cross Qwen→GPT2): C={results['C']:.0f}%")
    print(f"  H2b (cross GPT2→Qwen): D={results['D']:.0f}%")
    print(f"  H3 (ortho vs trunc):   E={results['E']:.0f}% vs B={results['B']:.0f}% "
          f"(Δ={results['E']-results['B']:+.0f}%)")
    print(f"  Baselines: Qwen={results['A']:.0f}%, GPT-2={results['F']:.0f}%")

    cross_avg = (results["C"] + results["D"]) / 2
    same_avg = (results["A"] + results["F"]) / 2
    print(f"\n  Same-model avg (native): {same_avg:.1f}%")
    print(f"  Cross-model avg:         {cross_avg:.1f}%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
