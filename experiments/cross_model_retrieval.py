#!/usr/bin/env python3
"""Cross-model retrieval: same-family + learned projection + CCA alignment.

Progressive Hypothesis Testing (Controlled Experiment pattern):

Hypotheses:
  H1: Same-family cross-model (Qwen3-0.6B → Qwen3-1.7B) works better
      than cross-family because they share tokenizer + training data.
  H2: A learned linear projection (closed-form least-squares) recovers
      cross-family recall.
  H3: CCA alignment (SVD-based, no training loop) finds a shared space
      where cross-model dot products are meaningful.

Results (run 2026-04-29):
  A: Qwen3-0.6B baseline:      100.0%
  B: Qwen3-1.7B baseline:      100.0%
  C: Same-family truncation:      0.0%  (energy distribution mismatch)
  D: GPT-2 baseline:            100.0%
  E: Cross-family raw:            6.7%
  F: Cross-family learned proj:  13.3%  (50 training samples)
  G: Cross-family CCA:            0.0%  (overfits on 50 samples)

Follow-up diagnostic:
  Same-family learned projection (100 training samples): 43.3%
  Truncation destroys signal because Qwen3-1.7B has 67% energy in
  upper dims (1024-2047), opposite of 0.6B (71% in dims 0-255).
  Cross-model is NOT a fundamental impossibility — it's a coordinate
  system mismatch solvable with learned projections.

Training data scaling (run 2026-04-29):
  Same-family (Qwen3-0.6B → 1.7B) with per-token projection (2433 tokens):
    90.0% R@5 — near baseline (96.7%) with zero model modification.
  Cross-family (Qwen3 → GPT-2) caps at ~23% with mean-pool projection.
  Per-token projection overfits cross-family (regresses to 10%).

  Same-family + per-token linear projection is a viable cross-model path.
  Cross-family remains model-specific without deeper alignment.

Conditions (100 memories, 30 queries, per-token encoding, skip pos 0):
  A: Qwen3-0.6B → Qwen3-0.6B native (baseline, known 96.7%)
  B: Qwen3-1.7B → Qwen3-1.7B native (H1 control)
  C: Qwen3-0.6B → Qwen3-1.7B same-family (H1, truncation to 1024)
  D: GPT-2 → GPT-2 native (baseline, known 96.7%)
  E: Qwen3-0.6B → GPT-2 raw truncation (known ~10%)
  F: Qwen3-0.6B → GPT-2 learned projection (H2)
  G: Qwen3-0.6B ↔ GPT-2 CCA alignment (H3)

Training/test split: first 50 memories for projection training,
remaining 50 for evaluation. Queries always from test set.

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

TRAIN_SPLIT = 50  # first 50 memories for projection training
TEST_SPLIT = 50   # remaining 50 for evaluation


# ── Model loading ────────────────────────────────────────────────────

def load_model(name):
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


# ── Hidden state extraction ──────────────────────────────────────────

def extract_hidden(model, tokenizer, text, ql):
    """Per-token hidden states from layer ql, skipping position 0."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[ql][0][1:].numpy().astype(np.float32)


def extract_all_hidden(model, tokenizer, texts, ql):
    """Extract hidden states for a list of texts. Returns list of (seq_len, dim) arrays."""
    return [extract_hidden(model, tokenizer, t, ql) for t in texts]


def mean_pool_all(hidden_list):
    """Mean-pool each (seq_len, dim) → (dim,). Returns (N, dim) matrix."""
    return np.array([h.mean(axis=0) for h in hidden_list], dtype=np.float32)


# ── Projection methods ───────────────────────────────────────────────

def truncate(vecs, target_dim):
    """Truncate per-token (seq_len, dim) to (seq_len, target_dim)."""
    if vecs.shape[1] <= target_dim:
        return vecs
    return vecs[:, :target_dim].copy()


def learn_projection(H_source, H_target):
    """Least-squares projection: W = H_target @ pinv(H_source).

    H_source: (N, dim_source), H_target: (N, dim_target).
    Returns W of shape (dim_target, dim_source).
    """
    W = H_target.T @ np.linalg.pinv(H_source.T)
    return W.astype(np.float32)


def cca_projections(H_a, H_b, n_components):
    """SVD-based CCA. Returns (W_a, W_b) projecting both to shared space.

    H_a: (N, dim_a), H_b: (N, dim_b). Both centered.
    Returns W_a (n_components, dim_a), W_b (n_components, dim_b).
    """
    H_a = H_a - H_a.mean(axis=0)
    H_b = H_b - H_b.mean(axis=0)

    n = H_a.shape[0]
    C_aa = (H_a.T @ H_a) / n + 1e-6 * np.eye(H_a.shape[1])
    C_bb = (H_b.T @ H_b) / n + 1e-6 * np.eye(H_b.shape[1])
    C_ab = (H_a.T @ H_b) / n

    # Whitening
    Ua, Sa, _ = np.linalg.svd(C_aa, full_matrices=False)
    inv_sqrt_a = Ua @ np.diag(1.0 / np.sqrt(Sa + 1e-8)) @ Ua.T

    Ub, Sb, _ = np.linalg.svd(C_bb, full_matrices=False)
    inv_sqrt_b = Ub @ np.diag(1.0 / np.sqrt(Sb + 1e-8)) @ Ub.T

    M = inv_sqrt_a @ C_ab @ inv_sqrt_b
    U, _, Vt = np.linalg.svd(M, full_matrices=False)

    W_a = (U[:, :n_components].T @ inv_sqrt_a).astype(np.float32)
    W_b = (Vt[:n_components, :] @ inv_sqrt_b).astype(np.float32)
    return W_a, W_b


# ── Engine operations ─────────────────────────────────────────────────

def store_per_token(engine, hidden_list, transform=None):
    """Store per-token encoded memories. transform: fn(seq_len, dim) → (seq_len, dim')."""
    for h in hidden_list:
        if transform is not None:
            h = transform(h)
        key = encode_per_token(h, h.shape[1])
        value = np.zeros(h.shape[1], dtype=np.float32)
        engine.mem_write(1, 0, key, value, 50.0, None)


def query_recall(engine, hidden_list, expected_ids, transform=None):
    """Query with per-token encoding, return (R@5, hits, total)."""
    hits, total = 0, 0
    for i, (query_text, expected, qtype) in enumerate(ALL_QUERIES):
        if qtype == "negative":
            continue
        if any(e >= TRAIN_SPLIT + TEST_SPLIT for e in expected):
            continue
        # Adjust expected IDs: test memories are stored at indices 0..TEST_SPLIT-1
        adjusted = [e - TRAIN_SPLIT for e in expected if e >= TRAIN_SPLIT]
        if not adjusted:
            continue
        total += 1
        h = hidden_list[i]
        if transform is not None:
            h = transform(h)
        key = encode_per_token(h, h.shape[1])
        results = engine.mem_read(key, 5, None)
        top5 = [r.cell_id for r in results[:5]]
        if any(m in adjusted for m in top5):
            hits += 1
    recall = 100 * hits / total if total else 0
    return recall, hits, total


# ── Experiment conditions ─────────────────────────────────────────────

def run_same_model(label, model_name):
    """Baseline: store and query with the same model, native dim."""
    print(f"\n--- {label} ---")
    model, tok, ql, dim = load_model(model_name)

    test_memories = MEMORIES[TRAIN_SPLIT:TRAIN_SPLIT + TEST_SPLIT]
    store_hidden = extract_all_hidden(model, tok, test_memories, ql)

    query_texts = [(q, e, t) for q, e, t in ALL_QUERIES if t != "negative"]
    query_hidden = [extract_hidden(model, tok, q, ql) for q, _, _ in query_texts]

    db_dir = tempfile.mkdtemp(prefix="tdb_xm_")
    engine = tardigrade_db.Engine(db_dir)
    store_per_token(engine, store_hidden)
    print(f"  Stored {engine.cell_count()} memories (dim={dim})")

    recall, hits, total = query_recall(engine, query_hidden, None)
    print(f"  R@5: {recall:.1f}% ({hits}/{total})")

    free_model(model, tok)
    shutil.rmtree(db_dir)
    return recall


def run_cross_model(label, store_name, query_name, store_transform, query_transform):
    """Cross-model: store with one model, query with another."""
    print(f"\n--- {label} ---")

    # Extract store hidden states
    model, tok, ql, _ = load_model(store_name)
    test_memories = MEMORIES[TRAIN_SPLIT:TRAIN_SPLIT + TEST_SPLIT]
    store_hidden = extract_all_hidden(model, tok, test_memories, ql)
    free_model(model, tok)

    # Extract query hidden states
    model, tok, ql, _ = load_model(query_name)
    query_texts = [(q, e, t) for q, e, t in ALL_QUERIES if t != "negative"]
    query_hidden = [extract_hidden(model, tok, q, ql) for q, _, _ in query_texts]
    free_model(model, tok)

    db_dir = tempfile.mkdtemp(prefix="tdb_xm_")
    engine = tardigrade_db.Engine(db_dir)
    store_per_token(engine, store_hidden, store_transform)
    print(f"  Stored {engine.cell_count()} memories")

    recall, hits, total = query_recall(engine, query_hidden, None, query_transform)
    print(f"  R@5: {recall:.1f}% ({hits}/{total})")

    shutil.rmtree(db_dir)
    return recall


def extract_training_data(model_name):
    """Extract hidden states for the training split (first 50 memories)."""
    model, tok, ql, dim = load_model(model_name)
    train_texts = MEMORIES[:TRAIN_SPLIT]
    hidden = extract_all_hidden(model, tok, train_texts, ql)
    pooled = mean_pool_all(hidden)
    free_model(model, tok)
    return pooled, dim


def main():
    qwen06 = "Qwen/Qwen3-0.6B"
    qwen17 = "Qwen/Qwen3-1.7B"
    gpt2 = "openai-community/gpt2"

    print("=" * 70)
    print("CROSS-MODEL RETRIEVAL: Same-Family + Learned Projection + CCA")
    print(f"Memories: {TEST_SPLIT} (test) + {TRAIN_SPLIT} (train), Queries: 30")
    print("=" * 70)

    results = {}

    # ── Baselines ──
    results["A"] = run_same_model("A: Qwen3-0.6B → Qwen3-0.6B (baseline)", qwen06)
    results["B"] = run_same_model("B: Qwen3-1.7B → Qwen3-1.7B (H1 control)", qwen17)
    results["D"] = run_same_model("D: GPT-2 → GPT-2 (baseline)", gpt2)

    # ── H1: Same-family ──
    results["C"] = run_cross_model(
        "C: Qwen3-0.6B → Qwen3-1.7B (H1: same-family, trunc to 1024)",
        qwen06, qwen17,
        store_transform=None,  # native 1024
        query_transform=lambda h: truncate(h, 1024),  # 2048 → 1024
    )

    # ── H2 + H3: need training data ──
    print("\n--- Training projection matrices ---")
    print("  Extracting Qwen3-0.6B training hidden states...")
    H_qwen, dim_qwen = extract_training_data(qwen06)
    print(f"  Qwen3 training data: {H_qwen.shape}")

    print("  Extracting GPT-2 training hidden states...")
    H_gpt2, dim_gpt2 = extract_training_data(gpt2)
    print(f"  GPT-2 training data: {H_gpt2.shape}")

    # H2: Learned linear projection (Qwen3 → GPT-2 space)
    print("  Learning projection W (least-squares)...")
    W_proj = learn_projection(H_qwen, H_gpt2)
    print(f"  W shape: {W_proj.shape}")

    # H3: CCA alignment
    n_cca = min(256, dim_gpt2)
    print(f"  Computing CCA alignment (n_components={n_cca})...")
    W_cca_a, W_cca_b = cca_projections(H_qwen, H_gpt2, n_cca)
    print(f"  CCA W_a: {W_cca_a.shape}, W_b: {W_cca_b.shape}")

    # ── E: Raw cross-family (known ~10%) ──
    results["E"] = run_cross_model(
        "E: Qwen3-0.6B → GPT-2 raw truncation (known baseline)",
        qwen06, gpt2,
        store_transform=lambda h: truncate(h, dim_gpt2),
        query_transform=None,
    )

    # ── F: Learned projection ──
    results["F"] = run_cross_model(
        "F: Qwen3-0.6B → GPT-2 learned projection (H2)",
        qwen06, gpt2,
        store_transform=lambda h: (h @ W_proj.T).astype(np.float32),
        query_transform=None,  # GPT-2 native 768
    )

    # ── G: CCA alignment ──
    results["G"] = run_cross_model(
        "G: Qwen3-0.6B ↔ GPT-2 CCA alignment (H3)",
        qwen06, gpt2,
        store_transform=lambda h: (h @ W_cca_a.T).astype(np.float32),
        query_transform=lambda h: (h @ W_cca_b.T).astype(np.float32),
    )

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    labels = {
        "A": "Qwen3-0.6B → Qwen3-0.6B (baseline)",
        "B": "Qwen3-1.7B → Qwen3-1.7B (H1 control)",
        "C": "Qwen3-0.6B → Qwen3-1.7B (H1: same-family)",
        "D": "GPT-2 → GPT-2 (baseline)",
        "E": "Qwen3-0.6B → GPT-2 raw (known ~10%)",
        "F": "Qwen3-0.6B → GPT-2 learned proj (H2)",
        "G": "Qwen3-0.6B ↔ GPT-2 CCA (H3)",
    }
    print(f"  {'Condition':<50} {'R@5':>8}")
    print(f"  {'─' * 60}")
    for k in "ABCDEFG":
        print(f"  {labels[k]:<50} {results[k]:>7.1f}%")

    print(f"\n  HYPOTHESIS TESTS:")
    print(f"  H1 same-family:   C={results['C']:.0f}% vs cross-family E={results['E']:.0f}% "
          f"(Δ={results['C']-results['E']:+.0f}%)")
    print(f"  H2 learned proj:  F={results['F']:.0f}% vs raw E={results['E']:.0f}% "
          f"(Δ={results['F']-results['E']:+.0f}%)")
    print(f"  H3 CCA:           G={results['G']:.0f}% vs raw E={results['E']:.0f}% "
          f"(Δ={results['G']-results['E']:+.0f}%)")
    print(f"  Baselines:        Qwen3-0.6B={results['A']:.0f}%, "
          f"Qwen3-1.7B={results['B']:.0f}%, GPT-2={results['D']:.0f}%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
