#!/usr/bin/env python3
"""Phase 34: Do entity tokens produce higher per-token matches than topic tokens?

Compares max per-token cosine similarity between:
  A) Cross-ref linking fact vs its related background memory (entity match)
  B) Same-domain unrelated memories (topic match)
  C) Cross-domain unrelated memories (baseline)

Usage:
    source .venv/bin/activate
    python experiments/token_level_entity_diagnostic.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "python"))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from corpus_100 import MEMORIES
from multi_memory_scale_test import CROSS_REF_PAIRS


def max_token_cosine(tokens_a, tokens_b):
    """Max cosine similarity between any token pair."""
    norms_a = np.linalg.norm(tokens_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(tokens_b, axis=1, keepdims=True)
    a_norm = tokens_a / np.maximum(norms_a, 1e-8)
    b_norm = tokens_b / np.maximum(norms_b, 1e-8)
    sim_matrix = a_norm @ b_norm.T
    return float(np.max(sim_matrix))


def get_tokens(model, tokenizer, text, query_layer):
    """Get per-token hidden states (skip pos 0)."""
    messages = [{"role": "system", "content": text}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    input_ids = tokenizer.encode(formatted, return_tensors="pt")
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
    return out.hidden_states[query_layer][0][1:].numpy().astype(np.float32)


def main():
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    query_layer = int(model.config.num_hidden_layers * 0.67)
    print(f"Query layer: {query_layer}\n", flush=True)

    # A: Entity matches (cross-ref fact vs related background)
    print("=" * 70, flush=True)
    print("A: ENTITY MATCHES (cross-ref vs related background)", flush=True)
    print("=" * 70, flush=True)

    entity_scores = []
    for i, entry in enumerate(CROSS_REF_PAIRS[:10]):
        link_fact = entry["facts"][0]
        tokens_link = get_tokens(model, tokenizer, link_fact, query_layer)

        best_score = 0
        best_bg = ""
        for mem in MEMORIES:
            tokens_bg = get_tokens(model, tokenizer, mem, query_layer)
            score = max_token_cosine(tokens_link, tokens_bg)
            if score > best_score:
                best_score = score
                best_bg = mem

        entity_scores.append(best_score)
        print(f"  Q{i+1} sim={best_score:.4f}: {link_fact[:40]}...", flush=True)
        print(f"    -> {best_bg[:40]}...", flush=True)

    # B: Topic matches (same-domain unrelated)
    print(f"\n{'='*70}", flush=True)
    print("B: TOPIC MATCHES (same-domain unrelated)", flush=True)
    print("=" * 70, flush=True)

    topic_scores = []
    domains = ["Work", "Parenting", "Cooking", "Health", "Legal"]
    for d, name in enumerate(domains):
        a, b = MEMORIES[d*10], MEMORIES[d*10 + 1]
        ta = get_tokens(model, tokenizer, a, query_layer)
        tb = get_tokens(model, tokenizer, b, query_layer)
        score = max_token_cosine(ta, tb)
        topic_scores.append(score)
        print(f"  {name}: sim={score:.4f}", flush=True)

    # C: Unrelated (cross-domain)
    print(f"\n{'='*70}", flush=True)
    print("C: UNRELATED (cross-domain)", flush=True)
    print("=" * 70, flush=True)

    unrelated_scores = []
    for ia, ib in [(0, 20), (10, 30), (20, 40), (30, 50), (40, 60)]:
        ta = get_tokens(model, tokenizer, MEMORIES[ia], query_layer)
        tb = get_tokens(model, tokenizer, MEMORIES[ib], query_layer)
        score = max_token_cosine(ta, tb)
        unrelated_scores.append(score)
        print(f"  sim={score:.4f}: [{ia}] vs [{ib}]", flush=True)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)
    e, t, u = np.mean(entity_scores), np.mean(topic_scores), np.mean(unrelated_scores)
    print(f"  Entity (same event):  {e:.4f} (min={min(entity_scores):.4f} max={max(entity_scores):.4f})", flush=True)
    print(f"  Topic (same domain):  {t:.4f} (min={min(topic_scores):.4f} max={max(topic_scores):.4f})", flush=True)
    print(f"  Unrelated (cross):    {u:.4f} (min={min(unrelated_scores):.4f} max={max(unrelated_scores):.4f})", flush=True)
    print(flush=True)

    if e > t * 1.1:
        print(f"  SIGNAL: entity ({e:.4f}) > topic ({t:.4f}) by {100*(e-t)/t:.1f}%", flush=True)
        print(f"  Threshold: {(e + t) / 2:.4f}", flush=True)
    else:
        print(f"  NO SIGNAL: entity ({e:.4f}) ~ topic ({t:.4f})", flush=True)


if __name__ == "__main__":
    main()
