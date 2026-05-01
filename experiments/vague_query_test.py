#!/usr/bin/env python3
"""Vague query retrieval experiment.

The most important untested assumption: does TardigradeDB work with
the kind of queries real agents actually ask?

Every existing test uses specific queries:
  "The pharmaceutical patent about enzyme inhibitors"
  "Lucia standing in front of the T-Rex at the Field Museum"

Real agents ask vague questions:
  "How has work been lately?"
  "What's going on with Lucia?"
  "Any health issues?"

Hypothesis: vague queries will show significant recall degradation
because per-token dot-product scoring relies on vocabulary overlap
between query and memory hidden states.

Design (Controlled Experiment):
  - Same 100 memories, same engine, same model (Qwen3-0.6B)
  - Three query tiers:
    A) Specific (existing corpus): "The birth certificate translation"
    B) Moderate: "Any translation work recently?"
    C) Vague: "How is work going?"
  - Each tier has queries covering all 10 domains
  - Compare R@5 across tiers

100 memories from corpus_100.py (Sonia's life across 10 domains).
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
from tardigrade_hooks.hf_kv_hook import HuggingFaceKVHook

MODEL_NAME = "Qwen/Qwen3-0.6B"
DOMAINS = [
    "Work", "Parenting", "Cooking", "Health", "Legal",
    "Social", "Fitness", "Dreams", "Errands", "Media",
]

# ── Vague queries ──────────────────────────────────────────────────────
# Each tuple: (query_text, expected_memory_indices, vagueness_tier)
# Tier "moderate": domain-relevant but no specific details
# Tier "vague": how a real agent/user would actually ask

MODERATE_QUERIES = [
    ("Any translation projects recently?", list(range(0, 10)), "moderate"),
    ("What's been happening with Lucia at school?", [11, 17, 19], "moderate"),
    ("Have you tried any new recipes?", list(range(20, 30)), "moderate"),
    ("Any doctor visits or health concerns?", [30, 31, 32, 33, 34, 35], "moderate"),
    ("What's the latest with the divorce paperwork?", [40, 42, 45, 47], "moderate"),
    ("Seen any friends lately?", list(range(50, 60)), "moderate"),
    ("How's the exercise routine going?", list(range(60, 70)), "moderate"),
    ("Any interesting dreams?", list(range(70, 80)), "moderate"),
    ("What errands needed doing?", list(range(80, 90)), "moderate"),
    ("Watched or read anything good?", list(range(90, 100)), "moderate"),
]

VAGUE_QUERIES = [
    ("How is work going?", list(range(0, 10)), "vague"),
    ("How is Lucia doing?", list(range(10, 20)), "vague"),
    ("What have you been eating?", list(range(20, 30)), "vague"),
    ("How are you feeling?", list(range(30, 40)), "vague"),
    ("How are things with Eduardo?", [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 72], "vague"),
    ("What's your social life like?", list(range(50, 60)), "vague"),
    ("Getting any exercise?", list(range(60, 70)), "vague"),
    ("Sleep well lately?", list(range(70, 80)), "vague"),
    ("Anything annoying happen?", list(range(80, 90)), "vague"),
    ("What are you into these days?", list(range(90, 100)), "vague"),
]

# Broad open-ended queries — no domain target, should retrieve SOMETHING relevant
OPEN_QUERIES = [
    ("Tell me about your week", list(range(100)), "open"),
    ("What's on your mind?", list(range(100)), "open"),
    ("Anything memorable happen recently?", list(range(100)), "open"),
    ("How are things?", list(range(100)), "open"),
    ("What's new?", list(range(100)), "open"),
]


def main():
    print("=" * 70)
    print("VAGUE QUERY RETRIEVAL EXPERIMENT")
    print(f"Model: {MODEL_NAME}, Memories: {len(MEMORIES)}")
    print("=" * 70)

    print(f"\n  Loading {MODEL_NAME}...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32, attn_implementation="eager",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    query_layer = int(n_layers * 0.67)
    print(f"OK ({n_layers}L, ql={query_layer})")

    db_dir = tempfile.mkdtemp(prefix="tdb_vague_")
    engine = tardigrade_db.Engine(db_dir)
    hook = HuggingFaceKVHook(
        engine, owner=1, model_config=model.config,
        model=model, use_hidden_states=True,
    )

    # Store all 100 memories
    print(f"\n--- STORING {len(MEMORIES)} MEMORIES ---")
    for i, memory in enumerate(MEMORIES):
        inputs = tokenizer(memory, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = model(**inputs, use_cache=True, output_hidden_states=True)
        d = hook.on_generate(
            layer=query_layer,
            past_key_values=out.past_key_values,
            model_hidden_states=out.hidden_states[query_layer],
        )
        if d.should_write:
            engine.mem_write(1, query_layer, d.key, d.value, d.salience, None)
    print(f"  Stored {engine.cell_count()} memories")

    # Query function
    def run_queries(queries, label):
        print(f"\n--- {label} ({len(queries)} queries) ---")
        hits_at_1, hits_at_5, hits_at_10 = 0, 0, 0
        total = len(queries)

        for query_text, expected, tier in queries:
            inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                out = model(**inputs, use_cache=True, output_hidden_states=True)
            handles = hook.on_prefill(
                layer=query_layer,
                past_key_values=out.past_key_values,
                model_hidden_states=out.hidden_states[query_layer],
            )
            retrieved = [h.cell_id for h in handles]
            top1 = retrieved[:1]
            top5 = retrieved[:5]
            top10 = retrieved[:10]

            hit1 = any(m in expected for m in top1)
            hit5 = any(m in expected for m in top5)
            hit10 = any(m in expected for m in top10)

            if hit1: hits_at_1 += 1
            if hit5: hits_at_5 += 1
            if hit10: hits_at_10 += 1

            domain = DOMAINS[expected[0] // 10] if expected and expected[0] < 100 else "Any"
            mark = "Y" if hit5 else "N"
            top_domain = DOMAINS[retrieved[0] // 10] if retrieved and retrieved[0] < 100 else "?"
            print(f"  {mark} [{domain:>10}] → [{top_domain:>10}] \"{query_text[:50]}\"")

        r1 = 100 * hits_at_1 / total if total else 0
        r5 = 100 * hits_at_5 / total if total else 0
        r10 = 100 * hits_at_10 / total if total else 0
        print(f"  R@1: {r1:.1f}%  R@5: {r5:.1f}%  R@10: {r10:.1f}%")
        return r1, r5, r10

    # Run all tiers
    # Tier A: Specific (existing corpus)
    specific = [(q, e, "specific") for q, e, t in ALL_QUERIES if t != "negative"]
    r1_a, r5_a, r10_a = run_queries(specific, "TIER A: SPECIFIC (existing corpus)")

    # Tier B: Moderate
    r1_b, r5_b, r10_b = run_queries(MODERATE_QUERIES, "TIER B: MODERATE (domain-relevant)")

    # Tier C: Vague
    r1_c, r5_c, r10_c = run_queries(VAGUE_QUERIES, "TIER C: VAGUE (how agents actually ask)")

    # Tier D: Open-ended
    r1_d, r5_d, r10_d = run_queries(OPEN_QUERIES, "TIER D: OPEN-ENDED (broadest possible)")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    print(f"  {'Tier':<35} {'R@1':>8} {'R@5':>8} {'R@10':>8}")
    print(f"  {'─' * 63}")
    print(f"  {'A: Specific (benchmark queries)':<35} {r1_a:>7.1f}% {r5_a:>7.1f}% {r10_a:>7.1f}%")
    print(f"  {'B: Moderate (domain-relevant)':<35} {r1_b:>7.1f}% {r5_b:>7.1f}% {r10_b:>7.1f}%")
    print(f"  {'C: Vague (real agent queries)':<35} {r1_c:>7.1f}% {r5_c:>7.1f}% {r10_c:>7.1f}%")
    print(f"  {'D: Open-ended (broadest)':<35} {r1_d:>7.1f}% {r5_d:>7.1f}% {r10_d:>7.1f}%")

    drop_b = r5_a - r5_b
    drop_c = r5_a - r5_c
    drop_d = r5_a - r5_d

    print(f"\n  Degradation from specific baseline:")
    print(f"    Moderate: {drop_b:+.1f}%")
    print(f"    Vague:    {drop_c:+.1f}%")
    print(f"    Open:     {drop_d:+.1f}%")
    print(f"{'=' * 70}")

    shutil.rmtree(db_dir)


if __name__ == "__main__":
    main()
