#!/usr/bin/env python3
"""Scale recall benchmark: parameterized test from 100 to 5K memories.

Property-Based Testing pattern: same test logic at multiple scale points.
Reports recall@1, recall@5, gravity well concentration, and latency.

Reuses the 100-memory Sonia corpus as the "signal" memories with known
ground-truth queries. Pads with N-100 generated distractor memories to
test whether retrieval holds as the haystack grows.

Requires: Qwen3-0.6B (CPU inference), ~2-10 min per scale point.
"""

import argparse
import shutil
import sys
import tempfile
import time
from collections import Counter
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

DISTRACTOR_TEMPLATES = [
    "On {day}, agent {name} recorded observation #{num}: the {adj} {noun} measured {val} {unit} in sector {sector}.",
    "Report from {name}: {noun} levels in {sector} reached {val} {unit} at {time} on {day}.",
    "Memo #{num} — {name} confirmed the {adj} {noun} shipment ({val} {unit}) arrived at warehouse {sector}.",
    "{name} logged {val} {unit} of {adj} {noun} during the {day} survey of zone {sector} (ref #{num}).",
    "Incident #{num}: {name} observed {adj} {noun} activity at {val} {unit} near checkpoint {sector} on {day}.",
]

NAMES = ["Kovacs", "Tanaka", "Okafor", "Bergstrom", "Medina", "Petrov",
         "Chen", "Nguyen", "Andersen", "Dlamini", "Moreau", "Ishikawa"]
ADJS = ["anomalous", "stable", "volatile", "residual", "latent",
        "compressed", "dispersed", "crystalline", "ambient", "thermal"]
NOUNS = ["particle", "compound", "alloy", "polymer", "catalyst",
         "substrate", "membrane", "filament", "isotope", "reagent"]
UNITS = ["mSv", "ppm", "kPa", "mol/L", "dB", "μm", "Hz", "lux", "ohm", "g/cm³"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
SECTORS = ["Alpha-7", "Beta-3", "Gamma-9", "Delta-1", "Epsilon-5",
           "Zeta-2", "Eta-8", "Theta-4", "Iota-6", "Kappa-0"]


def generate_distractors(count: int, seed: int = 42) -> list[str]:
    """Generate N unique distractor memories unrelated to the Sonia corpus."""
    rng = np.random.RandomState(seed)
    distractors = []
    for i in range(count):
        template = DISTRACTOR_TEMPLATES[i % len(DISTRACTOR_TEMPLATES)]
        text = template.format(
            name=NAMES[rng.randint(len(NAMES))],
            adj=ADJS[rng.randint(len(ADJS))],
            noun=NOUNS[rng.randint(len(NOUNS))],
            val=round(rng.uniform(0.1, 999.9), 1),
            unit=UNITS[rng.randint(len(UNITS))],
            day=DAYS[rng.randint(len(DAYS))],
            sector=SECTORS[rng.randint(len(SECTORS))],
            num=1000 + i,
            time=f"{rng.randint(0, 24):02d}:{rng.randint(0, 60):02d}",
        )
        distractors.append(text)
    return distractors


def run_scale_point(model, tokenizer, hook, query_layer, count: int) -> dict:
    """Run one scale point: store `count` memories, query with ground truth."""
    db_dir = Path(tempfile.mkdtemp(prefix=f"tdb_scale_{count}_"))
    engine = tardigrade_db.Engine(str(db_dir))
    hook.engine = engine

    # Store the 100 signal memories first (indices 0-99)
    signal_count = min(count, len(MEMORIES))
    distractor_count = max(0, count - len(MEMORIES))
    distractors = generate_distractors(distractor_count) if distractor_count > 0 else []

    all_memories = list(MEMORIES[:signal_count]) + distractors

    t0 = time.time()
    for i, memory in enumerate(all_memories):
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
    store_time = time.time() - t0

    # Query with ground truth
    hits_at_1, hits_at_5, all_top1, latencies = 0, 0, [], []
    total_queries = 0

    for query_text, expected, qtype in ALL_QUERIES:
        if qtype == "negative":
            continue
        # Skip queries whose expected memories are beyond our signal count
        if any(e >= signal_count for e in expected):
            continue

        total_queries += 1
        inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = model(**inputs, use_cache=True, output_hidden_states=True)

        t1 = time.time()
        handles = hook.on_prefill(
            layer=query_layer,
            past_key_values=out.past_key_values,
            model_hidden_states=out.hidden_states[query_layer],
        )
        latencies.append((time.time() - t1) * 1000)

        retrieved = [h.cell_id for h in handles]
        top1 = retrieved[0] if retrieved else -1
        top5 = retrieved[:5]

        all_top1.append(top1)
        if any(m in expected for m in [top1]):
            hits_at_1 += 1
        if any(m in expected for m in top5):
            hits_at_5 += 1

    recall_at_1 = 100 * hits_at_1 / total_queries if total_queries else 0
    recall_at_5 = 100 * hits_at_5 / total_queries if total_queries else 0

    top1c = Counter(all_top1)
    worst = top1c.most_common(1)[0] if top1c else (-1, 0)
    concentration = 100 * worst[1] / total_queries if total_queries else 0

    shutil.rmtree(db_dir)

    return {
        "count": count,
        "cells": engine.cell_count(),
        "queries": total_queries,
        "recall_at_1": recall_at_1,
        "recall_at_5": recall_at_5,
        "unique_top1": len(top1c),
        "gravity_well": worst,
        "concentration_pct": concentration,
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0,
        "store_time_s": store_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Scale recall benchmark")
    parser.add_argument(
        "--counts", default="100,500,1000",
        help="Comma-separated memory counts to test (default: 100,500,1000)",
    )
    args = parser.parse_args()
    counts = [int(c.strip()) for c in args.counts.split(",")]

    print("=" * 70)
    print("SCALE RECALL BENCHMARK: Hidden States + Top5Avg")
    print(f"Scale points: {counts}")
    print("=" * 70)

    print(f"\n  Loading {MODEL_NAME}...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32, attn_implementation="eager",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    query_layer = int(n_layers * 0.67)
    print(f"OK ({n_layers}L, query_layer={query_layer})")

    dummy_engine = tardigrade_db.Engine(tempfile.mkdtemp())
    hook = HuggingFaceKVHook(
        dummy_engine, owner=1, model_config=model.config,
        model=model, use_hidden_states=True,
    )

    results = []
    for count in counts:
        print(f"\n{'─' * 70}")
        print(f"  Scale point: {count} memories")
        print(f"{'─' * 70}")

        result = run_scale_point(model, tokenizer, hook, query_layer, count)
        results.append(result)

        print(f"  Cells stored:    {result['cells']}")
        print(f"  Queries:         {result['queries']}")
        print(f"  Recall@1:        {result['recall_at_1']:.1f}%")
        print(f"  Recall@5:        {result['recall_at_5']:.1f}%")
        print(f"  Unique top-1:    {result['unique_top1']}/{result['queries']}")
        print(f"  Gravity well:    mem {result['gravity_well'][0]} "
              f"({result['gravity_well'][1]}x, {result['concentration_pct']:.1f}%)")
        print(f"  Avg latency:     {result['avg_latency_ms']:.1f}ms")
        print(f"  Store time:      {result['store_time_s']:.1f}s")

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Count':>8} {'R@1':>8} {'R@5':>8} {'GravWell':>10} {'Latency':>10} {'Store':>8}")
    print(f"  {'─' * 54}")
    for r in results:
        print(
            f"  {r['count']:>8} "
            f"{r['recall_at_1']:>7.1f}% "
            f"{r['recall_at_5']:>7.1f}% "
            f"{r['concentration_pct']:>9.1f}% "
            f"{r['avg_latency_ms']:>8.1f}ms "
            f"{r['store_time_s']:>7.1f}s"
        )

    # Verdict
    baseline = results[0]["recall_at_5"] if results else 0
    degraded = any(r["recall_at_5"] < baseline * 0.7 for r in results[1:])
    print(f"\n  Baseline (100 memories): {baseline:.1f}% recall@5")
    if degraded:
        first_drop = next(r for r in results[1:] if r["recall_at_5"] < baseline * 0.7)
        print(f"  DEGRADATION at {first_drop['count']} memories: "
              f"{first_drop['recall_at_5']:.1f}% (>{30:.0f}% drop from baseline)")
    else:
        print(f"  NO SIGNIFICANT DEGRADATION across tested scale points")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
