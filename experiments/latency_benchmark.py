#!/usr/bin/env python3
"""Latency benchmark: compare pipeline configurations at scale.

Strategy Benchmark (Template Method pattern): same query workload
against two pipeline configurations:
  1. Default engine (Vamana threshold=10K, effectively brute-force at <10K cells)
  2. Vamana-activated engine (threshold lowered to trigger Vamana)

Reports latency and recall for each configuration at each scale point.
Proves whether acceleration structures actually reduce query time.

Requires: Qwen3-0.6B (CPU inference).
"""

import argparse
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
from scale_recall_benchmark import generate_distractors

MODEL_NAME = "Qwen/Qwen3-0.6B"


def store_memories(engine, hook, model, tokenizer, query_layer, count):
    """Store `count` memories into the engine."""
    signal_count = min(count, len(MEMORIES))
    distractor_count = max(0, count - len(MEMORIES))
    distractors = generate_distractors(distractor_count) if distractor_count > 0 else []
    all_memories = list(MEMORIES[:signal_count]) + distractors

    device = next(model.parameters()).device
    for memory in all_memories:
        inputs = tokenizer(memory, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, use_cache=True, output_hidden_states=True)
        d = hook.on_generate(
            layer=query_layer,
            past_key_values=out.past_key_values,
            model_hidden_states=out.hidden_states[query_layer],
        )
        if d.should_write:
            engine.mem_write(1, query_layer, d.key, d.value, d.salience, None)


def query_workload(engine, hook, model, tokenizer, query_layer):
    """Run the standard 30-query workload, return (latencies, recall_at_5)."""
    device = next(model.parameters()).device
    latencies = []
    hits = 0
    total = 0

    for query_text, expected, qtype in ALL_QUERIES:
        if qtype == "negative":
            continue
        if any(e >= len(MEMORIES) for e in expected):
            continue

        total += 1
        inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, use_cache=True, output_hidden_states=True)

        t0 = time.time()
        handles = hook.on_prefill(
            layer=query_layer,
            past_key_values=out.past_key_values,
            model_hidden_states=out.hidden_states[query_layer],
        )
        latencies.append((time.time() - t0) * 1000)

        top5 = [h.cell_id for h in handles[:5]]
        if any(m in expected for m in top5):
            hits += 1

    recall = 100 * hits / total if total else 0
    return latencies, recall


def run_config(label, model, tokenizer, hook, query_layer, count, vamana_threshold):
    """Run one configuration: store + query."""
    db_dir = Path(tempfile.mkdtemp(prefix=f"tdb_lat_{label}_{count}_"))
    engine = tardigrade_db.Engine(str(db_dir), vamana_threshold=vamana_threshold)
    hook.engine = engine

    t0 = time.time()
    store_memories(engine, hook, model, tokenizer, query_layer, count)
    store_time = time.time() - t0

    status = engine.status()
    vamana_active = status.get("vamana_active", False)

    latencies, recall = query_workload(engine, hook, model, tokenizer, query_layer)

    shutil.rmtree(db_dir)

    return {
        "label": label,
        "count": count,
        "cells": engine.cell_count(),
        "vamana_active": vamana_active,
        "recall_at_5": recall,
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0,
        "p50_latency_ms": float(np.median(latencies)) if latencies else 0,
        "p99_latency_ms": float(np.percentile(latencies, 99)) if latencies else 0,
        "store_time_s": store_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark: pipeline configs")
    parser.add_argument(
        "--counts", default="500,1000,2000",
        help="Comma-separated memory counts (default: 500,1000,2000)",
    )
    args = parser.parse_args()
    counts = [int(c.strip()) for c in args.counts.split(",")]

    print("=" * 70)
    print("LATENCY BENCHMARK: Default vs Vamana-Activated Pipeline")
    print(f"Scale points: {counts}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Loading {MODEL_NAME} on {device}...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32, attn_implementation="eager",
    )
    model.to(device).eval()
    n_layers = model.config.num_hidden_layers
    query_layer = int(n_layers * 0.67)
    print(f"OK ({n_layers}L, query_layer={query_layer})")

    dummy_engine = tardigrade_db.Engine(tempfile.mkdtemp())
    hook = HuggingFaceKVHook(
        dummy_engine, owner=1, model_config=model.config,
        model=model, use_hidden_states=True,
    )

    all_results = []

    for count in counts:
        print(f"\n{'─' * 70}")
        print(f"  Scale point: {count} memories")
        print(f"{'─' * 70}")

        # Config A: Default (brute-force, Vamana threshold=10K)
        r_default = run_config(
            "default", model, tokenizer, hook, query_layer,
            count, vamana_threshold=10_000,
        )
        print(f"  Default:  R@5={r_default['recall_at_5']:.0f}%  "
              f"avg={r_default['avg_latency_ms']:.0f}ms  "
              f"p99={r_default['p99_latency_ms']:.0f}ms  "
              f"vamana={r_default['vamana_active']}")

        # Config B: Vamana activated (threshold = count // 2)
        vamana_thresh = max(count // 2, 50)
        r_vamana = run_config(
            "vamana", model, tokenizer, hook, query_layer,
            count, vamana_threshold=vamana_thresh,
        )
        print(f"  Vamana:   R@5={r_vamana['recall_at_5']:.0f}%  "
              f"avg={r_vamana['avg_latency_ms']:.0f}ms  "
              f"p99={r_vamana['p99_latency_ms']:.0f}ms  "
              f"vamana={r_vamana['vamana_active']}")

        speedup = r_default["avg_latency_ms"] / r_vamana["avg_latency_ms"] if r_vamana["avg_latency_ms"] > 0 else 0
        print(f"  Speedup:  {speedup:.2f}x")

        all_results.append((r_default, r_vamana))

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Count':>8} {'Default avg':>12} {'Vamana avg':>12} {'Speedup':>10} {'R@5 Δ':>8}")
    print(f"  {'─' * 54}")
    for r_def, r_vam in all_results:
        speedup = r_def["avg_latency_ms"] / r_vam["avg_latency_ms"] if r_vam["avg_latency_ms"] > 0 else 0
        recall_delta = r_vam["recall_at_5"] - r_def["recall_at_5"]
        print(
            f"  {r_def['count']:>8} "
            f"{r_def['avg_latency_ms']:>10.0f}ms "
            f"{r_vam['avg_latency_ms']:>10.0f}ms "
            f"{speedup:>9.2f}x "
            f"{recall_delta:>+7.1f}%"
        )
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
