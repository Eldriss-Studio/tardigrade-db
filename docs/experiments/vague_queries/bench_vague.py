#!/usr/bin/env python3
"""Vague-query benchmark with refinement-mode comparison.

Wraps `experiments/vague_query_test.py` (which holds the canonical pinned
dataset of 100 memories + 300 queries across specific/moderate/vague tiers)
and runs each refinement mode through the same engine/model/queries.

Usage:
    python docs/experiments/vague_queries/bench_vague.py --modes none,centered,prf
    python docs/experiments/vague_queries/bench_vague.py --modes prf --alpha 0.7 --beta 0.3 --kprime 3
    python docs/experiments/vague_queries/bench_vague.py --sweep prf

Requires CUDA + Qwen3-0.6B (downloaded via HuggingFace).
"""

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "experiments"))

import tardigrade_db
from corpus_100 import ALL_QUERIES, MEMORIES
from tardigrade_hooks.hf_kv_hook import HuggingFaceKVHook
from vague_query_test import MODERATE_QUERIES, OPEN_QUERIES, VAGUE_QUERIES

MODEL_NAME = "Qwen/Qwen3-0.6B"


def configure_refinement(engine, mode, alpha, beta, kprime):
    if mode == "none":
        engine.set_refinement_mode("none")
    elif mode == "centered":
        engine.set_refinement_mode("centered")
    elif mode == "prf":
        engine.set_refinement_mode("prf", alpha=alpha, beta=beta, k_prime=kprime)
    else:
        raise ValueError(f"unknown mode: {mode}")


def run_tier(engine, hook, tokenizer, model, queries, query_layer, k=10):
    """Return (R@1, R@5, R@10, p95_latency_ms) for a tier."""
    import torch  # local import: only needed when running benchmark

    hits1 = hits5 = hits10 = 0
    latencies_ms = []
    total = len(queries)

    for query_text, expected, _tier in queries:
        inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k_: v.to(model.device) for k_, v in inputs.items()}
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(**inputs, use_cache=True, output_hidden_states=True)
        handles = hook.on_prefill(
            layer=query_layer,
            past_key_values=out.past_key_values,
            model_hidden_states=out.hidden_states[query_layer],
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        retrieved = [h.cell_id for h in handles[:k]]
        if any(m in expected for m in retrieved[:1]):
            hits1 += 1
        if any(m in expected for m in retrieved[:5]):
            hits5 += 1
        if any(m in expected for m in retrieved[:10]):
            hits10 += 1

    latencies_ms.sort()
    p95 = latencies_ms[int(0.95 * len(latencies_ms))] if latencies_ms else 0.0
    return (
        100.0 * hits1 / total if total else 0.0,
        100.0 * hits5 / total if total else 0.0,
        100.0 * hits10 / total if total else 0.0,
        p95,
    )


def populate(engine, hook, tokenizer, model, query_layer):
    import torch

    for memory in MEMORIES:
        inputs = tokenizer(memory, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k_: v.to(model.device) for k_, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, use_cache=True, output_hidden_states=True)
        decision = hook.on_generate(
            layer=query_layer,
            past_key_values=out.past_key_values,
            model_hidden_states=out.hidden_states[query_layer],
        )
        if decision.should_write:
            engine.mem_write(1, query_layer, decision.key, decision.value, decision.salience, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="none,centered,prf",
                        help="comma-separated: none,centered,prf")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--kprime", type=int, default=3)
    parser.add_argument("--sweep", choices=["prf"], default=None,
                        help="hyperparameter sweep")
    parser.add_argument("--device", default="cuda",
                        help="cuda or cpu (cpu is very slow)")
    args = parser.parse_args()

    import torch  # noqa: F401  (defer import so --help works without torch installed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32, attn_implementation="eager",
    )
    model = model.to(args.device).eval()
    n_layers = model.config.num_hidden_layers
    query_layer = int(n_layers * 0.67)
    print(f"  {n_layers} layers, query_layer={query_layer}")

    db_dir = tempfile.mkdtemp(prefix="tdb_vague_bench_")
    try:
        engine = tardigrade_db.Engine(db_dir)
        hook = HuggingFaceKVHook(
            engine, owner=1, k=10, model_config=model.config,
            model=model, use_hidden_states=True,
        )
        print(f"\nStoring {len(MEMORIES)} memories...")
        populate(engine, hook, tokenizer, model, query_layer)
        print(f"  {engine.cell_count()} cells")

        specific = [(q, e, "specific") for q, e, t in ALL_QUERIES if t != "negative"]
        tiers = [
            ("specific", specific),
            ("moderate", MODERATE_QUERIES),
            ("vague", VAGUE_QUERIES),
            ("open", OPEN_QUERIES),
        ]

        configurations = []
        if args.sweep == "prf":
            for a in (0.5, 0.7, 0.9):
                for b in (0.1, 0.3, 0.5):
                    for kp in (3, 5, 10):
                        configurations.append(("prf", a, b, kp))
        else:
            for mode in args.modes.split(","):
                mode = mode.strip()
                if mode == "prf":
                    configurations.append((mode, args.alpha, args.beta, args.kprime))
                else:
                    configurations.append((mode, None, None, None))

        print(f"\nRunning {len(configurations)} configuration(s)...")
        rows = []
        for mode, alpha, beta, kprime in configurations:
            configure_refinement(engine, mode, alpha or 0.7, beta or 0.3, kprime or 3)
            label = mode if mode != "prf" else f"prf(a={alpha},b={beta},k'={kprime})"
            print(f"\n=== {label} ===")
            row = [label]
            for tier_name, tier_queries in tiers:
                r1, r5, r10, p95 = run_tier(
                    engine, hook, tokenizer, model, tier_queries, query_layer
                )
                print(f"  {tier_name:<10} R@1={r1:5.1f}%  R@5={r5:5.1f}%  R@10={r10:5.1f}%  p95={p95:6.1f}ms")
                row.append((r5, p95))
            rows.append(row)

        # Compact summary table
        print(f"\n{'=' * 78}")
        print(f"{'mode':<32} {'specific':>10} {'moderate':>10} {'vague':>10} {'open':>10}")
        print("-" * 78)
        for row in rows:
            label = row[0]
            r5s = " ".join(f"{r5:>9.1f}%" for r5, _ in row[1:])
            print(f"{label:<32}{r5s}")
        print("=" * 78)
    finally:
        shutil.rmtree(db_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
