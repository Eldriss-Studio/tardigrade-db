#!/usr/bin/env python3
# 100-memory scale test -- Q*K retrieval at realistic density.
#
# Tests whether Q*K per-token retrieval holds at 100 memories across
# 10 life domains. Measures cross-domain recall, within-domain recall,
# gravity well severity, and latency.

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


def main():
    print("=" * 70)
    print("100-Memory Scale Test -- Q*K Per-Token Retrieval")
    print(f"Model: {MODEL_NAME} | Memories: {len(MEMORIES)} | Queries: {len(ALL_QUERIES)}")
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

    db_dir = Path(tempfile.mkdtemp(prefix="tardigrade_scale100_"))
    engine = tardigrade_db.Engine(str(db_dir))
    hook = HuggingFaceKVHook(engine, owner=1, model_config=model.config, model=model)

    # Store 100 memories
    print(f"\n--- STORING {len(MEMORIES)} MEMORIES ---")
    t0 = time.time()
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
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/100 stored...")
    store_time = time.time() - t0
    print(f"  Done: {engine.cell_count()} cells in {store_time:.1f}s")

    # Query
    print(f"\n--- QUERYING ({len(ALL_QUERIES)} queries) ---\n")

    cross_results = []
    within_results = []
    neg_scores = []
    all_top1 = []
    latencies = []

    for query_text, expected, qtype in ALL_QUERIES:
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
        top5 = retrieved[:5]
        top_mem = retrieved[0] if retrieved else -1

        if qtype == "negative":
            neg_scores.append(handles[0].score if handles else 0)
            continue

        all_top1.append(top_mem)
        hit = any(m in expected for m in top5)
        rank = "---"
        if hit:
            for j, m in enumerate(top5):
                if m in expected:
                    rank = f"#{j+1}"
                    break

        entry = {"q": query_text, "exp": expected, "hit": hit, "rank": rank, "top": top_mem}
        if qtype == "cross":
            cross_results.append(entry)
        else:
            within_results.append(entry)

        mark = f"Y{rank}" if hit else "N"
        domain = DOMAINS[expected[0] // 10] if expected else "?"
        print(f"  {mark:>5} [{qtype:>6}] [{domain:>10}] {query_text[:42]}")

    # Results
    c_hits = sum(1 for r in cross_results if r["hit"])
    w_hits = sum(1 for r in within_results if r["hit"])
    c_tot = len(cross_results)
    w_tot = len(within_results)
    total = c_hits + w_hits
    total_q = c_tot + w_tot

    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")
    print(f"\n  Cross-domain recall:   {c_hits}/{c_tot} ({100*c_hits/c_tot:.1f}%)")
    print(f"  Within-domain recall:  {w_hits}/{w_tot} ({100*w_hits/w_tot:.1f}%)")
    print(f"  Overall recall@5:      {total}/{total_q} ({100*total/total_q:.1f}%)")

    # Gravity well
    top1_counts = Counter(all_top1)
    worst = top1_counts.most_common(1)[0] if top1_counts else (-1, 0)
    unique = len(top1_counts)
    print(f"\n  Unique top-1 memories: {unique}/{total_q}")
    print(f"  Worst gravity well:   mem {worst[0]} ({worst[1]}x top-1)")
    print(f"  Gravity check:        {'PASS' if worst[1] <= 3 else 'FAIL'}")

    # Latency
    print(f"\n  Avg latency:           {np.mean(latencies):.1f}ms")
    print(f"  P99 latency:           {np.percentile(latencies, 99):.1f}ms")
    print(f"  Store time:            {store_time:.1f}s")

    # Per-domain
    print(f"\n  -- Per-Domain --")
    print(f"  {'Domain':<12} {'Cross':>6} {'Within':>8}")
    print(f"  {'-' * 28}")
    for d_idx in range(10):
        cd = [r for r in cross_results if r["exp"] and r["exp"][0] // 10 == d_idx]
        wd = [r for r in within_results if r["exp"] and r["exp"][0] // 10 == d_idx]
        ch = sum(1 for r in cd if r["hit"])
        wh = sum(1 for r in wd if r["hit"])
        print(f"  {DOMAINS[d_idx]:<12} {ch}/{len(cd) or 1:>3}  {wh}/{len(wd) or 1:>5}")

    # Misses
    misses = [r for r in cross_results + within_results if not r["hit"]]
    if misses:
        print(f"\n  Misses ({len(misses)}):")
        for r in misses[:20]:
            dom = DOMAINS[r["exp"][0] // 10] if r["exp"] else "?"
            print(f"    X [{dom:>10}] \"{r['q'][:48]}\" -> mem {r['top']}")
        if len(misses) > 20:
            print(f"    ... and {len(misses) - 20} more")

    shutil.rmtree(db_dir)

    pct = 100 * total / total_q
    verdict = "PASS" if pct >= 70 else "PARTIAL" if pct >= 50 else "NEEDS WORK"
    print(f"\n{'=' * 70}")
    print(f"  {verdict}: {total}/{total_q} ({pct:.1f}%)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
