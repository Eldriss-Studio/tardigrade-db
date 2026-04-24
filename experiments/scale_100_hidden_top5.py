#!/usr/bin/env python3
# 100-memory experiment: hidden states + Top5Avg through engine pipeline.
#
# Validates whether the diagnostic finding (100% recall, 10% negFP)
# survives Q4 quantization and the full retrieval path.

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
    print("100-Memory: Hidden States + Top5Avg Through Engine Pipeline")
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

    db_dir = Path(tempfile.mkdtemp(prefix="tardigrade_hidden_top5_"))
    engine = tardigrade_db.Engine(str(db_dir))
    hook = HuggingFaceKVHook(
        engine, owner=1, model_config=model.config,
        model=model, use_hidden_states=True,
    )

    # Store
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
            print(f"  {i + 1}/100...")
    store_time = time.time() - t0
    print(f"  Done: {engine.cell_count()} cells in {store_time:.1f}s")

    # Query
    print(f"\n--- QUERYING ---\n")
    cross_r, within_r, neg_scores, all_top1, lats = [], [], [], [], []

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
        lats.append((time.time() - t1) * 1000)
        retrieved = [h.cell_id for h in handles]
        top5, top_mem = retrieved[:5], (retrieved[0] if retrieved else -1)

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
        (cross_r if qtype == "cross" else within_r).append(entry)
        mark = f"Y{rank}" if hit else "N"
        dom = DOMAINS[expected[0] // 10] if expected else "?"
        print(f"  {mark:>5} [{qtype:>6}] [{dom:>10}] {query_text[:42]}")

    # Results
    ch, wh = sum(1 for r in cross_r if r["hit"]), sum(1 for r in within_r if r["hit"])
    ct, wt = len(cross_r), len(within_r)
    total, total_q = ch + wh, ct + wt
    pct = 100 * total / total_q

    print(f"\n{'=' * 70}")
    print(f"  Cross-domain:  {ch}/{ct} ({100*ch/ct:.1f}%)")
    print(f"  Within-domain: {wh}/{wt} ({100*wh/wt:.1f}%)")
    print(f"  Overall:       {total}/{total_q} ({pct:.1f}%)")

    top1c = Counter(all_top1)
    worst = top1c.most_common(1)[0] if top1c else (-1, 0)
    print(f"  Unique top-1:  {len(top1c)}/{total_q}")
    print(f"  Gravity well:  mem {worst[0]} ({worst[1]}x) {'PASS' if worst[1] <= 3 else 'FAIL'}")
    print(f"  Avg latency:   {np.mean(lats):.1f}ms")

    print(f"\n  -- Comparison --")
    print(f"  Q*K pipeline (Phase 22):        40.0%")
    print(f"  Diagnostic (outside engine):   100.0%")
    print(f"  RAG baseline:                  100.0%")
    print(f"  THIS RUN (engine pipeline):     {pct:.1f}%")

    misses = [r for r in cross_r + within_r if not r["hit"]]
    if misses:
        print(f"\n  Misses ({len(misses)}):")
        for r in misses[:15]:
            dom = DOMAINS[r["exp"][0] // 10] if r["exp"] else "?"
            print(f"    X [{dom:>10}] \"{r['q'][:48]}\" -> mem {r['top']}")

    shutil.rmtree(db_dir)
    v = "PASS" if pct >= 70 else "PARTIAL" if pct >= 50 else "NEEDS WORK"
    print(f"\n{'=' * 70}")
    print(f"  {v}: {total}/{total_q} ({pct:.1f}%)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
