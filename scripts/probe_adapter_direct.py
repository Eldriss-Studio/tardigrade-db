"""Direct bench-adapter probe: calls TardigradeAdapter.ingest + .query on 50
LoCoMo items, measures R@1 strict (answer matches expected ground_truth).

Bypasses the bench runner and evaluator entirely. Tells us whether the
recall leak (probe 20% vs bench 4%) is in the adapter itself or somewhere
downstream in the runner/eval.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from tdb_bench.adapters.tardigrade import TardigradeAdapter
from tdb_bench.models import BenchmarkItem

N_ITEMS = int(os.getenv("PROBE_N_ITEMS", "50"))
TOP_K = 5
DATASET = str(Path(__file__).resolve().parents[1]
              / "benchmarks/datasets/phase1_oracle/locomo_phase1.jsonl")


def load_items(path, n):
    items = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            items.append(BenchmarkItem(
                item_id=d["id"],
                dataset=d["dataset"],
                context=d["context"],
                question=d["question"],
                ground_truth=d["ground_truth"],
            ))
            if len(items) >= n:
                break
    return items


def main():
    items = load_items(DATASET, N_ITEMS)
    print(f"loaded {len(items)} items")

    adapter = TardigradeAdapter()
    print(f"mode: {adapter._mode}, refinement: {adapter._refinement}, "
          f"reranker: {adapter._reranker is not None}")

    print("=== ingest ===")
    adapter.ingest(items)
    print(f"cell_to_item size: {len(adapter._cell_to_item)}")

    print("=== query ===")
    r1_strict_hits = 0
    r1_overlap_hits = 0  # deterministic-style: >0% token overlap
    for i, item in enumerate(items):
        result = adapter.query(item, top_k=TOP_K)
        if result.status != "ok":
            continue
        if result.answer == item.ground_truth:
            r1_strict_hits += 1
        if result.answer and item.ground_truth:
            expected_tokens = set(item.ground_truth.lower().split())
            answer_tokens = set(result.answer.lower().split())
            if expected_tokens & answer_tokens:
                r1_overlap_hits += 1
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(items)}: R@1 strict={r1_strict_hits} "
                  f"overlap={r1_overlap_hits}")

    print()
    print(f"=== RESULT ===")
    print(f"R@1 strict (exact ground_truth match): "
          f"{r1_strict_hits}/{len(items)} = {r1_strict_hits / len(items):.1%}")
    print(f"R@1 any-overlap (≥1 shared token): "
          f"{r1_overlap_hits}/{len(items)} = {r1_overlap_hits / len(items):.1%}")


if __name__ == "__main__":
    main()
