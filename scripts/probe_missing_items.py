"""Phase 0+ #3: characterize the LongMemEval items where the right
item is NOT findable in deep top-K.

Uses item-level rank (matches bench scoring) at top-K=500 to see
the full distribution. For items still outside top-500:
  - What question shape do they have?
  - What ground-truth shape (date? proper noun? number? phrase?)?
  - What context size and chunk count?
  - Compared to easy items, what's different?

Output: structured JSON + summary statistics + 10 sampled "lost"
items printed for manual inspection.
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import torch

from tdb_bench.adapters.tardigrade import TardigradeAdapter, _load_model_cached
from tdb_bench.models import BenchmarkItem

DATASET = "benchmarks/datasets/phase1_oracle/longmemeval_phase1.jsonl"
N_ITEMS = int(os.getenv("MISSING_N_ITEMS", "200"))
DEEP_TOP_K = int(os.getenv("MISSING_TOP_K", "500"))
OUTPUT = os.getenv("MISSING_REPORT", "target/missing-items-forensic.json")
REFINEMENT = os.getenv("TDB_REFINEMENT_MODE", "whitened")

# Buckets for rank histogram
BUCKETS = [1, 5, 10, 25, 50, 100, 200, 500]


def load_items() -> list[BenchmarkItem]:
    items = []
    with open(DATASET) as f:
        for line in f:
            d = json.loads(line)
            items.append(BenchmarkItem(
                item_id=d["id"], dataset=d["dataset"], context=d["context"],
                question=d["question"], ground_truth=d["ground_truth"],
            ))
            if len(items) >= N_ITEMS:
                break
    return items


def characterize_ground_truth(gt: str) -> dict:
    """Heuristic classifiers for ground-truth shape."""
    has_digit = bool(re.search(r"\d", gt))
    has_year = bool(re.search(r"\b(19|20)\d{2}\b", gt))
    has_month = bool(re.search(
        r"\b(january|february|march|april|may|june|july|august|september|"
        r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|"
        r"nov|dec)\b",
        gt.lower(),
    ))
    has_proper_noun = bool(re.search(r"\b[A-Z][a-z]+\b", gt))
    tokens = re.findall(r"[a-z0-9]+", gt.lower())
    return {
        "len_chars": len(gt),
        "n_tokens": len(tokens),
        "has_digit": has_digit,
        "has_year": has_year,
        "has_month": has_month,
        "has_proper_noun": has_proper_noun,
    }


def characterize_question(q: str) -> dict:
    """Heuristic classifiers for question shape."""
    ql = q.lower()
    return {
        "len_chars": len(q),
        "n_tokens": len(re.findall(r"[a-z0-9]+", ql)),
        "starts_when": ql.startswith("when"),
        "starts_what": ql.startswith("what"),
        "starts_who": ql.startswith("who"),
        "starts_where": ql.startswith("where"),
        "starts_why": ql.startswith("why"),
        "starts_how": ql.startswith("how"),
        "has_temporal_cue": any(t in ql for t in [
            "yesterday", "last week", "last month", "last year",
            "before", "after", "when", "during",
        ]),
    }


def main():
    print(f"loading {N_ITEMS} LongMemEval items + ingesting "
          f"(refinement={REFINEMENT})...", flush=True)
    items = load_items()
    adapter = TardigradeAdapter()
    adapter.enable_chunk_text_tracking()
    t0 = time.perf_counter()
    adapter.ingest(items)
    print(f"ingested {len(adapter._cell_to_chunk_text)} cells in "
          f"{time.perf_counter() - t0:.0f}s", flush=True)

    model, tokenizer, query_layer = _load_model_cached()
    device = next(model.parameters()).device

    item_id_to_cells: dict[str, list[int]] = {}
    for cell_id, it in adapter._cell_to_item.items():
        item_id_to_cells.setdefault(it.item_id, []).append(cell_id)

    print(f"\nquerying {len(items)} items at top-K={DEEP_TOP_K}...", flush=True)

    records: list[dict] = []
    item_rank_distribution: list[int | None] = []

    for i, item in enumerate(items):
        cells_for_item = set(item_id_to_cells.get(item.item_id, []))

        inputs = tokenizer(
            item.question, return_tensors="pt", truncation=True, max_length=256,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, use_cache=True, output_hidden_states=True)

        old_k = adapter._hook.k
        adapter._hook.k = DEEP_TOP_K
        try:
            handles = adapter._hook.on_prefill(
                layer=query_layer,
                past_key_values=out.past_key_values,
                model_hidden_states=out.hidden_states[query_layer],
            )
        finally:
            adapter._hook.k = old_k

        ids = [int(h.cell_id) for h in handles]
        item_rank: int | None = None
        for rank, cell_id in enumerate(ids):
            if cell_id in cells_for_item:
                item_rank = rank
                break

        records.append({
            "item_id": item.item_id,
            "n_cells_for_item": len(cells_for_item),
            "context_len_chars": len(item.context),
            "item_rank": item_rank,
            "gt_shape": characterize_ground_truth(item.ground_truth),
            "q_shape": characterize_question(item.question),
            "ground_truth_preview": item.ground_truth[:120],
            "question": item.question,
        })
        item_rank_distribution.append(item_rank)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(items)} queried", flush=True)

    # --- analysis ---
    print(f"\n=== Item-level rank histogram (top-{DEEP_TOP_K}) ===")
    ranks = item_rank_distribution
    missing = sum(1 for r in ranks if r is None)
    found = [r for r in ranks if r is not None]
    print(f"  in top-{DEEP_TOP_K}: {len(found)}/{len(ranks)}")
    print(f"  outside top-{DEEP_TOP_K}: {missing}/{len(ranks)} = "
          f"{missing/len(ranks):.1%}")
    if found:
        srt = sorted(found)
        print(f"  median rank: {srt[len(srt)//2]}, "
              f"p25: {srt[len(srt)//4]}, p75: {srt[3*len(srt)//4]}, "
              f"min: {min(found)}, max: {max(found)}")

    print(f"\n=== Cumulative R@K (item-level) ===")
    for b in BUCKETS:
        count = sum(1 for r in found if r < b)
        print(f"  R@{b}: {count}/{len(ranks)} = {count/len(ranks):.1%}")

    # Compare missing items vs found-near-top items
    found_easy = [r for r in records
                  if r["item_rank"] is not None and r["item_rank"] < 10]
    not_found = [r for r in records if r["item_rank"] is None]
    rank_50_plus = [r for r in records
                    if r["item_rank"] is not None and r["item_rank"] >= 50]

    print(f"\n=== Easy items (rank < 10) vs hard items "
          f"(outside top-{DEEP_TOP_K}) ===")

    def avg_or(values, default=0):
        return sum(values) / len(values) if values else default

    print(f"  Easy items (n={len(found_easy)}):")
    print(f"    avg context chars: {avg_or([r['context_len_chars'] for r in found_easy]):.0f}")
    print(f"    avg gt tokens: "
          f"{avg_or([r['gt_shape']['n_tokens'] for r in found_easy]):.1f}")
    print(f"    %gt with year: "
          f"{sum(1 for r in found_easy if r['gt_shape']['has_year']) / max(len(found_easy), 1):.0%}")
    print(f"    %gt with month: "
          f"{sum(1 for r in found_easy if r['gt_shape']['has_month']) / max(len(found_easy), 1):.0%}")
    print(f"    %gt with proper noun: "
          f"{sum(1 for r in found_easy if r['gt_shape']['has_proper_noun']) / max(len(found_easy), 1):.0%}")
    print(f"    %q temporal cue: "
          f"{sum(1 for r in found_easy if r['q_shape']['has_temporal_cue']) / max(len(found_easy), 1):.0%}")

    print(f"\n  Hard items (n={len(not_found)}):")
    print(f"    avg context chars: {avg_or([r['context_len_chars'] for r in not_found]):.0f}")
    print(f"    avg gt tokens: "
          f"{avg_or([r['gt_shape']['n_tokens'] for r in not_found]):.1f}")
    print(f"    %gt with year: "
          f"{sum(1 for r in not_found if r['gt_shape']['has_year']) / max(len(not_found), 1):.0%}")
    print(f"    %gt with month: "
          f"{sum(1 for r in not_found if r['gt_shape']['has_month']) / max(len(not_found), 1):.0%}")
    print(f"    %gt with proper noun: "
          f"{sum(1 for r in not_found if r['gt_shape']['has_proper_noun']) / max(len(not_found), 1):.0%}")
    print(f"    %q temporal cue: "
          f"{sum(1 for r in not_found if r['q_shape']['has_temporal_cue']) / max(len(not_found), 1):.0%}")

    print(f"\n=== Sample 5 'lost' items (outside top-{DEEP_TOP_K}) ===")
    for r in not_found[:5]:
        print(f"\n  {r['item_id']}")
        print(f"    Q: {r['question'][:120]}")
        print(f"    GT: {r['ground_truth_preview']}")
        print(f"    n_cells_for_item: {r['n_cells_for_item']}, "
              f"context_chars: {r['context_len_chars']}")

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump({
            "config": {
                "n_items": N_ITEMS,
                "top_k_probe": DEEP_TOP_K,
                "refinement_mode": REFINEMENT,
                "buckets": BUCKETS,
            },
            "records": records,
        }, f, indent=2)
    print(f"\n=== report → {OUTPUT} ===")


if __name__ == "__main__":
    main()
