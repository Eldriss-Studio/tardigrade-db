"""Phase 0+ deeper forensic on retrieval failures.

Goes beyond rank histograms to answer five questions:

Q1. When the right chunk IS in top-25, what beats it to top-1?
    (Always the same hubs, or other chunks from the same item?)

Q2. When the right chunk is OUTSIDE top-25, where does it actually
    rank? Top-100? Top-1000? Or essentially random?

Q3. Is the "expected chunk" heuristic right? Check if the
    ground-truth answer text actually appears in the chunk that
    the heuristic picks as expected.

Q4. Distribution of expected-chunk ranks across the whole corpus —
    not just R@K cumulative buckets, but the actual rank values.

Q5. Are residual hubs concentrated in specific items, or spread
    across the corpus?

Runs on LongMemEval (smaller, faster, where the problem is most
visible). Uses top-K=200 to see deep ranks. Outputs structured
JSON + key insights to stdout.
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
N_ITEMS = int(os.getenv("FORENSIC_N_ITEMS", "100"))
DEEP_TOP_K = int(os.getenv("FORENSIC_TOP_K", "200"))
OUTPUT = os.getenv("FORENSIC_REPORT", "target/retrieval-forensic.json")


def _tok(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def find_expected_chunk_with_overlap(
    item: BenchmarkItem,
    cell_to_chunk_text: dict[int, str],
    cells_for_item: list[int],
) -> tuple[int | None, int]:
    """Same heuristic as probe_rank_diagnostic but also returns
    the actual overlap count so we can audit the heuristic."""
    if not cells_for_item:
        return None, 0
    gt_tokens = _tok(item.ground_truth)
    if not gt_tokens:
        return cells_for_item[0], 0
    best_cell = cells_for_item[0]
    best_overlap = -1
    for cell_id in cells_for_item:
        chunk = cell_to_chunk_text.get(cell_id, "")
        chunk_tokens = _tok(chunk)
        overlap = len(gt_tokens & chunk_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_cell = cell_id
    return best_cell, best_overlap


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


def main():
    print(f"loading {N_ITEMS} LongMemEval items + ingesting...", flush=True)
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

    print(f"\nrunning {len(items)} queries at top-K={DEEP_TOP_K}...", flush=True)

    records: list[dict] = []
    rank_distribution = []  # exact rank values (or None if past top-K)
    hub_co_occurrence = collections.Counter()  # cell_id -> times in top-25
    expected_overlap_dist = []  # heuristic-overlap values

    for i, item in enumerate(items):
        cells = item_id_to_cells.get(item.item_id, [])
        expected_cell, expected_overlap = find_expected_chunk_with_overlap(
            item, adapter._cell_to_chunk_text, cells,
        )
        expected_overlap_dist.append(expected_overlap)

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
        rank = ids.index(expected_cell) if expected_cell in ids else None
        rank_distribution.append(rank)

        # Co-occurrence: track which cells appear in top-25 alongside
        # the expected chunk (when present).
        for cell_id in ids[:25]:
            hub_co_occurrence[cell_id] += 1

        # Of the cells beating the expected chunk to top-1, which items
        # do they come from? Same-item or different?
        top_competitors = ids[:5]
        same_item_competitors = sum(
            1 for c in top_competitors
            if c != expected_cell and adapter._cell_to_item.get(c) == item
        )

        records.append({
            "item_id": item.item_id,
            "n_cells_for_item": len(cells),
            "expected_cell": expected_cell,
            "expected_cell_overlap": expected_overlap,
            "expected_rank": rank,
            "top5_ids": ids[:5],
            "top5_same_item": same_item_competitors,
        })

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(items)} queried", flush=True)

    # -------- analysis --------
    print(f"\n=== Q1: same-item vs cross-item competitors in top-5 ===")
    same_item_total = sum(r["top5_same_item"] for r in records)
    print(f"  same-item cells in top-5 (sum across queries): {same_item_total}")
    print(f"  average per query: {same_item_total / len(records):.2f}/5")
    # If average ~1+, intra-item competition is real. If ~0, hubs dominate.

    print(f"\n=== Q2: exact rank distribution ===")
    ranks_present = [r for r in rank_distribution if r is not None]
    print(f"  in top-{DEEP_TOP_K}: {len(ranks_present)}/{len(rank_distribution)}")
    if ranks_present:
        print(f"  median rank when found: {sorted(ranks_present)[len(ranks_present)//2]}")
        print(f"  min/max rank: {min(ranks_present)}/{max(ranks_present)}")
    print(f"  outside top-{DEEP_TOP_K}: "
          f"{sum(1 for r in rank_distribution if r is None)}/{len(rank_distribution)}")

    print(f"\n=== Q3: expected-chunk overlap audit ===")
    n_zero_overlap = sum(1 for v in expected_overlap_dist if v == 0)
    n_one_overlap = sum(1 for v in expected_overlap_dist if v == 1)
    print(f"  items where best chunk has 0 ground-truth overlap: "
          f"{n_zero_overlap}/{len(expected_overlap_dist)}")
    print(f"  items where best chunk has ONLY 1 ground-truth token: "
          f"{n_one_overlap}/{len(expected_overlap_dist)}")
    print(f"  ⚠ Heuristic is weak when these counts are high — the "
          f"'expected chunk' may not actually contain the answer.")

    print(f"\n=== Q4: rank histogram (deep) ===")
    buckets = [1, 3, 5, 10, 25, 50, 100, 200]
    for b in buckets:
        count = sum(1 for r in ranks_present if r < b)
        print(f"  R@{b}: {count}/{len(rank_distribution)} = "
              f"{count/len(rank_distribution):.1%}")

    print(f"\n=== Q5: hub-cell concentration in top-25 (post-fix) ===")
    n_queries = len(records)
    n_hub_threshold = max(1, int(n_queries * 0.5))
    super_hubs = [(c, n) for c, n in hub_co_occurrence.most_common(20)
                  if n >= n_hub_threshold]
    print(f"  cells appearing in top-25 of ≥50% of queries: {len(super_hubs)}")
    for c, n in super_hubs[:10]:
        print(f"    cell {c}: {n}/{n_queries} ({n/n_queries:.1%})")

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump({
            "config": {
                "n_items": N_ITEMS,
                "top_k_probe": DEEP_TOP_K,
                "refinement_mode": os.getenv("TDB_REFINEMENT_MODE", "centered"),
            },
            "records": records,
            "rank_distribution": rank_distribution,
            "hub_co_occurrence": dict(hub_co_occurrence.most_common(50)),
        }, f, indent=2)
    print(f"\n=== report → {OUTPUT} ===")


if __name__ == "__main__":
    main()
