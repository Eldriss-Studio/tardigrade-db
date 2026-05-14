"""Inspect the hub cells dominating LongMemEval retrieval.

Loads the diagnostic report, identifies the top-10 most-frequently-
retrieved cells, and prints their chunk text + cell metadata. Builds
the same engine state the diagnostic used so cell IDs line up.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import collections

from tdb_bench.adapters.tardigrade import TardigradeAdapter
from tdb_bench.models import BenchmarkItem

REPORT = "target/rank-diagnostic-full-k25.json"
DATASET = "benchmarks/datasets/phase1_oracle/longmemeval_phase1.jsonl"
N_HUBS = 15
N_ITEMS = 500


def main():
    d = json.load(open(REPORT))
    records = d["datasets"]["longmemeval"]["records"]

    # Identify the hub cells
    top10_dist = collections.Counter()
    for r in records:
        top10_dist.update(r["retrieved_top10"])
    hub_cells = [cell for cell, _ in top10_dist.most_common(N_HUBS)]
    hub_counts = {cell: top10_dist[cell] for cell in hub_cells}

    print(f"Hub cells (top-{N_HUBS} most-retrieved):")
    for cell in hub_cells:
        print(f"  cell {cell}: in top-10 of {hub_counts[cell]}/{len(records)} "
              f"queries ({hub_counts[cell]/len(records):.1%})")

    # Re-ingest to get chunk text per cell
    print("\nRe-ingesting to identify chunk text...")
    items = []
    with open(DATASET) as f:
        for line in f:
            items.append(json.loads(line))
            if len(items) >= N_ITEMS:
                break
    bench_items = [
        BenchmarkItem(
            item_id=d["id"], dataset=d["dataset"], context=d["context"],
            question=d["question"], ground_truth=d["ground_truth"],
        )
        for d in items
    ]
    adapter = TardigradeAdapter()
    adapter.enable_chunk_text_tracking()
    adapter.ingest(bench_items)
    print(f"Ingested {len(adapter._cell_to_chunk_text)} cells")

    print(f"\n=== HUB CHUNK TEXTS ===")
    for cell in hub_cells:
        text = adapter._cell_to_chunk_text.get(cell, "<not in map>")
        item = adapter._cell_to_item.get(cell)
        item_id = item.item_id if item else "?"
        print(f"\n--- cell {cell} (in {hub_counts[cell]/len(records):.0%} of top-10s, from {item_id}) ---")
        print(f"len={len(text)} chars")
        print(text[:400])
        if len(text) > 400:
            print(f"... [+{len(text)-400} more chars]")


if __name__ == "__main__":
    main()
