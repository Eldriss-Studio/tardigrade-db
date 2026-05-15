"""Inspect what the cross-encoder reranker sees and chooses for
specific LoCoMo queries — to diagnose why LoCoMo scores 38% even
with the rerank fix in place.

For 5 sample queries, dumps:
  - The question
  - Expected ground_truth
  - Top-15 latent candidates with chunk text + reranker score
  - The chosen top-1 and whether it matches the right item
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import torch

from tdb_bench.adapters.tardigrade import TardigradeAdapter, _load_model_cached
from tdb_bench.models import BenchmarkItem

DATASET = "benchmarks/datasets/phase1_oracle/locomo_phase1.jsonl"
N_ITEMS = int(os.getenv("PROBE_N", "100"))
N_SAMPLES = int(os.getenv("PROBE_SAMPLES", "5"))


def main():
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

    adapter = TardigradeAdapter()
    adapter.ingest(items)
    print(f"ingested {len(adapter._cell_to_chunk_text)} cells")

    model, tokenizer, query_layer = _load_model_cached()
    device = next(model.parameters()).device

    cell_to_item_id: dict[int, str] = {
        cid: it.item_id for cid, it in adapter._cell_to_item.items()
    }
    item_id_to_cells: dict[str, list[int]] = {}
    for cid, it in adapter._cell_to_item.items():
        item_id_to_cells.setdefault(it.item_id, []).append(cid)

    # Probe a spread of items: first, middle, late
    sample_indices = [0, N_ITEMS // 4, N_ITEMS // 2, 3 * N_ITEMS // 4, N_ITEMS - 1][:N_SAMPLES]

    for idx in sample_indices:
        item = items[idx]
        right_cells = set(item_id_to_cells.get(item.item_id, []))
        print(f"\n{'='*80}")
        print(f"ITEM {idx}: {item.item_id}")
        print(f"Q: {item.question}")
        print(f"GT: {item.ground_truth!r}")
        print(f"Right cell IDs: {sorted(right_cells)}")
        # Show right-item chunk text
        for cid in sorted(right_cells):
            text = adapter._cell_to_chunk_text.get(cid, "?")
            print(f"  cell {cid} text: {text[:150]!r}")

        # Run latent retrieval at top-15
        inputs = tokenizer(item.question, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, use_cache=True, output_hidden_states=True)
        old_k = adapter._hook.k
        adapter._hook.k = 15
        try:
            handles = adapter._hook.on_prefill(
                layer=query_layer,
                past_key_values=out.past_key_values,
                model_hidden_states=out.hidden_states[query_layer],
            )
        finally:
            adapter._hook.k = old_k

        print(f"\nTOP-15 LATENT (pre-rerank):")
        for r, h in enumerate(handles):
            cid = int(h.cell_id)
            item_id = cell_to_item_id.get(cid, "?")
            text = adapter._cell_to_chunk_text.get(cid, "?")
            is_right = "★" if cid in right_cells else " "
            print(f"  {is_right} rank {r:2d}: cell={cid:5d} item={item_id} "
                  f"score={h.score:.3f}")
            print(f"      text: {text[:120]!r}")

        # Apply reranker
        if adapter._reranker is not None:
            reranked = adapter._reranker.rerank(
                query_text=item.question,
                candidates=handles,
                get_text=lambda h: adapter._cell_to_chunk_text.get(int(h.cell_id)),
            )
            print(f"\nTOP-15 AFTER RERANK:")
            for r, h in enumerate(reranked):
                cid = int(h.cell_id)
                item_id = cell_to_item_id.get(cid, "?")
                text = adapter._cell_to_chunk_text.get(cid, "?")
                is_right = "★" if cid in right_cells else " "
                # The reranker should set h.score to the cross-encoder score
                print(f"  {is_right} rank {r:2d}: cell={cid:5d} item={item_id} "
                      f"score={h.score:.3f}")
                print(f"      text: {text[:120]!r}")

            chosen = reranked[0]
            chosen_item = cell_to_item_id.get(int(chosen.cell_id), "?")
            right_or_wrong = "RIGHT ITEM" if int(chosen.cell_id) in right_cells else "WRONG ITEM"
            print(f"\n>>> Top-1 chose: cell {int(chosen.cell_id)} "
                  f"(item {chosen_item}) — {right_or_wrong}")
            if int(chosen.cell_id) not in right_cells:
                chosen_item_obj = adapter._cell_to_item.get(int(chosen.cell_id))
                if chosen_item_obj:
                    print(f">>> Wrong answer would be: {chosen_item_obj.ground_truth!r}")


if __name__ == "__main__":
    main()
