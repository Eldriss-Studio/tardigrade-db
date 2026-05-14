"""Phase 0 diagnostic — per-query rank histograms for retrieval debugging.

For each query in LoCoMo and LongMemEval, retrieves top-K=100 cells
and records the rank at which the "expected" chunk appears.

Definitions:
- For LoCoMo: each item is one chunk (after the audit's evidence-only
  prep). The expected chunk is just the item's own cell. "Strict R@1"
  in audit-speak == this rank being 0.
- For LongMemEval: each item has many chunks. The "expected" chunk is
  the one whose text shares the most tokens with the ground_truth
  answer — a proxy for "the chunk a perfect retriever should find."
  This proxy is imperfect (the answer may be paraphrased across
  chunks) but is the best signal we have without per-item evidence
  annotation.

Output: a JSON report with per-item records and an aggregate rank
histogram per dataset. Reveals whether the right chunk is "near miss"
(top-10), "lost in the noise" (top-100), or "outside top-K entirely"
(infinity).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from tdb_bench.adapters.tardigrade import TardigradeAdapter
from tdb_bench.models import BenchmarkItem

import torch
from tardigrade_hooks.constants import DEFAULT_CAPTURE_LAYER_RATIO  # noqa: F401

DIAGNOSTIC_TOP_K_PROBE = int(os.getenv("TDB_DIAGNOSTIC_TOP_K", "100"))
DEFAULT_OUTPUT = "target/rank-diagnostic.json"
DEFAULT_DATASET_DIR = "benchmarks/datasets/phase1_oracle"
N_ITEMS_LOCOMO = int(os.getenv("TDB_DIAGNOSTIC_N_LOCOMO", "1533"))
N_ITEMS_LONGMEM = int(os.getenv("TDB_DIAGNOSTIC_N_LONGMEM", "500"))
RANK_BUCKETS = [1, 5, 10, 25, 50, 100]


def load_items(path: str, n: int, dataset_name: str) -> list[BenchmarkItem]:
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
    print(f"[{dataset_name}] loaded {len(items)} items", flush=True)
    return items


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def find_expected_chunk(item: BenchmarkItem, cell_to_chunk: dict[int, str],
                        cells_for_item: list[int]) -> int | None:
    """Pick the cell whose chunk text best matches the ground_truth.

    For multi-chunk items (LongMemEval), this is the proxy for "the
    chunk the retriever should ideally find." For single-chunk items
    (LoCoMo evidence-only), the single cell is the answer.
    """
    if not cells_for_item:
        return None
    if len(cells_for_item) == 1:
        return cells_for_item[0]
    gt_tokens = _tokenize(item.ground_truth)
    if not gt_tokens:
        return cells_for_item[0]
    best_cell = cells_for_item[0]
    best_overlap = -1
    for cell_id in cells_for_item:
        chunk = cell_to_chunk.get(cell_id, "")
        chunk_tokens = _tokenize(chunk)
        overlap = len(gt_tokens & chunk_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_cell = cell_id
    return best_cell


def run_dataset(adapter: TardigradeAdapter, items: list[BenchmarkItem],
                dataset_name: str) -> dict:
    """Ingest items, retrieve top-K=100 per query, record per-item rank."""
    print(f"\n=== {dataset_name}: ingest ===", flush=True)
    t0 = time.perf_counter()
    adapter.ingest(items)
    ingest_secs = time.perf_counter() - t0
    print(f"[{dataset_name}] ingest done in {ingest_secs:.1f}s, "
          f"cell_to_item size: {len(adapter._cell_to_item)}, "
          f"cell_to_chunk size: {len(adapter._cell_to_chunk_text)}",
          flush=True)

    # Pre-compute cells per item for chunk-matching.
    item_id_to_cells: dict[str, list[int]] = {}
    for cell_id, item in adapter._cell_to_item.items():
        item_id_to_cells.setdefault(item.item_id, []).append(cell_id)

    print(f"=== {dataset_name}: query (top-{DIAGNOSTIC_TOP_K_PROBE}) ===",
          flush=True)

    records: list[dict] = []
    rank_histogram = Counter()
    not_in_top_k = 0

    model, tokenizer, query_layer = _load_model()
    device = _model_device()

    for i, item in enumerate(items):
        cells_for_item = item_id_to_cells.get(item.item_id, [])
        expected_cell = find_expected_chunk(
            item, adapter._cell_to_chunk_text, cells_for_item,
        )

        # Use hook.on_prefill directly to control top-K — the production
        # adapter forces k=5 via the hook constructor.
        inputs = tokenizer(
            item.question, return_tensors="pt", truncation=True, max_length=256,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, use_cache=True, output_hidden_states=True)

        old_k = adapter._hook.k
        adapter._hook.k = DIAGNOSTIC_TOP_K_PROBE
        try:
            handles = adapter._hook.on_prefill(
                layer=query_layer,
                past_key_values=out.past_key_values,
                model_hidden_states=out.hidden_states[query_layer],
            )
        finally:
            adapter._hook.k = old_k

        retrieved_cell_ids = [int(h.cell_id) for h in handles]
        expected_rank = None
        if expected_cell is not None and expected_cell in retrieved_cell_ids:
            expected_rank = retrieved_cell_ids.index(expected_cell)

        records.append({
            "item_id": item.item_id,
            "n_cells_for_item": len(cells_for_item),
            "expected_cell": expected_cell,
            "expected_rank": expected_rank,
            "retrieved_top10": retrieved_cell_ids[:10],
        })

        if expected_rank is None:
            not_in_top_k += 1
        else:
            for bucket in RANK_BUCKETS:
                if expected_rank < bucket:
                    rank_histogram[bucket] += 1
                    break

        if (i + 1) % 100 == 0:
            print(f"[{dataset_name}] {i + 1}/{len(items)} queried", flush=True)

    n = len(items)
    histogram_pct = {
        f"<{b}": rank_histogram[b] / n if n else 0.0 for b in RANK_BUCKETS
    }
    histogram_pct[f">={RANK_BUCKETS[-1]}_or_missing"] = not_in_top_k / n if n else 0.0

    summary = {
        "n": n,
        "ingest_seconds": ingest_secs,
        "rank_histogram_counts": dict(rank_histogram),
        "not_in_top_k": not_in_top_k,
        "rank_histogram_pct": histogram_pct,
        "cumulative_in_top_k_pct": {
            f"R@{b}": sum(rank_histogram[bb] for bb in RANK_BUCKETS if bb <= b) / n
            for b in RANK_BUCKETS
        } if n else {},
    }

    return {"summary": summary, "records": records}


_MODEL_CACHE: dict = {}


def _load_model():
    if "model" not in _MODEL_CACHE:
        from tdb_bench.adapters.tardigrade import _load_model_cached
        model, tokenizer, query_layer = _load_model_cached()
        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["tokenizer"] = tokenizer
        _MODEL_CACHE["query_layer"] = query_layer
        _MODEL_CACHE["device"] = next(model.parameters()).device
    return (_MODEL_CACHE["model"], _MODEL_CACHE["tokenizer"],
            _MODEL_CACHE["query_layer"])


def _model_device():
    if "device" not in _MODEL_CACHE:
        _load_model()
    return _MODEL_CACHE["device"]


def main():
    dataset_dir = Path(os.getenv("TDB_DIAGNOSTIC_DATASET_DIR", DEFAULT_DATASET_DIR))
    output_path = os.getenv("TDB_DIAGNOSTIC_REPORT", DEFAULT_OUTPUT)

    locomo_items = load_items(str(dataset_dir / "locomo_phase1.jsonl"),
                              N_ITEMS_LOCOMO, "locomo")
    longmem_items = load_items(str(dataset_dir / "longmemeval_phase1.jsonl"),
                               N_ITEMS_LONGMEM, "longmemeval")

    _load_model()  # warm model cache before per-dataset runs

    report = {
        "config": {
            "top_k": DIAGNOSTIC_TOP_K_PROBE,
            "rank_buckets": RANK_BUCKETS,
            "n_items_locomo": N_ITEMS_LOCOMO,
            "n_items_longmem": N_ITEMS_LONGMEM,
            "rls_mode": os.getenv("TDB_RLS_MODE", "none"),
            "refinement_mode": os.getenv("TDB_REFINEMENT_MODE", "centered"),
            "reranker": os.getenv("TDB_BENCH_RERANK_MODEL", ""),
        },
        "datasets": {},
    }

    for name, items in (("locomo", locomo_items),
                         ("longmemeval", longmem_items)):
        adapter = TardigradeAdapter()
        adapter.enable_chunk_text_tracking()
        report["datasets"][name] = run_dataset(adapter, items, name)
        # Print the summary as soon as it's available
        print(f"\n[{name}] SUMMARY", flush=True)
        for k, v in report["datasets"][name]["summary"].items():
            if k == "records":
                continue
            print(f"  {k}: {v}", flush=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n=== report written to {output_path} ===", flush=True)


if __name__ == "__main__":
    main()
