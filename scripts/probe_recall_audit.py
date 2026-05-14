"""Audit probe: reproduce 2026-05-11 baseline ingestion pattern at small scale.

Bypasses bench evaluator, adapter pipeline, chunking, and batching. Goes
straight: tokenize one item context → forward pass → hook.on_generate →
engine.mem_write. Then queries with item.question and checks if top-1
retrieves the source item.

If this probe shows high R@1, the engine+hook serial path is healthy and
the regression is in adapter/chunking/batching layers.
If this probe shows low R@1, the engine itself is regressed and the bug
is in retrieval (encoding, INT8 tier, refinement, etc.).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import tardigrade_db
from tardigrade_hooks.constants import DEFAULT_CAPTURE_LAYER_RATIO
from tardigrade_hooks.hf_kv_hook import HuggingFaceKVHook

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = os.getenv("PROBE_DEVICE", "cuda")
N_ITEMS = int(os.getenv("PROBE_N_ITEMS", "30"))
TRUNCATE_TOKENS = int(os.getenv("PROBE_TRUNCATE", "256"))
TOP_K = 5
REFINEMENT = os.getenv("PROBE_REFINEMENT", "centered")
OWNER = 1
DATASET = os.getenv(
    "PROBE_DATASET",
    str(Path(__file__).resolve().parents[1]
        / "benchmarks/datasets/phase1_oracle/locomo_phase1.jsonl"),
)


def load_items(path: str, n: int):
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
            if len(items) >= n:
                break
    return items


def main():
    print(f"=== probe: {N_ITEMS} items, truncate={TRUNCATE_TOKENS}, device={DEVICE}")
    print(f"     refinement={REFINEMENT}")

    items = load_items(DATASET, N_ITEMS)
    print(f"loaded {len(items)} items")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32, attn_implementation="eager",
    ).to(DEVICE).eval()
    n_layers = model.config.num_hidden_layers
    query_layer = int(n_layers * DEFAULT_CAPTURE_LAYER_RATIO)
    print(f"model loaded, query_layer={query_layer}/{n_layers}")

    data_dir = tempfile.mkdtemp(prefix="probe_audit_")
    engine = tardigrade_db.Engine(data_dir)
    try:
        engine.set_refinement_mode(REFINEMENT)
    except Exception as exc:
        print(f"set_refinement_mode failed: {exc}")

    hook = HuggingFaceKVHook(
        engine, owner=OWNER, k=5,
        model_config=model.config, model=model,
        use_hidden_states=True,
    )

    cell_to_item: dict[int, int] = {}

    # ---- ingest: one cell per item, truncated, NO chunking, NO batching ----
    for idx, item in enumerate(items):
        inputs = tokenizer(
            item["context"], return_tensors="pt",
            truncation=True, max_length=TRUNCATE_TOKENS,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, use_cache=True, output_hidden_states=True)
        decision = hook.on_generate(
            layer=query_layer,
            past_key_values=out.past_key_values,
            model_hidden_states=out.hidden_states[query_layer],
        )
        if not decision.should_write or len(decision.key) == 0:
            continue
        cell_id = engine.mem_write(
            OWNER, query_layer, decision.key, decision.value, decision.salience, None,
        )
        cell_to_item[int(cell_id)] = idx
        if (idx + 1) % 10 == 0:
            print(f"  ingest {idx + 1}/{len(items)}")
    print(f"ingested {len(cell_to_item)} cells")

    # ---- query: item.question → handles → does top-1 map back to item? ----
    r1_hits = 0
    r5_hits = 0
    for idx, item in enumerate(items):
        inputs = tokenizer(
            item["question"], return_tensors="pt",
            truncation=True, max_length=TRUNCATE_TOKENS,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, use_cache=True, output_hidden_states=True)
        handles = hook.on_prefill(
            layer=query_layer,
            past_key_values=out.past_key_values,
            model_hidden_states=out.hidden_states[query_layer],
        )
        retrieved = [cell_to_item.get(int(h.cell_id), -1) for h in handles[:TOP_K]]
        if retrieved and retrieved[0] == idx:
            r1_hits += 1
        if idx in retrieved:
            r5_hits += 1
        if (idx + 1) % 10 == 0:
            print(f"  query {idx + 1}/{len(items)}: R@1={r1_hits} R@5={r5_hits}")

    print()
    print(f"=== RESULT ===")
    print(f"R@1 = {r1_hits}/{len(items)} = {r1_hits / len(items):.1%}")
    print(f"R@5 = {r5_hits}/{len(items)} = {r5_hits / len(items):.1%}")


if __name__ == "__main__":
    main()
