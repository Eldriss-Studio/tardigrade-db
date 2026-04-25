#!/usr/bin/env python3
"""Phase 33: Calibrate auto-link threshold.

Measure retrieval scores between cross-ref detail facts and existing
background memories. Determine threshold separating related from unrelated.

Usage:
    source .venv/bin/activate
    python experiments/auto_link_calibration.py
"""

import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "python"))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import tardigrade_db
from tardigrade_hooks.kp_injector import KnowledgePackStore
from tardigrade_hooks.encoding import encode_per_token
from corpus_100 import MEMORIES
from multi_memory_scale_test import CROSS_REF_PAIRS


def main():
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    tmpdir = tempfile.mkdtemp()
    engine = tardigrade_db.Engine(tmpdir)
    kps = KnowledgePackStore(engine, model, tokenizer, owner=1)

    print("Storing 100 background memories...", flush=True)
    for mem in MEMORIES:
        kps.store(mem, auto_link=False)
    print(f"Background packs: {engine.pack_count()}", flush=True)

    print(flush=True)
    print("=" * 70, flush=True)
    print("CALIBRATION: linking facts vs background memories", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    top_scores = []
    fifth_scores = []

    for i, entry in enumerate(CROSS_REF_PAIRS):
        linking_fact = entry["facts"][0]

        messages = [{"role": "system", "content": linking_fact}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_ids = tokenizer.encode(formatted, return_tensors="pt")
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
        hidden = out.hidden_states[kps.query_layer][0]
        h_tokens = hidden[1:].numpy().astype(np.float32)
        query_key = encode_per_token(h_tokens, kps.hidden_size)

        results = engine.mem_read_pack(query_key, 5, 1)

        if results:
            top_score = results[0]["score"]
            top_text = kps._text_registry.get(results[0]["pack_id"], "?")[:50]
            top_scores.append(top_score)
            print(f"  Q{i+1} (score={top_score:.1f}): {linking_fact[:45]}...", flush=True)
            print(f"    -> {top_text}...", flush=True)

            if len(results) >= 5:
                fifth_scores.append(results[4]["score"])

    print(flush=True)
    print("=" * 70, flush=True)
    print("SCORE DISTRIBUTION", flush=True)
    print("=" * 70, flush=True)

    if top_scores:
        print(f"  Top-1 scores: min={min(top_scores):.1f} max={max(top_scores):.1f} "
              f"mean={np.mean(top_scores):.1f} median={np.median(top_scores):.1f}", flush=True)
    if fifth_scores:
        print(f"  5th scores:   min={min(fifth_scores):.1f} max={max(fifth_scores):.1f} "
              f"mean={np.mean(fifth_scores):.1f}", flush=True)
    if top_scores and fifth_scores:
        gap = np.mean(top_scores) - np.mean(fifth_scores)
        mid = (min(top_scores) + max(fifth_scores)) / 2
        print(f"  Gap: {gap:.1f}", flush=True)
        print(f"  Suggested threshold: {mid:.1f}", flush=True)


if __name__ == "__main__":
    main()
