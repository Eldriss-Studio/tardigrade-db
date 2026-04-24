#!/usr/bin/env python3
"""Decompose the 55% scale failure: is it retrieval or injection?

For each of the 9 failing queries at 140 memories, determine:
1. Did first-hop retrieval find the correct linking pack?
2. Did trace link load the correct second pack?
3. If both packs are correct, why did injection fail?

Usage:
    source .venv/bin/activate
    python experiments/multi_memory_scale_debug.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "python"))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import tardigrade_db
from tardigrade_hooks.kp_injector import KnowledgePackStore
from tardigrade_hooks.encoding import encode_per_token
from multi_memory_scale_test import CROSS_REF_PAIRS


def load_model():
    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    return model, tokenizer


def main():
    from corpus_100 import MEMORIES

    model, tokenizer = load_model()

    tmpdir = tempfile.mkdtemp()
    engine = tardigrade_db.Engine(tmpdir)
    kps = KnowledgePackStore(engine, model, tokenizer, owner=1)

    # Store background
    print(f"Storing {len(MEMORIES)} background memories...")
    for mem in MEMORIES:
        kps.store(mem)

    # Store cross-ref pairs with trace links
    print(f"Storing {len(CROSS_REF_PAIRS)} cross-ref pairs...")
    for entry in CROSS_REF_PAIRS:
        kps.store_linked(entry["facts"])

    fact_to_pack = {v: k for k, v in kps._text_registry.items()}
    print(f"Total: {engine.pack_count()} packs")

    print()
    print("=" * 70)
    print("FAILURE DECOMPOSITION")
    print("=" * 70)
    print()

    retrieval_ok = 0
    trace_ok = 0
    injection_ok = 0

    for i, entry in enumerate(CROSS_REF_PAIRS):
        query = entry["query"]
        expected_ids = {fact_to_pack[f] for f in entry["facts"]}

        # Step 1: first-hop retrieval
        query_input = tokenizer.encode(query, return_tensors="pt")
        with torch.no_grad():
            out = model(query_input, output_hidden_states=True)
        hidden = out.hidden_states[kps.query_layer][0]
        h_tokens = hidden[1:].numpy().astype(np.float32)
        query_key = encode_per_token(h_tokens, kps.hidden_size)

        first_hop = engine.mem_read_pack(query_key, 1, 1)
        first_hop_ids = {p["pack_id"] for p in first_hop}

        first_hop_correct = bool(first_hop_ids & expected_ids)

        # Step 2: trace links
        linked_ids = set()
        for pid in first_hop_ids:
            linked_ids.update(kps._trace_links.get(pid, set()))
        all_found_ids = first_hop_ids | linked_ids

        all_expected_found = expected_ids <= all_found_ids

        # Step 3: generation
        text, tokens, had_memory = kps.generate_with_trace(
            query + " /no_think", k=1, max_new_tokens=100, do_sample=False,
        )
        correct = entry["expected"].lower() in text.lower()

        if first_hop_correct:
            retrieval_ok += 1
        if all_expected_found:
            trace_ok += 1
        if correct:
            injection_ok += 1

        status = "PASS" if correct else "MISS"
        r1 = "R1:ok" if first_hop_correct else "R1:WRONG"
        r2 = "trace:ok" if all_expected_found else "trace:MISS"

        print(f"  [{status}] [{r1}] [{r2}] Q{i+1}: {query[:50]}...")

        if not correct:
            first_text = kps._text_registry.get(
                list(first_hop_ids)[0], "?"
            ) if first_hop_ids else "NONE"
            print(f"    First hop: {first_text[:60]}...")
            if not first_hop_correct:
                for eid in expected_ids:
                    print(f"    Expected:  {kps._text_registry[eid][:60]}...")
            if not all_expected_found:
                missing = expected_ids - all_found_ids
                for mid in missing:
                    print(f"    Trace miss: {kps._text_registry[mid][:60]}...")
            print(f"    Expected answer: {entry['expected']}")
            print(f"    Got: {text[:80]}")

    print()
    print("=" * 70)
    print("DECOMPOSITION SUMMARY")
    print("=" * 70)
    n = len(CROSS_REF_PAIRS)
    print(f"  First-hop retrieval correct: {retrieval_ok}/{n}")
    print(f"  All packs found (trace):     {trace_ok}/{n}")
    print(f"  Final answer correct:        {injection_ok}/{n}")
    print()
    retrieval_failures = n - retrieval_ok
    trace_only_failures = retrieval_ok - trace_ok if retrieval_ok > trace_ok else 0
    injection_failures = trace_ok - injection_ok
    print(f"  Breakdown of {n - injection_ok} failures:")
    print(f"    Retrieval wrong (first hop finds wrong pack): {retrieval_failures}")
    print(f"    Trace miss (link not followed correctly):     {trace_only_failures}")
    print(f"    Injection fail (correct packs, wrong answer): {injection_failures}")


if __name__ == "__main__":
    main()
