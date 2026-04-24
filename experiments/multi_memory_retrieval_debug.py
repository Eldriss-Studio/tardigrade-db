#!/usr/bin/env python3
"""Phase 30 diagnostic: are the right packs being retrieved?

Hypothesis: multi-memory injection fails not because of cross-attention
or position corruption, but because the retriever returns the WRONG
second pack. E.g., "What car does Lucia's instructor drive?" retrieves
pack A ("Lucia's instructor is Tomoko") but NOT pack B ("Tomoko drives
a Honda Civic") because pack B's retrieval key doesn't mention "Lucia."

This script stores all facts, then for each query:
1. Shows which packs were retrieved (by pack_id -> fact text)
2. Shows which packs were EXPECTED (the facts from the corpus)
3. Reports retrieval accuracy: did we find ALL needed packs?

Usage:
    source .venv/bin/activate
    python experiments/multi_memory_retrieval_debug.py
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
from multi_memory_corpus import MULTI_FACTS


def main():
    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    tmpdir = tempfile.mkdtemp()
    engine = tardigrade_db.Engine(tmpdir)
    kps = KnowledgePackStore(engine, model, tokenizer, owner=1)

    # Store all facts
    print("Storing all facts...")
    for entry in MULTI_FACTS:
        for fact in entry["facts"]:
            kps.store(fact)

    pack_to_fact = dict(kps._text_registry)
    fact_to_pack = {v: k for k, v in pack_to_fact.items()}

    total_facts = sum(len(e["facts"]) for e in MULTI_FACTS)
    print(f"Stored {total_facts} facts ({engine.pack_count()} packs)")
    print()

    print("=" * 70)
    print("RETRIEVAL DIAGNOSTIC")
    print("=" * 70)
    print()

    all_facts_found = 0
    some_facts_found = 0
    no_facts_found = 0

    for i, entry in enumerate(MULTI_FACTS):
        query = entry["query"]
        expected_facts = set(entry["facts"])
        expected_pack_ids = {fact_to_pack[f] for f in expected_facts}
        k = len(entry["facts"])

        # Compute query key and retrieve
        query_input = tokenizer.encode(query, return_tensors="pt")
        with torch.no_grad():
            query_out = model(query_input, output_hidden_states=True)
        hidden = query_out.hidden_states[kps.query_layer][0]
        h_tokens = hidden[1:].numpy().astype(np.float32)
        query_key = encode_per_token(h_tokens, kps.hidden_size)

        packs = engine.mem_read_pack(query_key, k, 1)
        retrieved_pack_ids = {p["pack_id"] for p in packs}

        # Also get top-5 to see ranking
        packs_top5 = engine.mem_read_pack(query_key, 5, 1)

        found_ids = expected_pack_ids & retrieved_pack_ids
        missing_ids = expected_pack_ids - retrieved_pack_ids

        if len(missing_ids) == 0:
            status = "ALL FOUND"
            all_facts_found += 1
        elif len(found_ids) > 0:
            status = "PARTIAL"
            some_facts_found += 1
        else:
            status = "NONE FOUND"
            no_facts_found += 1

        print(f"Q{i+1} [{status}]: {query[:60]}...")
        print(f"  Expected packs ({k}):")
        for pid in sorted(expected_pack_ids):
            marker = "found" if pid in retrieved_pack_ids else "MISSING"
            print(f"    [{marker}] pack {pid}: {pack_to_fact[pid][:60]}...")
        wrong_ids = retrieved_pack_ids - expected_pack_ids
        if wrong_ids:
            print(f"  Wrong packs retrieved:")
            for pid in sorted(wrong_ids):
                print(f"    [wrong] pack {pid}: {pack_to_fact[pid][:60]}...")

        print(f"  Top-5 ranking:")
        for rank, p in enumerate(packs_top5):
            pid = p["pack_id"]
            score = p["score"]
            is_expected = "***" if pid in expected_pack_ids else "   "
            print(f"    {is_expected} #{rank+1}: pack {pid} (score={score:.1f}) {pack_to_fact[pid][:50]}...")
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  All facts retrieved:  {all_facts_found}/{len(MULTI_FACTS)}")
    print(f"  Partial retrieval:    {some_facts_found}/{len(MULTI_FACTS)}")
    print(f"  No facts retrieved:   {no_facts_found}/{len(MULTI_FACTS)}")
    print()

    if all_facts_found < 7:
        print("  DIAGNOSIS: RETRIEVAL IS THE BOTTLENECK")
        print("  The retriever cannot find all needed packs.")
        print("  Fix retrieval before attempting injection fixes.")
    else:
        print("  DIAGNOSIS: RETRIEVAL IS FINE")
        print("  The right packs are retrieved.")
        print("  The problem is in composition/injection.")


if __name__ == "__main__":
    main()
