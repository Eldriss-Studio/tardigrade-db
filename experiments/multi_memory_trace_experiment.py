#!/usr/bin/env python3
"""Phase 31: Trace-linked multi-hop retrieval experiment.

Compares: baseline retrieval (0/10 find both) vs trace-linked (should
find both via links). Uses Qwen3-0.6B.

Decision gate G8: trace retrieval finds both packs 10/10
Decision gate G9: trace + injection >= 6/10 (matching oracle ceiling)

Usage:
    source .venv/bin/activate
    python experiments/multi_memory_trace_experiment.py
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


def load_model():
    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    return model, tokenizer


def run_text_rag(model, tokenizer, entry):
    facts_text = "\n".join(f"- {f}" for f in entry["facts"])
    messages = [
        {"role": "system", "content": f"Use these facts to answer:\n{facts_text}"},
        {"role": "user", "content": entry["query"]},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(formatted, return_tensors="pt")
    prompt_tokens = input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=100, do_sample=False)
    response = tokenizer.decode(out[0][prompt_tokens:], skip_special_tokens=True).strip()
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()
    correct = entry["expected"].lower() in response.lower()
    return correct, response, prompt_tokens


def main():
    model, tokenizer = load_model()

    print(f"\n{'='*70}")
    print("PHASE 31: TRACE-LINKED MULTI-HOP RETRIEVAL")
    print(f"{'='*70}")
    print()

    # -- Text RAG baseline --
    print("--- TEXT RAG (baseline) ---")
    rag_correct = 0
    for i, entry in enumerate(MULTI_FACTS):
        correct, response, tokens = run_text_rag(model, tokenizer, entry)
        rag_correct += int(correct)
        status = "PASS" if correct else "MISS"
        print(f"  [{status}] Q{i+1}: {entry['query'][:50]}...")
    print(f"  Text RAG: {rag_correct}/{len(MULTI_FACTS)}")

    # -- Store with trace links --
    print()
    print("--- STORING WITH TRACE LINKS ---")
    tmpdir = tempfile.mkdtemp()
    engine = tardigrade_db.Engine(tmpdir)
    kps = KnowledgePackStore(engine, model, tokenizer, owner=1)

    for entry in MULTI_FACTS:
        kps.store_linked(entry["facts"])
    print(f"  Stored {engine.pack_count()} packs with trace links")

    # -- Verify trace retrieval finds both packs --
    print()
    print("--- RETRIEVAL DIAGNOSTIC ---")
    fact_to_pack = {v: k for k, v in kps._text_registry.items()}
    retrieval_all_found = 0

    for i, entry in enumerate(MULTI_FACTS):
        expected_ids = {fact_to_pack[f] for f in entry["facts"]}

        query_input = tokenizer.encode(entry["query"], return_tensors="pt")
        with torch.no_grad():
            out = model(query_input, output_hidden_states=True)
        hidden = out.hidden_states[kps.query_layer][0]
        h_tokens = hidden[1:].numpy().astype(np.float32)
        query_key = encode_per_token(h_tokens, kps.hidden_size)

        direct_packs = engine.mem_read_pack(query_key, 1, 1)
        direct_ids = {p["pack_id"] for p in direct_packs}
        linked_ids = set()
        for pid in direct_ids:
            linked_ids.update(kps._trace_links.get(pid, set()))
        all_ids = direct_ids | linked_ids

        missing = expected_ids - all_ids
        if len(missing) == 0:
            retrieval_all_found += 1
            print(f"  [ALL FOUND] Q{i+1}: {entry['query'][:50]}...")
        else:
            print(f"  [PARTIAL]   Q{i+1}: {entry['query'][:50]}...")
            for mid in missing:
                print(f"    MISSING: {kps._text_registry[mid][:50]}...")

    print(f"\n  Trace retrieval: {retrieval_all_found}/{len(MULTI_FACTS)} find all packs")

    # -- Generate with trace --
    print()
    print("--- GENERATE WITH TRACE ---")
    trace_correct = 0
    trace_tokens = 0

    for i, entry in enumerate(MULTI_FACTS):
        text, tokens, had_memory = kps.generate_with_trace(
            entry["query"] + " /no_think", k=1,
            max_new_tokens=100, do_sample=False,
        )
        correct = entry["expected"].lower() in text.lower()
        trace_correct += int(correct)
        trace_tokens += tokens
        status = "PASS" if correct else "MISS"
        mem_status = "mem" if had_memory else "NO_MEM"
        print(f"  [{status}][{mem_status}] Q{i+1}: {entry['query'][:50]}...")
        if not correct:
            print(f"    Expected: {entry['expected']}")
            print(f"    Got: {text[:80]}")

    print(f"\n  Trace injection: {trace_correct}/{len(MULTI_FACTS)} correct, {trace_tokens} total tokens")

    # -- Summary --
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Text RAG:               {rag_correct}/{len(MULTI_FACTS)}")
    print(f"  Baseline injection:     3/10 (Phase 30)")
    print(f"  Oracle injection:       6/10 (Phase 30B)")
    print(f"  Trace retrieval found:  {retrieval_all_found}/10 all packs")
    print(f"  Trace + injection:      {trace_correct}/{len(MULTI_FACTS)}")
    print()

    if trace_correct >= 6:
        print("  DECISION GATE G9: PASS")
    elif trace_correct > 3:
        print("  DECISION GATE G9: IMPROVEMENT over baseline 3/10")
    else:
        print("  DECISION GATE G9: NO IMPROVEMENT")


if __name__ == "__main__":
    main()
