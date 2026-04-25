#!/usr/bin/env python3
"""Phase 33: Auto-linking scale test.

Store 100 background memories, then 40 cross-ref detail facts
individually (NOT with store_linked). Auto-linking should connect
detail facts to related background memories automatically.

Usage:
    source .venv/bin/activate
    python experiments/auto_link_scale_test.py
"""

import sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "python"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import tardigrade_db
from tardigrade_hooks.kp_injector import KnowledgePackStore
from corpus_100 import MEMORIES
from multi_memory_scale_test import CROSS_REF_PAIRS


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
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    print(f"\n{'='*70}", flush=True)
    print("PHASE 33: AUTO-LINKING SCALE TEST", flush=True)
    print(f"{'='*70}", flush=True)

    # Text RAG baseline
    print("\n--- TEXT RAG (baseline) ---", flush=True)
    rag_correct = 0
    for i, entry in enumerate(CROSS_REF_PAIRS):
        correct, response, tokens = run_text_rag(model, tokenizer, entry)
        rag_correct += int(correct)
        status = "PASS" if correct else "MISS"
        print(f"  [{status}] Q{i+1}: {entry['query'][:50]}...", flush=True)
    print(f"  Text RAG: {rag_correct}/{len(CROSS_REF_PAIRS)}", flush=True)

    # Store with auto-linking
    print("\n--- STORING WITH AUTO-LINKING ---", flush=True)
    tmpdir = tempfile.mkdtemp()
    engine = tardigrade_db.Engine(tmpdir)
    kps = KnowledgePackStore(engine, model, tokenizer, owner=1)

    print(f"  Storing {len(MEMORIES)} background memories...", flush=True)
    t0 = time.time()
    for mem in MEMORIES:
        kps.store(mem, auto_link=True)
    bg_time = time.time() - t0
    bg_links = sum(len(engine.pack_links(pid)) for pid in range(1, engine.pack_count() + 1)) // 2
    print(f"  Background: {engine.pack_count()} packs, {bg_links} auto-links, {bg_time:.1f}s", flush=True)

    print(f"  Storing {len(CROSS_REF_PAIRS) * 2} cross-ref facts individually...", flush=True)
    t0 = time.time()
    fact_to_pack = {}
    for entry in CROSS_REF_PAIRS:
        for fact in entry["facts"]:
            pack_id = kps.store(fact, auto_link=True)
            fact_to_pack[fact] = pack_id
    cr_time = time.time() - t0
    total_links = sum(len(engine.pack_links(pid)) for pid in range(1, engine.pack_count() + 1)) // 2
    print(f"  Total: {engine.pack_count()} packs, {total_links} auto-links, {cr_time:.1f}s", flush=True)

    # Auto-link diagnostic
    print("\n--- AUTO-LINK DIAGNOSTIC ---", flush=True)
    correct_links = 0
    for i, entry in enumerate(CROSS_REF_PAIRS):
        pid_a = fact_to_pack[entry["facts"][0]]
        pid_b = fact_to_pack[entry["facts"][1]]
        links_a = set(engine.pack_links(pid_a))
        links_b = set(engine.pack_links(pid_b))

        directly_linked = pid_b in links_a
        shared = links_a & links_b
        indirectly_linked = len(shared) > 0
        linked = directly_linked or indirectly_linked

        if linked:
            correct_links += 1
        status = "LINKED" if linked else "NOT LINKED"
        method = "direct" if directly_linked else ("shared" if indirectly_linked else "none")
        print(f"  [{status}][{method}] Q{i+1}: {entry['facts'][0][:40]}...", flush=True)
    print(f"\n  Auto-linked pairs: {correct_links}/{len(CROSS_REF_PAIRS)}", flush=True)

    # Generate with trace
    print("\n--- GENERATE WITH TRACE (auto-linked) ---", flush=True)
    trace_correct = 0
    for i, entry in enumerate(CROSS_REF_PAIRS):
        text, tokens, had_memory = kps.generate_with_trace(
            entry["query"] + " /no_think", k=1,
            max_new_tokens=100, do_sample=False,
        )
        correct = entry["expected"].lower() in text.lower()
        trace_correct += int(correct)
        status = "PASS" if correct else "MISS"
        print(f"  [{status}] Q{i+1}: {entry['query'][:50]}...", flush=True)
        if not correct:
            print(f"    Expected: {entry['expected']}", flush=True)
            print(f"    Got: {text[:80]}", flush=True)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Total memories: {engine.pack_count()}", flush=True)
    print(f"  Auto-links formed: {total_links}", flush=True)
    print(f"  Pairs correctly linked: {correct_links}/20", flush=True)
    print(f"\n  Text RAG:            {rag_correct}/{len(CROSS_REF_PAIRS)} ({100*rag_correct//len(CROSS_REF_PAIRS)}%)", flush=True)
    print(f"  store_linked (prev): 14/20 (70%) [Phase 32]", flush=True)
    print(f"  Auto-linked:         {trace_correct}/{len(CROSS_REF_PAIRS)} ({100*trace_correct//len(CROSS_REF_PAIRS)}%)", flush=True)

    if trace_correct > 14:
        print("\n  DECISION GATE G13: PASS -- auto-linking beats 70% plateau", flush=True)
    elif trace_correct >= 14:
        print("\n  DECISION GATE G13: EQUAL -- matches plateau", flush=True)
    else:
        print("\n  DECISION GATE G13: WORSE", flush=True)


if __name__ == "__main__":
    main()
