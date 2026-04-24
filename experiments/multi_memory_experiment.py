#!/usr/bin/env python3
"""Multi-memory KV injection experiment (Phase 30).

Compares naive KV concatenation vs text RAG on cross-referencing facts
that require synthesizing information from 2+ memories.

Decision gate G1:
  >= 70% of text RAG accuracy -> ship naive concat
  < 50% -> proceed to sequential recomputation
  50-70% -> consider hybrid approach

Usage:
    source .venv/bin/activate
    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop -m crates/tdb-python/Cargo.toml
    python experiments/multi_memory_experiment.py
"""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "python"))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import tardigrade_db
from tardigrade_hooks.kp_injector import KnowledgePackStore
from tardigrade_hooks.multi_composer import NaiveConcatComposer, SequentialRecomputeComposer
from multi_memory_corpus import MULTI_FACTS


def load_model():
    """Load Qwen3-0.6B for the experiment."""
    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    return model, tokenizer


def run_text_rag(model, tokenizer, entry):
    """Text RAG baseline: paste all facts into the prompt."""
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
        out = model.generate(
            input_ids, max_new_tokens=100, do_sample=False,
        )
    response = tokenizer.decode(out[0][prompt_tokens:], skip_special_tokens=True).strip()
    # Strip thinking tags if present
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()
    correct = entry["expected"].lower() in response.lower()
    return correct, response, prompt_tokens


def run_kv_injection_multi(kps, entry, k, composer=None):
    """Multi-memory KV injection: retrieve and inject k memories."""
    if composer is None:
        composer = NaiveConcatComposer()
    text, prompt_tokens, had_memory = kps.generate_multi(
        entry["query"] + " /no_think",
        k=k,
        composer=composer,
        max_new_tokens=100,
        do_sample=False,
    )
    correct = entry["expected"].lower() in text.lower()
    return correct, text, prompt_tokens, had_memory


def main():
    model, tokenizer = load_model()

    print(f"\n{'='*70}")
    print("MULTI-MEMORY KV INJECTION EXPERIMENT")
    print(f"{'='*70}")
    print(f"Corpus: {len(MULTI_FACTS)} cross-referencing fact sets")
    print(f"Model: {model.config._name_or_path}")
    print(f"Facts per query: 2-3")
    print()

    # -- Text RAG baseline --
    print("=" * 40)
    print("PATH A: TEXT RAG (baseline)")
    print("=" * 40)
    rag_correct = 0
    rag_tokens = 0
    for i, entry in enumerate(MULTI_FACTS):
        correct, response, tokens = run_text_rag(model, tokenizer, entry)
        rag_correct += int(correct)
        rag_tokens += tokens
        status = "PASS" if correct else "MISS"
        print(f"  [{status}] Q{i+1}: {entry['query'][:50]}...")
        if not correct:
            print(f"         Expected: {entry['expected']}")
            print(f"         Got: {response[:80]}")

    print(f"\nText RAG: {rag_correct}/{len(MULTI_FACTS)} correct, {rag_tokens} total prompt tokens")

    # -- KV injection (naive concat, k=2) --
    print()
    print("=" * 40)
    print("PATH B: KV INJECTION (naive concat, k=2)")
    print("=" * 40)

    tmpdir = tempfile.mkdtemp()
    engine = tardigrade_db.Engine(tmpdir)
    kps = KnowledgePackStore(engine, model, tokenizer, owner=1)

    # Store all facts (each fact is a separate pack)
    print("  Storing facts...")
    t0 = time.time()
    for entry in MULTI_FACTS:
        for fact in entry["facts"]:
            kps.store(fact)
    store_time = time.time() - t0
    total_facts = sum(len(e["facts"]) for e in MULTI_FACTS)
    print(f"  Stored {total_facts} facts ({engine.pack_count()} packs) in {store_time:.1f}s")

    inj_correct = 0
    inj_tokens = 0
    for i, entry in enumerate(MULTI_FACTS):
        k = len(entry["facts"])  # request as many packs as facts needed
        correct, response, tokens, had_memory = run_kv_injection_multi(kps, entry, k)
        inj_correct += int(correct)
        inj_tokens += tokens
        status = "PASS" if correct else "MISS"
        mem_status = "mem" if had_memory else "NO_MEM"
        print(f"  [{status}][{mem_status}] Q{i+1}: {entry['query'][:50]}...")
        if not correct:
            print(f"         Expected: {entry['expected']}")
            print(f"         Got: {response[:80]}")

    print(f"\nKV Injection (naive): {inj_correct}/{len(MULTI_FACTS)} correct, {inj_tokens} total prompt tokens")

    # -- KV injection (sequential recomputation, k=2) --
    print()
    print("=" * 40)
    print("PATH C: KV INJECTION (sequential recompute)")
    print("=" * 40)

    tmpdir2 = tempfile.mkdtemp()
    engine2 = tardigrade_db.Engine(tmpdir2)
    kps2 = KnowledgePackStore(engine2, model, tokenizer, owner=1)

    print("  Storing facts...")
    for entry in MULTI_FACTS:
        for fact in entry["facts"]:
            kps2.store(fact)
    print(f"  Stored {engine2.pack_count()} packs")

    seq_composer = SequentialRecomputeComposer(model, tokenizer, kps2._text_registry)
    seq_correct = 0
    seq_tokens = 0
    for i, entry in enumerate(MULTI_FACTS):
        k = len(entry["facts"])
        correct, response, tokens, had_memory = run_kv_injection_multi(
            kps2, entry, k, composer=seq_composer
        )
        seq_correct += int(correct)
        seq_tokens += tokens
        status = "PASS" if correct else "MISS"
        mem_status = "mem" if had_memory else "NO_MEM"
        print(f"  [{status}][{mem_status}] Q{i+1}: {entry['query'][:50]}...")
        if not correct:
            print(f"         Expected: {entry['expected']}")
            print(f"         Got: {response[:80]}")

    print(f"\nKV Injection (sequential): {seq_correct}/{len(MULTI_FACTS)} correct, {seq_tokens} total prompt tokens")

    # -- Summary --
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Text RAG:              {rag_correct}/{len(MULTI_FACTS)} ({100*rag_correct/len(MULTI_FACTS):.0f}%)")
    print(f"  KV Naive Concat:       {inj_correct}/{len(MULTI_FACTS)} ({100*inj_correct/len(MULTI_FACTS):.0f}%)")
    print(f"  KV Sequential Recomp:  {seq_correct}/{len(MULTI_FACTS)} ({100*seq_correct/len(MULTI_FACTS):.0f}%)")
    print(f"  Token savings (naive): {rag_tokens - inj_tokens} tokens ({100*(rag_tokens-inj_tokens)/rag_tokens:.0f}%)")
    print(f"  Token savings (seq):   {rag_tokens - seq_tokens} tokens ({100*(rag_tokens-seq_tokens)/rag_tokens:.0f}%)")
    print()

    seq_ratio = seq_correct / max(rag_correct, 1)
    if seq_ratio >= 0.8:
        print("  DECISION GATE G3: PASS -- sequential recompute >= 80% of text RAG")
        print("  Ship SequentialRecomputeComposer as default strategy.")
    elif seq_ratio >= 0.5:
        print("  DECISION GATE G3: BORDERLINE -- sequential recompute 50-80%")
    else:
        print("  DECISION GATE G3: FAIL -- sequential recompute < 50%")
        print("  Multi-memory KV injection not viable. Fall back to text RAG.")


if __name__ == "__main__":
    main()
