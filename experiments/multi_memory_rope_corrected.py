#!/usr/bin/env python3
"""Phase 30A: RoPE-corrected multi-memory injection experiment.

Tests whether fixing RoPE positions before concatenating KV packs
restores cross-fact reasoning. Based on CacheBlend (EuroSys 2025).

Decision gate G4a:
  >= 7/10 -> ship RoPECorrectedConcatComposer. Skip HKVD.
  5-6/10 -> proceed to Phase 30B (HKVD selective recomputation)
  < 5/10 -> position is not the primary issue

Usage:
    source .venv/bin/activate
    python experiments/multi_memory_rope_corrected.py
"""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "python"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import tardigrade_db
from tardigrade_hooks.kp_injector import KnowledgePackStore
from tardigrade_hooks.multi_composer import (
    NaiveConcatComposer,
    RoPECorrectedConcatComposer,
)
from tardigrade_hooks.position import RoPEPositionEncoder
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


def run_injection(kps, entry, k, composer):
    text, prompt_tokens, had_memory = kps.generate_multi(
        entry["query"] + " /no_think",
        k=k, composer=composer,
        max_new_tokens=100, do_sample=False,
    )
    correct = entry["expected"].lower() in text.lower()
    return correct, text, prompt_tokens, had_memory


def run_path(name, kps, composer):
    correct_count = 0
    total_tokens = 0
    for i, entry in enumerate(MULTI_FACTS):
        k = len(entry["facts"])
        correct, response, tokens, had_memory = run_injection(kps, entry, k, composer)
        correct_count += int(correct)
        total_tokens += tokens
        status = "PASS" if correct else "MISS"
        mem_status = "mem" if had_memory else "NO_MEM"
        print(f"  [{status}][{mem_status}] Q{i+1}: {entry['query'][:50]}...")
        if not correct:
            print(f"         Expected: {entry['expected']}")
            print(f"         Got: {response[:80]}")
    print(f"\n{name}: {correct_count}/{len(MULTI_FACTS)} correct, {total_tokens} total prompt tokens")
    return correct_count, total_tokens


def main():
    model, tokenizer = load_model()
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    rope_theta = getattr(cfg, "rope_theta", 10000.0)

    print(f"\n{'='*70}")
    print("PHASE 30A: RoPE-CORRECTED MULTI-MEMORY INJECTION")
    print(f"{'='*70}")
    print(f"Corpus: {len(MULTI_FACTS)} cross-referencing fact sets")
    print(f"Model: {cfg._name_or_path}")
    print(f"RoPE base (rope_theta): {rope_theta}")
    print(f"Head dim: {head_dim}")
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

    # -- Store facts for injection paths --
    tmpdir = tempfile.mkdtemp()
    engine = tardigrade_db.Engine(tmpdir)
    kps = KnowledgePackStore(engine, model, tokenizer, owner=1)

    print("\nStoring facts...")
    t0 = time.time()
    for entry in MULTI_FACTS:
        for fact in entry["facts"]:
            kps.store(fact)
    store_time = time.time() - t0
    total_facts = sum(len(e["facts"]) for e in MULTI_FACTS)
    print(f"Stored {total_facts} facts ({engine.pack_count()} packs) in {store_time:.1f}s")

    # -- Naive concat (comparison) --
    print()
    print("=" * 40)
    print("PATH B: NAIVE CONCAT (comparison)")
    print("=" * 40)
    naive_correct, naive_tokens = run_path("Naive Concat", kps, NaiveConcatComposer())

    # -- RoPE-corrected concat (decision gate) --
    print()
    print("=" * 40)
    print("PATH C: RoPE-CORRECTED CONCAT (CacheBlend)")
    print("=" * 40)
    rope_encoder = RoPEPositionEncoder(head_dim=head_dim, base=rope_theta)
    rope_composer = RoPECorrectedConcatComposer(rope_encoder)
    rope_correct, rope_tokens = run_path("RoPE Corrected", kps, rope_composer)

    # -- Summary --
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Text RAG:           {rag_correct}/{len(MULTI_FACTS)} ({100*rag_correct/len(MULTI_FACTS):.0f}%)")
    print(f"  Naive Concat:       {naive_correct}/{len(MULTI_FACTS)} ({100*naive_correct/len(MULTI_FACTS):.0f}%)")
    print(f"  RoPE Corrected:     {rope_correct}/{len(MULTI_FACTS)} ({100*rope_correct/len(MULTI_FACTS):.0f}%)")
    print(f"  Token savings:      {rag_tokens - rope_tokens} tokens ({100*(rag_tokens-rope_tokens)/rag_tokens:.0f}%)")
    print()

    if rope_correct >= 7:
        print("  DECISION GATE G4a: PASS -- RoPE correction >= 7/10")
        print("  Ship RoPECorrectedConcatComposer as default. Skip Phase 30B.")
    elif rope_correct >= 5:
        print("  DECISION GATE G4b: BORDERLINE -- RoPE correction 5-6/10")
        print("  Position helps but insufficient. Proceed to Phase 30B (HKVD).")
    else:
        print("  DECISION GATE G4c: FAIL -- RoPE correction < 5/10")
        print("  Position is not the primary issue. Fall back to hybrid approach.")


if __name__ == "__main__":
    main()
