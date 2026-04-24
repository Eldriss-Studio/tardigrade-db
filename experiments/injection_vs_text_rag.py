#!/usr/bin/env python3
# Phase 25: Injection vs Text RAG -- the existential test.
#
# Same novel facts, same queries, two paths:
#   A) Text RAG: paste memory text into prompt, generate
#   B) KV Injection: inject stored KV tensors into cache, generate
#
# Uses GPT-2 (proven injection pipeline from Phase 18).

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(".").resolve() / "python"))
sys.path.insert(0, str(Path(".").resolve() / "experiments"))

import tardigrade_db
from novel_facts_corpus import NOVEL_FACTS
from tardigrade_hooks.hf_hook import HuggingFaceHook
from tardigrade_hooks.injector import MemoryInjector
from tardigrade_hooks.position import AbsolutePositionEncoder


def run_text_rag(model, tokenizer, facts):
    results = []
    for memory_text, query, expected in facts:
        prompt = f"Based on this information: {memory_text}\n\nQuestion: {query}\nAnswer:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        prompt_tokens = input_ids.shape[1]

        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=30, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(out[0][prompt_tokens:], skip_special_tokens=True)
        correct = expected.lower() in generated.lower()
        results.append({
            "query": query, "expected": expected,
            "generated": generated.strip(), "correct": correct,
            "prompt_tokens": prompt_tokens,
        })
    return results


def run_kv_injection(model, tokenizer, facts):
    results = []
    for memory_text, query, expected in facts:
        db_dir = tempfile.mkdtemp(prefix="tardigrade_inject_")
        engine = tardigrade_db.Engine(db_dir)
        hook = HuggingFaceHook(engine, owner=1, norm_threshold=0.0)

        # Store memory KV cache across all layers.
        mem_inputs = tokenizer(memory_text, return_tensors="pt")
        with torch.no_grad():
            mem_out = model(**mem_inputs, output_hidden_states=True)

        for layer_idx in range(model.config.n_layer):
            h = mem_out.hidden_states[layer_idx + 1].numpy()
            decision = hook.on_generate(layer=layer_idx, hidden_states=h)
            if decision.should_write:
                engine.mem_write(
                    1, layer_idx, decision.key, decision.value,
                    decision.salience, None,
                )

        # Inject and generate.
        injector = MemoryInjector(
            model=model, engine=engine, owner=1,
            position_encoder=AbsolutePositionEncoder(),
        )

        query_ids = tokenizer.encode(query, return_tensors="pt")
        prompt_tokens = query_ids.shape[1]

        with torch.no_grad():
            out = injector.generate(
                query_ids, max_new_tokens=30, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(out[0][prompt_tokens:], skip_special_tokens=True)
        correct = expected.lower() in generated.lower()
        results.append({
            "query": query, "expected": expected,
            "generated": generated.strip(), "correct": correct,
            "prompt_tokens": prompt_tokens,
        })
        shutil.rmtree(db_dir)
    return results


def main():
    print("=" * 70)
    print("Phase 25: Injection vs Text RAG")
    print(f"Novel facts: {len(NOVEL_FACTS)} | Model: GPT-2")
    print("=" * 70)

    print("\n  Loading GPT-2...", end=" ", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()
    print("OK")

    # Path A
    print("\n--- TEXT RAG ---\n")
    text_r = run_text_rag(model, tokenizer, NOVEL_FACTS)
    for r in text_r:
        m = "Y" if r["correct"] else "N"
        print(f"  {m}  [{r['prompt_tokens']:>3} tok] {r['query'][:45]}")
        print(f"       -> \"{r['generated'][:60]}\"")

    # Path B
    print("\n--- KV INJECTION ---\n")
    inject_r = run_kv_injection(model, tokenizer, NOVEL_FACTS)
    for r in inject_r:
        m = "Y" if r["correct"] else "N"
        print(f"  {m}  [{r['prompt_tokens']:>3} tok] {r['query'][:45]}")
        print(f"       -> \"{r['generated'][:60]}\"")

    # Summary
    tc = sum(1 for r in text_r if r["correct"])
    ic = sum(1 for r in inject_r if r["correct"])
    tt = sum(r["prompt_tokens"] for r in text_r)
    it = sum(r["prompt_tokens"] for r in inject_r)
    n = len(NOVEL_FACTS)

    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")
    print(f"\n  {'Metric':<25} {'Text RAG':>12} {'Injection':>12}")
    print(f"  {'-' * 51}")
    print(f"  {'Correct':<25} {tc:>10}/{n} {ic:>10}/{n}")
    print(f"  {'Total prompt tokens':<25} {tt:>12} {it:>12}")
    print(f"  {'Avg tokens/query':<25} {tt//n:>12} {it//n:>12}")
    print(f"  {'Token savings':<25} {'':>12} {tt - it:>12}")

    print(f"\n  -- Per-Fact --\n")
    print(f"  {'Expected':<25} {'Text':>5} {'Inject':>7} {'Saved':>6}")
    print(f"  {'-' * 45}")
    for t, i in zip(text_r, inject_r):
        tm = "Y" if t["correct"] else "N"
        im = "Y" if i["correct"] else "N"
        print(f"  {t['expected'][:23]:<25} {tm:>5} {im:>7} {t['prompt_tokens'] - i['prompt_tokens']:>6}")

    print(f"\n{'=' * 70}")
    if ic >= 7 and (tt - it) >= 30:
        verdict = "KV INJECTION WORKS -- recalls facts AND saves tokens"
    elif ic >= 5:
        verdict = "PARTIAL -- injection recalls some facts, needs improvement"
    else:
        verdict = "INJECTION DOES NOT RELIABLY TRANSFER KNOWLEDGE"
    print(f"  {verdict}")
    print(f"  Text RAG: {tc}/{n} | Injection: {ic}/{n} | Tokens saved: {tt - it}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
