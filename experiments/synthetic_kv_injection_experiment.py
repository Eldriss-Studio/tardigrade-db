#!/usr/bin/env python3
"""Path 1: Verify KV injection transfers truly novel knowledge.

Uses fully synthetic gibberish facts that cannot exist in any training data.
Compares text RAG (fact pasted in prompt) vs KV injection (stored KV tensors
injected as past_key_values via KnowledgePackStore).

Success criterion: KV injection recalls >= 70% of what text RAG recalls.

Usage:
    source .venv/bin/activate
    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop -m crates/tdb-python/Cargo.toml
    python experiments/synthetic_kv_injection_experiment.py
"""

import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer

import tardigrade_db
from synthetic_facts_corpus import SYNTHETIC_FACTS
from tardigrade_hooks.kp_injector import KnowledgePackStore

GPT2_CHAT_TEMPLATE = '{% for message in messages %}{{ message["content"] }}{% endfor %}'


def load_model(model_name="Qwen/Qwen3-0.6B"):
    if model_name == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.chat_template = GPT2_CHAT_TEMPLATE
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32, attn_implementation="eager"
        )
    model.eval()
    return model, tokenizer


def _strip_thinking(text):
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


def run_text_rag(model, tokenizer, fact, query, expected):
    messages = [
        {"role": "system", "content": f"Use this fact to answer: {fact}"},
        {"role": "user", "content": query},
    ]

    enable_thinking = {}
    if hasattr(tokenizer, "apply_chat_template"):
        import inspect
        sig = inspect.signature(tokenizer.apply_chat_template)
        if "enable_thinking" in sig.parameters:
            enable_thinking["enable_thinking"] = False

    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **enable_thinking
    )
    input_ids = tokenizer.encode(formatted, return_tensors="pt")
    prompt_tokens = input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=100, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(out[0][prompt_tokens:], skip_special_tokens=True).strip()
    response = _strip_thinking(response)
    correct = expected.lower() in response.lower()
    return {"correct": correct, "response": response, "prompt_tokens": prompt_tokens}


def run_kv_injection(model, tokenizer, fact, query, expected):
    db_dir = tempfile.mkdtemp(prefix="tardigrade_synth_")
    try:
        engine = tardigrade_db.Engine(db_dir)
        kps = KnowledgePackStore(engine, model, tokenizer, owner=1)
        kps.store(fact)

        query_with_nothink = query + " /no_think"
        text, prompt_tokens, had_memory = kps.generate(
            query_with_nothink, max_new_tokens=100, do_sample=False,
        )
        text = _strip_thinking(text)
        correct = expected.lower() in text.lower()
        return {
            "correct": correct, "response": text,
            "prompt_tokens": prompt_tokens, "had_memory": had_memory,
        }
    finally:
        shutil.rmtree(db_dir, ignore_errors=True)


def run_experiment(facts=None, model_name="Qwen/Qwen3-0.6B"):
    if facts is None:
        facts = SYNTHETIC_FACTS

    model, tokenizer = load_model(model_name)
    per_fact = []

    for fact, query, expected in facts:
        rag = run_text_rag(model, tokenizer, fact, query, expected)
        kv = run_kv_injection(model, tokenizer, fact, query, expected)
        per_fact.append({
            "fact": fact, "query": query, "expected": expected,
            "rag": rag, "kv": kv,
        })

    rag_correct = sum(1 for p in per_fact if p["rag"]["correct"])
    kv_correct = sum(1 for p in per_fact if p["kv"]["correct"])
    rag_tokens = sum(p["rag"]["prompt_tokens"] for p in per_fact)
    kv_tokens = sum(p["kv"]["prompt_tokens"] for p in per_fact)

    return {
        "rag_correct": rag_correct,
        "kv_correct": kv_correct,
        "rag_total": len(facts),
        "kv_total": len(facts),
        "token_savings": rag_tokens - kv_tokens,
        "per_fact_results": per_fact,
    }


def main():
    print("=" * 70)
    print("Path 1: Synthetic-Fact KV Injection Verification")
    print(f"Corpus: {len(SYNTHETIC_FACTS)} fully synthetic gibberish facts")
    print("=" * 70)

    t0 = time.time()
    result = run_experiment()
    elapsed = time.time() - t0

    n = result["rag_total"]
    rc = result["rag_correct"]
    kc = result["kv_correct"]
    savings = result["token_savings"]

    print(f"\n{'#':<4} {'Expected':<25} {'RAG':>5} {'KV':>5} {'Saved':>6}")
    print(f"{'-' * 50}")
    for i, pf in enumerate(result["per_fact_results"], 1):
        rm = "Y" if pf["rag"]["correct"] else "N"
        km = "Y" if pf["kv"]["correct"] else "N"
        saved = pf["rag"]["prompt_tokens"] - pf["kv"]["prompt_tokens"]
        print(f"{i:<4} {pf['expected'][:24]:<25} {rm:>5} {km:>5} {saved:>6}")

    # Show generated responses for debugging
    print(f"\n{'=' * 70}")
    print("  RESPONSES (first 80 chars)")
    print(f"{'=' * 70}")
    for i, pf in enumerate(result["per_fact_results"], 1):
        rm = "Y" if pf["rag"]["correct"] else "N"
        km = "Y" if pf["kv"]["correct"] else "N"
        print(f"\n  [{i}] {pf['expected']}")
        print(f"  RAG ({rm}): {pf['rag']['response'][:80]}")
        print(f"  KV  ({km}): {pf['kv']['response'][:80]}")

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Metric':<25} {'Text RAG':>12} {'KV Inject':>12}")
    print(f"  {'-' * 51}")
    print(f"  {'Correct':<25} {rc:>10}/{n} {kc:>10}/{n}")
    print(f"  {'Token savings':<25} {'':>12} {savings:>12}")
    print(f"  {'Time (s)':<25} {elapsed:>12.1f}")

    ratio = kc / max(rc, 1)
    print(f"\n  Recall ratio: {kc}/{max(rc,1)} = {ratio:.0%}")

    if ratio >= 0.70 and rc >= 7:
        verdict = "PASS — KV injection transfers truly novel knowledge"
    elif ratio >= 0.50:
        verdict = "BORDERLINE — injection works partially, needs investigation"
    else:
        verdict = "FAIL — injection does not reliably transfer synthetic knowledge"

    print(f"\n  {verdict}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
