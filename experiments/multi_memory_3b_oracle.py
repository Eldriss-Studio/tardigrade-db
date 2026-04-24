#!/usr/bin/env python3
"""Does a bigger model handle multi-memory KV injection better?

Runs oracle injection (correct packs) on Qwen2.5-3B (5x larger)
and compares to Qwen3-0.6B. Tests whether model capacity affects
cross-entity attention over independently-computed KV caches.

Usage:
    source .venv/bin/activate
    python experiments/multi_memory_3b_oracle.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "python"))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import tardigrade_db
from tardigrade_hooks.kp_injector import KnowledgePackStore
from tardigrade_hooks.encoding import encode_per_token
from tardigrade_hooks.multi_composer import NaiveConcatComposer
from multi_memory_corpus import MULTI_FACTS


def load_model(model_name):
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


def oracle_generate(kps, query_text, correct_pack_ids, composer):
    query_input = kps.tokenizer.encode(query_text, return_tensors="pt")
    with torch.no_grad():
        query_out = kps.model(query_input, output_hidden_states=True)
    hidden = query_out.hidden_states[kps.query_layer][0]
    h_tokens = hidden[1:].numpy().astype(np.float32)
    query_key = encode_per_token(h_tokens, kps.hidden_size)

    all_packs = kps.engine.mem_read_pack(query_key, 50, kps.owner)
    correct_packs = [p for p in all_packs if p["pack_id"] in correct_pack_ids]

    if not correct_packs:
        return "NO PACKS FOUND", 0, False

    cache = composer.compose(
        correct_packs, kps.num_kv_heads, kps.head_dim, kps.kv_dim, kps.n_layers
    )

    fact_messages = [{"role": "system", "content": "placeholder"}]
    fact_fmt = kps.tokenizer.apply_chat_template(
        fact_messages, tokenize=False, add_generation_prompt=False
    )
    messages = [
        {"role": "system", "content": "placeholder"},
        {"role": "user", "content": query_text},
    ]
    full_fmt = kps.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_ids = kps.tokenizer.encode(full_fmt, return_tensors="pt")
    fact_len = len(kps.tokenizer.encode(fact_fmt))
    query_ids = full_ids[:, fact_len:]

    kv_len = cache.get_seq_length()
    q_len = query_ids.shape[1]
    attention_mask = torch.ones(1, kv_len + q_len, dtype=torch.long)

    clone = DynamicCache()
    for li in range(len(cache.layers)):
        layer = cache.layers[li]
        clone.update(layer.keys.clone(), layer.values.clone(), li)

    with torch.no_grad():
        out = kps.model.generate(
            query_ids,
            past_key_values=clone,
            attention_mask=attention_mask,
            max_new_tokens=100,
            do_sample=False,
        )

    text = kps.tokenizer.decode(out[0][q_len:], skip_special_tokens=True).strip()
    return text, q_len, True


def run_model(model_name):
    model, tokenizer = load_model(model_name)
    cfg = model.config

    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    print(f"  Layers: {cfg.num_hidden_layers}, Heads: {cfg.num_attention_heads}, "
          f"KV Heads: {kv_heads}, Hidden: {cfg.hidden_size}")
    print()

    # Text RAG baseline
    print("--- TEXT RAG ---")
    rag_correct = 0
    for i, entry in enumerate(MULTI_FACTS):
        correct, response, tokens = run_text_rag(model, tokenizer, entry)
        rag_correct += int(correct)
        status = "PASS" if correct else "MISS"
        print(f"  [{status}] Q{i+1}: {entry['query'][:50]}...")
        if not correct:
            print(f"    Expected: {entry['expected']}")
            print(f"    Got: {response[:80]}")
    print(f"  Text RAG: {rag_correct}/{len(MULTI_FACTS)}")

    # Oracle injection
    print()
    print("--- ORACLE INJECTION ---")
    tmpdir = tempfile.mkdtemp()
    engine = tardigrade_db.Engine(tmpdir)
    kps = KnowledgePackStore(engine, model, tokenizer, owner=1)

    print("  Storing facts...")
    for entry in MULTI_FACTS:
        for fact in entry["facts"]:
            kps.store(fact)
    print(f"  Stored {engine.pack_count()} packs")

    fact_to_pack = {v: k for k, v in kps._text_registry.items()}
    composer = NaiveConcatComposer()
    oracle_correct = 0

    for i, entry in enumerate(MULTI_FACTS):
        query = entry["query"] + " /no_think"
        correct_pack_ids = {fact_to_pack[f] for f in entry["facts"]}
        text, tokens, had_memory = oracle_generate(kps, query, correct_pack_ids, composer)
        correct = entry["expected"].lower() in text.lower()
        oracle_correct += int(correct)
        status = "PASS" if correct else "MISS"
        print(f"  [{status}] Q{i+1}: {entry['query'][:50]}...")
        if not correct:
            print(f"    Expected: {entry['expected']}")
            print(f"    Got: {text[:80]}")
    print(f"  Oracle injection: {oracle_correct}/{len(MULTI_FACTS)}")

    # Cleanup
    del model
    del kps
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()

    return rag_correct, oracle_correct


def main():
    results = {}

    for model_name in ["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-3B"]:
        rag, oracle = run_model(model_name)
        results[model_name] = (rag, oracle)

    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    for name, (rag, oracle) in results.items():
        print(f"  {name:25s}  Text RAG: {rag}/10  Oracle Injection: {oracle}/10")


if __name__ == "__main__":
    main()
