#!/usr/bin/env python3
"""Phase 30B: Oracle injection + higher-k retrieval diagnostic.

Two tests in one script:

1. ORACLE INJECTION: bypass retrieval, manually inject the correct packs.
   If the model answers correctly -> injection works, fix retrieval.
   If not -> injection is broken regardless of retrieval.

2. HIGHER-K RETRIEVAL: test if missing facts appear at k=10 or k=20.
   If they do -> just increase k, no Trace needed.
   If not -> need Trace-linked retrieval.

Usage:
    source .venv/bin/activate
    python experiments/multi_memory_oracle_injection.py
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


def load_model():
    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    return model, tokenizer


def oracle_generate(kps, query_text, correct_pack_ids, composer):
    """Inject specific packs by ID, bypassing retrieval scoring."""
    query_input = kps.tokenizer.encode(query_text, return_tensors="pt")
    with torch.no_grad():
        query_out = kps.model(query_input, output_hidden_states=True)
    hidden = query_out.hidden_states[kps.query_layer][0]
    h_tokens = hidden[1:].numpy().astype(np.float32)
    query_key = encode_per_token(h_tokens, kps.hidden_size)

    # Get ALL packs, filter to correct ones
    all_packs = kps.engine.mem_read_pack(query_key, 50, kps.owner)
    correct_packs = [p for p in all_packs if p["pack_id"] in correct_pack_ids]

    if not correct_packs:
        return "NO PACKS FOUND", 0, False

    cache = composer.compose(
        correct_packs, kps.num_kv_heads, kps.head_dim, kps.kv_dim, kps.n_layers
    )

    # Build query_ids (chat template continuation)
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

    # Clone and generate
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


def main():
    model, tokenizer = load_model()

    tmpdir = tempfile.mkdtemp()
    engine = tardigrade_db.Engine(tmpdir)
    kps = KnowledgePackStore(engine, model, tokenizer, owner=1)

    print("Storing all facts...")
    for entry in MULTI_FACTS:
        for fact in entry["facts"]:
            kps.store(fact)

    pack_to_fact = dict(kps._text_registry)
    fact_to_pack = {v: k for k, v in pack_to_fact.items()}
    print(f"Stored {engine.pack_count()} packs")

    # ================================================================
    # TEST 1: Oracle injection
    # ================================================================
    print()
    print("=" * 70)
    print("TEST 1: ORACLE INJECTION (correct packs, bypass retrieval)")
    print("=" * 70)
    print()

    composer = NaiveConcatComposer()
    oracle_correct = 0

    for i, entry in enumerate(MULTI_FACTS):
        query = entry["query"] + " /no_think"
        correct_pack_ids = {fact_to_pack[f] for f in entry["facts"]}

        text, tokens, had_memory = oracle_generate(
            kps, query, correct_pack_ids, composer
        )

        correct = entry["expected"].lower() in text.lower()
        oracle_correct += int(correct)
        status = "PASS" if correct else "MISS"
        print(f"  [{status}] Q{i+1}: {entry['query'][:55]}...")
        print(f"    Packs injected: {sorted(correct_pack_ids)}")
        if not correct:
            print(f"    Expected: {entry['expected']}")
            print(f"    Got: {text[:80]}")

    print(f"\nOracle injection: {oracle_correct}/{len(MULTI_FACTS)} correct")

    # ================================================================
    # TEST 2: Higher-k retrieval
    # ================================================================
    print()
    print("=" * 70)
    print("TEST 2: HIGHER-K RETRIEVAL")
    print("=" * 70)
    print()

    for test_k in [2, 5, 10, 20]:
        all_found = 0
        for i, entry in enumerate(MULTI_FACTS):
            query = entry["query"]
            expected_pack_ids = {fact_to_pack[f] for f in entry["facts"]}

            query_input = tokenizer.encode(query, return_tensors="pt")
            with torch.no_grad():
                query_out = model(query_input, output_hidden_states=True)
            hidden = query_out.hidden_states[kps.query_layer][0]
            h_tokens = hidden[1:].numpy().astype(np.float32)
            query_key = encode_per_token(h_tokens, kps.hidden_size)

            packs = engine.mem_read_pack(query_key, test_k, 1)
            retrieved_ids = {p["pack_id"] for p in packs}
            missing = expected_pack_ids - retrieved_ids

            if len(missing) == 0:
                all_found += 1

        print(f"  k={test_k:2d}: {all_found}/{len(MULTI_FACTS)} queries find ALL needed packs")

    # ================================================================
    # SUMMARY
    # ================================================================
    print()
    print("=" * 70)
    print("DECISION MATRIX")
    print("=" * 70)

    if oracle_correct >= 7:
        print(f"  G6: Oracle injection {oracle_correct}/10 -- PASS")
        print("    Injection works with correct packs. The problem is retrieval.")
    elif oracle_correct >= 5:
        print(f"  G6: Oracle injection {oracle_correct}/10 -- BORDERLINE")
    else:
        print(f"  G6: Oracle injection {oracle_correct}/10 -- FAIL")
        print("    Injection broken even with correct packs.")


if __name__ == "__main__":
    main()
