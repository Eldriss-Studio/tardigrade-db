# ATDD acceptance tests for Path 1: synthetic-fact KV injection.
#
# Design pattern: Facade (KnowledgePackStore) + Strategy (text RAG baseline).
#
# Tests 1-6 use GPT-2 for fast structural validation (CI-safe, CPU-only).
# Test 7 is the gate test: runs the full experiment on Qwen3-0.6B.

import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

import tardigrade_db
from synthetic_facts_corpus import SYNTHETIC_FACTS
from tardigrade_hooks.kp_injector import KnowledgePackStore

CHAT_TEMPLATE = '{% for message in messages %}{{ message["content"] }}{% endfor %}'


@pytest.fixture
def gpt2():
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()
    return model


@pytest.fixture
def tokenizer():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.chat_template = CHAT_TEMPLATE
    tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def engine(tmp_path):
    return tardigrade_db.Engine(str(tmp_path))


@pytest.fixture
def kps(engine, gpt2, tokenizer):
    return KnowledgePackStore(engine, gpt2, tokenizer, owner=1)


# -- 1: corpus answers are unique gibberish -----------------------------------


def test_corpus_answers_are_unique_gibberish():
    """Each expected answer is unique, >= 4 chars, and contains non-alpha."""
    answers = [expected for _, _, expected in SYNTHETIC_FACTS]
    assert len(answers) == len(set(answers)), "duplicate answers in corpus"
    for ans in answers:
        assert len(ans) >= 4, f"answer too short: {ans}"
        assert not ans.isalpha(), f"answer is pure alpha (no gibberish markers): {ans}"


# -- 2: answers are not single tokens -----------------------------------------


def test_corpus_answers_not_single_token(tokenizer):
    """No expected answer should encode as a single token."""
    for _, _, expected in SYNTHETIC_FACTS:
        token_ids = tokenizer.encode(expected)
        assert len(token_ids) > 1, f"'{expected}' is a single token"


# -- 3: store and retrieve structural -----------------------------------------


def test_kv_store_and_retrieve_structural(kps, gpt2):
    """GIVEN all 10 synthetic facts stored,
    WHEN retrieve_and_inject(query) for each,
    THEN cache is non-None with correct layer count."""
    for fact, query, _ in SYNTHETIC_FACTS:
        kps.store(fact)

    for fact, query, _ in SYNTHETIC_FACTS[:3]:
        cache, query_ids, attn_mask = kps.retrieve_and_inject(query)
        assert cache is not None, f"no cache for query: {query}"
        assert len(cache) == gpt2.config.n_layer


# -- 4: injection returns with memory flag ------------------------------------


def test_kv_injection_returns_with_memory_flag(kps):
    """GIVEN a stored fact, WHEN generate(query), THEN had_memory is True."""
    fact, query, _ = SYNTHETIC_FACTS[0]
    kps.store(fact)
    text, prompt_tokens, had_memory = kps.generate(
        query, max_new_tokens=20, do_sample=False
    )
    assert had_memory is True
    assert len(text) > 0


# -- 5: text RAG baseline produces output --------------------------------------


def test_text_rag_baseline_produces_output(gpt2, tokenizer):
    """Text RAG helper returns a non-empty string."""
    fact, query, _ = SYNTHETIC_FACTS[0]
    messages = [
        {"role": "system", "content": f"Use this fact to answer: {fact}"},
        {"role": "user", "content": query},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(formatted, return_tensors="pt")
    prompt_tokens = input_ids.shape[1]

    with torch.no_grad():
        out = gpt2.generate(
            input_ids, max_new_tokens=30, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(out[0][prompt_tokens:], skip_special_tokens=True)
    assert len(response.strip()) > 0


# -- 6: experiment result schema -----------------------------------------------


def test_experiment_result_schema():
    """run_experiment() returns dict with required keys."""
    from synthetic_kv_injection_experiment import run_experiment

    result = run_experiment(facts=SYNTHETIC_FACTS[:1], model_name="gpt2")
    required_keys = {"rag_correct", "kv_correct", "rag_total", "kv_total",
                     "token_savings", "per_fact_results"}
    assert required_keys.issubset(result.keys())
    assert len(result["per_fact_results"]) == 1
    pf = result["per_fact_results"][0]
    assert "rag" in pf and "kv" in pf
    assert "correct" in pf["rag"] and "correct" in pf["kv"]


# -- 7: the gate — injection recall >= 70% of text RAG -----------------------


@pytest.mark.slow
def test_injection_recall_gate():
    """KV injection must recall >= 70% of what text RAG recalls.
    Runs full experiment on Qwen3-0.6B. Skip if model unavailable."""
    try:
        from transformers import AutoModelForCausalLM
        AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float32)
    except Exception:
        pytest.skip("Qwen3-0.6B not available")

    from synthetic_kv_injection_experiment import run_experiment

    result = run_experiment()
    rag = result["rag_correct"]
    kv = result["kv_correct"]

    assert rag >= 7, f"text RAG baseline too low ({rag}/10) — corpus may be too hard"
    ratio = kv / max(rag, 1)
    assert ratio >= 0.70, (
        f"KV injection recall {kv}/{result['kv_total']} is only {ratio:.0%} "
        f"of text RAG {rag}/{result['rag_total']} — below 70% gate"
    )
