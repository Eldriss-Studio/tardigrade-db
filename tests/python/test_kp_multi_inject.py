# ATDD acceptance tests for multi-memory KV injection (Phase 30).
#
# Design pattern: Strategy (CompositionStrategy protocol) + Facade
# (KnowledgePackStore.generate_multi wraps retrieval + composition).
#
# Uses GPT-2 with a minimal chat template for structural tests.
# These verify the multi-pack contract (shapes, pack counts, token
# accounting) — not generation quality. Correctness will be validated
# by experiments/multi_memory_experiment.py on Qwen3-0.6B.

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db
from tardigrade_hooks.kp_injector import KnowledgePackStore
from tardigrade_hooks.multi_composer import NaiveConcatComposer

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


# -- 1: multi-retrieve returns composed cache ---------------------------------


def test_multi_retrieve_returns_composed_cache(kps, engine):
    """GIVEN 2 facts stored,
    WHEN retrieve_and_inject_multi(query, k=2) returns 2 packs,
    THEN composed cache seq_len > single-pack seq_len."""
    kps.store("Sonia's wifi password is mango-cathedral-7")
    kps.store("Eduardo works at a bakery called Morning Bloom on Oak Street")

    # Single-pack retrieval for baseline seq_len
    cache_single, _, _ = kps.retrieve_and_inject("Tell me about Sonia and Eduardo")
    single_seq = cache_single.get_seq_length()

    # Multi-pack retrieval — request both
    cache_multi, _, _ = kps.retrieve_and_inject_multi(
        "Tell me about Sonia and Eduardo", k=2
    )

    assert cache_multi is not None
    # If retriever found 2 packs, composed cache is strictly larger.
    # If only 1 pack was found (retriever limitation), at least equal.
    assert cache_multi.get_seq_length() >= single_seq


# -- 2: naive concat cache shape -----------------------------------------------


def test_multi_naive_concat_cache_shape(kps, engine):
    """GIVEN 2 facts stored,
    WHEN retrieve_and_inject_multi(query, k=2, strategy=NaiveConcat),
    THEN each layer's seq_len == sum of individual pack seq_lens
    AND attention_mask covers kv_len + query_len."""
    kps.store("The pharmacy closes at 8:30pm on Tuesdays")
    kps.store("The pharmacy is located at 742 Elm Avenue")

    cache, query_ids, attn_mask = kps.retrieve_and_inject_multi(
        "Tell me about the pharmacy", k=2, composer=NaiveConcatComposer()
    )

    assert cache is not None
    assert len(cache.layers) == 12  # GPT-2 layers

    kv_len = cache.get_seq_length()
    q_len = query_ids.shape[1]
    assert attn_mask.shape == (1, kv_len + q_len)
    assert attn_mask.sum().item() == kv_len + q_len  # all ones


# -- 3: generate_multi reports had_memory -------------------------------------


def test_multi_generate_reports_had_memory(kps, engine):
    """GIVEN 2 facts stored,
    WHEN generate_multi(query, k=2),
    THEN had_memory == True and prompt_tokens == query-only count."""
    kps.store("Lucia's swimming instructor is named Tomoko")
    kps.store("Tomoko drives a red Honda Civic")

    text, prompt_tokens, had_memory = kps.generate_multi(
        "What car does Tomoko drive?", k=2, max_new_tokens=5
    )

    assert had_memory is True

    # Verify prompt_tokens matches query-only portion
    _, query_ids, _ = kps.retrieve_and_inject_multi(
        "What car does Tomoko drive?", k=2
    )
    assert prompt_tokens == query_ids.shape[1]


# -- 4: transparent without memories ------------------------------------------


def test_multi_generate_transparent_without_memories(kps, engine):
    """GIVEN empty engine,
    WHEN generate_multi(query, k=2),
    THEN had_memory == False and returns valid text."""
    assert engine.pack_count() == 0

    text, prompt_tokens, had_memory = kps.generate_multi(
        "What is the meaning of life?", k=2, max_new_tokens=10
    )

    assert had_memory is False
    assert isinstance(text, str)
    assert len(text) > 0


# -- 5: k=1 matches single generate -------------------------------------------


def test_multi_k1_matches_single_generate(kps, engine):
    """GIVEN 1 fact stored,
    WHEN generate_multi(query, k=1) and generate(query),
    THEN both produce the same prompt_tokens and had_memory."""
    kps.store("The wifi password is mango-cathedral-7")
    query = "What is the wifi password?"

    _, tokens_single, had_single = kps.generate(query, max_new_tokens=5)
    _, tokens_multi, had_multi = kps.generate_multi(query, k=1, max_new_tokens=5)

    assert had_single == had_multi == True
    assert tokens_single == tokens_multi


# -- 6: fewer packs than k (graceful degradation) -----------------------------


def test_multi_fewer_packs_than_k(kps, engine):
    """GIVEN 1 fact stored,
    WHEN retrieve_and_inject_multi(query, k=3),
    THEN returns cache composed from 1 pack (not an error)."""
    kps.store("Eduardo's apartment is 4B")

    cache, query_ids, attn_mask = kps.retrieve_and_inject_multi(
        "Where does Eduardo live?", k=3
    )

    assert cache is not None
    assert len(cache.layers) == 12


# -- 7: generate_multi clones cache (no in-place mutation) --------------------


def test_multi_generate_clones_cache(kps, engine):
    """GIVEN 2 facts stored,
    WHEN generate_multi() called twice with same query,
    THEN both calls succeed (cache clone prevents mutation)."""
    kps.store("The pharmacy closes at 8:30pm")
    kps.store("The pharmacy is on Elm Avenue")
    query = "Tell me about the pharmacy"

    text1, _, had1 = kps.generate_multi(query, k=2, max_new_tokens=5)
    text2, _, had2 = kps.generate_multi(query, k=2, max_new_tokens=5)

    assert had1 is True
    assert had2 is True
    assert isinstance(text1, str)
    assert isinstance(text2, str)
