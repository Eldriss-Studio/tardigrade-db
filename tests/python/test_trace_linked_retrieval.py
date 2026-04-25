# ATDD tests for trace-linked multi-hop retrieval (Phase 31).
#
# Design pattern: Graph Traversal (follow Python-side trace links
# from retrieved packs to discover related packs).
#
# Uses GPT-2 for structural tests. Correctness validated by
# experiments/multi_memory_trace_experiment.py on Qwen3-0.6B.

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db
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


# -- 1: store_linked creates retrievable packs --------------------------------


def test_store_linked_creates_packs(kps, engine):
    """GIVEN store_linked(["fact A", "fact B"]),
    THEN pack_count == 2 and both pack_ids returned."""
    pack_ids = kps.store_linked([
        "Lucia's instructor is Tomoko",
        "Tomoko drives a Honda Civic",
    ])

    assert len(pack_ids) == 2
    assert engine.pack_count() == 2
    assert pack_ids[0] != pack_ids[1]


# -- 2: store_linked creates trace links --------------------------------------


def test_store_linked_creates_bidirectional_links(kps):
    """GIVEN store_linked(["fact A", "fact B"]),
    THEN both packs are linked to each other."""
    pack_ids = kps.store_linked(["fact A about Sonia", "fact B about Eduardo"])

    assert pack_ids[1] in kps.engine.pack_links(pack_ids[0])
    assert pack_ids[0] in kps.engine.pack_links(pack_ids[1])


# -- 3: retrieve_with_trace finds linked packs --------------------------------


def test_retrieve_with_trace_finds_linked_pack(kps, engine):
    """GIVEN 2 linked facts,
    WHEN retrieve_with_trace(query matching fact A),
    THEN cache contains KV from BOTH packs."""
    kps.store_linked([
        "The pharmacy closes at 8:30pm on Tuesdays",
        "The pharmacy is located at 742 Elm Avenue",
    ])

    cache_single, _, _ = kps.retrieve_and_inject("Tell me about the pharmacy")
    single_seq = cache_single.get_seq_length()

    cache_trace, _, _ = kps.retrieve_with_trace("Tell me about the pharmacy", k=1)
    trace_seq = cache_trace.get_seq_length()

    assert trace_seq > single_seq


# -- 4: unlinked facts no trace hop -------------------------------------------


def test_unlinked_facts_no_trace_hop(kps, engine):
    """GIVEN 2 facts stored separately (NOT linked),
    WHEN retrieve_with_trace(query, k=1),
    THEN only 1 pack returned (no trace links to follow)."""
    kps.store("The pharmacy closes at 8:30pm", auto_link=False)
    kps.store("Eduardo works at Morning Bloom bakery", auto_link=False)

    cache_single, _, _ = kps.retrieve_and_inject("Tell me about the pharmacy")
    cache_trace, _, _ = kps.retrieve_with_trace("Tell me about the pharmacy", k=1)

    assert cache_single.get_seq_length() == cache_trace.get_seq_length()


# -- 5: generate_with_trace returns valid output -------------------------------


def test_generate_with_trace_returns_valid_output(kps, engine):
    """GIVEN 2 linked facts,
    WHEN generate_with_trace(query),
    THEN had_memory == True and returns valid text."""
    kps.store_linked([
        "Sonia's wifi password is mango-cathedral-7",
        "Sonia lives in apartment 4B",
    ])

    text, tokens, had_memory = kps.generate_with_trace(
        "What is Sonia's wifi password?", k=1, max_new_tokens=5
    )

    assert had_memory is True
    assert isinstance(text, str)
    assert len(text) > 0
