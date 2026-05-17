# ATDD tests for store_and_link() API.
#
# The agent decides what to link. The engine records the link.
# This is the canonical way to attach new details to existing memories.

import sys
from pathlib import Path

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


def test_store_and_link_creates_bidirectional_link(kps, engine):
    """GIVEN an existing memory,
    WHEN store_and_link(new_detail, existing_pack_id),
    THEN both packs are trace-linked bidirectionally."""
    existing = kps.store("Went to a poetry reading at a bookstore in Pilsen")
    detail = kps.store_and_link("The bookstore is called Casa Azul", existing)

    assert detail != existing
    assert engine.pack_count() == 2
    assert existing in engine.pack_links(detail)
    assert detail in engine.pack_links(existing)


def test_store_and_link_retrieval_finds_both(kps, engine):
    """GIVEN store_and_link(detail, existing),
    WHEN retrieve_with_trace(query matching existing),
    THEN cache contains KV from both packs."""
    existing = kps.store("The pharmacy closes at 8:30pm on Tuesdays")
    kps.store_and_link("The pharmacy is on Elm Avenue", existing)

    cache_single, _, _ = kps.retrieve_and_inject("pharmacy")
    cache_trace, _, _ = kps.retrieve_with_trace("pharmacy", k=1)

    assert cache_trace.get_seq_length() > cache_single.get_seq_length()


def test_store_and_link_multiple_details(kps, engine):
    """GIVEN an existing memory with two details linked to it,
    THEN all three packs are connected via the hub memory."""
    existing = kps.store("Went to a bookstore in Pilsen")
    detail_a = kps.store_and_link("The bookstore is called Casa Azul", existing)
    detail_b = kps.store_and_link("Casa Azul has open mic on Fridays", existing)

    assert engine.pack_count() == 3
    assert existing in engine.pack_links(detail_a)
    assert existing in engine.pack_links(detail_b)
    assert detail_a in engine.pack_links(existing)
    assert detail_b in engine.pack_links(existing)
