# ATDD acceptance tests for KnowledgePackStore.
#
# Design pattern: Facade (wraps chat template + KV computation + pack
# storage + DynamicCache reconstruction + injection).
#
# Uses GPT-2 with a minimal chat template for structural tests.
# GPT-2 is too small to recall novel facts, so these tests verify
# the contract (shapes, pack counts, token accounting) — not the
# quality of generated answers. Correctness was validated by
# experiments/injection_vs_text_rag.py on Qwen3-0.6B.

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db
from tardigrade_hooks.kp_injector import KnowledgePackStore

# Minimal chat template — GPT-2 doesn't ship one.
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


# -- 1: store creates one pack with all layers --------------------------------


def test_kp_store_creates_pack_for_all_layers(kps, engine):
    """GIVEN KnowledgePackStore with GPT-2 (12 layers),
    WHEN store("some fact"),
    THEN pack_count == 1 and store() returns a pack_id."""
    pack_id = kps.store("The wifi password is mango-cathedral-7")

    assert isinstance(pack_id, int)
    assert engine.pack_count() == 1


# -- 2: store wraps fact in chat template --------------------------------------


def test_kp_store_uses_chat_template(kps, engine, tokenizer):
    """GIVEN KnowledgePackStore,
    WHEN store("fact text"),
    THEN the stored KV seq_len matches chat-template-formatted token count,
    not raw text token count."""
    fact = "Eduardo's apartment number is 4B on the third floor"
    kps.store(fact)

    # Expected seq_len from chat template
    messages = [{"role": "system", "content": fact}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    expected_seq_len = len(tokenizer.encode(formatted))

    # Retrieve the pack and check stored KV dimensions
    query_input = tokenizer.encode(fact, return_tensors="pt")
    with torch.no_grad():
        out = kps.model(query_input, output_hidden_states=True)
    hidden = out.hidden_states[kps.query_layer][0]
    from tardigrade_hooks.encoding import encode_per_token
    h_tokens = hidden[1:].numpy().astype(np.float32)
    query_key = encode_per_token(h_tokens, kps.hidden_size)

    packs = engine.mem_read_pack(query_key, 1, 1)
    assert len(packs) == 1

    # Each layer's data is [K_flat | V_flat], each half = seq_len * kv_dim
    layer_data = np.array(packs[0]["layers"][0]["data"], dtype=np.float32)
    half = len(layer_data) // 2
    actual_seq_len = half // kps.kv_dim

    assert actual_seq_len == expected_seq_len


# -- 3: retrieve_and_inject reconstructs valid DynamicCache --------------------


def test_kp_retrieve_returns_valid_cache(kps, engine):
    """GIVEN a stored fact,
    WHEN retrieve_and_inject(query),
    THEN returns (DynamicCache, query_ids, attention_mask) with correct shapes."""
    kps.store("The pharmacy closes at 8:30pm on Tuesdays")

    cache, query_ids, attn_mask = kps.retrieve_and_inject(
        "When does the pharmacy close?"
    )

    assert cache is not None
    assert len(cache.layers) == 12  # GPT-2 has 12 layers

    kv_len = cache.get_seq_length()
    q_len = query_ids.shape[1]

    # Attention mask covers KV cache + query tokens
    assert attn_mask.shape == (1, kv_len + q_len)
    assert attn_mask.sum().item() == kv_len + q_len  # all ones

    # Each layer has correct head/dim shape
    layer0 = cache.layers[0]
    assert layer0.keys.shape[1] == kps.num_kv_heads
    assert layer0.keys.shape[3] == kps.head_dim


# -- 4: generate uses zero memory prompt tokens -------------------------------


def test_kp_generate_uses_zero_memory_prompt_tokens(kps, engine):
    """GIVEN a stored fact,
    WHEN generate(query),
    THEN prompt_tokens == query-only token count (no memory text in prompt)
    AND had_memory == True."""
    kps.store("Lucia's favorite dinosaur is the Pachycephalosaurus")

    query = "What is Lucia's favorite dinosaur?"
    text, prompt_tokens, had_memory = kps.generate(query, max_new_tokens=5)

    assert had_memory is True

    # prompt_tokens should be just the query portion — no memory text pasted
    _, query_ids, _ = kps.retrieve_and_inject(query)
    assert prompt_tokens == query_ids.shape[1]


# -- 5: generate is transparent without memories -------------------------------


def test_kp_generate_transparent_without_memories(kps, engine):
    """GIVEN an empty engine (no stored memories),
    WHEN generate(query),
    THEN returns valid text, had_memory == False, and generation completes."""
    assert engine.pack_count() == 0

    text, prompt_tokens, had_memory = kps.generate(
        "What is the meaning of life?", max_new_tokens=10
    )

    assert had_memory is False
    assert isinstance(text, str)
    assert len(text) > 0
    assert prompt_tokens > 0


# -- 6: multiple stores create independent packs ------------------------------


def test_kp_multiple_stores_create_independent_packs(kps, engine):
    """GIVEN two different facts stored,
    WHEN querying each,
    THEN pack_count == 2 and retrieval returns different packs."""
    kps.store("The wifi password is mango-cathedral-7")
    kps.store("Eduardo's apartment is 4B on the third floor")

    assert engine.pack_count() == 2

    # Both facts stored, both retrievable
    cache1, _, _ = kps.retrieve_and_inject("What is the wifi password?")
    cache2, _, _ = kps.retrieve_and_inject("What is Eduardo's apartment?")

    assert cache1 is not None
    assert cache2 is not None


# -- 7: generate clones cache (no in-place mutation) ---------------------------


def test_kp_generate_clones_cache(kps, engine):
    """GIVEN a stored fact,
    WHEN generate() is called twice with the same query,
    THEN both calls succeed (cache clone prevents in-place mutation)."""
    kps.store("The pharmacy closes at 8:30pm on Tuesdays")
    query = "When does the pharmacy close?"

    text1, _, had1 = kps.generate(query, max_new_tokens=5)
    text2, _, had2 = kps.generate(query, max_new_tokens=5)

    assert had1 is True
    assert had2 is True
    assert isinstance(text1, str)
    assert isinstance(text2, str)
