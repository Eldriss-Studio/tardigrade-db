# ATDD acceptance tests for real-KV hook (Issue #9).
#
# Design pattern: Adapter (translates HF DynamicCache into TardigradeDB API).
# Dual-store: mean-pooled K as search index, flattened K+V as injection payload.

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db
from tardigrade_hooks.hf_kv_hook import HuggingFaceKVHook
from tardigrade_hooks.hook import MemoryCellHandle, WriteDecision


@pytest.fixture
def gpt2():
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()
    return model


@pytest.fixture
def tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")


@pytest.fixture
def engine(tmp_path):
    return tardigrade_db.Engine(str(tmp_path))


def get_past_kv(model, tokenizer, text):
    """Run inference and return past_key_values (real KV cache)."""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    return out.past_key_values


# -- ATDD 1: Hook extracts real K projections from past_key_values -----------


def test_kv_hook_extracts_real_k_projections(gpt2, tokenizer, engine):
    """GIVEN past_key_values from GPT-2,
    WHEN HuggingFaceKVHook.on_generate() processes layer 0,
    THEN WriteDecision.key dimension = num_heads * head_dim."""
    past_kv = get_past_kv(gpt2, tokenizer, "The capital of France is")

    hook = HuggingFaceKVHook(engine, owner=1, model_config=gpt2.config)
    decision = hook.on_generate(layer=0, past_key_values=past_kv)

    assert decision.should_write is True
    assert decision.key is not None

    # GPT-2: 12 heads * 64 head_dim = 768
    expected_dim = gpt2.config.n_head * (gpt2.config.n_embd // gpt2.config.n_head)
    assert len(decision.key) == expected_dim
    assert decision.key.dtype == np.float32


# -- ATDD 2: Dual-store: key (index) is smaller than value (payload) ---------


def test_kv_hook_dual_store_key_and_value(gpt2, tokenizer, engine):
    """GIVEN past_key_values,
    WHEN on_generate() produces WriteDecision,
    THEN value (full K+V payload) is larger than key (mean-pooled K index)."""
    past_kv = get_past_kv(gpt2, tokenizer, "Hello world")

    hook = HuggingFaceKVHook(engine, owner=1, model_config=gpt2.config)
    decision = hook.on_generate(layer=0, past_key_values=past_kv)

    assert decision.should_write is True
    assert len(decision.value) > len(decision.key), (
        f"Value ({len(decision.value)}) should be larger than key ({len(decision.key)})"
    )


# -- ATDD 3: Round-trip store and retrieve with real KV ----------------------


def test_kv_hook_round_trip_store_retrieve(gpt2, tokenizer, engine):
    """GIVEN memories stored via KV hook from prompt A,
    WHEN queried with prompt B,
    THEN retrieval returns results with non-zero scores."""
    hook = HuggingFaceKVHook(engine, owner=1, model_config=gpt2.config)

    # Store.
    past_kv_a = get_past_kv(gpt2, tokenizer, "The capital of France is Paris")
    for layer in range(gpt2.config.n_layer):
        decision = hook.on_generate(layer=layer, past_key_values=past_kv_a)
        if decision.should_write:
            engine.mem_write(1, layer, decision.key, decision.value, decision.salience, None)

    assert engine.cell_count() > 0

    # Retrieve.
    past_kv_b = get_past_kv(gpt2, tokenizer, "What is the main city of France")
    handles = hook.on_prefill(layer=8, past_key_values=past_kv_b)

    assert len(handles) > 0
    assert all(isinstance(h, MemoryCellHandle) for h in handles)
    assert all(h.score != 0.0 for h in handles)


# -- ATDD 4: KV hook produces different keys than hidden-state hook ----------


def test_kv_hook_differs_from_hidden_state_hook(gpt2, tokenizer, engine):
    """GIVEN same prompt,
    WHEN processed by KV hook vs hidden-state hook,
    THEN keys are different (K projections != raw hidden states)."""
    from tardigrade_hooks.hf_hook import HuggingFaceHook

    text = "The capital of France is"
    past_kv = get_past_kv(gpt2, tokenizer, text)

    # KV hook.
    kv_hook = HuggingFaceKVHook(engine, owner=1, model_config=gpt2.config)
    kv_decision = kv_hook.on_generate(layer=0, past_key_values=past_kv)

    # Hidden-state hook.
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = gpt2(**inputs, output_hidden_states=True)
    hidden = out.hidden_states[1].numpy()

    hs_hook = HuggingFaceHook(engine, owner=1, norm_threshold=0.0)
    hs_decision = hs_hook.on_generate(layer=0, hidden_states=hidden)

    assert not np.allclose(kv_decision.key, hs_decision.key, atol=1e-4), (
        "KV hook key should differ from hidden-state hook key"
    )


# -- ATDD 5: Real KV injection changes model output -------------------------


def test_real_kv_injection_changes_output(gpt2, tokenizer, engine):
    """GIVEN memories stored via KV hook,
    WHEN injected into a different prompt via MemoryInjector,
    THEN logits differ from un-injected model."""
    from tardigrade_hooks.injector import MemoryInjector
    from tardigrade_hooks.position import AbsolutePositionEncoder

    hook = HuggingFaceKVHook(engine, owner=1, model_config=gpt2.config)

    # Store.
    past_kv_a = get_past_kv(gpt2, tokenizer, "The capital of France is Paris")
    for layer in range(gpt2.config.n_layer):
        decision = hook.on_generate(layer=layer, past_key_values=past_kv_a)
        if decision.should_write:
            engine.mem_write(1, layer, decision.key, decision.value, decision.salience, None)

    # Inject.
    injector = MemoryInjector(
        model=gpt2, engine=engine, owner=1,
        position_encoder=AbsolutePositionEncoder(),
    )

    input_ids = tokenizer("What is the main city of France", return_tensors="pt")["input_ids"]

    with torch.no_grad():
        logits_raw = gpt2(input_ids).logits
        logits_injected = injector(input_ids).logits

    assert not torch.allclose(logits_raw, logits_injected, atol=1e-4), (
        "Real KV injection should change model output"
    )
