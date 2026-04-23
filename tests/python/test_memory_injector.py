"""ATDD acceptance tests for MemoryInjector — Decorator pattern.

The MemoryInjector wraps a HuggingFace model, transparently retrieving
stored KV memories from TardigradeDB and injecting them into the attention
cache before the forward pass. Callers use model.generate() as usual.

Approach C (hook-based): Uses the model's embedding layer as a cheap
query to TardigradeDB, then injects retrieved KV into a DynamicCache
for a single full forward pass. No double-forward.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db
from tardigrade_hooks.injector import MemoryInjector
from tardigrade_hooks.position import AbsolutePositionEncoder

# GPT-2 constants.
NUM_HEADS = 12
HEAD_DIM = 64
D_MODEL = NUM_HEADS * HEAD_DIM  # 768
NUM_LAYERS = 12


@pytest.fixture
def gpt2_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()
    return model


@pytest.fixture
def tokenizer():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def engine(tmp_path):
    return tardigrade_db.Engine(str(tmp_path))


def store_memories_from_prompt(engine, model, tokenizer, text, owner=1):
    """Capture hidden states from a prompt and store in TardigradeDB."""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    cell_ids = []
    for layer_idx in range(model.config.n_layer):
        hidden = outputs.hidden_states[layer_idx + 1].numpy()[0]
        mean_hidden = hidden.mean(axis=0).astype(np.float32)
        cell_id = engine.mem_write(owner, layer_idx, mean_hidden, mean_hidden, 80.0, None)
        cell_ids.append(cell_id)
    return cell_ids


# ── ATDD Test 1: Decorator is transparent on empty DB ──────────────────────


def test_injector_transparent_on_empty_db(gpt2_model, tokenizer, engine):
    """GIVEN: Empty TardigradeDB, WHEN: generate via MemoryInjector,
    THEN: output is identical to unwrapped model (Decorator transparency)."""
    injector = MemoryInjector(
        model=gpt2_model,
        engine=engine,
        owner=1,
        position_encoder=AbsolutePositionEncoder(),
    )

    input_ids = tokenizer("The capital of France is", return_tensors="pt")["input_ids"]

    with torch.no_grad():
        logits_raw = gpt2_model(input_ids).logits
        logits_wrapped = injector(input_ids).logits

    assert torch.allclose(logits_raw, logits_wrapped, atol=1e-5), (
        "Empty DB injection should be a no-op"
    )


# ── ATDD Test 2: Injection changes model output ───────────────────────────


def test_injector_changes_output_with_memories(gpt2_model, tokenizer, engine):
    """GIVEN: Memories stored from 'The capital of France is Paris',
    WHEN: query 'What is the main city of France' via MemoryInjector,
    THEN: output logits differ from unwrapped model."""
    store_memories_from_prompt(engine, gpt2_model, tokenizer, "The capital of France is Paris")

    injector = MemoryInjector(
        model=gpt2_model,
        engine=engine,
        owner=1,
        position_encoder=AbsolutePositionEncoder(),
    )

    input_ids = tokenizer("What is the main city of France", return_tensors="pt")["input_ids"]

    with torch.no_grad():
        logits_raw = gpt2_model(input_ids).logits
        logits_injected = injector(input_ids).logits

    assert not torch.allclose(logits_raw, logits_injected, atol=1e-4), (
        "Injection should change output when memories exist"
    )


# ── ATDD Test 3: generate() works through the Decorator ───────────────────


def test_injector_generate_produces_tokens(gpt2_model, tokenizer, engine):
    """GIVEN: Memories stored, WHEN: injector.generate(),
    THEN: produces valid token sequence (no crash, reasonable length)."""
    store_memories_from_prompt(engine, gpt2_model, tokenizer, "The capital of France is Paris")

    injector = MemoryInjector(
        model=gpt2_model,
        engine=engine,
        owner=1,
        position_encoder=AbsolutePositionEncoder(),
    )

    input_ids = tokenizer("What is the main city of France", return_tensors="pt")["input_ids"]

    with torch.no_grad():
        output_ids = injector.generate(input_ids, max_new_tokens=10)

    assert output_ids.shape[1] > input_ids.shape[1], "generate() should produce new tokens"
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    assert len(decoded) > 0, "Decoded output should be non-empty"


# ── ATDD Test 4: Multi-layer alignment ────────────────────────────────────


def test_injector_multi_layer_alignment(gpt2_model, tokenizer, engine):
    """GIVEN: Memories stored per-layer, WHEN: injection,
    THEN: each layer's cache gets its own K/V (layer 0's K != layer 5's K)."""
    store_memories_from_prompt(engine, gpt2_model, tokenizer, "Machine learning is fascinating")

    injector = MemoryInjector(
        model=gpt2_model,
        engine=engine,
        owner=1,
        position_encoder=AbsolutePositionEncoder(),
    )

    input_ids = tokenizer("Deep learning models are", return_tensors="pt")["input_ids"]

    # Access the built cache before forward to check layer alignment.
    cache = injector.build_memory_cache(input_ids)

    if cache is not None:
        # Verify different layers have different KV content.
        assert len(cache) >= 2, "Should have entries for multiple layers"
        layer0_k = cache.layers[0].keys
        layer5_k = cache.layers[5].keys
        assert not torch.allclose(layer0_k, layer5_k, atol=1e-6), (
            "Different layers should have different K vectors"
        )


# ── ATDD Test 5: Owner isolation ──────────────────────────────────────────


def test_injector_respects_owner_isolation(gpt2_model, tokenizer):
    """GIVEN: Agent 1 stores memories, WHEN: Agent 2's injector queries,
    THEN: no memories found, output identical to unwrapped model."""
    with tempfile.TemporaryDirectory() as tmp:
        engine = tardigrade_db.Engine(tmp)

        # Agent 1 stores memories.
        store_memories_from_prompt(engine, gpt2_model, tokenizer, "Secret agent data", owner=1)

        # Agent 2's injector.
        injector = MemoryInjector(
            model=gpt2_model,
            engine=engine,
            owner=2,  # Different owner — should see no memories.
            position_encoder=AbsolutePositionEncoder(),
        )

        input_ids = tokenizer("Tell me the secret", return_tensors="pt")["input_ids"]

        with torch.no_grad():
            logits_raw = gpt2_model(input_ids).logits
            logits_isolated = injector(input_ids).logits

        assert torch.allclose(logits_raw, logits_isolated, atol=1e-5), (
            "Agent 2 should not see Agent 1's memories"
        )
