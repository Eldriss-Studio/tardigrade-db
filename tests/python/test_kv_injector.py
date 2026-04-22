"""ATDD acceptance tests for KV cache injection (Adapter pattern)."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import DynamicCache, GPT2Config, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db
from tardigrade_hooks.hook import MemoryCellHandle
from tardigrade_hooks.kv_injector import (
    build_injection_cache,
    inject_into_cache,
    prepare_injection,
    reshape_to_kv,
)

# GPT-2 config constants.
NUM_HEADS = 12
HEAD_DIM = 64
D_MODEL = NUM_HEADS * HEAD_DIM  # 768
NUM_LAYERS = 12


def make_handle(cell_id=0, layer=0, dim=D_MODEL):
    """Create a test MemoryCellHandle with random key/value."""
    return MemoryCellHandle(
        cell_id=cell_id,
        owner=1,
        layer=layer,
        score=0.9,
        key=np.random.randn(dim).astype(np.float32),
        value=np.random.randn(dim).astype(np.float32),
    )


def test_injector_reshapes_to_past_key_values():
    """ATDD 1: flat 768-dim vector → (1, 12, 1, 64) tensor."""
    flat = np.random.randn(D_MODEL).astype(np.float32)
    tensor = reshape_to_kv(flat, NUM_HEADS, HEAD_DIM)

    assert tensor.shape == (1, NUM_HEADS, 1, HEAD_DIM)
    assert tensor.dtype == torch.float32

    with pytest.raises(ValueError, match="Vector length"):
        reshape_to_kv(np.zeros(100, dtype=np.float32), NUM_HEADS, HEAD_DIM)


def test_injector_extends_kv_cache():
    """ATDD 2: existing cache seq_len=5, inject 1 cell → seq_len=6."""
    cache = DynamicCache()
    existing_k = torch.randn(1, NUM_HEADS, 5, HEAD_DIM)
    existing_v = torch.randn(1, NUM_HEADS, 5, HEAD_DIM)
    cache.update(key_states=existing_k, value_states=existing_v, layer_idx=0)
    assert cache.get_seq_length() == 5

    handle = make_handle()
    inject_into_cache(cache, layer_idx=0, handles=[handle], num_heads=NUM_HEADS, head_dim=HEAD_DIM)
    assert cache.get_seq_length() == 6


def test_injector_with_real_gpt2():
    """ATDD 3: GPT-2 forward with injection produces different logits."""
    # Use eager attention to avoid SDPA shape constraints with partial cache.
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()

    input_ids = torch.tensor([[464, 3139, 286, 4881, 318]])

    with torch.no_grad():
        logits_no = model(input_ids).logits

    # Inject 1 cell at ALL layers (required: GPT-2 expects uniform cache across layers).
    handles_by_layer = {i: [make_handle(layer=i)] for i in range(NUM_LAYERS)}
    cache = build_injection_cache(
        handles_by_layer=handles_by_layer,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_layers=NUM_LAYERS,
    )

    fwd_kwargs = prepare_injection(cache, input_ids)
    with torch.no_grad():
        logits_yes = model(input_ids, **fwd_kwargs).logits

    assert not torch.allclose(logits_no, logits_yes, atol=1e-4), "Injection should change output"


def test_injector_multiple_handles():
    """ATDD 4: Inject 3 handles at layer 0. Cache seq_len increases by 3."""
    cache = DynamicCache()

    handles = [make_handle(cell_id=i) for i in range(3)]
    inject_into_cache(cache, layer_idx=0, handles=handles, num_heads=NUM_HEADS, head_dim=HEAD_DIM)

    assert cache.get_seq_length() == 3

    # Each handle contributed one KV entry. Access via layers API.
    layer0 = cache.layers[0]
    assert layer0.keys.shape == (1, NUM_HEADS, 3, HEAD_DIM)
    assert layer0.values.shape == (1, NUM_HEADS, 3, HEAD_DIM)


def test_round_trip_capture_inject():
    """ATDD 5: Capture KV from prompt A via engine, inject into prompt B."""
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()
    config = GPT2Config.from_pretrained("gpt2")

    # Capture from prompt A.
    prompt_a = torch.tensor([[464, 3139, 286, 4881, 318]])
    with torch.no_grad():
        out_a = model(prompt_a, output_hidden_states=True)

    hidden = out_a.hidden_states[1].numpy()[0]
    mean_hidden = hidden.mean(axis=0).astype(np.float32)

    engine = tardigrade_db.Engine(tempfile.mkdtemp())
    engine.mem_write(1, 0, mean_hidden, mean_hidden, 80.0, None)

    # Retrieve for prompt B.
    prompt_b = torch.tensor([[2061, 318, 262, 1388, 1748]])
    with torch.no_grad():
        logits_no = model(prompt_b).logits

    results = engine.mem_read(mean_hidden, 1, 1)
    assert len(results) > 0

    handles = [
        MemoryCellHandle(
            cell_id=r.cell_id,
            owner=r.owner,
            layer=r.layer,
            score=r.score,
            key=np.array(r.key(), dtype=np.float32),
            value=np.array(r.value(), dtype=np.float32),
        )
        for r in results
    ]

    # Inject at all layers (GPT-2 requires uniform cache across layers).
    cache = build_injection_cache(
        handles_by_layer={i: handles for i in range(config.n_layer)},
        num_heads=config.n_head,
        head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer,
    )

    fwd_kwargs = prepare_injection(cache, prompt_b)
    with torch.no_grad():
        logits_yes = model(prompt_b, **fwd_kwargs).logits

    assert not torch.allclose(logits_no, logits_yes, atol=1e-4), "Injection should change output"
