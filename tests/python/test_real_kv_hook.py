# ATDD acceptance tests for real-KV hook (Issue #9).
#
# Design pattern: Adapter (translates HF DynamicCache into TardigradeDB API).
# Dual-store: mean-pooled K as search index, flattened K+V as injection payload.

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from types import SimpleNamespace
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db
from tardigrade_hooks.encoding import HEADER_SIZE, SENTINEL_IDX, DIM_IDX
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


class RecordingEngine:
    """Minimal engine double that records the query key passed to mem_read."""

    def __init__(self):
        self.query_key = None

    def mem_read(self, query_key, k, owner):
        self.query_key = np.array(query_key, dtype=np.float32)
        return []


class DummyAttention(torch.nn.Module):
    def __init__(self, hidden_size, q_dim, kv_dim):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_size, q_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, kv_dim, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.arange(q_dim * hidden_size, dtype=torch.float32).reshape(q_dim, hidden_size) / 100.0)
            self.k_proj.weight.copy_(torch.arange(kv_dim * hidden_size, dtype=torch.float32).reshape(kv_dim, hidden_size) / 100.0)


class DummyModel:
    def __init__(self, hidden_size, q_dim, kv_dim):
        attn = DummyAttention(hidden_size, q_dim, kv_dim)
        layer = SimpleNamespace(self_attn=attn)
        self.model = SimpleNamespace(layers=[layer])


class DummyCache:
    def __init__(self, heads, seq, head_dim):
        keys = torch.arange(heads * seq * head_dim, dtype=torch.float32).reshape(1, heads, seq, head_dim)
        values = keys + 100.0
        self.layers = [SimpleNamespace(keys=keys, values=values)]


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

    # Per-token encoded: 64-float Q4-safe header + N_tokens * kv_dim.
    kv_dim = gpt2.config.n_head * (gpt2.config.n_embd // gpt2.config.n_head)
    assert len(decision.key) > kv_dim, "Per-token key should be larger than single-token dim"
    # Header contract: sentinel at [0], dim at [33]. n_tokens is NOT stored
    # in the header (Q4 corrupts it); readers compute n from data length.
    assert decision.key[SENTINEL_IDX] < -1.0e8, "Header should start with sentinel"
    dim_from_header = int(round(decision.key[DIM_IDX]))
    assert dim_from_header == kv_dim, f"Header dim {dim_from_header} should match kv_dim {kv_dim}"
    data_len = len(decision.key) - HEADER_SIZE
    assert data_len % kv_dim == 0, "Encoded data must be a whole number of tokens"
    n_tokens = data_len // kv_dim
    assert n_tokens >= 1, "Per-token key should carry at least one token"
    assert decision.key.dtype == np.float32


def test_kv_hook_prefill_sends_encoded_per_token_q_query():
    """Regression: on_prefill must not collapse Q tokens to one mean vector."""
    hidden_size = 8
    q_dim = 8
    kv_dim = 8
    model_config = SimpleNamespace(
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=hidden_size,
        head_dim=4,
    )
    recorder = RecordingEngine()
    hook = HuggingFaceKVHook(
        recorder,
        owner=1,
        model_config=model_config,
        model=DummyModel(hidden_size, q_dim, kv_dim),
    )
    hidden = torch.arange(3 * hidden_size, dtype=torch.float32).reshape(1, 3, hidden_size)

    hook.on_prefill(layer=0, model_hidden_states=hidden)

    assert recorder.query_key is not None
    assert recorder.query_key[SENTINEL_IDX] < -1.0e8
    assert int(round(recorder.query_key[DIM_IDX])) == q_dim
    # n_tokens is inferred from data length (header field is unreliable through Q4).
    assert len(recorder.query_key) == HEADER_SIZE + 2 * q_dim  # 3 tokens minus position 0


def test_kv_hook_expands_gqa_k_for_retrieval_but_keeps_payload_compact():
    """Regression: GQA storage expands K search keys instead of averaging Q down."""
    hidden_size = 8
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 2
    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    seq = 3
    model_config = SimpleNamespace(
        num_attention_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
        hidden_size=hidden_size,
        head_dim=head_dim,
    )
    hook = HuggingFaceKVHook(
        RecordingEngine(),
        owner=1,
        model_config=model_config,
        model=DummyModel(hidden_size, q_dim, kv_dim),
    )
    hidden = torch.arange(seq * hidden_size, dtype=torch.float32).reshape(1, seq, hidden_size)
    cache = DummyCache(num_kv_heads, seq, head_dim)

    decision = hook.on_generate(layer=0, past_key_values=cache, model_hidden_states=hidden)

    assert decision.should_write is True
    assert int(round(decision.key[DIM_IDX])) == q_dim
    # n_tokens is encoded as data-length / dim, not as a header field.
    assert len(decision.key) == HEADER_SIZE + (seq - 1) * q_dim
    assert len(decision.value) == 2 * seq * kv_dim


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

    # Keys should differ: KV hook produces per-token encoded (longer),
    # hidden-state hook produces mean-pooled (shorter).
    assert len(kv_decision.key) != len(hs_decision.key), (
        "KV hook key (per-token encoded) should differ in length from hidden-state hook key (mean-pooled)"
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
