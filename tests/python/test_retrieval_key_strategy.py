# Acceptance tests for the RetrievalKeyStrategy abstraction.
#
# Note on ATDD discipline: these tests were written POST-implementation
# during the session that shipped retrieval_key_strategy.py. The user
# called out the ATDD slip explicitly; see memory note
# feedback-atdd-also-when-excited.md. Going forward in this PR, any
# remaining work (kp_injector integration) goes RED-first.

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tardigrade_hooks.retrieval_key_strategy import (
    HiddenStateKeyStrategy,
    KVectorKeyStrategy,
    RetrievalKeyStrategy,
)


# ---- ABC ----------------------------------------------------------------

def test_retrieval_key_strategy_is_abstract():
    with pytest.raises(TypeError):
        RetrievalKeyStrategy()  # type: ignore[abstract]


# ---- HiddenStateKeyStrategy --------------------------------------------

def test_hidden_state_strategy_describe_format():
    s = HiddenStateKeyStrategy(query_layer=17)
    assert s.describe() == "hidden_state@17"


def test_hidden_state_strategy_query_layer_is_readable():
    s = HiddenStateKeyStrategy(query_layer=5)
    assert s.query_layer == 5


def test_hidden_state_strategy_compute_returns_encoded_key():
    # Hidden states list: 3 layers' worth, each (4 tokens, hidden_size=8).
    rng = np.random.RandomState(42)
    hidden_states = [rng.randn(4, 8).astype("float32") for _ in range(3)]

    s = HiddenStateKeyStrategy(query_layer=1)
    key = s.compute(hidden_states, kv=None, hidden_size=8)
    # encode_per_token prepends a 64-float header; data is (4-1)*8 = 24 floats.
    assert isinstance(key, np.ndarray)
    assert key.dtype == np.float32
    assert key.shape == (64 + 24,)


def test_hidden_state_strategy_raises_on_out_of_range_layer():
    hidden_states = [np.zeros((4, 8), dtype=np.float32) for _ in range(3)]
    s = HiddenStateKeyStrategy(query_layer=99)
    with pytest.raises(ValueError, match="out of range"):
        s.compute(hidden_states, kv=None, hidden_size=8)


def test_hidden_state_strategy_accepts_degenerate_single_token_input():
    """Single-token input → h[1:] is empty, encoding is header-only.

    Matches the pre-Phase-2 inline-extraction behavior in
    KnowledgePackStore. Raising here would regress consumers that
    store single-token facts (test fixtures with one-word stubs)."""
    hidden_states = [np.zeros((1, 8), dtype=np.float32) for _ in range(3)]
    s = HiddenStateKeyStrategy(query_layer=1)
    key = s.compute(hidden_states, kv=None, hidden_size=8)
    # Header is 64 floats; no data follows.
    assert key.shape == (64,)


# ---- KVectorKeyStrategy ------------------------------------------------

def test_kvector_strategy_describe_format():
    s = KVectorKeyStrategy(softmax_layer_idx=11)
    assert s.describe() == "k_vector@11"


def test_kvector_strategy_softmax_layer_idx_is_readable():
    s = KVectorKeyStrategy(softmax_layer_idx=3)
    assert s.softmax_layer_idx == 3


def _make_fake_cache(softmax_layers_with_k: dict):
    """Build a fake DynamicCache-like with specific .keys tensors."""
    class _SoftmaxLayer:
        def __init__(self, k):
            self.keys = k
    class _LinearLayer:
        keys = None  # not a tensor → fails _is_softmax_cache_layer
    class _FakeCache:
        pass
    cache = _FakeCache()
    layers = []
    max_idx = max(softmax_layers_with_k.keys()) if softmax_layers_with_k else 0
    for i in range(max_idx + 1):
        if i in softmax_layers_with_k:
            layers.append(_SoftmaxLayer(softmax_layers_with_k[i]))
        else:
            layers.append(_LinearLayer())
    cache.layers = layers
    return cache


def test_kvector_strategy_compute_returns_encoded_key():
    # cache.layers[2].keys is (1, num_kv_heads=2, seq_len=4, head_dim=3)
    keys_tensor = torch.randn(1, 2, 4, 3, dtype=torch.float32)
    cache = _make_fake_cache({2: keys_tensor})

    s = KVectorKeyStrategy(softmax_layer_idx=2)
    key = s.compute(hidden_states=None, kv=cache, hidden_size=12)
    # encode_per_token prepends 64-float header. K reshape: (seq=4, kv_heads*head_dim=6).
    # After [1:] slice: (3, 6) = 18 floats.
    assert isinstance(key, np.ndarray)
    assert key.dtype == np.float32
    assert key.shape == (64 + 18,)


def test_kvector_strategy_raises_on_non_softmax_layer():
    # Layer 0 has no .keys tensor (linear/recurrent).
    cache = _make_fake_cache({2: torch.randn(1, 2, 4, 3)})
    s = KVectorKeyStrategy(softmax_layer_idx=0)
    with pytest.raises(ValueError, match="not a softmax-attention"):
        s.compute(hidden_states=None, kv=cache, hidden_size=12)


def test_kvector_strategy_raises_on_out_of_range_layer():
    cache = _make_fake_cache({2: torch.randn(1, 2, 4, 3)})
    s = KVectorKeyStrategy(softmax_layer_idx=99)
    with pytest.raises(ValueError, match="out of range"):
        s.compute(hidden_states=None, kv=cache, hidden_size=12)


def test_kvector_strategy_accepts_degenerate_single_token_input():
    """Same degenerate-but-valid policy as HiddenStateKeyStrategy."""
    keys_tensor = torch.randn(1, 2, 1, 3, dtype=torch.float32)
    cache = _make_fake_cache({0: keys_tensor})
    s = KVectorKeyStrategy(softmax_layer_idx=0)
    key = s.compute(hidden_states=None, kv=cache, hidden_size=6)
    # Header only — no data after skipping pos 0 on a 1-token input.
    assert key.shape == (64,)
