# ATDD tests for RoPE-corrected multi-memory composition (Phase 30A).
#
# Design pattern: Strategy (RoPECorrectedConcatComposer implements
# CompositionStrategy, accepts PositionEncoder via DI).
#
# Based on CacheBlend (EuroSys 2025): fix RoPE positions before
# concatenating independently-computed KV packs. V vectors unchanged
# since RoPE only affects K.

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db
from tardigrade_hooks.kp_injector import KnowledgePackStore
from tardigrade_hooks.multi_composer import (
    NaiveConcatComposer,
    RoPECorrectedConcatComposer,
)
from tardigrade_hooks.position import AbsolutePositionEncoder, RoPEPositionEncoder

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


def _get_packs(kps, engine, query_text, k=2):
    """Helper: retrieve k packs for a query."""
    from tardigrade_hooks.encoding import encode_per_token

    query_input = kps.tokenizer.encode(query_text, return_tensors="pt")
    with torch.no_grad():
        out = kps.model(query_input, output_hidden_states=True)
    hidden = out.hidden_states[kps.query_layer][0]
    h_tokens = hidden[1:].numpy().astype(np.float32)
    query_key = encode_per_token(h_tokens, kps.hidden_size)
    return engine.mem_read_pack(query_key, k, 1)


# -- 1: same shape as naive ---------------------------------------------------


def test_rope_corrected_produces_same_shape_as_naive(kps, engine):
    """GIVEN 2 facts stored,
    WHEN compose() with AbsolutePositionEncoder (GPT-2, no-op remap),
    THEN cache shape matches NaiveConcatComposer."""
    kps.store("The pharmacy closes at 8:30pm on Tuesdays")
    kps.store("The pharmacy is located at 742 Elm Avenue")

    packs = _get_packs(kps, engine, "Tell me about the pharmacy")
    naive = NaiveConcatComposer()
    corrected = RoPECorrectedConcatComposer(AbsolutePositionEncoder())

    cache_naive = naive.compose(packs, kps.num_kv_heads, kps.head_dim, kps.kv_dim, kps.n_layers)
    cache_corr = corrected.compose(packs, kps.num_kv_heads, kps.head_dim, kps.kv_dim, kps.n_layers)

    assert cache_naive.get_seq_length() == cache_corr.get_seq_length()
    assert len(cache_naive.layers) == len(cache_corr.layers)


# -- 2: plugs into generate_multi ---------------------------------------------


def test_rope_corrected_plugs_into_generate_multi(kps, engine):
    """GIVEN 2 facts stored,
    WHEN generate_multi with RoPECorrectedConcatComposer,
    THEN had_memory == True and returns valid text."""
    kps.store("Sonia's wifi password is mango-cathedral-7")
    kps.store("Eduardo works at Morning Bloom bakery")

    composer = RoPECorrectedConcatComposer(AbsolutePositionEncoder())
    text, tokens, had_memory = kps.generate_multi(
        "Tell me about Sonia", k=2, composer=composer, max_new_tokens=5
    )

    assert had_memory is True
    assert isinstance(text, str)


# -- 3: cumulative position offsets --------------------------------------------


def test_position_encoder_invoked_with_cumulative_offsets():
    """GIVEN 3 synthetic packs with known seq_lens,
    WHEN compose() with a recording PositionEncoder,
    THEN remap_keys called with correct cumulative offsets."""

    class RecordingEncoder:
        def __init__(self):
            self.calls = []

        def remap_keys(self, keys, old_positions, new_start):
            self.calls.append({
                "seq_len": keys.shape[2],
                "old_pos_len": len(old_positions),
                "new_start": new_start,
            })
            return keys  # pass-through

    head_dim = 4
    num_heads = 2
    kv_dim = num_heads * head_dim
    n_layers = 1

    def make_pack(seq_len):
        data = np.random.randn(2 * seq_len * kv_dim).astype(np.float32)
        return {"pack_id": 0, "layers": [{"layer_idx": 0, "data": data}]}

    packs = [make_pack(10), make_pack(8), make_pack(12)]
    recorder = RecordingEncoder()
    composer = RoPECorrectedConcatComposer(recorder)
    composer.compose(packs, num_heads, head_dim, kv_dim, n_layers)

    assert len(recorder.calls) == 3
    assert recorder.calls[0]["new_start"] == 0
    assert recorder.calls[0]["seq_len"] == 10
    assert recorder.calls[1]["new_start"] == 10
    assert recorder.calls[1]["seq_len"] == 8
    assert recorder.calls[2]["new_start"] == 18
    assert recorder.calls[2]["seq_len"] == 12


# -- 4: V vectors unchanged ---------------------------------------------------


def test_v_vectors_unchanged_by_position_correction(kps, engine):
    """GIVEN 2 facts stored,
    WHEN compose() with RoPECorrectedConcatComposer,
    THEN V tensors are identical to NaiveConcatComposer."""
    kps.store("The pharmacy closes at 8:30pm")
    kps.store("The pharmacy is on Elm Avenue")

    packs = _get_packs(kps, engine, "pharmacy")
    naive = NaiveConcatComposer()
    corrected = RoPECorrectedConcatComposer(AbsolutePositionEncoder())

    cache_naive = naive.compose(packs, kps.num_kv_heads, kps.head_dim, kps.kv_dim, kps.n_layers)
    cache_corr = corrected.compose(packs, kps.num_kv_heads, kps.head_dim, kps.kv_dim, kps.n_layers)

    for li in range(kps.n_layers):
        assert torch.allclose(
            cache_naive.layers[li].values, cache_corr.layers[li].values, atol=1e-6
        )


# -- 5: AbsoluteEncoder makes K identical to naive ----------------------------


def test_absolute_encoder_produces_identical_k_to_naive(kps, engine):
    """GIVEN 2 packs and AbsolutePositionEncoder (no-op remap),
    THEN K tensors are identical to NaiveConcatComposer."""
    kps.store("Fact one about Sonia")
    kps.store("Fact two about Eduardo")

    packs = _get_packs(kps, engine, "Sonia Eduardo")
    naive = NaiveConcatComposer()
    corrected = RoPECorrectedConcatComposer(AbsolutePositionEncoder())

    cache_naive = naive.compose(packs, kps.num_kv_heads, kps.head_dim, kps.kv_dim, kps.n_layers)
    cache_corr = corrected.compose(packs, kps.num_kv_heads, kps.head_dim, kps.kv_dim, kps.n_layers)

    for li in range(kps.n_layers):
        assert torch.allclose(
            cache_naive.layers[li].keys, cache_corr.layers[li].keys, atol=1e-6
        )


# -- 6: single pack unchanged -------------------------------------------------


def test_single_pack_no_correction_needed(kps, engine):
    """GIVEN 1 pack,
    WHEN compose() with RoPECorrectedConcatComposer,
    THEN output seq_len matches the single pack."""
    kps.store("Eduardo's apartment is 4B")

    cache_single, _, _ = kps.retrieve_and_inject("Where does Eduardo live?")
    composer = RoPECorrectedConcatComposer(AbsolutePositionEncoder())
    cache_multi, _, _ = kps.retrieve_and_inject_multi(
        "Where does Eduardo live?", k=1, composer=composer
    )

    assert cache_single.get_seq_length() == cache_multi.get_seq_length()


# -- 7: RoPE correction actually changes K for second pack --------------------


def test_rope_correction_changes_k_for_second_pack():
    """GIVEN 2 synthetic packs,
    WHEN compose() with RoPEPositionEncoder,
    THEN K values for pack 2 differ from NaiveConcatComposer."""
    head_dim = 64
    num_heads = 2
    kv_dim = num_heads * head_dim
    n_layers = 1
    seq_a, seq_b = 5, 4

    np.random.seed(42)
    data_a = np.random.randn(2 * seq_a * kv_dim).astype(np.float32)
    data_b = np.random.randn(2 * seq_b * kv_dim).astype(np.float32)

    packs = [
        {"pack_id": 0, "layers": [{"layer_idx": 0, "data": data_a}]},
        {"pack_id": 1, "layers": [{"layer_idx": 0, "data": data_b}]},
    ]

    naive = NaiveConcatComposer()
    corrected = RoPECorrectedConcatComposer(RoPEPositionEncoder(head_dim=head_dim))

    cache_naive = naive.compose(packs, num_heads, head_dim, kv_dim, n_layers)
    cache_corr = corrected.compose(packs, num_heads, head_dim, kv_dim, n_layers)

    k_naive = cache_naive.layers[0].keys
    k_corr = cache_corr.layers[0].keys

    # Pack 1 K (positions 0..4): new_start=0, so remap is identity
    assert torch.allclose(k_naive[:, :, :seq_a, :], k_corr[:, :, :seq_a, :], atol=1e-4)

    # Pack 2 K (positions 0..3 vs 5..8): should differ after re-rotation
    assert not torch.allclose(k_naive[:, :, seq_a:, :], k_corr[:, :, seq_a:, :], atol=1e-4)
