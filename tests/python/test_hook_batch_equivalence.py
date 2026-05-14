"""AT-B1 — `HuggingFaceKVHook` batch path equivalence with the serial path.

Slice B1 of the ingestion correctness/perf workstream. Pins that
batched forward passes through the hook produce the same
`WriteDecision` per chunk as the previous serial path, so the GPU-
batching speedup is correctness-preserving.

Behavioral test (Kent Dodds): runs the *same* model on two paths
(serial unbatched, batched with `batch_index` + `seq_len`) and
asserts each pair of decisions matches to the precision the model
itself provides. No internal-state probes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tardigrade_hooks.hf_kv_hook import HuggingFaceKVHook  # noqa: E402

# ---------- fixture constants ----------

FIXTURE_TEXTS = [
    "The capital of France is Paris.",
    "Photosynthesis converts sunlight into chemical energy stored in glucose.",
    "Mount Everest stands at 8,848 meters above sea level.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
]
FIXTURE_LAYER_INDEX = 6  # mid-stack GPT-2 layer
FIXTURE_PAD_TO_LONGEST = "longest"
KEY_COSINE_FLOOR = 0.999
VALUE_COSINE_FLOOR = 0.999


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    norm = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / norm)


@pytest.fixture(scope="module")
def gpt2_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()
    return model


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="module")
def hook(gpt2_model):
    # use_hidden_states=True matches the engine's production wiring
    # (TardigradeAdapter._init_native).
    return HuggingFaceKVHook(
        engine=None,
        owner=1,
        model_config=gpt2_model.config,
        model=gpt2_model,
        use_hidden_states=True,
    )


def _serial_decision(model, tokenizer, hook, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, use_cache=True, output_hidden_states=True)
    return hook.on_generate(
        layer=FIXTURE_LAYER_INDEX,
        past_key_values=out.past_key_values,
        model_hidden_states=out.hidden_states[FIXTURE_LAYER_INDEX],
    )


def _batched_decisions(model, tokenizer, hook, texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=FIXTURE_PAD_TO_LONGEST,
    )
    with torch.no_grad():
        out = model(**inputs, use_cache=True, output_hidden_states=True)
    decisions = []
    for batch_index in range(inputs["input_ids"].shape[0]):
        seq_len = int(inputs["attention_mask"][batch_index].sum().item())
        decision = hook.on_generate(
            layer=FIXTURE_LAYER_INDEX,
            past_key_values=out.past_key_values,
            model_hidden_states=out.hidden_states[FIXTURE_LAYER_INDEX],
            batch_index=batch_index,
            seq_len=seq_len,
        )
        decisions.append(decision)
    return decisions


def test_batched_hook_matches_serial_per_chunk(gpt2_model, gpt2_tokenizer, hook):
    serial_decisions = [
        _serial_decision(gpt2_model, gpt2_tokenizer, hook, t) for t in FIXTURE_TEXTS
    ]
    batched_decisions = _batched_decisions(gpt2_model, gpt2_tokenizer, hook, FIXTURE_TEXTS)

    assert len(serial_decisions) == len(batched_decisions)
    for index, (s, b) in enumerate(zip(serial_decisions, batched_decisions)):
        assert s.should_write == b.should_write, f"chunk {index}: should_write mismatch"
        assert len(s.key) == len(b.key), (
            f"chunk {index}: key length mismatch — serial={len(s.key)} batched={len(b.key)}"
        )
        assert len(s.value) == len(b.value), (
            f"chunk {index}: value length mismatch — serial={len(s.value)} batched={len(b.value)}"
        )
        key_cos = cosine(s.key, b.key)
        value_cos = cosine(s.value, b.value)
        assert key_cos >= KEY_COSINE_FLOOR, (
            f"chunk {index}: key cosine {key_cos} below floor {KEY_COSINE_FLOOR}"
        )
        assert value_cos >= VALUE_COSINE_FLOOR, (
            f"chunk {index}: value cosine {value_cos} below floor {VALUE_COSINE_FLOOR}"
        )
