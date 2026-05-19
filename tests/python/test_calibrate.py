# ATDD acceptance tests for the per-model query-layer calibration.
#
# Design patterns under test:
#   Factory Method  — `select_query_layer(model, tokenizer, …)`
#   Strategy        — `CalibrationStrategy` ABC + `LinearSweepStrategy`
#   Repository      — `CalibrationRegistry` (covered separately)
#
# Plan: ~/.claude/plans/methodical-sleuthing-cuttlefish.md

import dataclasses
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tardigrade_hooks.calibrate import (
    CalibrationResult,
    CalibrationStrategy,
    LayerScore,
    LinearSweepStrategy,
    select_query_layer,
)
from tardigrade_hooks.calibration_registry import CalibrationRegistry


# ---- dataclass shape contracts ----

def test_layer_score_is_frozen_dataclass():
    score = LayerScore(layer=3, kind="attention", top1=18, top5=20)
    assert dataclasses.is_dataclass(score)
    with pytest.raises(dataclasses.FrozenInstanceError):
        score.layer = 99  # type: ignore[misc]


def test_calibration_result_is_frozen_dataclass():
    result = CalibrationResult(
        model_id="x",
        tardigrade_db_version="0.0.0",
        timestamp_iso="2026-05-19T00:00:00",
        n_layers=2,
        hidden_size=4,
        best_layer=1,
        scores=(LayerScore(layer=1, kind="attention", top1=1, top5=1),),
    )
    assert dataclasses.is_dataclass(result)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.best_layer = 99  # type: ignore[misc]


def test_calibration_result_round_trips_through_as_dict_from_dict():
    original = CalibrationResult(
        model_id="x",
        tardigrade_db_version="0.0.0",
        timestamp_iso="2026-05-19T00:00:00",
        n_layers=2,
        hidden_size=4,
        best_layer=1,
        scores=(
            LayerScore(layer=0, kind="embedding", top1=0, top5=0),
            LayerScore(layer=1, kind="attention", top1=1, top5=1),
        ),
    )
    restored = CalibrationResult.from_dict(original.as_dict())
    assert restored == original


def test_calibration_strategy_is_abstract():
    with pytest.raises(TypeError):
        CalibrationStrategy()  # type: ignore[abstract]


# ---- Factory behaviour ----

def test_select_query_layer_with_none_tokenizer_returns_default_layer():
    # Hosted-API mode: no tokenizer to probe, no sweep possible.
    # Returns a no-sweep result whose best_layer matches the library
    # default ratio.
    model = MagicMock()
    model.config.num_hidden_layers = 28
    model.config.hidden_size = 1024
    result = select_query_layer(model, tokenizer=None)
    from tardigrade_hooks.constants import DEFAULT_CAPTURE_LAYER_RATIO
    assert result.best_layer == int(28 * DEFAULT_CAPTURE_LAYER_RATIO)
    assert result.scores == ()  # no sweep was run


def test_select_query_layer_consults_registry_first(tmp_path):
    import tardigrade_hooks.calibration_registry as _reg_mod
    reg = CalibrationRegistry(tmp_path / "calib.json")
    cached = CalibrationResult(
        model_id="cached/model",
        tardigrade_db_version=_reg_mod._current_version(),
        timestamp_iso="2026-05-19T00:00:00",
        n_layers=12,
        hidden_size=512,
        best_layer=7,
        scores=(LayerScore(layer=7, kind="attention", top1=20, top5=20),),
    )
    reg.save(cached)

    # Mock model whose name_or_path matches the cached key.
    model = MagicMock()
    model.config.name_or_path = "cached/model"
    model.config.num_hidden_layers = 12
    model.config.hidden_size = 512
    tokenizer = MagicMock()

    strategy = MagicMock(spec=CalibrationStrategy)
    result = select_query_layer(
        model, tokenizer, registry=reg, strategy=strategy
    )
    assert result.best_layer == 7
    strategy.run.assert_not_called()  # cache hit → no sweep


def test_select_query_layer_saves_to_registry_when_provided(tmp_path):
    reg = CalibrationRegistry(tmp_path / "calib.json")

    import tardigrade_hooks.calibration_registry as _reg_mod
    fake_result = CalibrationResult(
        model_id="new/model",
        tardigrade_db_version=_reg_mod._current_version(),
        timestamp_iso="2026-05-19T00:00:00",
        n_layers=4,
        hidden_size=8,
        best_layer=2,
        scores=(LayerScore(layer=2, kind="attention", top1=10, top5=20),),
    )

    strategy = MagicMock(spec=CalibrationStrategy)
    strategy.run.return_value = fake_result

    model = MagicMock()
    model.config.name_or_path = "new/model"
    model.config.num_hidden_layers = 4
    model.config.hidden_size = 8
    tokenizer = MagicMock()

    select_query_layer(model, tokenizer, registry=reg, strategy=strategy)

    cached = reg.load("new/model")
    assert cached is not None
    assert cached.best_layer == 2


# ---- LinearSweepStrategy behaviour ----

def test_linear_sweep_returns_one_score_per_hidden_state_index():
    # Use a trivial pseudo-model that returns a fixed hidden-state
    # sequence per layer. We don't need a real transformer for this
    # contract test — just an object with the right interface.
    from tardigrade_hooks.calibrate import LinearSweepStrategy

    strategy = LinearSweepStrategy()
    sentinel = object()

    # Patch the internal forward helper to avoid loading any model.
    import tardigrade_hooks.calibrate as cal_mod

    n_layers = 4
    n_hidden_states = n_layers + 1  # embeddings + per-layer outputs

    def fake_forward(model, tokenizer, text, *, wrap_chat, adapter):
        # Return a list of n_hidden_states arrays, each of shape (3, 4)
        # — three tokens, hidden_size=4. Distinct content per index so
        # retrieval can discriminate.
        import numpy as np
        rng = np.random.RandomState(hash(text) & 0xFFFFFFFF)
        return [rng.randn(3, 4).astype("float32") for _ in range(n_hidden_states)], [], 3

    import tardigrade_hooks.calibrate as cal_mod
    original = cal_mod._compute_per_layer_hidden_states
    cal_mod._compute_per_layer_hidden_states = fake_forward
    try:
        model = MagicMock()
        model.config.num_hidden_layers = n_layers
        model.config.hidden_size = 4
        model.config.name_or_path = "fake"
        result = strategy.run(
            model, tokenizer=MagicMock(),
            corpus=[("fact-a", "query-a"), ("fact-b", "query-b")],
        )
    finally:
        cal_mod._compute_per_layer_hidden_states = original

    assert len(result.scores) == n_hidden_states
    layers_seen = [s.layer for s in result.scores]
    assert layers_seen == list(range(n_hidden_states))


# ---- GPU end-to-end ----

@pytest.mark.gpu
def test_calibrate_on_qwen3_picks_layer_with_high_recall():
    # On a known-good uniform-softmax model, the calibration should
    # empirically find a layer that gets at least 80% top-5 on the
    # bundled corpus.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda")
    model.train(False)

    result = select_query_layer(model, tok)
    best_score = result.scores[result.best_layer]
    assert best_score.top5 >= int(0.8 * 20), (
        f"Expected top-5 ≥ 16/20 on Qwen3-0.6B; got {best_score.top5}/20 "
        f"at best_layer={result.best_layer}"
    )


@pytest.mark.gpu
def test_calibrate_on_recurrentgemma_does_not_crash():
    # On a hybrid model where the default layer may be wrong, calibration
    # still returns SOMETHING. Recall numbers are not asserted — those
    # are empirical findings the test catalogues, not invariants.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        model_id = "google/recurrentgemma-2b-it"
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda")
    except Exception as e:
        pytest.skip(f"RecurrentGemma not available locally: {e}")
    model.train(False)

    result = select_query_layer(model, tok)
    assert 0 <= result.best_layer < result.n_layers + 1
    assert len(result.scores) == result.n_layers + 1
