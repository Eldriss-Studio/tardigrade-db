# ATDD acceptance tests for `tardigrade_hooks.is_supported`.
#
# The Inspector returns a typed CompatibilityReport so consumers can
# pre-flight a (model, tokenizer) pair before deploying it into a
# KnowledgePackStore. Pure-config — no forward pass, no device hop.
#
# Coverage shape:
#
#   - one real-model test (GPT-2) for the uniform-softmax happy path;
#   - mocked configs for hybrid (Griffin 3:1) and pure-recurrent shapes,
#     which don't require pulling Gemma / Mamba weights at test time;
#   - tokenizer-shape tests covering the chat-template probes.

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tardigrade_hooks import CompatibilityReport, is_supported


# -- fixtures -----------------------------------------------------------------


class _WorkingTokenizer:
    """Tokenizer mock with a callable ``apply_chat_template`` and a chat_template
    string (the common production case)."""

    chat_template = (
        '{% for message in messages %}{{ message["content"] }}{% endfor %}'
    )

    def apply_chat_template(self, messages, **kwargs):
        return "".join(m["content"] for m in messages)


class _NoApplyTokenizer:
    """Some legacy tokenizers expose no chat-template surface at all."""

    chat_template = None


class _ApplyButNoTemplateTokenizer:
    """Modern tokenizer object that exposes ``apply_chat_template`` but
    has not had a template assigned (``chat_template`` is None). Recall
    can still work, but every fact will be encoded without a wrap."""

    chat_template = None

    def apply_chat_template(self, messages, **kwargs):
        return "".join(m["content"] for m in messages)


def _mock_uniform_softmax_model(n_layers: int = 12):
    """Build a minimal model-shaped mock for the Inspector — only the
    config attributes the probe reads. No forward, no parameters."""
    cfg = SimpleNamespace(
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        hidden_size=64,
    )
    return SimpleNamespace(config=cfg)


def _mock_griffin_hybrid_model(n_attn: int = 8, n_recurrent: int = 18):
    """Build a hybrid-shape config mirroring RecurrentGemma-2B-it's
    Griffin layout: 8 attention layers spaced every 3rd position
    among 26 total."""
    # Spacing: ['recurrent', 'recurrent', 'attention'] × 8 + leftover recurrents.
    block_types: list[str] = []
    attn_emitted = 0
    while attn_emitted < n_attn:
        block_types.extend(["recurrent", "recurrent", "attention"])
        attn_emitted += 1
    # Pad with extra recurrents so total length matches Griffin's 26.
    total = n_attn + n_recurrent
    while len(block_types) < total:
        block_types.append("recurrent")
    cfg = SimpleNamespace(
        num_hidden_layers=total,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        hidden_size=64,
        layers_block_type=block_types,
    )
    return SimpleNamespace(config=cfg)


def _mock_pure_recurrent_model(n_layers: int = 8):
    """SSM/Mamba/RWKV-style config — every layer is recurrent."""
    cfg = SimpleNamespace(
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        hidden_size=64,
        layer_types=["recurrent"] * n_layers,
    )
    return SimpleNamespace(config=cfg)


# -- 1: real GPT-2 happy path -------------------------------------------------


def test_is_supported_gpt2_uniform_softmax():
    """GIVEN GPT-2 (uniform-softmax, 12 layers),
    WHEN is_supported runs,
    THEN architecture=uniform_softmax, recommended_strategy=hidden_state,
         is_supported=True, no blockers."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.chat_template = (
        '{% for message in messages %}{{ message["content"] }}{% endfor %}'
    )

    report = is_supported(model, tok)

    assert report.is_supported is True
    assert report.architecture == "uniform_softmax"
    assert report.n_hidden_layers == 12
    assert report.n_softmax_layers == 12
    assert report.recommended_strategy == "hidden_state"
    assert report.adapter_type in ("LegacySystemAdapter", "UserMessageAdapter")
    assert report.blockers == ()


# -- 2: hybrid model (Griffin shape) ------------------------------------------


def test_is_supported_hybrid_recommends_calibration():
    """GIVEN a hybrid config (Griffin 3:1, 8 softmax of 26),
    WHEN is_supported runs,
    THEN architecture=hybrid, n_softmax_layers=8,
         recommended_strategy=calibration_required, is_supported=True,
         notes mention K-vector / calibration."""
    model = _mock_griffin_hybrid_model(n_attn=8, n_recurrent=18)
    tok = _WorkingTokenizer()

    report = is_supported(model, tok)

    assert report.is_supported is True
    assert report.architecture == "hybrid"
    assert report.n_hidden_layers == 26
    assert report.n_softmax_layers == 8
    assert report.recommended_strategy == "calibration_required"
    # Notes should point the consumer at the next step.
    assert any("calibration" in n.lower() for n in report.notes)
    assert report.blockers == ()


# -- 3: pure-recurrent model (Mamba shape) ------------------------------------


def test_is_supported_pure_recurrent_is_unsupported():
    """GIVEN a pure-SSM/Mamba config (no softmax layers),
    WHEN is_supported runs,
    THEN is_supported=False, blocker explains the cause."""
    model = _mock_pure_recurrent_model(n_layers=8)
    tok = _WorkingTokenizer()

    report = is_supported(model, tok)

    assert report.is_supported is False
    assert report.n_softmax_layers == 0
    assert any("softmax" in b.lower() for b in report.blockers)


# -- 4: tokenizer without apply_chat_template ---------------------------------


def test_is_supported_tokenizer_without_apply_chat_template_is_unsupported():
    """GIVEN a tokenizer with no ``apply_chat_template`` method,
    WHEN is_supported runs,
    THEN is_supported=False, blocker mentions chat template."""
    model = _mock_uniform_softmax_model()
    tok = _NoApplyTokenizer()

    report = is_supported(model, tok)

    assert report.is_supported is False
    assert any("chat" in b.lower() for b in report.blockers)


# -- 5: tokenizer with apply_chat_template but chat_template is None ----------


def test_is_supported_tokenizer_chat_template_none_produces_note():
    """GIVEN a tokenizer that has ``apply_chat_template`` but
    ``chat_template is None`` (a common pre-finetune state),
    WHEN is_supported runs,
    THEN is_supported=True, but a note flags the wrap will degrade."""
    model = _mock_uniform_softmax_model()
    tok = _ApplyButNoTemplateTokenizer()

    report = is_supported(model, tok)

    assert report.is_supported is True
    assert any("chat_template" in n.lower() or "wrap" in n.lower()
               for n in report.notes)


# -- 6: missing num_hidden_layers ---------------------------------------------


def test_is_supported_config_missing_num_hidden_layers_is_unsupported():
    """GIVEN a model config without ``num_hidden_layers``,
    WHEN is_supported runs,
    THEN is_supported=False, blocker mentions the missing attribute."""
    model = SimpleNamespace(config=SimpleNamespace())
    tok = _WorkingTokenizer()

    report = is_supported(model, tok)

    assert report.is_supported is False
    assert any("num_hidden_layers" in b for b in report.blockers)


# -- 7: report is a value object ----------------------------------------------


def test_compatibility_report_is_frozen_and_hashable():
    """CompatibilityReport must be a Value Object — immutable + hashable
    so consumers can stash it in sets / dict keys / cache it across
    boots without worrying about identity vs equality."""
    a = CompatibilityReport(
        is_supported=True,
        architecture="uniform_softmax",
        n_hidden_layers=12,
        n_softmax_layers=12,
        recommended_strategy="hidden_state",
        adapter_type="UserMessageAdapter",
        notes=(),
        blockers=(),
    )
    b = CompatibilityReport(
        is_supported=True,
        architecture="uniform_softmax",
        n_hidden_layers=12,
        n_softmax_layers=12,
        recommended_strategy="hidden_state",
        adapter_type="UserMessageAdapter",
        notes=(),
        blockers=(),
    )

    assert a == b
    assert hash(a) == hash(b)
    # Frozen: assignment must raise.
    with pytest.raises((AttributeError, Exception)):
        a.is_supported = False  # type: ignore[misc]
