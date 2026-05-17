"""ATDD: build_answerer_from_env factory (part of the bench harness).

Env-driven Factory Method: assembles the answerer chain
(``Retry → Cache → DeepSeek``) from environment variables so the
bench runner can flip providers/models without code changes.

Env contract:

* ``TDB_LLM_GATE_PROVIDER`` ∈ {``deepseek``, ``openai``, ``mock``}
  (default ``deepseek``).
* ``TDB_LLM_GATE_MODEL`` overrides the provider's default model.
* ``TDB_LLM_GATE_CACHE_DIR`` enables response cache when set.

A returned tuple carries ``(generator, model_label)`` so the adapter
records the actual model in its metadata.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from tdb_bench.answerers import (
    AnswerGenerator,
    CachedAnswerGenerator,
    DeepSeekAnswerer,
    MockAnswerGenerator,
    OpenAIAnswerer,
    RetryingGenerator,
)
from tdb_bench.answerers.constants import LLM_GATE_DEFAULT_MODEL
from tdb_bench.answerers.factory import build_answerer_from_env


class TestProviderSelection:
    @patch.dict(os.environ, {"TDB_LLM_GATE_PROVIDER": "mock"}, clear=False)
    def test_mock_provider_returns_mock_generator(self):
        gen, label = build_answerer_from_env()
        # Strip retry/cache decorators to inspect core.
        assert label == "mock"
        assert _has_inner(gen, MockAnswerGenerator)

    @patch.dict(os.environ, {"TDB_LLM_GATE_PROVIDER": "deepseek"}, clear=False)
    def test_deepseek_provider_returns_deepseek(self):
        gen, label = build_answerer_from_env()
        assert label == LLM_GATE_DEFAULT_MODEL
        assert _has_inner(gen, DeepSeekAnswerer)

    @patch.dict(os.environ, {"TDB_LLM_GATE_PROVIDER": "openai"}, clear=False)
    def test_openai_provider_returns_openai(self):
        gen, label = build_answerer_from_env()
        assert _has_inner(gen, OpenAIAnswerer)
        assert "gpt" in label.lower()

    @patch.dict(os.environ, {}, clear=True)
    def test_default_provider_is_deepseek(self):
        gen, label = build_answerer_from_env()
        assert _has_inner(gen, DeepSeekAnswerer)
        assert label == LLM_GATE_DEFAULT_MODEL

    @patch.dict(os.environ, {"TDB_LLM_GATE_PROVIDER": "bogus"}, clear=False)
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="provider"):
            build_answerer_from_env()


class TestModelOverride:
    @patch.dict(
        os.environ,
        {"TDB_LLM_GATE_PROVIDER": "deepseek", "TDB_LLM_GATE_MODEL": "deepseek-reasoner"},
        clear=False,
    )
    def test_model_env_overrides_default(self):
        _, label = build_answerer_from_env()
        assert label == "deepseek-reasoner"

    @patch.dict(
        os.environ,
        {"TDB_LLM_GATE_PROVIDER": "mock", "TDB_LLM_GATE_MODEL": "custom-mock"},
        clear=False,
    )
    def test_model_env_used_for_mock_too(self):
        # Even mock records the model label so test runs reflect intent.
        _, label = build_answerer_from_env()
        assert label == "custom-mock"


class TestRetryWrapping:
    @patch.dict(os.environ, {"TDB_LLM_GATE_PROVIDER": "mock"}, clear=False)
    def test_retry_is_outermost_wrapper(self):
        gen, _ = build_answerer_from_env()
        # The factory always wraps with retry so transient API errors
        # don't fail the whole bench run.
        assert isinstance(gen, RetryingGenerator)


class TestCacheWrapping:
    @patch.dict(os.environ, {"TDB_LLM_GATE_PROVIDER": "mock"}, clear=True)
    def test_no_cache_when_env_unset(self):
        os.environ.pop("TDB_LLM_GATE_CACHE_DIR", None)
        gen, _ = build_answerer_from_env()
        # Cache layer should not be present.
        assert not _has_inner(gen, CachedAnswerGenerator)

    def test_cache_layer_present_when_dir_env_set(self, tmp_path: Path):
        with patch.dict(
            os.environ,
            {
                "TDB_LLM_GATE_PROVIDER": "mock",
                "TDB_LLM_GATE_CACHE_DIR": str(tmp_path),
            },
            clear=False,
        ):
            gen, _ = build_answerer_from_env()
            assert _has_inner(gen, CachedAnswerGenerator)


def _has_inner(gen: AnswerGenerator, cls: type) -> bool:
    """Walk through wrapped decorators looking for an instance of ``cls``."""
    seen: set[int] = set()
    cursor: object = gen
    while cursor is not None and id(cursor) not in seen:
        seen.add(id(cursor))
        if isinstance(cursor, cls):
            return True
        cursor = getattr(cursor, "_inner", None)
    return False
