"""ATDD tests for JudgeProvider — Strategy pattern for LLM judge APIs."""

import os
from unittest.mock import patch

import pytest

from tdb_bench.evaluators.providers import (
    DeepSeekProvider,
    JudgeProvider,
    OpenAIProvider,
)


class TestJudgeProviderABC:
    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            JudgeProvider()

    def test_deepseek_is_substitutable(self):
        assert isinstance(DeepSeekProvider(), JudgeProvider)

    def test_openai_is_substitutable(self):
        assert isinstance(OpenAIProvider(), JudgeProvider)


class TestDeepSeekProvider:
    def test_name(self):
        assert DeepSeekProvider().name() == "deepseek"

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"})
    def test_available_when_key_set(self):
        assert DeepSeekProvider().is_available()

    @patch.dict(os.environ, {}, clear=True)
    def test_unavailable_when_key_missing(self):
        p = DeepSeekProvider()
        os.environ.pop("DEEPSEEK_API_KEY", None)
        assert not p.is_available()

    def test_judge_raises_when_unavailable(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("DEEPSEEK_API_KEY", None)
            p = DeepSeekProvider()
            with pytest.raises(ValueError, match="not available"):
                p.judge("test prompt")


class TestOpenAIProvider:
    def test_name(self):
        assert OpenAIProvider().name() == "openai"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
    def test_available_when_key_set(self):
        assert OpenAIProvider().is_available()

    @patch.dict(os.environ, {}, clear=True)
    def test_unavailable_when_key_missing(self):
        p = OpenAIProvider()
        os.environ.pop("OPENAI_API_KEY", None)
        assert not p.is_available()

    def test_judge_raises_when_unavailable(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            p = OpenAIProvider()
            with pytest.raises(ValueError, match="not available"):
                p.judge("test prompt")

    def test_custom_model(self):
        p = OpenAIProvider(model="gpt-4o")
        assert p.name() == "openai"
