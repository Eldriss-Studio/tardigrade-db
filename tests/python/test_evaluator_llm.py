"""ATDD tests for LLMGatedEvaluator — Chain of Responsibility over providers."""

from unittest.mock import MagicMock

import pytest

from tdb_bench.evaluators.llm import LLMGatedEvaluator, _parse_score
from tdb_bench.evaluators.providers import JudgeProvider
from tdb_bench.models import BenchmarkItem


def _make_item():
    return BenchmarkItem(
        item_id="test-1",
        dataset="test",
        context="Test context.",
        question="What color is the sky?",
        ground_truth="blue",
    )


def _mock_provider(name: str, available: bool, response: str | None = None, error: Exception | None = None):
    p = MagicMock(spec=JudgeProvider)
    p.name.return_value = name
    p.is_available.return_value = available
    if error:
        p.judge.side_effect = error
    elif response is not None:
        p.judge.return_value = response
    return p


class TestChainOfResponsibility:
    def test_uses_first_available_provider(self):
        p1 = _mock_provider("alpha", available=True, response='{"score": 0.9}')
        p2 = _mock_provider("beta", available=True, response='{"score": 0.5}')
        ev = LLMGatedEvaluator(providers=[p1, p2])
        result = ev.score(_make_item(), "blue", [])
        assert result.evaluator_mode == "llm_alpha"
        assert result.score == 0.9

    def test_skips_unavailable_provider(self):
        p1 = _mock_provider("alpha", available=False)
        p2 = _mock_provider("beta", available=True, response='{"score": 0.8}')
        ev = LLMGatedEvaluator(providers=[p1, p2])
        result = ev.score(_make_item(), "blue", [])
        assert result.evaluator_mode == "llm_beta"
        p1.judge.assert_not_called()

    def test_falls_through_on_provider_error(self):
        p1 = _mock_provider("alpha", available=True, error=ConnectionError("timeout"))
        p2 = _mock_provider("beta", available=True, response='{"score": 0.7}')
        ev = LLMGatedEvaluator(providers=[p1, p2])
        result = ev.score(_make_item(), "blue", [])
        assert result.evaluator_mode == "llm_beta"

    def test_deterministic_fallback_when_all_fail(self):
        p1 = _mock_provider("alpha", available=True, error=ConnectionError("down"))
        ev = LLMGatedEvaluator(providers=[p1])
        result = ev.score(_make_item(), "blue", [])
        assert result.evaluator_mode == "deterministic_fallback"

    def test_deterministic_fallback_when_no_providers(self):
        ev = LLMGatedEvaluator(providers=[])
        result = ev.score(_make_item(), "blue", [])
        assert result.evaluator_mode == "deterministic_fallback"

    def test_judgment_pass_above_threshold(self):
        p = _mock_provider("alpha", available=True, response='{"score": 0.85}')
        ev = LLMGatedEvaluator(providers=[p])
        result = ev.score(_make_item(), "blue", [])
        assert result.judgment == "llm_pass"

    def test_judgment_fail_below_threshold(self):
        p = _mock_provider("alpha", available=True, response='{"score": 0.3}')
        ev = LLMGatedEvaluator(providers=[p])
        result = ev.score(_make_item(), "blue", [])
        assert result.judgment == "llm_fail"


class TestParseScore:
    def test_valid_json(self):
        assert _parse_score('{"score": 0.85}') == 0.85

    def test_json_in_markdown_code_fence(self):
        assert _parse_score('```json\n{"score": 0.9}\n```') == 0.9

    def test_json_with_surrounding_text(self):
        assert _parse_score('The answer is correct. {"score": 1.0}') == 1.0

    def test_malformed_json_returns_zero(self):
        assert _parse_score("not json at all") == 0.0

    def test_empty_string_returns_zero(self):
        assert _parse_score("") == 0.0

    def test_clamps_to_range(self):
        assert _parse_score('{"score": 1.5}') == 1.0
        assert _parse_score('{"score": -0.3}') == 0.0
