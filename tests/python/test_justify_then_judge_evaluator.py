"""ATDD: JustifyThenJudgeEvaluator (Track B, slice B3).

Implements the canonical leaderboard pipeline:
``retrieve → answer → justify → judge`` — Mem0/Memobase/ByteRover
all use a justify (reasoning trace) step between answer generation
and final judgment.

Patterns under test:

* **Strategy** — `JustifyThenJudgeEvaluator(Evaluator)` plugs into
  the same slot as `LLMGatedEvaluator`.
* **Chain of Responsibility** — reuses the existing
  ``JudgeProvider`` list pattern for both stages.
* **Template Method** — ``JustifyPromptBuilder`` and
  ``JudgeWithJustificationPromptBuilder`` carry version-pinned
  templates (cache-key safe).

ATDD slices covered here: B3.1, B3.2, B3.3, B3.4, B3.6.
The live integration slice B3.5 lives in
``tests/python/test_justify_then_judge_live.py``.
"""

from __future__ import annotations

import hashlib

import pytest

from tdb_bench.evaluators.justify_then_judge import (
    JUDGE_WITH_JUSTIFICATION_TEMPLATE_VERSION,
    JUSTIFY_MAX_TOKENS,
    JUSTIFY_TEMPLATE_VERSION,
    JudgeWithJustificationPromptBuilder,
    JustifyPromptBuilder,
    JustifyThenJudgeEvaluator,
)
from tdb_bench.evaluators.providers import JudgeProvider
from tdb_bench.models import BenchmarkItem, ScoreResult


# ─── Test doubles ─────────────────────────────────────────────────────────


class _MockProvider(JudgeProvider):
    """Records prompts; returns canned response or raises configured error."""

    def __init__(self, response, name: str = "mock", available: bool = True) -> None:
        self._response = response
        self._name = name
        self._available = available
        self.calls: list[tuple[str, int]] = []

    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available

    def judge(self, prompt: str, max_tokens: int = 60) -> str:
        self.calls.append((prompt, max_tokens))
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _item(q: str = "Who is Sonia's husband?", gt: str = "Aaron") -> BenchmarkItem:
    return BenchmarkItem(
        item_id="t1", dataset="locomo", context="ctx", question=q, ground_truth=gt
    )


# ─── B3.1 — JustifyPromptBuilder ─────────────────────────────────────────


class TestJustifyPromptBuilder:
    _Q = "Who is Sonia's husband?"
    _E = ["Sonia married Aaron in 2019.", "Aaron and Sonia have a dog."]
    _A = "Aaron"

    def test_contains_question_evidence_and_answer(self):
        p = JustifyPromptBuilder().build(question=self._Q, evidence=self._E, answer=self._A)
        assert self._Q in p
        for chunk in self._E:
            assert chunk in p
        assert self._A in p

    def test_instructs_reasoning_trace(self):
        # The whole point of the justify step: ask for a reasoning
        # trace, not a yes/no.
        p = JustifyPromptBuilder().build(question=self._Q, evidence=self._E, answer=self._A)
        lower = p.lower()
        assert any(token in lower for token in ("reason", "justif", "explain", "trace"))

    def test_is_deterministic(self):
        a = JustifyPromptBuilder().build(question=self._Q, evidence=self._E, answer=self._A)
        b = JustifyPromptBuilder().build(question=self._Q, evidence=self._E, answer=self._A)
        assert a == b
        assert hashlib.sha256(a.encode()).hexdigest() == hashlib.sha256(b.encode()).hexdigest()

    def test_changing_inputs_changes_prompt(self):
        a = JustifyPromptBuilder().build(question=self._Q, evidence=self._E, answer=self._A)
        b = JustifyPromptBuilder().build(question=self._Q, evidence=self._E, answer="Brian")
        assert a != b

    def test_template_version_pinned(self):
        assert JustifyPromptBuilder().template_version() == JUSTIFY_TEMPLATE_VERSION


# ─── B3.2 — JudgeWithJustificationPromptBuilder ───────────────────────────


class TestJudgeWithJustificationPromptBuilder:
    def test_includes_question_gt_answer_and_justification(self):
        p = JudgeWithJustificationPromptBuilder().build(
            question="Q?", ground_truth="GT", answer="A", justification="J"
        )
        for marker in ("Q?", "GT", "A", "J"):
            assert marker in p

    def test_template_version_pinned(self):
        assert (
            JudgeWithJustificationPromptBuilder().template_version()
            == JUDGE_WITH_JUSTIFICATION_TEMPLATE_VERSION
        )


# ─── B3.3 — happy path with two mock providers ────────────────────────────


class TestEvaluatorHappyPath:
    def test_runs_justify_then_judge_and_returns_score(self):
        justify = _MockProvider(response="Because the evidence says so.", name="mock_just")
        judge = _MockProvider(response='{"score": 0.9}', name="mock_judge")
        evaluator = JustifyThenJudgeEvaluator(
            justify_providers=[justify],
            judge_providers=[judge],
        )

        result = evaluator.score(
            _item(), answer="Aaron", evidence=["Sonia married Aaron in 2019."]
        )

        assert isinstance(result, ScoreResult)
        assert result.score == 0.9
        assert result.judgment == "llm_pass"
        assert "justify" in result.evaluator_mode  # mode signals the pipeline used
        assert len(justify.calls) == 1
        assert len(judge.calls) == 1
        # The justify call requests the longer max_tokens budget.
        assert justify.calls[0][1] == JUSTIFY_MAX_TOKENS
        # The judge prompt incorporates the justification.
        judge_prompt = judge.calls[0][0]
        assert "Because the evidence says so." in judge_prompt

    def test_pass_threshold_below_returns_fail_judgment(self):
        justify = _MockProvider(response="some trace", name="mock_just")
        judge = _MockProvider(response='{"score": 0.4}', name="mock_judge")
        evaluator = JustifyThenJudgeEvaluator(
            justify_providers=[justify], judge_providers=[judge]
        )

        result = evaluator.score(_item(), answer="x", evidence=["e"])

        assert result.score == 0.4
        assert result.judgment == "llm_fail"


# ─── B3.4 — failure isolation ─────────────────────────────────────────────


class TestEvaluatorFailureIsolation:
    def test_justify_failure_falls_back_to_plain_judge(self):
        # When the justify stage exhausts, the evaluator should still
        # try to judge without a justification. The judge prompt won't
        # contain the (missing) justification but the score path still
        # functions — preserves LLMGatedEvaluator's robustness.
        justify = _MockProvider(response=RuntimeError("justify down"), name="mock_just")
        judge = _MockProvider(response='{"score": 0.5}', name="mock_judge")
        evaluator = JustifyThenJudgeEvaluator(
            justify_providers=[justify], judge_providers=[judge]
        )

        result = evaluator.score(_item(), answer="x", evidence=["e"])

        assert result.score == 0.5
        # Mode signals fallback so analysis can split scored items by
        # whether justification was active.
        assert "no_justification" in result.evaluator_mode

    def test_justify_unavailable_falls_back_to_plain_judge(self):
        justify = _MockProvider(response="ignored", name="mock_just", available=False)
        judge = _MockProvider(response='{"score": 0.7}', name="mock_judge")
        evaluator = JustifyThenJudgeEvaluator(
            justify_providers=[justify], judge_providers=[judge]
        )

        result = evaluator.score(_item(), answer="x", evidence=["e"])

        assert result.score == 0.7
        assert "no_justification" in result.evaluator_mode

    def test_both_stages_fail_falls_back_to_deterministic(self):
        # Last-line defence: when no LLM responds anywhere, scoring
        # still produces a deterministic answer rather than crashing
        # the bench run. Matches LLMGatedEvaluator semantics.
        justify = _MockProvider(response=RuntimeError("down"), name="mock_just")
        judge = _MockProvider(response=RuntimeError("down"), name="mock_judge")
        evaluator = JustifyThenJudgeEvaluator(
            justify_providers=[justify], judge_providers=[judge]
        )

        result = evaluator.score(
            _item(q="What is the capital of France?", gt="Paris"),
            answer="Paris",
            evidence=["Paris is the capital."],
        )

        assert "fallback" in result.evaluator_mode


# ─── B3.6 — Registry wiring ───────────────────────────────────────────────


class TestRegistryWiring:
    def test_create_evaluator_returns_justify_then_judge(self):
        from tdb_bench.registry import RegistryFactory

        evaluator = RegistryFactory.create_evaluator(
            {"mode": "justify_then_judge", "judge_model": "gpt-4.1-mini"}
        )

        assert isinstance(evaluator, JustifyThenJudgeEvaluator)
