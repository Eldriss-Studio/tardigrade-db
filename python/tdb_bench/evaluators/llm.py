"""LLM-gated evaluator with provider chain and deterministic fallback.

Chain of Responsibility: iterates JudgeProvider instances until one
succeeds. Falls back to DeterministicEvaluator if all providers fail
or none are available.
"""

from __future__ import annotations

import re

from tdb_bench.contracts import Evaluator
from tdb_bench.models import BenchmarkItem, ScoreResult

from .deterministic import DeterministicEvaluator
from .providers import JudgeProvider

_JUDGE_PROMPT = (
    'Score answer correctness from 0.0 to 1.0 as JSON {{"score": number}}.\n'
    "Question: {question}\n"
    "Ground truth: {ground_truth}\n"
    "Answer: {answer}"
)

_PASS_THRESHOLD = 0.8


class LLMGatedEvaluator(Evaluator):
    """Chain of Responsibility over LLM judge providers.

    Tries each provider in order. First successful response wins.
    Falls back to deterministic scoring if all providers fail.
    """

    def __init__(self, providers: list[JudgeProvider]) -> None:
        self._providers = providers
        self._fallback = DeterministicEvaluator()

    def score(self, item: BenchmarkItem, answer: str, evidence: list[str]) -> ScoreResult:
        prompt = _JUDGE_PROMPT.format(
            question=item.question,
            ground_truth=item.ground_truth,
            answer=answer,
        )

        for provider in self._providers:
            if not provider.is_available():
                continue
            try:
                text = provider.judge(prompt)
                parsed = _parse_score(text)
                verdict = "pass" if parsed >= _PASS_THRESHOLD else "fail"
                return ScoreResult(
                    score=parsed,
                    judgment=f"llm_{verdict}",
                    evaluator_mode=f"llm_{provider.name()}",
                )
            except Exception:
                continue

        fallback = self._fallback.score(item, answer, evidence)
        return ScoreResult(
            score=fallback.score,
            judgment=f"{fallback.judgment}_fallback",
            evaluator_mode="deterministic_fallback",
        )


def _parse_score(text: str) -> float:
    """Extract score from LLM judge response. Returns 0.0 on parse failure."""
    if not text:
        return 0.0
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text)
    match = re.search(r'\{\s*"score"\s*:\s*([0-9]*\.?[0-9]+)\s*\}', cleaned)
    if not match:
        return 0.0
    score = float(match.group(1))
    return max(0.0, min(1.0, round(score, 6)))
