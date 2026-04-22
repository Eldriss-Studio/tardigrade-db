"""Deterministic evaluator strategy."""

from __future__ import annotations

import re

from tdb_bench.contracts import Evaluator
from tdb_bench.models import BenchmarkItem, ScoreResult


class DeterministicEvaluator(Evaluator):
    """Pure deterministic lexical overlap scorer for smoke portability."""

    def score(self, item: BenchmarkItem, answer: str, evidence: list[str]) -> ScoreResult:
        expected_tokens = _norm_tokens(item.ground_truth)
        answer_tokens = _norm_tokens(answer)

        if not expected_tokens:
            score = 0.0
        else:
            overlap = len(expected_tokens & answer_tokens)
            score = overlap / len(expected_tokens)

        verdict = "pass" if score >= 0.8 else "fail"
        return ScoreResult(
            score=round(score, 6),
            judgment=f"deterministic_{verdict}",
            evaluator_mode="deterministic",
        )


def _norm_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t}
