"""LLM-gated evaluator strategy with deterministic fallback."""

from __future__ import annotations

import json
import os
import urllib.request

from tdb_bench.contracts import Evaluator
from tdb_bench.models import BenchmarkItem, ScoreResult

from .deterministic import DeterministicEvaluator


class LLMGatedEvaluator(Evaluator):
    """Uses LLM judge when credentials exist, otherwise deterministic fallback."""

    def __init__(self, answerer_model: str, judge_model: str) -> None:
        self.answerer_model = answerer_model
        self.judge_model = judge_model
        self.fallback = DeterministicEvaluator()

    def score(self, item: BenchmarkItem, answer: str, evidence: list[str]) -> ScoreResult:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            fallback = self.fallback.score(item, answer, evidence)
            return ScoreResult(
                score=fallback.score,
                judgment=f"{fallback.judgment}_fallback_no_api_key",
                evaluator_mode="deterministic_fallback",
            )

        try:
            score = self._score_with_openai(api_key, item, answer)
            verdict = "pass" if score >= 0.8 else "fail"
            return ScoreResult(score=score, judgment=f"llm_{verdict}", evaluator_mode="llm")
        except Exception:
            fallback = self.fallback.score(item, answer, evidence)
            return ScoreResult(
                score=fallback.score,
                judgment=f"{fallback.judgment}_fallback_llm_error",
                evaluator_mode="deterministic_fallback",
            )

    def _score_with_openai(self, api_key: str, item: BenchmarkItem, answer: str) -> float:
        """Minimal Responses API call; falls back safely when unavailable."""
        payload = {
            "model": self.judge_model,
            "input": (
                "Score answer correctness from 0.0 to 1.0 as JSON {\\\"score\\\": number}.\\n"
                f"Question: {item.question}\\n"
                f"Ground truth: {item.ground_truth}\\n"
                f"Answer: {answer}"
            ),
            "max_output_tokens": 60,
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as response:  # noqa: S310
            body = json.loads(response.read().decode("utf-8"))

        text = _extract_text(body)
        parsed = json.loads(text)
        score = float(parsed.get("score", 0.0))
        return max(0.0, min(1.0, round(score, 6)))


def _extract_text(payload: dict) -> str:
    if "output_text" in payload and payload["output_text"]:
        return str(payload["output_text"])

    outputs = payload.get("output", [])
    for segment in outputs:
        content = segment.get("content", [])
        for part in content:
            if part.get("type") == "output_text":
                return str(part.get("text", "{}"))
    return "{\"score\": 0.0}"
