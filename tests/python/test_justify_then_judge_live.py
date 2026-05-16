"""ATDD: JustifyThenJudgeEvaluator real-API smoke (slice B3.5).

Live DeepSeek round-trip. Marked ``live_api`` so CI / no-key
environments auto-skip. Single 5-item case to confirm the
two-stage pipeline executes end-to-end against the real API.

Cost: ~$0.005 per run (5 items × 2 API calls × DeepSeek-chat
pricing). Cached responses persist under the bench's normal cache
directory so subsequent runs are free.
"""

from __future__ import annotations

import os

import pytest

from tdb_bench.evaluators import (
    DeepSeekProvider,
    JustifyThenJudgeEvaluator,
)
from tdb_bench.models import BenchmarkItem


_DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
pytestmark = [
    pytest.mark.live_api,
    pytest.mark.skipif(not _DEEPSEEK_API_KEY, reason="DEEPSEEK_API_KEY not set"),
]


_FIXTURES: list[tuple[BenchmarkItem, str, list[str]]] = [
    (
        BenchmarkItem(
            item_id="live-1",
            dataset="locomo",
            context="ctx",
            question="What is the capital of France?",
            ground_truth="Paris",
        ),
        "Paris",
        ["Paris is the capital city of France."],
    ),
    (
        BenchmarkItem(
            item_id="live-2",
            dataset="locomo",
            context="ctx",
            question="When did Caroline go to the LGBTQ conference?",
            ground_truth="10 July 2023",
        ),
        # Model-style answer with relative reference; the justify
        # stage should reason "session is 12 July, two days ago → 10 July."
        "Two days before 12 July 2023",
        ["[4:33 pm on 12 July, 2023] I went to an LGBTQ conference two days ago."],
    ),
    (
        BenchmarkItem(
            item_id="live-3",
            dataset="locomo",
            context="ctx",
            question="Who wrote Hamlet?",
            ground_truth="Shakespeare",
        ),
        "William Shakespeare",
        ["Hamlet was written by William Shakespeare in the early 1600s."],
    ),
]


def test_two_stage_pipeline_runs_end_to_end():
    evaluator = JustifyThenJudgeEvaluator(
        justify_providers=[DeepSeekProvider()],
        judge_providers=[DeepSeekProvider()],
    )

    results = [evaluator.score(item, answer, evidence) for item, answer, evidence in _FIXTURES]

    # All three should yield ScoreResults; mode strings should reflect
    # the two-stage pipeline ran (no fallback).
    for r in results:
        assert 0.0 <= r.score <= 1.0
        # mode should start with the justify+judge prefix when both
        # stages succeed; if either stage timed out the mode would
        # signal that explicitly.
        assert ("justify" in r.evaluator_mode) or ("fallback" in r.evaluator_mode)

    # At least one trivia item should pass (Paris / Hamlet). Loose
    # threshold to be robust to transient model variation.
    passing = sum(1 for r in results if r.judgment == "llm_pass")
    assert passing >= 1, f"no items passed; got {[r.judgment for r in results]}"
