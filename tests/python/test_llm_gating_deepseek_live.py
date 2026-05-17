"""ATDD: DeepSeekAnswerer real-API smoke.

Live API call. Marked ``live_api`` so CI / no-key environments
auto-skip. Single-shot trivia question to confirm the wiring; the
expected substring is robust to common phrasings ("Paris", "the
capital is Paris", "Paris is the capital").

Cost: ~$0.0001 per run.
"""

from __future__ import annotations

import os

import pytest

from tdb_bench.answerers import AnswerGenerator
from tdb_bench.answerers.constants import (
    LLM_GATE_DEEPSEEK_API_KEY_ENV,
    LLM_GATE_DEFAULT_MODEL,
)
from tdb_bench.answerers.deepseek import DeepSeekAnswerer


_LIVE = os.getenv(LLM_GATE_DEEPSEEK_API_KEY_ENV, "").strip()
pytestmark = [
    pytest.mark.live_api,
    pytest.mark.skipif(not _LIVE, reason="DEEPSEEK_API_KEY not set"),
]


def test_deepseek_answers_trivia_from_evidence():
    gen = DeepSeekAnswerer()

    prompt = (
        "Answer the question using only the evidence below. "
        "Be concise — one short phrase.\n\n"
        "Evidence:\n[1] Paris is the capital city of France.\n\n"
        "Question: What is the capital of France?\n"
        "Answer:"
    )
    answer = gen.generate(prompt)

    assert answer
    assert "paris" in answer.lower()


def test_deepseek_is_substitutable():
    assert isinstance(DeepSeekAnswerer(), AnswerGenerator)


def test_deepseek_default_model():
    assert DeepSeekAnswerer().model_name == LLM_GATE_DEFAULT_MODEL
