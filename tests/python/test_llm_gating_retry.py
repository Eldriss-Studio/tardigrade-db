"""ATDD: RetryingGenerator decorator.

Bounded retry with exponential backoff. Surfaces ``GeneratorExhausted``
after ``max_attempts`` so the bench adapter can record a per-item
failure rather than crashing the run.
"""

from __future__ import annotations

import pytest

from tdb_bench.answerers import AnswerGenerator, GeneratorExhausted
from tdb_bench.answerers.constants import (
    LLM_GATE_RETRY_BACKOFF_FACTOR,
    LLM_GATE_RETRY_INITIAL_DELAY_S,
    LLM_GATE_RETRY_MAX_ATTEMPTS,
)
from tdb_bench.answerers.retry import RetryingGenerator


class _FlakyGen(AnswerGenerator):
    """Raises ``RuntimeError`` ``fail_n`` times then returns ``"ok"``."""

    def __init__(self, fail_n: int) -> None:
        self._fail_n = fail_n
        self.attempts = 0

    def generate(self, prompt: str) -> str:
        self.attempts += 1
        if self.attempts <= self._fail_n:
            raise RuntimeError(f"transient failure {self.attempts}")
        return "ok"


class _AlwaysFailGen(AnswerGenerator):
    def __init__(self) -> None:
        self.attempts = 0

    def generate(self, prompt: str) -> str:
        self.attempts += 1
        raise RuntimeError("persistent failure")


class TestRetryingGeneratorRecovers:
    def test_succeeds_after_transient_failure(self):
        inner = _FlakyGen(fail_n=2)
        wrapped = RetryingGenerator(inner, max_attempts=3, sleep=lambda s: None)

        result = wrapped.generate("x")

        assert result == "ok"
        assert inner.attempts == 3

    def test_succeeds_first_try_no_retry(self):
        inner = _FlakyGen(fail_n=0)
        wrapped = RetryingGenerator(inner, max_attempts=3, sleep=lambda s: None)

        assert wrapped.generate("x") == "ok"
        assert inner.attempts == 1


class TestRetryingGeneratorExhausts:
    def test_raises_generator_exhausted_after_max_attempts(self):
        inner = _AlwaysFailGen()
        wrapped = RetryingGenerator(inner, max_attempts=3, sleep=lambda s: None)

        with pytest.raises(GeneratorExhausted):
            wrapped.generate("x")
        assert inner.attempts == 3

    def test_max_attempts_default_from_constants(self):
        inner = _AlwaysFailGen()
        wrapped = RetryingGenerator(inner, sleep=lambda s: None)

        with pytest.raises(GeneratorExhausted):
            wrapped.generate("x")
        assert inner.attempts == LLM_GATE_RETRY_MAX_ATTEMPTS


class TestRetryingGeneratorBackoff:
    def test_observes_exponential_backoff_delays(self):
        inner = _FlakyGen(fail_n=3)
        observed_sleeps: list[float] = []
        wrapped = RetryingGenerator(
            inner,
            max_attempts=4,
            initial_delay_s=1.0,
            backoff_factor=2.0,
            sleep=observed_sleeps.append,
        )

        wrapped.generate("x")

        # 3 failures → 3 sleeps. Each = initial × factor^attempt_index.
        assert observed_sleeps == [1.0, 2.0, 4.0]

    def test_default_backoff_pins_to_constants(self):
        inner = _FlakyGen(fail_n=1)
        observed_sleeps: list[float] = []
        wrapped = RetryingGenerator(inner, sleep=observed_sleeps.append)

        wrapped.generate("x")

        assert observed_sleeps == [LLM_GATE_RETRY_INITIAL_DELAY_S]


class TestRetryingGeneratorSubstitutability:
    def test_is_an_answer_generator(self):
        wrapped = RetryingGenerator(_FlakyGen(fail_n=0))
        assert isinstance(wrapped, AnswerGenerator)
