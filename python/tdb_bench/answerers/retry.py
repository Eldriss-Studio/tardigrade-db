"""``RetryingGenerator`` — Circuit Breaker over any :class:`AnswerGenerator`.

Bounded exponential backoff. After ``max_attempts`` consecutive
failures, raises :class:`GeneratorExhausted` (distinct from arbitrary
``RuntimeError``) so the bench adapter can record a per-item failure
and continue the run instead of crashing.

The ``sleep`` callable is injected (defaults to :func:`time.sleep`) so
unit tests can observe backoff delays without actually waiting.
"""

from __future__ import annotations

import time
from typing import Callable

from .base import AnswerGenerator, GeneratorExhausted
from .constants import (
    LLM_GATE_RETRY_BACKOFF_FACTOR,
    LLM_GATE_RETRY_INITIAL_DELAY_S,
    LLM_GATE_RETRY_MAX_ATTEMPTS,
)


class RetryingGenerator(AnswerGenerator):
    """Decorator that retries a wrapped generator with exponential backoff."""

    def __init__(
        self,
        inner: AnswerGenerator,
        *,
        max_attempts: int = LLM_GATE_RETRY_MAX_ATTEMPTS,
        initial_delay_s: float = LLM_GATE_RETRY_INITIAL_DELAY_S,
        backoff_factor: float = LLM_GATE_RETRY_BACKOFF_FACTOR,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._inner = inner
        self._max_attempts = max_attempts
        self._initial_delay_s = initial_delay_s
        self._backoff_factor = backoff_factor
        self._sleep = sleep

    def generate(self, prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                return self._inner.generate(prompt)
            except Exception as exc:  # noqa: BLE001 — retry on any failure
                last_error = exc
                is_last_attempt = attempt == self._max_attempts - 1
                if is_last_attempt:
                    break
                delay = self._initial_delay_s * (self._backoff_factor ** attempt)
                self._sleep(delay)
        raise GeneratorExhausted(
            f"generator failed after {self._max_attempts} attempts: {last_error}"
        ) from last_error
