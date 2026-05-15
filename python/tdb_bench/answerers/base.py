"""``AnswerGenerator`` Strategy protocol + retry-exhaustion exception."""

from __future__ import annotations

from abc import ABC, abstractmethod


class AnswerGenerator(ABC):
    """Strategy: turn a fully-assembled prompt into an answer string.

    Subtypes implement only :meth:`generate`. The protocol is
    deliberately thin (one method, no vendor-specific state) so any
    LLM API or local model satisfies it. Decorators (retry, cache)
    wrap a generator with the same interface — Liskov-substitutable.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return the model's answer to ``prompt``.

        Raises:
            GeneratorExhausted: after all retries fail (when wrapped in
                :class:`RetryingGenerator`).
            ValueError: when the underlying provider is unavailable
                (e.g. missing API key).
        """


class GeneratorExhausted(RuntimeError):
    """Raised when an :class:`AnswerGenerator` retry policy gives up.

    Distinct from transient HTTP errors so callers can decide to
    record a per-item ``status="generator_failed"`` and continue the
    run rather than aborting.
    """
