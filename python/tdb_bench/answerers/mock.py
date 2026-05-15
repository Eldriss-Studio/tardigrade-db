"""Test doubles for :class:`AnswerGenerator`.

Two doubles:

* :class:`MockAnswerGenerator` — returns a configured answer and
  records calls. Used by unit tests for the Decorator adapter
  (Slice L3) and the cache + retry decorators (Slices L6/L7).
* :class:`NoOpAnswerGenerator` — Null Object. Returns ``""``. Lets
  the adapter's negative path be tested without API calls.
"""

from __future__ import annotations

from .base import AnswerGenerator


class MockAnswerGenerator(AnswerGenerator):
    """Returns a canned string; records call count and last prompt."""

    def __init__(self, canned: str) -> None:
        self._canned = canned
        self.call_count = 0
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return self._canned


class NoOpAnswerGenerator(AnswerGenerator):
    """Null Object: returns the empty string for any prompt."""

    def generate(self, prompt: str) -> str:  # noqa: ARG002 — interface contract
        return ""
