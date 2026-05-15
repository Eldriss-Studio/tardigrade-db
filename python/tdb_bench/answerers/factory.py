"""Factory Method assembling an :class:`AnswerGenerator` chain from env vars.

Returned chain (outer → inner):

    RetryingGenerator
      └ CachedAnswerGenerator (only when ``TDB_LLM_GATE_CACHE_DIR`` set)
          └ DeepSeekAnswerer | OpenAIAnswerer | MockAnswerGenerator

Retry is always outermost so transient API errors are absorbed before
they reach the bench adapter. Cache (when enabled) sits *inside* retry
because a cached hit doesn't need retrying.

Env contract:

* ``TDB_LLM_GATE_PROVIDER`` ∈ {``deepseek`` | ``openai`` | ``mock``},
  default ``deepseek``.
* ``TDB_LLM_GATE_MODEL`` overrides the provider's default model name.
* ``TDB_LLM_GATE_CACHE_DIR`` enables disk-backed cache when set.
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import AnswerGenerator
from .cache import CachedAnswerGenerator
from .constants import (
    LLM_GATE_CACHE_DIR_ENV,
    LLM_GATE_DEFAULT_MODEL,
    LLM_GATE_DEFAULT_PROVIDER,
)
from .deepseek import DeepSeekAnswerer
from .mock import MockAnswerGenerator
from .openai import OpenAIAnswerer
from .retry import RetryingGenerator


_PROVIDER_ENV = "TDB_LLM_GATE_PROVIDER"
_MODEL_ENV = "TDB_LLM_GATE_MODEL"

_PROVIDER_DEEPSEEK = "deepseek"
_PROVIDER_OPENAI = "openai"
_PROVIDER_MOCK = "mock"

_OPENAI_DEFAULT_MODEL = "gpt-4.1-mini"
_MOCK_DEFAULT_RESPONSE = "I don't know"


def build_answerer_from_env() -> tuple[AnswerGenerator, str]:
    """Return ``(generator, model_label)`` driven by env vars.

    ``model_label`` is the literal model name (or ``"mock"``) recorded
    in adapter metadata so run records pin which model produced which
    answers.
    """
    provider = os.getenv(_PROVIDER_ENV, LLM_GATE_DEFAULT_PROVIDER).lower()
    model_env = os.getenv(_MODEL_ENV, "").strip()

    base: AnswerGenerator
    label: str
    if provider == _PROVIDER_DEEPSEEK:
        model = model_env or LLM_GATE_DEFAULT_MODEL
        base = DeepSeekAnswerer(model=model)
        label = model
    elif provider == _PROVIDER_OPENAI:
        model = model_env or _OPENAI_DEFAULT_MODEL
        base = OpenAIAnswerer(model=model)
        label = model
    elif provider == _PROVIDER_MOCK:
        base = MockAnswerGenerator(canned=_MOCK_DEFAULT_RESPONSE)
        label = model_env or _PROVIDER_MOCK
    else:
        raise ValueError(
            f"unknown provider {provider!r}; expected one of "
            f"{_PROVIDER_DEEPSEEK!r}, {_PROVIDER_OPENAI!r}, {_PROVIDER_MOCK!r}"
        )

    cache_dir = os.getenv(LLM_GATE_CACHE_DIR_ENV, "").strip()
    if cache_dir:
        base = CachedAnswerGenerator(
            inner=base, model_name=label, cache_dir=Path(cache_dir)
        )

    return RetryingGenerator(base), label
