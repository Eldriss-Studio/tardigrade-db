"""OpenAI Chat Completions answerer.

Default model ``gpt-4.1-mini`` to match the bench config's
``evaluator.answerer_model`` default; ``gpt-4o-mini`` is the Mem0-
comparable target — swap via env var when publishing a citeable
LoCoMo number against Mem0/Memobase/ByteRover.
"""

from __future__ import annotations

from .constants import (
    LLM_GATE_OPENAI_API_KEY_ENV,
    LLM_GATE_OPENAI_URL,
)
from .openai_compatible import OpenAICompatibleAnswerer


_DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


class OpenAIAnswerer(OpenAICompatibleAnswerer):
    """Answerer backed by api.openai.com."""

    def __init__(self, model: str = _DEFAULT_OPENAI_MODEL) -> None:
        super().__init__(
            url=LLM_GATE_OPENAI_URL,
            api_key_env=LLM_GATE_OPENAI_API_KEY_ENV,
            model=model,
        )
