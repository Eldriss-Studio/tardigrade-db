"""DeepSeek Chat Completions answerer.

OpenAI-compatible wire format, so this is a thin config layer over
:class:`OpenAICompatibleAnswerer`. Default model: ``deepseek-chat``
(DeepSeek-V3); swap to ``deepseek-reasoner`` via env var for ~2×
the cost and the reasoning trace.
"""

from __future__ import annotations

from .constants import (
    LLM_GATE_DEEPSEEK_API_KEY_ENV,
    LLM_GATE_DEEPSEEK_URL,
    LLM_GATE_DEFAULT_MODEL,
)
from .openai_compatible import OpenAICompatibleAnswerer


class DeepSeekAnswerer(OpenAICompatibleAnswerer):
    """Answerer backed by api.deepseek.com."""

    def __init__(self, model: str = LLM_GATE_DEFAULT_MODEL) -> None:
        super().__init__(
            url=LLM_GATE_DEEPSEEK_URL,
            api_key_env=LLM_GATE_DEEPSEEK_API_KEY_ENV,
            model=model,
        )
