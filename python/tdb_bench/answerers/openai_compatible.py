"""Base class for OpenAI Chat Completions–compatible answerer APIs.

DeepSeek's API is wire-compatible with OpenAI's Chat Completions
schema, so a single base class covers both with config-only swaps.
Concrete classes :class:`DeepSeekAnswerer` and :class:`OpenAIAnswerer`
supply the URL and env-var name; everything else is shared.

Why duplicate the HTTP call from :mod:`tdb_bench.evaluators.providers`:
the judge uses ``max_tokens=60``, the answerer uses
``LLM_GATE_MAX_TOKENS`` (256 by default). Different roles → different
parameter sets → separate codepath that documents the intent. The
shared 12-line ``urllib`` call isn't worth the cross-module coupling.
"""

from __future__ import annotations

import json
import os
import urllib.request

from .base import AnswerGenerator
from .constants import (
    LLM_GATE_HTTP_TIMEOUT_S,
    LLM_GATE_MAX_TOKENS,
    LLM_GATE_TEMPERATURE,
)


class OpenAICompatibleAnswerer(AnswerGenerator):
    """Template Method base for OpenAI Chat Completions wire protocol."""

    def __init__(
        self,
        *,
        url: str,
        api_key_env: str,
        model: str,
        max_tokens: int = LLM_GATE_MAX_TOKENS,
        temperature: float = LLM_GATE_TEMPERATURE,
        timeout_s: int = LLM_GATE_HTTP_TIMEOUT_S,
    ) -> None:
        self._url = url
        self._api_key_env = api_key_env
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout_s = timeout_s

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str) -> str:
        api_key = os.getenv(self._api_key_env, "").strip()
        if not api_key:
            raise ValueError(
                f"{type(self).__name__} unavailable: {self._api_key_env} not set"
            )

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        req = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout_s) as response:  # noqa: S310
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]
