"""LLM judge providers — Strategy pattern for multi-provider evaluation.

Each provider encapsulates one LLM API's HTTP call. The evaluator
iterates providers via Chain of Responsibility until one succeeds.
"""

from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod


class JudgeProvider(ABC):
    """Strategy: sends a judge prompt to an LLM API, returns raw text."""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def judge(self, prompt: str) -> str: ...


class DeepSeekProvider(JudgeProvider):
    """DeepSeek Chat Completions API."""

    _URL = "https://api.deepseek.com/v1/chat/completions"
    _MODEL = "deepseek-chat"
    _ENV_VAR = "DEEPSEEK_API_KEY"
    _TIMEOUT = 15

    def name(self) -> str:
        return "deepseek"

    def is_available(self) -> bool:
        return bool(os.getenv(self._ENV_VAR, "").strip())

    def judge(self, prompt: str) -> str:
        api_key = os.getenv(self._ENV_VAR, "").strip()
        if not api_key:
            raise ValueError(f"{self.name()} provider not available: {self._ENV_VAR} not set")
        return _chat_completions(self._URL, api_key, self._MODEL, prompt, self._TIMEOUT)


class OpenAIProvider(JudgeProvider):
    """OpenAI Chat Completions API."""

    _URL = "https://api.openai.com/v1/chat/completions"
    _ENV_VAR = "OPENAI_API_KEY"
    _TIMEOUT = 10

    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        self._model = model

    def name(self) -> str:
        return "openai"

    def is_available(self) -> bool:
        return bool(os.getenv(self._ENV_VAR, "").strip())

    def judge(self, prompt: str) -> str:
        api_key = os.getenv(self._ENV_VAR, "").strip()
        if not api_key:
            raise ValueError(f"{self.name()} provider not available: {self._ENV_VAR} not set")
        return _chat_completions(self._URL, api_key, self._model, prompt, self._TIMEOUT)


def _chat_completions(url: str, api_key: str, model: str, prompt: str, timeout: int) -> str:
    """Shared Chat Completions call — same format for DeepSeek and OpenAI."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 60,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]
