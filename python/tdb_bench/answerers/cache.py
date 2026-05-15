"""``CachedAnswerGenerator`` — Cache-Aside over any :class:`AnswerGenerator`.

Keyed by ``(model_name, prompt_sha256, template_version)``. Three
invalidation axes:

* prompt changes → different hash → miss;
* model swap (e.g. ``deepseek-chat`` → ``gpt-4o-mini``) → different
  partition → miss (cross-model leakage would corrupt comparisons);
* prompt template bumped → different ``PROMPT_TEMPLATE_VERSION`` →
  miss.

Cache lives on disk so it survives process restarts (dev iteration
loop is the primary use case — a full LoCoMo bench is $0.57 even on
DeepSeek; re-running with cache is free).

Layout::

    <cache_dir>/<model_name>/<template_version>/<prompt_hash>.txt

Plain ``.txt`` files chosen over a database for inspectability — a
human can ``cat`` any cache entry to see what the model returned.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .base import AnswerGenerator
from .constants import PROMPT_TEMPLATE_VERSION


class CachedAnswerGenerator(AnswerGenerator):
    """Disk-backed Cache-Aside decorator."""

    def __init__(
        self,
        *,
        inner: AnswerGenerator,
        model_name: str,
        cache_dir: Path,
        template_version: str = PROMPT_TEMPLATE_VERSION,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must be non-empty (cache partition key)")
        self._inner = inner
        self._model_name = model_name
        self._template_version = template_version
        self._partition_dir = Path(cache_dir) / model_name / template_version
        self._partition_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, prompt: str) -> str:
        cache_file = self._partition_dir / f"{_hash(prompt)}.txt"
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")
        answer = self._inner.generate(prompt)
        cache_file.write_text(answer, encoding="utf-8")
        return answer


def _hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
