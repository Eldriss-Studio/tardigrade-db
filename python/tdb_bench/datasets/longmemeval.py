"""LongMemEval dataset adapter."""

from __future__ import annotations

from .jsonl import JsonlDatasetAdapter


class LongMemEvalDatasetAdapter(JsonlDatasetAdapter):
    def __init__(self, revision: str, path: str) -> None:
        super().__init__(name="longmemeval", revision=revision, path=path)
