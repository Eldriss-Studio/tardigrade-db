"""LoCoMo dataset adapter."""

from __future__ import annotations

from .jsonl import JsonlDatasetAdapter


class LoCoMoDatasetAdapter(JsonlDatasetAdapter):
    def __init__(self, revision: str, path: str) -> None:
        super().__init__(name="locomo", revision=revision, path=path)
