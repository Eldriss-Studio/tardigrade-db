"""Core contracts using explicit design patterns.

Patterns:
- Adapter: BenchmarkAdapter implementations for each system.
- Strategy: Evaluator implementations.
- Template Method: Runner orchestrates invariant flow around these contracts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import AdapterQueryResult, BenchmarkItem, ScoreResult


class BenchmarkAdapter(ABC):
    """Adapter contract for memory systems under evaluation."""

    name: str

    @abstractmethod
    def ingest(self, items: list[BenchmarkItem]) -> None:
        """Populate system memory with dataset items."""

    @abstractmethod
    def query(self, item: BenchmarkItem, top_k: int) -> AdapterQueryResult:
        """Query system memory for a benchmark item."""

    @abstractmethod
    def reset(self) -> None:
        """Reset adapter state between datasets/runs."""

    @abstractmethod
    def metadata(self) -> dict[str, str]:
        """Return adapter identity/version metadata."""


class DatasetAdapter(ABC):
    """Dataset provider contract."""

    @abstractmethod
    def load_items(self, max_items: int | None = None) -> list[BenchmarkItem]:
        """Load normalized benchmark items."""

    @abstractmethod
    def metadata(self) -> dict[str, str]:
        """Return dataset metadata (name/revision/source)."""


class Evaluator(ABC):
    """Strategy contract for scoring answer quality."""

    @abstractmethod
    def score(self, item: BenchmarkItem, answer: str, evidence: list[str]) -> ScoreResult:
        """Score and judge the produced answer."""
