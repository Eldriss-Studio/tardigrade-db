"""Typed models for benchmark v1 results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkItem:
    """Normalized dataset item.

    ``category`` is optional metadata used by the runner's per-category
    aggregate (e.g., LoCoMo single_hop / multi_hop / temporal /
    open_domain / adversarial; LongMemEval ``question_type``). Defaults
    to ``"unknown"`` so legacy fixtures without the field still flow
    through the pipeline.
    """

    item_id: str
    dataset: str
    context: str
    question: str
    ground_truth: str
    category: str = "unknown"
    # Gold evidence snippets the retriever is supposed to surface.
    # Powers the audit-resistant retrieval-only metrics (#88). Empty
    # list when the source dataset has no evidence references for
    # this item — the runner skips such rows from retrieval averages.
    gold_evidence: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AdapterQueryResult:
    """Adapter query output before evaluation."""

    answer: str
    evidence: list[str]
    latency_ms: float
    status: str
    error: str | None = None


@dataclass(frozen=True)
class ScoreResult:
    """Evaluator output."""

    score: float
    judgment: str
    evaluator_mode: str


@dataclass
class RunResultV1:
    """Versioned run schema contract (v1)."""

    version: int
    manifest: dict[str, Any]
    items: list[dict[str, Any]]
    aggregates: dict[str, Any]
    comparisons: dict[str, Any]
    status_summary: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "manifest": self.manifest,
            "items": self.items,
            "aggregates": self.aggregates,
            "comparisons": self.comparisons,
            "status_summary": self.status_summary,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunResultV1":
        return cls(
            version=payload["version"],
            manifest=payload["manifest"],
            items=payload["items"],
            aggregates=payload["aggregates"],
            comparisons=payload["comparisons"],
            status_summary=payload["status_summary"],
        )
