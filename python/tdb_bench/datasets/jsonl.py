"""JSONL dataset adapter base (Template Method hook style)."""

from __future__ import annotations

import json
from pathlib import Path

from tdb_bench.contracts import DatasetAdapter
from tdb_bench.errors import DatasetUnavailableError
from tdb_bench.models import BenchmarkItem


class JsonlDatasetAdapter(DatasetAdapter):
    """Load normalized benchmark items from JSONL.

    Expected keys: id, dataset, context, question, ground_truth.
    """

    def __init__(self, name: str, revision: str, path: str) -> None:
        self.name = name
        self.revision = revision
        self.path = Path(path)

    def load_items(self, max_items: int | None = None) -> list[BenchmarkItem]:
        if not self.path.exists() or not self.path.is_file():
            raise DatasetUnavailableError(f"DATASET_NOT_FOUND: {self.path}")

        items: list[BenchmarkItem] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                items.append(
                    BenchmarkItem(
                        item_id=str(raw["id"]),
                        dataset=str(raw.get("dataset", self.name)),
                        context=str(raw["context"]),
                        question=str(raw["question"]),
                        ground_truth=str(raw["ground_truth"]),
                        category=str(raw.get("category", "unknown")),
                    )
                )
                if max_items and len(items) >= max_items:
                    break
        return items

    def metadata(self) -> dict[str, str]:
        return {
            "dataset": self.name,
            "revision": self.revision,
            "source": str(self.path),
        }
