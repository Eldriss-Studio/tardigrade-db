"""``FootprintReporter`` — engine + process footprint at a moment in time.

Companion to the latency-bench v2 harness (Track A slice A1): same
shape of Strategy + Repository primitives, but the measurement is
bytes (not seconds). Combines:

* engine-level arena bytes / segment count / per-cell average from
  :func:`engine.status()` (added in commit ``965ed7e``);
* process-level RSS via :mod:`psutil` so the bench reports what
  the OS actually sees the Python process holding.

## Pattern stack

* **Repository / Value Object** — :class:`FootprintSnapshot` carries
  the snapshot shape; serializable to JSON for the positioning doc.
* **Template Method** — :class:`GrowthCurve` sequences "ingest →
  snapshot" at each sample point in :data:`FOOTPRINT_SAMPLE_POINTS`.
* **Dependency Injection** — `GrowthCurve.run(ingest_fn=...)` lets
  callers plug in any "do N units of work" callback (real model
  inference in the bench script, lightweight cell writes in tests).

## SOLID

* SRP — engine bytes vs process RSS are different concerns; the
  snapshot exposes both but doesn't blur them.
* OCP — adding a new footprint axis (e.g. GPU memory via
  ``torch.cuda.max_memory_allocated``) plugs in as a new dataclass
  field + reporter step.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import psutil


# Corpus sizes at which the growth audit records snapshots. Pinned
# here so the positioning doc and the bench script stay aligned.
# Adding more points is a constant change; ingest cost grows with the
# largest point.
FOOTPRINT_SAMPLE_POINTS: list[int] = [0, 100, 1000, 5000]


# ─── Value Object: FootprintSnapshot ──────────────────────────────────────


@dataclass
class FootprintSnapshot:
    """A single moment-in-time view of engine + process footprint."""

    cell_count: int
    segment_count: int
    arena_bytes: int
    arena_bytes_per_cell: int
    process_rss_bytes: int

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Strategy: FootprintReporter ──────────────────────────────────────────


class FootprintReporter:
    """Snapshots engine + process footprint together."""

    def __init__(self, engine) -> None:
        self._engine = engine
        self._proc = psutil.Process(os.getpid())

    def snapshot(self) -> FootprintSnapshot:
        s = self._engine.status()
        return FootprintSnapshot(
            cell_count=int(s["cell_count"]),
            segment_count=int(s["segment_count"]),
            arena_bytes=int(s["arena_bytes"]),
            arena_bytes_per_cell=int(s["arena_bytes_per_cell"]),
            process_rss_bytes=int(self._proc.memory_info().rss),
        )


# ─── Template Method: GrowthCurve ─────────────────────────────────────────


_IngestFn = Callable[[int], None]


@dataclass
class GrowthCurve:
    """Captures ``FootprintSnapshot`` at each sample point during ingest."""

    reporter: FootprintReporter
    sample_points: list[int] = field(default_factory=lambda: list(FOOTPRINT_SAMPLE_POINTS))
    snapshots: list[FootprintSnapshot] = field(default_factory=list)

    def run(self, *, ingest_fn: _IngestFn) -> None:
        """Walk sample points in order; ingest the delta; snapshot.

        ``ingest_fn(i)`` is called once for each integer ``i`` between
        the previous sample point and the current one. The callback is
        responsible for the actual write; the reporter handles
        snapshotting between calls.
        """
        if not self.sample_points:
            return
        sorted_points = sorted(self.sample_points)
        ingested = 0
        for target in sorted_points:
            while ingested < target:
                ingest_fn(ingested)
                ingested += 1
            self.snapshots.append(self.reporter.snapshot())

    def to_dict(self) -> dict:
        return {"snapshots": [s.to_dict() for s in self.snapshots]}

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
