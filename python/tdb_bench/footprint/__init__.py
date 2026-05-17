"""Footprint reporting primitives for the positioning reframe.

Combines engine-level :class:`FootprintSnapshot` (arena bytes,
segment count, per-cell averages — all from
``engine.status()``) with process-level RSS via :mod:`psutil` so the
positioning doc can quote a single, internally-consistent footprint
number per scale.
"""

from .reporter import (
    FOOTPRINT_SAMPLE_POINTS,
    FootprintReporter,
    FootprintSnapshot,
    GrowthCurve,
)

__all__ = [
    "FOOTPRINT_SAMPLE_POINTS",
    "FootprintReporter",
    "FootprintSnapshot",
    "GrowthCurve",
]
