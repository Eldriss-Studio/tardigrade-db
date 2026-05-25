#!/usr/bin/env python3
"""Compute the median 8-thread b=1 speedup across N scaling runs.

Reads ``target/perf/run-{i}.log`` files (produced by the pre-release-perf
workflow), parses the "8 threads — b=1: X.XXx" speedup line from each,
takes the median across the runs, and either prints the verdict to the
GitHub step summary or exits non-zero if the median falls below the
configured threshold.

Inputs are read from the environment, never from shell arguments —
matches the safe pattern for handling workflow_dispatch inputs without
shell interpolation. See ``pre-release-perf.yml`` for context.
"""

from __future__ import annotations

import os
import re
import statistics
import sys
from pathlib import Path

# The scaling script's stdout includes one line like
# "  8 threads — b=1: 2.81x | ..." per run. We match the speedup
# value with a tight regex so noise in the rest of the line doesn't
# leak in.
SPEEDUP_PATTERN = re.compile(r"8 threads\s*—\s*b=1:\s+([0-9]+\.[0-9]+)x")


def _parse_one(path: Path) -> float:
    """Extract the 8-thread b=1 speedup from one run log."""
    match = SPEEDUP_PATTERN.search(path.read_text(encoding="utf-8"))
    if match is None:
        print(
            f"::error::Could not parse 8-thread b=1 speedup from {path.name}",
            file=sys.stderr,
        )
        sys.exit(1)
    return float(match.group(1))


def _write_step_summary(measurements: list[float], median: float, threshold: float) -> None:
    """Append a markdown summary to GITHUB_STEP_SUMMARY if available."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = ["## cp313t 8-thread b=1 speedup\n", "| Run | Speedup |\n|---|---|\n"]
    for idx, value in enumerate(measurements, start=1):
        lines.append(f"| {idx} | {value:.2f}x |\n")
    lines.append(f"\n**Median: {median:.3f}x** (threshold: {threshold:.3f}x)\n")
    with open(summary_path, "a", encoding="utf-8") as fh:
        fh.writelines(lines)


def main() -> int:
    threshold = float(os.environ["MEDIAN_CP313T_THRESHOLD"])
    repeats = int(os.environ["REPEATS"])

    log_dir = Path("target/perf")
    measurements = [_parse_one(log_dir / f"run-{i}.log") for i in range(1, repeats + 1)]
    median = statistics.median(measurements)

    print(f"measurements (each run): {measurements}")
    print(f"median: {median:.3f}x")
    print(f"threshold: {threshold:.3f}x")

    _write_step_summary(measurements, median, threshold)

    if median < threshold:
        # The error annotation tells the maintainer what to do, not
        # just that something failed. Per the project's
        # `feedback_explain_the_why` discipline.
        print(
            f"::error::Median 8-thread b=1 speedup {median:.3f}x is below "
            f"threshold {threshold:.3f}x. Either real perf erosion or the "
            "threshold is stale; investigate via "
            "experiments/concurrent_reads_scaling.py on dedicated hardware "
            "before bumping the floor.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
