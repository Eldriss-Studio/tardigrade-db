#!/usr/bin/env python3
"""Append today's cp313t scaling measurement to the trend history.

The history is a JSONL file at ``target/perf/cp313t-history.jsonl`` —
one row per day, schema ``{"date": "YYYY-MM-DD", "metric": "cp313t_8t_b1_speedup", "value": float}``.
Because this runs on a fresh runner each invocation, the file starts
empty; previous-day rows are restored from the workflow's prior
artifact via the upload-artifact / download-artifact dance (a Phase 3
follow-up — for now the artifact just shows the latest measurement,
not the multi-day rolling median).

Renders a markdown table into ``GITHUB_STEP_SUMMARY`` so the run page
shows today's value alongside any history that survived.

This script is observational only: never exits non-zero on
measurement value (no threshold). The only failure mode is "couldn't
parse the scaling diagnostic output," which is a build problem worth
surfacing loudly.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

SPEEDUP_PATTERN = re.compile(r"8 threads\s*—\s*b=1:\s+([0-9]+\.[0-9]+)x")
METRIC_NAME = "cp313t_8t_b1_speedup"
LOG_PATH = Path("target/perf/trend.log")
HISTORY_PATH = Path("target/perf/cp313t-history.jsonl")


def _parse_speedup(log: str) -> float:
    match = SPEEDUP_PATTERN.search(log)
    if match is None:
        print(
            "::error::Could not parse 8-thread b=1 speedup from trend log",
            file=sys.stderr,
        )
        sys.exit(1)
    return float(match.group(1))


def _load_history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _append_today(history: list[dict], today_iso: str, value: float) -> list[dict]:
    """Append today's row, replacing any prior row with the same date."""
    history = [row for row in history if row.get("date") != today_iso]
    history.append({"date": today_iso, "metric": METRIC_NAME, "value": value})
    history.sort(key=lambda row: row.get("date", ""))
    return history


def _write_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in history:
            fh.write(json.dumps(row) + "\n")


def _write_step_summary(history: list[dict], today_iso: str, today_value: float) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    # Show the last 7 entries — enough for the human reader to spot a
    # trend, short enough to render cleanly without scrolling.
    recent = history[-7:]
    lines = [
        "## cp313t parallel-read trend\n",
        f"Today ({today_iso}): **{today_value:.2f}x** 8-thread b=1 speedup\n",
        "\n",
        "| Date | Speedup |\n|---|---|\n",
    ]
    for row in recent:
        lines.append(f"| {row['date']} | {row['value']:.2f}x |\n")
    with open(summary_path, "a", encoding="utf-8") as fh:
        fh.writelines(lines)


def main() -> int:
    log = LOG_PATH.read_text(encoding="utf-8")
    speedup = _parse_speedup(log)

    today_iso = dt.date.today().isoformat()
    history = _load_history(HISTORY_PATH)
    history = _append_today(history, today_iso, speedup)
    _write_history(HISTORY_PATH, history)

    print(f"recorded {today_iso}: {speedup:.3f}x ({METRIC_NAME})")
    _write_step_summary(history, today_iso, speedup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
