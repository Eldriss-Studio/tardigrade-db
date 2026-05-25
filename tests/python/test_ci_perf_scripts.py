"""Unit tests for the CI perf measurement scripts.

The workflow ATs in ``test_ci_workflow_structure.py`` check that the
right files exist with the right triggers. These tests cover the
actual computation: parsing the scaling diagnostic output, computing
the median, appending to the JSONL trend history. Lets us catch
regressions in the scripts without waiting for a real CI run.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PERF_MEDIAN = _REPO_ROOT / ".github" / "workflows" / "scripts" / "perf_median.py"
_PERF_TREND_APPEND = (
    _REPO_ROOT / ".github" / "workflows" / "scripts" / "perf_trend_append.py"
)


def _load_module(name: str, path: Path):
    """Import a script-style file as a module so we can exercise its functions."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def perf_median():
    return _load_module("perf_median", _PERF_MEDIAN)


@pytest.fixture
def perf_trend_append():
    return _load_module("perf_trend_append", _PERF_TREND_APPEND)


def _sample_scaling_log(speedup: float) -> str:
    """Synthesize a scaling diagnostic stdout line we can parse."""
    return (
        "Some preamble noise that the parser ignores.\n"
        f"   8 threads — b=1: {speedup:.2f}x | rest of the line is ignored\n"
        "More noise after.\n"
    )


class TestPerfMedianParser:
    """The pre-release median computation parses runs and exits correctly."""

    def test_parses_speedup_from_a_single_log(self, perf_median, tmp_path) -> None:
        log = tmp_path / "run-1.log"
        log.write_text(_sample_scaling_log(2.81), encoding="utf-8")
        assert perf_median._parse_one(log) == pytest.approx(2.81)

    def test_passes_when_median_meets_threshold(
        self, perf_median, tmp_path, monkeypatch
    ) -> None:
        # Five runs, median = 2.81 (the middle of a sorted list).
        speedups = [2.70, 2.75, 2.81, 2.90, 2.95]
        for idx, speedup in enumerate(speedups, start=1):
            (tmp_path / f"run-{idx}.log").write_text(
                _sample_scaling_log(speedup), encoding="utf-8"
            )

        monkeypatch.setenv("MEDIAN_CP313T_THRESHOLD", "2.5")
        monkeypatch.setenv("REPEATS", "5")
        monkeypatch.chdir(tmp_path.parent)
        # Symlink target/perf to where the runs landed so the script's
        # hardcoded relative path resolves correctly under tmp_path.
        target_perf = tmp_path.parent / "target" / "perf"
        target_perf.parent.mkdir(exist_ok=True)
        if target_perf.exists() or target_perf.is_symlink():
            target_perf.unlink()
        target_perf.symlink_to(tmp_path, target_is_directory=True)

        assert perf_median.main() == 0

    def test_fails_when_median_falls_below_threshold(
        self, perf_median, tmp_path, monkeypatch
    ) -> None:
        speedups = [1.0, 1.1, 1.2, 1.3, 1.4]  # median 1.2 < 2.5
        for idx, speedup in enumerate(speedups, start=1):
            (tmp_path / f"run-{idx}.log").write_text(
                _sample_scaling_log(speedup), encoding="utf-8"
            )

        monkeypatch.setenv("MEDIAN_CP313T_THRESHOLD", "2.5")
        monkeypatch.setenv("REPEATS", "5")
        monkeypatch.chdir(tmp_path.parent)
        target_perf = tmp_path.parent / "target" / "perf"
        target_perf.parent.mkdir(exist_ok=True)
        if target_perf.exists() or target_perf.is_symlink():
            target_perf.unlink()
        target_perf.symlink_to(tmp_path, target_is_directory=True)

        assert perf_median.main() == 1


class TestPerfTrendAppend:
    """The nightly trend script appends new rows and replaces same-day entries."""

    def test_parses_speedup_from_a_trend_log(self, perf_trend_append) -> None:
        assert perf_trend_append._parse_speedup(_sample_scaling_log(2.81)) == pytest.approx(2.81)

    def test_appends_to_empty_history(self, perf_trend_append) -> None:
        out = perf_trend_append._append_today([], "2026-05-25", 2.81)
        assert out == [
            {"date": "2026-05-25", "metric": "cp313t_8t_b1_speedup", "value": 2.81}
        ]

    def test_replaces_same_day_row(self, perf_trend_append) -> None:
        existing = [
            {"date": "2026-05-24", "metric": "cp313t_8t_b1_speedup", "value": 2.79},
            {"date": "2026-05-25", "metric": "cp313t_8t_b1_speedup", "value": 2.81},
        ]
        out = perf_trend_append._append_today(existing, "2026-05-25", 2.95)
        # Old 2026-05-25 row dropped, replaced with the new value.
        assert {row["date"]: row["value"] for row in out} == {
            "2026-05-24": 2.79,
            "2026-05-25": 2.95,
        }

    def test_history_round_trips_through_disk(
        self, perf_trend_append, tmp_path
    ) -> None:
        history = [
            {"date": "2026-05-24", "metric": "cp313t_8t_b1_speedup", "value": 2.79},
            {"date": "2026-05-25", "metric": "cp313t_8t_b1_speedup", "value": 2.81},
        ]
        path = tmp_path / "history.jsonl"
        perf_trend_append._write_history(path, history)
        loaded = perf_trend_append._load_history(path)
        assert loaded == history

    def test_load_missing_history_returns_empty_list(
        self, perf_trend_append, tmp_path
    ) -> None:
        assert perf_trend_append._load_history(tmp_path / "nope.jsonl") == []


class TestScriptsRunUnderShebang:
    """The scripts are invocable directly — catches syntax regressions cheaply."""

    @pytest.mark.parametrize("script", [_PERF_MEDIAN, _PERF_TREND_APPEND])
    def test_script_imports_cleanly(self, script: Path) -> None:
        # ``python -c "import ast; ast.parse(open(...).read())"`` is the
        # cheapest possible smoke test for "this file is well-formed
        # Python." Catches typos that wouldn't surface until a real
        # CI run.
        result = subprocess.run(
            [sys.executable, "-c", f"import ast; ast.parse(open({str(script)!r}).read())"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
