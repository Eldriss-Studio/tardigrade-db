"""ATs for the CI performance-measurement restructure.

Per ``~/.claude/plans/ci-perf-measurement-restructure.md``:

- Per-PR gating CI must NOT run the cp313t scaling assertion. It is too
  flaky on shared runners to gate release-blocking PRs.
- A dedicated pre-release workflow runs on tag pushes and workflow_dispatch
  and uses median-of-N methodology instead of single-shot.
- A nightly trend workflow runs unattended and posts trend data without
  asserting — observational only.

These tests parse the workflow YAML files and assert structural
properties. They are not pytest-discovered "CI" tests; they assert the
contract a contributor would otherwise verify by reading the workflow
files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# PyYAML is widely available in Python toolchains. If it isn't installed,
# fall back to a string-search check that still catches the most common
# regressions (the job name `test-freethreaded` reappearing in ci.yml).
try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    yaml = None


_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = _REPO_ROOT / ".github" / "workflows"


def _load_yaml(path: Path) -> dict | None:
    if not path.is_file():
        return None
    if yaml is None:
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class TestPerPrCiDoesNotGatePerf:
    """``ci.yml`` no longer contains the cp313t scaling assertion."""

    def test_test_freethreaded_job_is_removed_from_ci_yml(self) -> None:
        ci_text = (_WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
        assert "test-freethreaded:" not in ci_text, (
            "ci.yml still defines the `test-freethreaded` job. "
            "Per the perf-measurement restructure plan, this job moves "
            "to pre-release-perf.yml — single-shot speedup assertions on "
            "shared CI runners flake too often to gate per-PR work."
        )
        assert "Assert 8-thread single-call speedup" not in ci_text, (
            "The 3.0x speedup assertion text persists in ci.yml. The "
            "assertion moves to pre-release-perf.yml with median-of-N "
            "methodology."
        )


class TestPreReleasePerfWorkflowExists:
    """``pre-release-perf.yml`` exists and triggers on tag + dispatch."""

    def test_workflow_file_exists(self) -> None:
        assert (_WORKFLOWS / "pre-release-perf.yml").is_file(), (
            "Expected .github/workflows/pre-release-perf.yml to exist "
            "as the destination for the cp313t scaling assertion. See "
            "~/.claude/plans/ci-perf-measurement-restructure.md."
        )

    @pytest.mark.skipif(yaml is None, reason="PyYAML not available")
    def test_workflow_triggers_on_tag_push_and_dispatch(self) -> None:
        data = _load_yaml(_WORKFLOWS / "pre-release-perf.yml")
        assert data is not None, "pre-release-perf.yml failed to parse"
        # YAML parsers convert the unquoted ``on`` key to True (the YAML
        # boolean) under some library versions. Handle both forms.
        triggers = data.get("on") or data.get(True)
        assert triggers, "pre-release-perf.yml has no `on:` triggers"
        assert "workflow_dispatch" in triggers, (
            "pre-release-perf.yml must support workflow_dispatch so a "
            "maintainer can run it manually before tagging."
        )
        push_cfg = triggers.get("push")
        assert push_cfg and "tags" in push_cfg, (
            "pre-release-perf.yml must trigger on tag pushes so every "
            "release version is perf-validated."
        )

    @pytest.mark.skipif(yaml is None, reason="PyYAML not available")
    def test_workflow_runs_scaling_script_with_repeats(self) -> None:
        text = (_WORKFLOWS / "pre-release-perf.yml").read_text(encoding="utf-8")
        assert "concurrent_reads_scaling.py" in text, (
            "pre-release-perf.yml must invoke the existing scaling "
            "measurement script. Keep the script as the single source "
            "of truth."
        )


class TestNightlyTrendWorkflowExists:
    """``perf-trend.yml`` exists, runs on cron, and is observational."""

    def test_workflow_file_exists(self) -> None:
        assert (_WORKFLOWS / "perf-trend.yml").is_file(), (
            "Expected .github/workflows/perf-trend.yml to exist for "
            "nightly trend measurement."
        )

    @pytest.mark.skipif(yaml is None, reason="PyYAML not available")
    def test_workflow_runs_on_cron(self) -> None:
        data = _load_yaml(_WORKFLOWS / "perf-trend.yml")
        assert data is not None, "perf-trend.yml failed to parse"
        triggers = data.get("on") or data.get(True)
        assert triggers, "perf-trend.yml has no `on:` triggers"
        sched = triggers.get("schedule")
        assert sched and any("cron" in entry for entry in sched), (
            "perf-trend.yml must have a schedule.cron entry so it runs "
            "nightly without manual intervention."
        )


class TestPerformanceContractsDoc:
    """``docs/guide/performance-contracts.md`` exists and names workflows."""

    def test_doc_file_exists(self) -> None:
        doc = _REPO_ROOT / "docs" / "guide" / "performance-contracts.md"
        assert doc.is_file(), (
            "Expected docs/guide/performance-contracts.md to document "
            "which workflows validate which contract metrics."
        )

    def test_doc_names_each_workflow(self) -> None:
        doc = _REPO_ROOT / "docs" / "guide" / "performance-contracts.md"
        text = doc.read_text(encoding="utf-8")
        for name in ("pre-release-perf.yml", "perf-trend.yml"):
            assert name in text, (
                f"performance-contracts.md must reference `{name}` so "
                "a reader can navigate from the doc to the source."
            )
