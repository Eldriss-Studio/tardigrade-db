"""Report and comparison rendering."""

from __future__ import annotations

import json
from pathlib import Path

from .models import RunResultV1


def classify_run_validity(run: RunResultV1) -> dict[str, str]:
    systems = run.aggregates.get("systems", {})
    if not isinstance(systems, dict) or not systems:
        return {"state": "invalid", "reason": "missing_system_aggregates"}

    systems_without_success: list[str] = []
    systems_with_fail_or_skip: list[str] = []
    for system, agg in sorted(systems.items()):
        ok = int(agg.get("ok", 0))
        skipped = int(agg.get("skipped", 0))
        failed = int(agg.get("failed", 0))
        total = ok + skipped + failed

        if total == 0 or (ok == 0 and (skipped > 0 or failed > 0)):
            systems_without_success.append(system)
            continue
        if skipped > 0 or failed > 0:
            systems_with_fail_or_skip.append(system)

    if systems_without_success:
        reason = "no_successful_results_for=" + ",".join(systems_without_success)
        return {"state": "invalid", "reason": reason}

    if systems_with_fail_or_skip:
        reason = "partial_failures_or_skips_for=" + ",".join(systems_with_fail_or_skip)
        return {"state": "degraded", "reason": reason}

    return {"state": "comparable", "reason": "all_systems_ok"}


def render_report_markdown(run: RunResultV1) -> str:
    validity = classify_run_validity(run)
    lines = [
        "# Benchmark Report",
        "",
        f"- mode: `{run.manifest.get('mode', '')}`",
        f"- git_sha: `{run.manifest.get('git_sha', '')}`",
        f"- repeats: `{run.manifest.get('repeats', 1)}`",
        f"- seeds: `{run.manifest.get('seeds', [run.manifest.get('seed', 0)])}`",
        f"- run_validity: `{validity['state']}`",
        f"- run_validity_reason: `{validity['reason']}`",
        "",
        "## Aggregates",
        "",
        "| system | avg_score | stddev | ci95 | runs | ok | skipped | failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for system, agg in sorted(run.aggregates.get("systems", {}).items()):
        lines.append(
            f"| {system} | {agg.get('avg_score', 0.0):.4f} | {agg.get('score_stddev', 0.0):.4f} | {agg.get('score_ci95', 0.0):.4f} | {agg.get('run_count', 1)} | {agg.get('ok', 0)} | {agg.get('skipped', 0)} | {agg.get('failed', 0)} |"
        )

    return "\n".join(lines) + "\n"


def render_report_json(run: RunResultV1) -> str:
    payload = run.to_dict()
    payload["run_validity"] = classify_run_validity(run)
    return json.dumps(payload, indent=2, sort_keys=True)


def compare_runs(baseline: RunResultV1, candidate: RunResultV1) -> dict:
    result: dict[str, dict] = {}
    baseline_systems = baseline.aggregates.get("systems", {})
    candidate_systems = candidate.aggregates.get("systems", {})

    for system in sorted(set(baseline_systems) | set(candidate_systems)):
        b = float(baseline_systems.get(system, {}).get("avg_score", 0.0))
        c = float(candidate_systems.get(system, {}).get("avg_score", 0.0))
        b_std = float(baseline_systems.get(system, {}).get("score_stddev", 0.0))
        c_std = float(candidate_systems.get(system, {}).get("score_stddev", 0.0))
        b_ci = float(baseline_systems.get(system, {}).get("score_ci95", 0.0))
        c_ci = float(candidate_systems.get(system, {}).get("score_ci95", 0.0))
        result[system] = {
            "baseline_avg_score": b,
            "candidate_avg_score": c,
            "baseline_stddev": b_std,
            "candidate_stddev": c_std,
            "baseline_ci95": b_ci,
            "candidate_ci95": c_ci,
            "delta": round(c - b, 6),
        }

    return result


def render_compare_markdown(comparison: dict) -> str:
    lines = [
        "# Benchmark Comparison",
        "",
        "| system | baseline_avg_score | candidate_avg_score | baseline_ci95 | candidate_ci95 | delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for system, row in sorted(comparison.items()):
        lines.append(
            f"| {system} | {row['baseline_avg_score']:.4f} | {row['candidate_avg_score']:.4f} | {row.get('baseline_ci95', 0.0):.4f} | {row.get('candidate_ci95', 0.0):.4f} | {row['delta']:+.4f} |"
        )
    return "\n".join(lines) + "\n"


def load_run(path: Path) -> RunResultV1:
    return RunResultV1.from_dict(json.loads(path.read_text(encoding="utf-8")))
