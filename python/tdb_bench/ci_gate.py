"""CI gate policies for benchmark smoke runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GatePolicy:
    """Threshold configuration for CI benchmark gating."""

    fail_invalid_validity: bool = True
    warn_non_ok_ratio: float = 0.05
    fail_non_ok_ratio: float = 0.20


@dataclass(frozen=True)
class GateDecision:
    """Decision object for CI pass/fail and annotations."""

    should_fail: bool
    run_validity: str
    run_validity_reason: str
    total: int
    ok: int
    skipped: int
    failed: int
    non_ok_ratio: float
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def evaluate_report_payload(payload: dict[str, Any], policy: GatePolicy) -> GateDecision:
    run_validity = payload.get("run_validity", {})
    if not isinstance(run_validity, dict):
        run_validity = {}
    run_validity_state = str(run_validity.get("state", "invalid"))
    run_validity_reason = str(run_validity.get("reason", "missing_run_validity"))

    summary = payload.get("status_summary", {})
    if not isinstance(summary, dict):
        summary = {}

    ok = int(summary.get("ok", 0))
    skipped = int(summary.get("skipped", 0))
    failed = int(summary.get("failed", 0))
    known_total = ok + skipped + failed
    raw_total = sum(int(v) for v in summary.values() if isinstance(v, int))
    total = max(known_total, raw_total)
    non_ok = skipped + failed
    non_ok_ratio = float(non_ok / total) if total > 0 else 1.0

    warnings: list[str] = []
    failures: list[str] = []

    if policy.fail_invalid_validity and run_validity_state == "invalid":
        failures.append(f"run_validity=invalid ({run_validity_reason})")

    if non_ok_ratio >= policy.fail_non_ok_ratio:
        failures.append(
            f"non_ok_ratio={non_ok_ratio:.4f} exceeds fail threshold {policy.fail_non_ok_ratio:.4f}"
        )
    elif non_ok_ratio >= policy.warn_non_ok_ratio:
        warnings.append(
            f"non_ok_ratio={non_ok_ratio:.4f} exceeds warn threshold {policy.warn_non_ok_ratio:.4f}"
        )

    return GateDecision(
        should_fail=bool(failures),
        run_validity=run_validity_state,
        run_validity_reason=run_validity_reason,
        total=total,
        ok=ok,
        skipped=skipped,
        failed=failed,
        non_ok_ratio=round(non_ok_ratio, 6),
        warnings=warnings,
        failures=failures,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench-ci-gate")
    parser.add_argument("--input", required=True)
    parser.add_argument("--warn-non-ok-ratio", type=float, default=0.05)
    parser.add_argument("--fail-non-ok-ratio", type=float, default=0.20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    policy = GatePolicy(
        fail_invalid_validity=True,
        warn_non_ok_ratio=float(args.warn_non_ok_ratio),
        fail_non_ok_ratio=float(args.fail_non_ok_ratio),
    )
    decision = evaluate_report_payload(payload, policy)

    print(
        "ci_gate_summary "
        f"validity={decision.run_validity} "
        f"reason={decision.run_validity_reason} "
        f"total={decision.total} ok={decision.ok} skipped={decision.skipped} failed={decision.failed} "
        f"non_ok_ratio={decision.non_ok_ratio:.4f}"
    )
    for msg in decision.warnings:
        print(f"::warning::{msg}")
    for msg in decision.failures:
        print(f"::error::{msg}")

    return 1 if decision.should_fail else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

