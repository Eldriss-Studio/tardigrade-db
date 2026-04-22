"""CLI for benchmark run/report/compare."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .reporting import compare_runs, load_run, render_compare_markdown, render_report_json, render_report_markdown
from .runner import BenchmarkRunner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run")
    run.add_argument("--mode", required=True, choices=["smoke", "full"])
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--system", action="append", default=[])
    run.add_argument("--dataset", action="append", default=[])
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--seed", type=int, action="append", default=[])

    report = sub.add_parser("report")
    report.add_argument("--input", required=True)
    report.add_argument("--format", required=True, choices=["md", "json"])
    report.add_argument("--output", required=True)

    compare = sub.add_parser("compare")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--format", required=True, choices=["md", "json"])
    compare.add_argument("--output", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.cmd == "run":
        runner = BenchmarkRunner.from_config_file(Path(args.config))
        runner.run(
            mode=args.mode,
            output_path=Path(args.output),
            systems=args.system or None,
            datasets=args.dataset or None,
            repeat=int(args.repeat),
            seeds=args.seed or None,
        )
        return 0

    if args.cmd == "report":
        run = load_run(Path(args.input))
        payload = render_report_markdown(run) if args.format == "md" else render_report_json(run)
        Path(args.output).write_text(payload, encoding="utf-8")
        return 0

    if args.cmd == "compare":
        baseline = load_run(Path(args.baseline))
        candidate = load_run(Path(args.candidate))
        comparison = compare_runs(baseline, candidate)
        if args.format == "md":
            text = render_compare_markdown(comparison)
        else:
            text = json.dumps(comparison, indent=2, sort_keys=True)
        Path(args.output).write_text(text, encoding="utf-8")
        return 0

    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
