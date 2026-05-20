"""Aggregate the per-model logprob results into a cross-model comparison.

Reads every `results_logprob_*.json` in this directory, computes the
per-cell mean / median Δlog-ratio (= log(P_inject / P_control)) and the
exponentiated boost factor (geometric mean of P_inject / P_control).

Output: a markdown table printed to stdout, suitable for pasting into
the research doc.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_all() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(HERE.glob("results_logprob_*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"  ! skipping malformed file {path.name}: {e}", file=sys.stderr)
            continue
        out[path.name] = data
    return out


def summarise_trials(trials: list[dict]) -> dict[int, dict]:
    """Group trials by n_filler; return summary stats per N."""
    by_n: dict[int, list[float]] = {}
    inj_lp: dict[int, list[float]] = {}
    ctl_lp: dict[int, list[float]] = {}
    for t in trials:
        n = t.get("n_filler")
        if n is None:
            continue
        delta = t.get("delta_log_ratio")
        if not isinstance(delta, (int, float)) or not math.isfinite(delta):
            continue
        by_n.setdefault(n, []).append(float(delta))
        if math.isfinite(t.get("logprob_inject", float("-inf"))):
            inj_lp.setdefault(n, []).append(float(t["logprob_inject"]))
        if math.isfinite(t.get("logprob_control", float("-inf"))):
            ctl_lp.setdefault(n, []).append(float(t["logprob_control"]))

    result: dict[int, dict] = {}
    for n in sorted(by_n.keys()):
        deltas = by_n[n]
        if not deltas:
            continue
        result[n] = {
            "n_trials": len(deltas),
            "mean_delta": statistics.mean(deltas),
            "median_delta": statistics.median(deltas),
            "stdev_delta": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
            "boost_factor": math.exp(statistics.mean(deltas)),
            "mean_logp_inj": statistics.mean(inj_lp.get(n, [0])),
            "mean_logp_ctl": statistics.mean(ctl_lp.get(n, [0])),
        }
    return result


def main() -> int:
    all_results = load_all()
    if not all_results:
        print("No results_logprob_*.json files found.", file=sys.stderr)
        return 1

    print(f"# Cross-model logprob comparison\n")
    print(f"Found {len(all_results)} result file(s):\n")
    for fname in all_results.keys():
        meta = all_results[fname]
        print(f"  - {fname}  (model={meta.get('model_id', '?')}, "
              f"quant={meta.get('quantization', '?')}, "
              f"n={meta.get('n_trials_per_cell', '?')})")
    print()

    # Per-model summaries
    summaries: dict[str, dict] = {}
    for fname, data in all_results.items():
        model_id = data.get("model_id", fname)
        quant = data.get("quantization", "?")
        trials = data.get("q_b_logprob_trials", [])
        summary = summarise_trials(trials)
        key = f"{model_id} ({quant})"
        summaries[key] = summary

    # Find all N values
    all_ns = set()
    for s in summaries.values():
        all_ns.update(s.keys())
    sorted_ns = sorted(all_ns)

    # Boost-factor table (the headline)
    print(f"## Injection boost factor (exp of mean log-ratio)\n")
    header = "| Model (quant) | " + " | ".join(f"N={n}" for n in sorted_ns) + " |"
    sep = "|" + "|".join(["---"] * (1 + len(sorted_ns))) + "|"
    print(header)
    print(sep)
    for key, summary in summaries.items():
        cells = []
        for n in sorted_ns:
            if n in summary:
                cells.append(f"**{summary[n]['boost_factor']:.2f}×**")
            else:
                cells.append("—")
        print(f"| `{key}` | " + " | ".join(cells) + " |")
    print()

    # Mean Δlog-ratio with std-dev table
    print(f"## Mean Δlog-ratio ± stdev (signal strength + variance)\n")
    print(header)
    print(sep)
    for key, summary in summaries.items():
        cells = []
        for n in sorted_ns:
            if n in summary:
                cells.append(f"{summary[n]['mean_delta']:+.2f} ± {summary[n]['stdev_delta']:.2f}")
            else:
                cells.append("—")
        print(f"| `{key}` | " + " | ".join(cells) + " |")
    print()

    # Raw probabilities for sanity
    print(f"## Raw mean log-probabilities (inject / control)\n")
    print(header)
    print(sep)
    for key, summary in summaries.items():
        cells = []
        for n in sorted_ns:
            if n in summary:
                inj = summary[n]["mean_logp_inj"]
                ctl = summary[n]["mean_logp_ctl"]
                cells.append(f"{inj:.2f} / {ctl:.2f}")
            else:
                cells.append("—")
        print(f"| `{key}` | " + " | ".join(cells) + " |")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
