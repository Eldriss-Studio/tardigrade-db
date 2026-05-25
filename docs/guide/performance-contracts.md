# Performance Contracts and Where They're Validated

TardigradeDB makes a small number of performance promises that justify shipping specific build artifacts and architectural choices. Those promises need validation — but validation in the wrong place becomes noise that erodes trust in CI. This page documents which promises are contract-grade, which workflow validates each, and how to interpret a failure.

## The philosophical position

**Per-PR performance gating is rejected.** Shared CI runners (GitHub's `ubuntu-latest` and similar) are shared VMs whose CPU model, scheduler behavior, and neighbor noise vary across runs and rotate over weeks. Free-threaded parallel scaling is especially sensitive — a 3.0× speedup measured on one runner becomes 2.81× the next time GitHub rotates hardware, and a release that has nothing wrong with its code is blocked. Single-shot performance assertions on shared infrastructure flake by construction.

Two things go in CI gating: correctness (tests, lints, type-checks) and constant-time invariants (does the snapshot format round-trip, does the wheel build). Numbers that depend on hardware behavior live elsewhere.

The contract validation moves to two places:

1. **Pre-release gating** — runs on every `v*` tag push and on `workflow_dispatch`. Uses median-of-5 across runs of the measurement script. If the median falls below the documented floor, the release is blocked. The methodology removes single-shot variance as a false-positive source while still catching real regressions.
2. **Nightly trend** — runs unattended once per day. Records the measurement in an append-only JSONL history. No assertion. Erosion that wouldn't trip a single-shot gate (a 2% drop per week, silent in any one measurement but 20% over two months) shows up as a sloped line in the trend.

These run on the same shared runner CI uses, so they inherit the variance — but median-of-5 statistically dampens it, and the trend file lets a human spot regressions that no single-shot threshold could catch.

## Contract metrics

### cp313t 8-thread b=1 parallel-read speedup

**What it is.** With a free-threaded Python 3.13t interpreter and a hot retrieval corpus, eight threads issuing `mem_read_pack` calls in parallel should complete faster than one thread doing the same work sequentially. The cp313t wheel exists specifically to deliver this speedup; without it, the cp313t target collapses into the cp312 target.

**The floor.** Median of 5 runs must be at least **2.5×**. Lower than that, the cp313t wheel is not earning its keep.

**Why 2.5× and not 3.0×.** The original number (3.0×) was measured on initial cp313t-launch hardware. CI hardware rotates; the runner pool today produces median measurements closer to 2.81× even when the engine itself hasn't moved. 2.5× sits below the current observed band with enough headroom that a real regression triggers the gate and runner-rotation alone does not.

**Where it's validated.** [`.github/workflows/pre-release-perf.yml`](../../.github/workflows/pre-release-perf.yml). The threshold is overridable per `workflow_dispatch` invocation if a maintainer wants to validate a candidate against a different bar.

**Where the trend lives.** [`.github/workflows/perf-trend.yml`](../../.github/workflows/perf-trend.yml) appends one measurement per day to `target/perf/cp313t-history.jsonl`, uploaded as a workflow artifact.

**Underlying measurement.** [`experiments/concurrent_reads_scaling.py`](../../experiments/concurrent_reads_scaling.py). The same script powers both workflows and is runnable locally for ad-hoc verification.

### Engine retrieval latency (positioning claim)

**What it is.** Sub-millisecond p99 retrieval at 5K packs is the positioning claim documented in [`docs/positioning/latency_first.md`](../positioning/latency_first.md).

**Where it's validated.** Not yet a CI gate. The claim is supported by `cargo bench` outputs on the author's development hardware. Adding it to the pre-release-perf workflow is a tracked follow-up — the measurement methodology is already in `cargo bench -p tdb-engine`, just not wired into a workflow yet.

## How to interpret a failure

### Pre-release gate fails

The release is blocked. **Read the step summary on the workflow run page** — it shows each of the 5 run measurements and the median. Three possible diagnoses:

- **Real regression.** All five runs cluster tightly below the floor. Bisect against recent commits — the parallel-read path is small enough that a regression typically lands in a single change.
- **Runner rotation.** One or two of the five runs is significantly lower than the others, dragging the median down. Trigger another `workflow_dispatch` run. If the median bounces back, accept the result; if it stays low, treat as a real regression and investigate the new runner's CPU model.
- **Threshold is stale.** If the trend (see below) shows the metric has drifted gradually but consistently downward over weeks without any specific commit being responsible, the cp313t wheel may genuinely be delivering less on current GitHub-runner hardware. Recalibrate the threshold, document the new floor here, and tag a new release.

### Trend artifact shows erosion

Not a release blocker — the trend is observational. But a sustained downward slope is signal worth investigating before the pre-release gate catches it. The history JSONL is intentionally append-only so you can git-blame the engine commits that bracket any inflection point.

## When to add a new contract metric

Add a metric to this page when:

1. It justifies a specific build target, dependency, or architectural choice.
2. It can be measured with a script that produces a parseable number on stdout.
3. The floor is defensible — not "the best number we ever saw" but "the worst number we can accept and still claim the contract."

Wire it into `pre-release-perf.yml` for the gate, into `perf-trend.yml` for the trend, and write a section here matching the cp313t shape above.

## Related

- [Latency-first positioning](../positioning/latency_first.md) — the broader claim the cp313t wheel supports.
- [Free-threaded Python proof](../experiments/2026-05-22-freethreaded-python-proof.md) — the original measurement that established the cp313t contract.
- The `~/.claude/plans/ci-perf-measurement-restructure.md` plan documents the rationale for splitting per-PR gating, pre-release gating, and nightly trend across three workflows.
