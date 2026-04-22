# Benchmark V1 — Observed Results So Far

Last updated: 2026-04-22

This page reports **completed runs we actually executed**.  
It is intentionally separate from `latest-full-*` so we do not mislabel partial runs as full matrix results.

## Reading Guide

- `run_validity=comparable`: safe to compare systems in that run scope.
- `run_validity=invalid`: at least one system had no successful outputs; not a fair quality comparison.
- Evaluator mode matters:
  - `llm`: model-judged scoring
  - `deterministic_fallback`: lexical overlap fallback (used when no API key available)

## Completed Runs

| Date | Scope | Systems | Evaluator | Run validity | Key outcome |
|---|---|---|---|---|---|
| 2026-04-22 | Smoke fixture, 3 repeats | Tardigrade vs Letta | deterministic_fallback | comparable | Quality tie (`1.0000` vs `1.0000`), Tardigrade lower latency |
| 2026-04-22 | Official-data sample (25 LoCoMo + 25 LongMemEval) | Tardigrade vs Letta | deterministic_fallback | comparable | Tardigrade `1.0000`, Letta `0.2002`; no failures |
| 2026-04-22 | Official full Phase-1 (attempt before ingest chunk fix) | Tardigrade vs Letta | deterministic_fallback | invalid | Letta ingest failures on all items (`2042/2042`) |

## Sample Run Metrics (Comparable Run)

Source: official-data sample run (`25 + 25`, both systems `ok=50`, `failed=0`, `skipped=0`).

| System | Avg score | LoCoMo avg latency | LongMemEval avg latency |
|---|---:|---:|---:|
| Tardigrade | `1.0000` | `~7.44 ms` | `~6.67 ms` |
| Letta | `0.2002` | `~81.10 ms` | `~77.63 ms` |

## Important Caveats

- The full matrix run is compute-heavy and may not finish reliably on constrained local machines.
- Sample results are useful and transparent, but they are not a substitute for full-matrix publication.
- We always keep failed/skipped outcomes visible; we do not silently drop them.
