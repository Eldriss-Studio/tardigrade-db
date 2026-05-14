> **⚠️ RETRACTED — 2026-05-14 (partial).** The 2026-05-14 bench audit found two dataset-preparation bugs that invalidate LoCoMo-derived numbers in this file:
>
> - The 2026-04-22 Tardigrade-vs-Letta rows showing `1.0000` Tardigrade scores were already flagged in this file as **stub adapter / lexical fallback** (see "Important Caveats" below). That caveat stands; the bench audit reinforces it — those `1.0000` rows do not reflect engine quality.
> - The **2026-05-02 "Tardigrade adapter rewrite" smoke fixture** (6-item, RTX 3070 Ti, native mode: none=0.833, centered=1.000) used the same `prepare_phase1_datasets.py` that contained the LoCoMo evidence-extraction bug. **However**, the smoke fixture is documented as a 6-item sample (the file already says "too small to be statistically meaningful"), so it is best read as a mechanism check rather than a real measurement. Treat the centered=+0.167 delta as anecdotal until re-run on the fixed dataset.
> - **Any claim or follow-up implying TardigradeDB achieves ~68% on LoCoMo via the native engine is retracted.** Honest 50-item native-engine number on the clean dataset is ~36% R@1 (bench adapter, full feature stack) or ~20% R@1 (minimal probe). The full-corpus number is not yet measured on clean data as of 2026-05-14.
> - The "Mem0 / Letta head-to-head pending operator setup" block is unaffected mechanically, but a fair three-way comparison must use the **fixed** dataset (post-2026-05-14 `prepare_phase1_datasets.py`).
>
> Forensic record: [`../experiments/2026-05-14-bench-audit.md`](../experiments/2026-05-14-bench-audit.md). The "Pre-2026-05-02 stub adapter" caveat below is preserved as-is — it correctly identified the lexical-fallback problem; the new audit extends that finding to the post-rewrite LoCoMo runs as well.

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
- **Pre-2026-05-02 Tardigrade scores were inflated by a stub adapter** that returned `item.ground_truth` after lexical word-overlap on the question. From 2026-05-02 the `tardigrade` adapter actually invokes Qwen3-0.6B and the engine retrieval pipeline (see "Tardigrade adapter rewrite" below). All earlier `1.0000` scores should be interpreted as reflecting the lexical fallback, not engine quality.

---

## Tardigrade adapter rewrite (2026-05-02)

The `tardigrade` adapter in `python/tdb_bench/adapters/tardigrade.py` now has two execution modes:

- **native** (default when CUDA + transformers available): loads Qwen3-0.6B, captures per-token hidden states for each ingested item via `HuggingFaceKVHook`, writes to the engine, retrieves at query time via `engine.mem_read_tokens`, maps the top-1 cell back to its source item, returns that item's `ground_truth` as the answer.
- **in_memory** (fallback for CPU-only CI runners): honest lexical word-overlap. Surfaced via `metadata.mode = "in_memory"` so it can never be confused with engine results.

`TDB_REFINEMENT_MODE=none|centered|prf` env var selects the engine refinement mode. Defaults to `none`.

### Smoke fixture (6 items, native mode, RTX 3070 Ti, deterministic_fallback evaluator)

| Refinement | avg score | passed |
|---|---:|---|
| none | 0.833 | 5/6 |
| centered | **1.000** | **6/6** |

The single miss at `none` (`How was ticket #8842 resolved?` retrieved the maintenance-window cell instead of the ticket-resolution cell) is recovered by mean-centering — consistent with the +31pp moderate-tier gain measured on the 100-memory vague-query corpus (`docs/experiments/vague_queries/results.md`).

This is a **6-item fixture**, too small to be statistically meaningful on its own, but it confirms the mechanism transfers from the synthetic vague-query corpus to the LoCoMo/LongMemEval question shape.

### Mem0 / Letta head-to-head — pending operator setup

To run the 3-way comparison the operator needs:
1. Docker stack up: `docker compose -f benchmarks/docker-compose.external.yml up -d`
2. `OPENAI_API_KEY` exported (Mem0 uses OpenAI for extraction + embeddings)
3. `LETTA_BASE_URL=http://localhost:8283 MEM0_BASE_URL=http://localhost:8888`
4. Run: `PYTHONPATH=python TDB_REFINEMENT_MODE=centered python -m tdb_bench run --mode smoke --config python/tdb_bench/config/default.json --output target/bench-v2/three-way.json`

The previous Tardigrade-vs-Letta numbers in this file should be re-run before being cited, since the adapter has changed.
