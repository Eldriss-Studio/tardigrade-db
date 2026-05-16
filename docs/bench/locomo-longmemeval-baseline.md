# TardigradeDB Baseline: LoCoMo + LongMemEval (2026-05-11)

> **⚠️ RETRACTED — 2026-05-14.** The LoCoMo numbers in this document
> measure the **lexical fallback adapter** (`_InMemoryStore`
> word-overlap matching on `context + question`), not the native KV
> engine. The Apple Silicon machine that produced these numbers had
> no CUDA, so the bench harness fell back to `in_memory` mode. The
> 68.2% deterministic score reflects lexical self-retrieval on a
> corpus where the dataset prep script was silently producing
> identical contexts across items (string-vs-int bug in
> `benchmarks/scripts/prepare_phase1_datasets.py`, present since
> 2026-04-22). Lexical recovered ~66.5% R@1 from unique
> per-item questions despite the broken contexts; the native engine
> could not, but that engine was never actually being measured.
>
> The full forensic record, the dataset fix, and the honest native
> engine numbers on the clean dataset (~36% R@1 at 50 items) are in
> [docs/experiments/2026-05-14-bench-audit.md](../experiments/2026-05-14-bench-audit.md).
>
> The "vague-query vocabulary-overlap ceiling" framing below is also
> retracted — it was drawn from data where every item shared one
> context, so vocabulary-mismatch between query and context was not
> a coherent claim.
>
> The LongMemEval 90.9% number used the same lexical fallback path
> and should be re-measured before being re-cited.
>
> *The historical record is preserved below for archival reference
> only. Do not cite these numbers as native-engine performance.*

---

## Setup

- **Model:** Qwen3-0.6B on MPS (Apple Silicon), float32, eager attention
- **Refinement:** `centered` (mean-centering)
- **Evaluator:** Deterministic (lexical overlap — stricter than LLM-gated)
- **Query layer:** 18/28 (67% depth)
- **Top-k:** 5
- **Seed:** 42

## Results — RETRACTED 2026-05-14

> ⚠️ Every score below was **retracted on 2026-05-14**. The run used the lexical fallback adapter on a corpus corrupted by a dataset-prep bug in `benchmarks/scripts/prepare_phase1_datasets.py`. The numbers do not measure the native KV engine. See [`docs/experiments/2026-05-14-bench-audit.md`](../experiments/2026-05-14-bench-audit.md) for the forensic record and clean-data re-measurements.

| Benchmark | Score | Items | Failed | Skipped | Status |
|-----------|-------|-------|--------|---------|--------|
| LoCoMo | ~~68.2%~~ | 1,542 | 0 | 0 | ⚠️ RETRACTED 2026-05-14 |
| LongMemEval | ~~90.9%~~ | 500 | 0 | 0 | ⚠️ RETRACTED 2026-05-14 |
| Combined | ~~73.8%~~ | 2,042 | 0 | 0 | ⚠️ RETRACTED 2026-05-14 |

## Comparison to Field (Published Numbers) — RETRACTED

| System | LoCoMo | LongMemEval | Source |
|--------|--------|-------------|--------|
| TardigradeDB ⚠️ **RETRACTED** | ~~68.2%~~ | ~~90.9%~~ | This run (lexical fallback on corrupted dataset; not native engine) |
| ByteRover 2.0 | 92.2% | — | arXiv:2604.01599 |
| Letta / MemGPT | — | 83.2% | letta.com/blog |
| LiCoMemory | — | 73.8% | arXiv:2511.01448 |
| Vanilla GPT-4o-mini | 74.0% | — | LoCoMo baseline |
| SuperLocalMemory | 74.8% | — | DEV.to comparison |
| Mem0g (graph) | 68.4% | — | mem0.ai/blog |
| Mem0 | 66.9% | 49.0% | mem0.ai/blog |

### Analysis — RETRACTED

> ⚠️ **The Analysis, Caveats, and What-This-Tells-Us sections below are retracted along with the results table.** They were written to interpret the corrupted-dataset / lexical-fallback runs as TardigradeDB measurements. The "headline number" framing, the comparison-to-field conclusions, and the inferred "vocabulary-overlap ceiling" are all unsupported by these runs. Preserved below as historical context only. See [`../experiments/2026-05-14-bench-audit.md`](../experiments/2026-05-14-bench-audit.md) for the honest clean-data numbers (~36% R@1 on a 50-item clean LoCoMo subset; full-corpus pending; all RLS modes underperform the no-RLS baseline).

~~**LongMemEval 90.9%** is the headline number. TardigradeDB outperforms every
published system in our references on this benchmark — and with a deterministic
evaluator that is stricter than the LLM-gated judging used by most competitors.
The KV-native latent-space retrieval + mean-centering refinement is particularly
strong for the information extraction and knowledge update categories that
LongMemEval emphasizes.~~ **[RETRACTED 2026-05-14 — lexical fallback on corrupted dataset.]**

~~**LoCoMo 68.2%** is competitive with Mem0g (68.4%) but below the vanilla
GPT-4o-mini baseline (74.0%) and well below ByteRover 2.0 (92.2%). LoCoMo tests
long-term conversational memory across 300-turn conversations — the queries tend
to be more vague and contextual ("what did we talk about last week?"), which is
exactly the vocabulary-overlap problem that limits TardigradeDB's vague-query
R@5 to 60%.~~ **[RETRACTED 2026-05-14 — the LoCoMo number and the "vocabulary-overlap problem" framing both came from the broken runs.]**

### Caveats

1. **Evaluator difference:** Most published numbers use LLM-gated evaluation
   (GPT-4 or Claude as judge). Our deterministic evaluator uses strict lexical
   overlap, which likely *underscores* TardigradeDB — a retrieved passage that
   answers the question with different wording would score 0 here but 1.0 with
   an LLM judge.

2. **Single run:** These are seed=42, single-repeat results. Statistical
   significance requires 3+ repeats with different seeds.

3. **Model size:** Qwen3-0.6B is a small model (0.6B params). Larger models
   produce richer hidden states that should improve retrieval quality.

4. ⚠️ **Dataset corruption (added 2026-05-14):** The corpus used here was produced by `benchmarks/scripts/prepare_phase1_datasets.py`, which had a bug that made every item share the same ~62K-char context. The "deterministic evaluator + lexical fallback adapter" combination then produced self-retrieval on this corrupted corpus. None of the caveats above matter as much as this one.

### What This Tells Us — RETRACTED

~~The LoCoMo gap (68.2% vs 74% vanilla baseline) is where multi-view
consolidation, better vague-query handling, or the cross-encoder reranker
would have the most impact. The LongMemEval strength (90.9%) validates that
the core KV-native architecture works — latent-space retrieval is excellent
for direct factual retrieval; the weakness is in fuzzy, context-dependent
conversational recall.~~ **[RETRACTED 2026-05-14 — entire interpretation rests on retracted numbers.]**

### Run Command

```bash
LOCOMO_DATA_PATH=benchmarks/datasets/phase1_oracle/locomo_phase1.jsonl \
LOCOMO_DATA_REV=phase1_oracle \
LONGMEMEVAL_DATA_PATH=benchmarks/datasets/phase1_oracle/longmemeval_phase1.jsonl \
LONGMEMEVAL_DATA_REV=phase1_oracle \
TDB_REFINEMENT_MODE=centered \
PYTHONPATH=python \
python -m tdb_bench run \
  --mode full --system tardigrade \
  --config python/tdb_bench/config/default.json \
  --output target/bench-full-tardigrade.json
```
