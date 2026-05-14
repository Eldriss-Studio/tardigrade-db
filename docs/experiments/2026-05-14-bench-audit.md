# Benchmark Audit — 2026-05-14

**Status:** authoritative. Supersedes earlier baseline and RLS result docs.
**Audit duration:** ~6 hours across two machines (RTX 3070 Ti + WSL2 / Apple Silicon).

This document is the forensic record of a bench audit that started from
"RLS isn't helping, recall is 4%" and ended with two distinct dataset
bugs, one retracted baseline, and one bench-adapter feature stack that
turned out to be net-positive once measured against honest data.

---

## TL;DR

1. **The dataset preparation script
   `benchmarks/scripts/prepare_phase1_datasets.py` had a two-stage
   bug** that corrupted every LoCoMo benchmark run since it was first
   committed (2026-04-22, commit `d9ecc1c`).
2. **The "68.2% LoCoMo baseline" measured the lexical fallback
   adapter, not the native KV engine.** On the Apple Silicon machine
   that ran the historical numbers, no CUDA was available, so the
   bench fell back to in-memory word-overlap. That fallback genuinely
   scored 66.5% R@1 on self-questions; the number is real but it
   measured a *different system* than the public narrative implied.
3. **The native KV engine, measured honestly on a clean dataset,
   scores ~36% R@1 at 50 LoCoMo items** through the production bench
   adapter, and ~20% R@1 through a minimal probe with no reranker /
   chunking / quantization tier. The full feature stack is net
   positive on clean data — opposite of the conclusion the broken
   dataset implied.
4. **All four reformulation strategies (RLS) underperform the
   no-RLS baseline on the clean dataset.** Keyword expansion loses
   5.3pp; the DeepSeek agent reformulator loses 12.7pp while adding
   ~1.7s latency per query. The "RLS = 0% improvement" experiments
   recorded in prior docs were measuring the lexical fallback or a
   broken dataset; the honest verdict is that RLS as currently
   implemented hurts retrieval, not "is neutral".

---

## How we got here

The session began with a CUDA bench run on RTX 3070 Ti showing
~4% deterministic score on LoCoMo, vs a claimed 68.2% baseline. After
killing several long-running benches and burning ~3 hours auditing
retrieval / hook / quantization code that turned out to be fine, the
root cause was found in the dataset prep script.

### Stage 1 of the bug — string vs int

LoCoMo source items reference evidence by string keys like `"D1:3"`
(session 1, dialogue turn 3). The prep script accepted only `int`
keys:

```python
# benchmarks/scripts/prepare_phase1_datasets.py (broken)
for eid in evidence_ids:
    if isinstance(eid, int) and eid in dia_to_text:
        evidence_lines.append(dia_to_text[eid])
```

Every `eid` was a string, so the check always failed, `evidence_lines`
was always empty, and every item silently fell through to
`context = full_context`. In `--locomo-context evidence` mode the
output was supposed to be per-item evidence sentences (30-200 chars
each); it was actually the full ~62,000-character conversation for
every item in a 10-conversation, 1542-question corpus.

**Effect:** 20 items per conversation share one 62K-char context.
For the native KV engine this destroys retrieval — every cell in a
conversation has indistinguishable hidden-state keys. For the lexical
fallback adapter, contexts are also identical, but the *question*
field is unique per item, and `_InMemoryStore.scored_best_match`
scores on `context + question`. The exact question wins by
overlap-with-itself, so lexical recovers 66.5% R@1.

**Fix:** accept both `int` and `str` keys.

```python
if isinstance(eid, (int, str)) and eid in dia_to_text:
    evidence_lines.append(dia_to_text[eid])
```

Same fix applied to `dia_id` accumulation a few lines up.

### Stage 2 of the bug — empty-evidence carpet bombing

After Stage 1 was fixed, 1538 of 1542 items had proper per-item
evidence (typically 30-200 chars). The remaining 4 had empty
evidence arrays in the LoCoMo source (`"evidence": []`). The prep
script still fell through to `full_context` for those 4 items.

In native-engine mode, items go through `TextChunker` at ingest.
A 200-char evidence text produces 1 chunk; the 62K-char full
conversation produces 128 chunks. Four items × 128 chunks = 512
"noise" cells from items with no oracle context, competing against
1538 single-cell items in the same corpus. The four noise items
dominated retrieval because their cells covered the entire
conversation's vocabulary.

**Concrete measurement:** before this fix, the bench adapter scored
R@1 = 6% (strict) on 50 LoCoMo items, of which only 3 items had
empty-evidence carpet bombing. After the fix, R@1 = 36%.

**Fix:** in evidence mode, skip items with no evidence rather than
falling back to full conversation. They have no oracle context
anyway; including them is silently destructive.

```python
if context_mode == "evidence":
    if not evidence_lines:
        continue  # oracle mode: no evidence = no item
    context = "\n".join(dict.fromkeys(evidence_lines)).strip()
else:
    context = full_context
```

Resulting corpus: 1533 LoCoMo rows (0.6% loss) + unchanged 500
LongMemEval rows.

---

## What 68.2% actually measured

On the Apple Silicon machine that produced
`docs/bench/locomo-longmemeval-baseline.md`, the bench harness ran
the `TardigradeAdapter` in `in_memory` mode because the machine had
no CUDA. The lexical store's `best_match` returns the top-scoring
item's `ground_truth` verbatim. For self-retrieval against the same
1542 items, with unique questions per item, the lexical word-overlap
matcher picks the right item ~66% of the time and returns the
verbatim correct answer, scoring exactly 1.0 on the deterministic
evaluator.

Distribution of scores on the 1542-item historical run:

- 66.5% of items: score = 1.0 (lexical picked correct item)
- 28.3% of items: score = 0.0 (wrong item, zero token overlap)
- 5.1% of items: partial (0.0 < score < 1.0)

**The 68.2% number is a real measurement of the lexical
fallback's self-retrieval performance. It is not a measurement of
the native KV engine.** The error in the published baseline doc was
framing it as the native engine's performance. The bench output
includes `evaluator_mode: "deterministic_fallback"` per item, which
should have been read as "the LLM judge never ran" — but the
adapter-mode metadata was not surfaced in the headline.

This also invalidates:

- "Vague-query vocabulary mismatch is the retrieval ceiling" — the
  conclusion was drawn from data where every item had identical
  context, so "vocabulary mismatch between query and context" was
  not a coherent claim. The DeepMind LIMIT paper citation in
  `docs/refs/external-references.md` as theoretical support for our
  empirical ceiling is unsupported by our data; the LIMIT paper
  itself is unaffected, the citation context is wrong.
- The 67.2% LLM-judge follow-up
  (`docs/bench/locomo-longmemeval-llm-judged.md`): also lexical
  mode, also a different-system measurement.

---

## Native engine performance on clean data

Two probe paths, both Qwen3-0.6B on CUDA, refinement=centered,
query_layer=18, layer ratio 0.67:

| Path | Features | R@1 (50 items) | R@5 |
|---|---|---|---|
| Minimal probe | 1 cell/item, no chunking, no batching, no INT8 tier, no reranker | 20% | 58% |
| Bench adapter | Chunking + GPU batch=8 + INT8 lazy tier + cross-encoder reranker + mem_write_batch | **36%** | not measured |

The full bench-adapter stack adds 16pp of R@1 vs the minimal probe.
The reranker and feature stack are net positive when the corpus is
clean. The historical claim that the bench adapter was "leaking
recall" was an artifact of the empty-evidence carpet-bombing —
items with 128 chunks each dominate the cell pool and bias the top-1
toward the wrong items.

### Full-corpus headline numbers (2026-05-14)

Same configuration, full clean dataset, single seed (42):

| Dataset | n | avg score | score = 1.0 (R@1 exact) | score = 0.0 (miss) | partial |
|---|---|---|---|---|---|
| **LoCoMo** | 1533 | **29.62%** | 410 (26.7%) | 946 (61.7%) | 177 |
| **LongMemEval** | 500 | **3.14%** | 5 (1.0%) | 439 (87.8%) | 56 |
| Combined | 2033 | 23.11% | — | — | — |

**LoCoMo 29.62%** (deterministic) is the honest tardigrade-on-Qwen3-0.6B
LoCoMo number on the clean evidence-only corpus. Comparable to
published baselines (with the heavy caveat that those use different
evaluators): Mem0 66.9%, Letta 74%, vanilla GPT-4o-mini 74%,
ByteRover 92.2%. The gap to even the vanilla baseline is real and
points to either a larger capture model, a trained retrieval head,
or a different retrieval signal entirely.

**LongMemEval 3.14%** is the bigger story. The historical "90.9%"
number was the lexical fallback adapter scoring on the question
field; the native KV engine, asked to find the one specific evidence
chunk among ~40,000 cells (500 items × 25-150 chunks each), almost
never wins. Per-token Top5Avg on raw hidden states cannot resolve
needle-in-haystack at this model size. Hypotheses for the collapse:

- Chunking dilutes per-item identity (no single chunk contains both
  the question semantics and the answer text).
- Top-K is dominated by chunks from items with similar surface
  vocabulary, not the item that contains the actual answer.
- Raw next-token-prediction hidden states aren't trained to be a
  retrieval signal; the haystack task amplifies that mismatch.

A trained retrieval head, hybrid lexical-plus-latent retrieval, or a
larger capture model are the architectural levers worth testing for
LongMemEval.

---

## RLS verdict on clean data

50-item LoCoMo, deterministic evaluator, all bench-adapter features
enabled, single ablation per row:

| TDB_RLS_MODE | Avg score | Latency (median) | Delta vs none |
|---|---|---|---|
| none | **21.95%** | ~70 ms | — |
| keyword | 16.62% | ~250 ms | -5.3 pp |
| agent (DeepSeek) | 9.29% | ~1700 ms | **-12.7 pp** |

All RLS modes underperform the no-RLS baseline. The agent
reformulator (DeepSeek `deepseek-chat` API) is the worst — it adds
~1.7s/query of LLM latency and loses 12.7pp of recall. Hypotheses
worth testing if RLS is to be salvaged:

1. **Fusion logic discards good answers.** The original query likely
   retrieves the correct item; reformulations add wrong items and the
   fusion picks one of them.
2. **Reformulations dilute the latent signal.** Replacing
   "When did Caroline go to LGBTQ support group?" with synonyms
   changes the Q-side hidden states in ways that don't move closer
   to the K-side stored representation.
3. **Confidence threshold is mis-calibrated.** It was tuned
   (1.5 → 1.10) against the broken baseline, so the calibration is
   meaningless.

Recommended action: deprioritize RLS until either the fusion logic
is rebuilt or evidence emerges that it helps on a different
retrieval architecture (e.g., a trained embedding head rather than
raw hidden states).

---

## What survives the audit

The bench bug invalidates a narrow set of LoCoMo-specific
conclusions, not the broader engineering work.

**Still valid:**

- 663 tests passing on the core engine (304 Rust + 359 Python).
  None depend on the LoCoMo data.
- Synthetic-corpus recall: 100% at 100 memories, 100% at 5K
  memories (Top5Avg, Q4 pipeline). These use synthetic distinct
  corpora and measure what the engine actually does on
  well-separated keys.
- KV injection equivalence to text-in-prompt (9/10 recall on
  synthetic gibberish facts with Qwen3-0.6B). Distinct
  experimental setup, not affected.
- Cross-model retrieval (same-family 90% R@5 via linear
  projection, cross-family 76.7% via MLP adapter). Synthetic
  corpora, unaffected.
- vLLM connector hardening, multi-agent isolation, semantic edges,
  multi-view consolidation v2, file ingestion (`TextChunker`,
  `FileIngestor`, `TardigradeClient`). Each measured against its
  own targeted experiment, not LoCoMo.
- Cross-encoder reranker, mean-centering refinement, INT8 lazy
  tier, GPU batching, `mem_write_batch` — each demonstrably
  improves recall over the minimal probe on the clean dataset
  (probe 20% → bench 36% at 50 items).
- Architectural patterns (Strategy, Bridge, Factory, Active
  Object, Decorator, Adapter, Repository). Code quality is
  independent of the dataset.

**Retracted or invalidated:**

- 68.2% LoCoMo deterministic baseline
  (`docs/bench/locomo-longmemeval-baseline.md`) — measured the
  lexical fallback adapter, framed as native engine.
- 67.2% LoCoMo LLM-judge baseline
  (`docs/bench/locomo-longmemeval-llm-judged.md`) — same.
- "Vague-query vocabulary mismatch is the retrieval ceiling" — the
  conclusion was drawn from data where context was identical
  across items; the claim has no empirical support from these
  runs.
- All RLS technique-elimination experiments documented in
  `docs/experiments/` (keyword, multiphrasing, embedding,
  generative 3B, agent) recorded as "0% improvement on LoCoMo".
  Those numbers were measured against the broken baseline. On the
  clean dataset, RLS is actively harmful (-5 to -13 pp).
- The DeepMind LIMIT paper citation in
  `docs/refs/external-references.md` as theoretical support for our
  measured ceiling. The paper is unaffected; the application of it
  to our empirical results is not justified by data.

---

## Recommendations going forward

1. **Run the full 1533-item LoCoMo bench on the clean dataset** with
   the production adapter (no RLS) to establish the honest headline
   number. Expected range: ~25-35% deterministic, based on the
   50-item subset.
2. **Strip or quarantine RLS.** Three reformulation strategies +
   DeepSeek integration + confidence machinery now have measured
   evidence of being harmful. Recommend marking experimental and
   default-off until a fusion redesign is shipped.
3. **Re-run cross-encoder, mean-centering, and the rest of the
   refinement experiments on the clean dataset.** They've been
   validated as net-positive in the adapter ablation; the absolute
   improvement numbers in
   `docs/experiments/vague_queries/results.md` need to be
   re-measured.
4. **Update CLAUDE.md status line** — the "vague-query refinement"
   numbers and the "vocabulary overlap cliff" claim need either new
   measurements or retraction.
5. **Field comparisons (Letta 74%, Mem0 67%, ByteRover 92%)** are
   from those systems' own papers with their own evaluators. Until
   they are re-measured on the same clean dataset with the same
   evaluator, do not claim tardigrade ranks against them.

---

## Verification artifacts

Reproducible probes saved at:

- `scripts/probe_recall_audit.py` — minimal native-engine R@1/R@5
  probe (bypasses bench adapter and evaluator).
- `scripts/probe_adapter_direct.py` — direct bench-adapter probe
  (bypasses runner and evaluator, exposes `cell_to_item` cardinality).
- `scripts/probe_key_inspect.py` — inspects encoded per-token keys
  for a single LoCoMo item; used during the initial diagnosis.

Bench configs:

- `target/bench-rls-probe-config.json` — 50-item LoCoMo subset,
  deterministic eval, native-only.

Dataset fingerprints on this machine after the fix (1533 items):

- `locomo_phase1.jsonl`: SHA256 changed (see `manifest.json`)
- `longmemeval_phase1.jsonl`: SHA256
  `7038ff132e9fba12f082db1047b0a4c707769cb96cdcaade05494759e88b460a`
  (unchanged — LongMemEval prep was not affected)

Commits relevant to the audit:

- `aa50951` — the historical "68.2% baseline" commit (2026-05-11)
- `d9ecc1c` — the first appearance of the prep-script bug
  (2026-04-22)
- HEAD as of this writing: `d2e0635`
