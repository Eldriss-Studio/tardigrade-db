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

## Phase 0 diagnostic + Phase 1A chunker fixes (2026-05-14 → 2026-05-15)

Detailed in `docs/superpowers/plans/2026-05-14-retrieval-debug-plan.md`. This section
records the measured results.

### Phase 0 — per-query rank diagnostic

Built `scripts/probe_rank_diagnostic.py` that, for each query,
records the rank at which the "expected" chunk (proxy: max
ground-truth-token overlap) appears in the latent retriever's top-K.
Bypasses the cross-encoder reranker so we can see the raw retrieval
signal.

**LoCoMo (1533 items, top-K=100):**

| Rank | Cumulative R@K |
|---|---|
| R@1 | 5.87% |
| R@5 | 19.83% |
| R@10 | 27.40% |
| R@25 | 38.69% |
| R@50 | 50.24% |
| R@100 | 65.56% |
| Outside top-100 | 34.44% |

The cross-encoder reranker lifts deterministic-eval bench score from
5.87% (raw R@1) to 29.62% — a 5× boost by pulling the right chunk
from rank 5-100 to rank 1. Reranker does nearly all the recall
work; the 34% of items where the right chunk isn't in top-100 are
the headroom Phase 1 needs to address.

**LongMemEval (500 items, top-K=25 due to compute cost at full top-K):**

| Rank | Cumulative R@K |
|---|---|
| R@1 | 0% |
| R@5 | 0% |
| R@10 | 0% |
| R@25 | 0% |
| Outside top-25 | 100% |

**Catastrophic — the expected chunk never appears in top-25.**
Forensic inspection (`scripts/probe_hub_cells.py`) showed only 9
unique cells appeared in top-1 across all 500 queries; cell 225
alone was in top-10 of 99.2% of queries. A handful of "hub cells"
dominate all retrievals regardless of query content.

All identified hub cells started with mid-word fragments — typically
mid-number — e.g., `"0000 is a great option..."`, `"000 miles..."`,
`"88 as a rough estimate..."`, `"000 PQMs..."`. Pattern traced to
the chunker producing fragment chunks that carry anomalous hidden
states which score against any query.

### Phase 1A.1 — chunker END boundary fix

`TextChunker._split_tokens` (`python/tardigrade_hooks/chunker.py`)
accepted a `BoundaryStrategy` in `__init__` but never called it
inside the splitter. Chunks were sliced purely by token count.
Fix: invoke `self._boundary.find_split` on every non-final chunk
whose char-space end would otherwise fall mid-word.

Also added `ParagraphBoundaryStrategy` — prefers `\n\n` (turn
boundary in conversational transcripts) → sentence end → whitespace.
Wired as default in the bench adapter.

**Re-measured diagnostic with end-trim only:**

LoCoMo: R@25 38.69% → **54.79%** (+16.1pp middle-rank lift).

LongMemEval: still **0% R@25**. Hub-cell pattern unchanged in
shape — 8 cells in 87-100% of top-10s, just different cell IDs
(13940, 24108, 23874, etc.). All still numeric-fragment starts.

The end-only fix changed cell *ends* but not cell *starts*. Each
non-first chunk's start is determined by overlap-token rollback
into the previous chunk — which lands mid-word almost every time
for sub-word tokenizers.

### Phase 1A.1b — chunker START boundary fix

Forward-snap chunk `start_char` to the next whitespace within the
`BOUNDARY_LOOKBACK_RATIO`-sized window when the preceding character
is alphanumeric. Mirrors the end-trim's lookback logic. Non-zero
overlap no longer produces mid-word starts.

**Re-measured diagnostic with both ends and starts trimmed:**

| Dataset | Metric | Broken | End-only | **Start-trim (Phase 1A.1b)** |
|---|---|---|---|---|
| LoCoMo | R@25 | 38.69% | 54.79% | **54.67%** |
| LongMemEval | R@1 | 0% | 0% | **0%** |
| LongMemEval | R@5 | 0% | 0% | **0.2%** |
| LongMemEval | R@10 | 0% | 0% | **29.4%** |
| LongMemEval | R@25 | 0% | 0% | **39.2%** |
| LongMemEval | unique cells in top-10 | 19 | 21 | **1685** |

**The fragment hypothesis was correct.** Fixing chunk starts moved
LongMemEval R@25 from 0% to 39.2%, and top-10 cell diversity from
19 distinct cells to 1685 (80× more diverse). The hub-cell
catastrophe is broken.

### Residual finding — numeric-token anisotropy

After both ends and starts are trim-aware, 5 cells still appear in
99.8% of LongMemEval top-10s:

- cell 3606: `"55-inch 4K UHD TV..."`
- cell 19047: `"11 kg (24 lbs)..."`
- cell 17411: `"99.9% dust-free..."`
- cell 16350: `"99.9% dust-free..."` (duplicate from variant item)
- cell 4260: `"22. Avengers: Age of Ultron..."`

These chunks start at clean word boundaries (no mid-word fragments)
but the words themselves are **numbers**. Numeric BPE tokens
produce anomalous high-norm hidden states regardless of chunk
cleanliness — genuine latent-space anisotropy. The hook already
skips position 0 (BOS attention sink); position 1 may carry a
similar but less-skipped outlier signal for numeric-leading
chunks.

This explains why LongMemEval R@10 caps at 29.4% rather than
reaching 50-60%: 5 of every 10 top-10 slots are occupied by these
residual hubs, leaving only 5 "real" slots for actual content.

**Two candidate interventions for the next slice:**

1. **ZCA whitening** — already implemented in
   `crates/tdb-retrieval/src/refinement.rs::WhitenedRescore`.
   Removes the strongest correlated directions in the latent
   space; if the "number-token direction" is one of them,
   whitening will defuse the residual hubs. Activated via
   `TDB_REFINEMENT_MODE=whitened`. Zero new code.
2. **Skip position 1 in addition to position 0** in the hook. If
   numeric-leading chunks have position-1 attention-sink behavior,
   skipping it kills the anisotropy at source. ~3 lines of code
   in `python/tardigrade_hooks/hf_kv_hook.py`.

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
