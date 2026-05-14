# Plan: Staged Retrieval Debug — Indexing First, Architecture Second

**Date:** 2026-05-14
**Anchor docs:**
- [`docs/experiments/2026-05-14-bench-audit.md`](../../experiments/2026-05-14-bench-audit.md)
  — what we measured
- [`docs/refs/retrieval-architecture-research-2026-05-14.md`](../../refs/retrieval-architecture-research-2026-05-14.md)
  — what the literature says
- [`docs/refs/external-references.md`](../../refs/external-references.md)
  §A3g — gating citations

---

## Premise

The audit established honest TardigradeDB numbers on the clean
dataset:

| Dataset | Score | vs weakest published baseline (BM25) |
|---|---|---|
| LoCoMo | 29.62% | ~35-45pp below typical hybrid systems |
| LongMemEval | 3.14% | **22× below BM25 R@5 = 68.3%** |

The research review found one dominant finding for guiding the
debug: **the LongMemEval paper's own ablation shows indexing
strategy moves recall ~15 points; swapping a strong retriever for a
weak one moves it ~5-8 points.** A 22× gap to the weakest baseline
cannot be explained by retriever choice. It is dominantly an
indexing problem.

This plan is the staged debug derived from that finding. Each
phase has a published-supported hypothesis, an ATDD-style success
criterion, and a clear gate to the next phase. The cost grows with
each phase; the leverage drops. We stop at the first phase that
delivers the target.

The plan is named-pattern, no-magic-values, ATDD-first per
`CLAUDE.md`. Each phase ships independently; no phase requires the
next.

---

## Phase 0 — Re-baseline & instrument

**Goal:** lock in a reproducible measurement of where we are *now*,
on the clean dataset, with the current production adapter, with
enough instrumentation to answer "where did recall actually go?"
when later phases shift it.

**Why:** Today we have aggregate scores (29.62% / 3.14%) but no
visibility into *why* each query missed. Per-query diagnostics are
the foundation for evaluating every subsequent phase.

### Acceptance test (AT-0)

Behavioral: a single command produces a structured per-query
report — for each item, record `expected_chunk_id`,
`retrieved_top_k_chunk_ids`, `expected_rank` (None if not in top-K),
and `score`. Aggregates to a histogram of `expected_rank` per
dataset. Runs cleanly on both LoCoMo and LongMemEval at full scale
in <60 min.

```
GIVEN  the production bench adapter and clean datasets
WHEN   the diagnostic bench is run on LoCoMo (1533) + LongMemEval (500)
THEN   a JSON report is written with per-item retrieval rank histograms,
       AND the histogram for LongMemEval reveals where the right chunk
       lands (top-1? top-10? top-100? out of top-K entirely?)
```

### Design patterns

- **Observer / Audit Log** for the per-query diagnostic stream.
- **Strategy** for the rank-classifier (LoCoMo treats item-as-target;
  LongMemEval needs chunk-as-target with the chunk-to-item map).

### Constants (no magic values)

- `DIAGNOSTIC_TOP_K_PROBE = 100` — enough to see if the right chunk
  is "near miss" (top-10) or "completely lost" (out of top-100).
- `DIAGNOSTIC_OUTPUT_PATH` from env var
  `TDB_DIAGNOSTIC_REPORT` with documented default.

### Gating criterion

Pass → Phase 1. The instrument is the prerequisite for everything
that follows.

---

## Phase 1 — Indexing fixes (the dominant published lever)

**Goal:** Test the LongMemEval paper's headline finding — that
indexing strategy outranks retriever choice — by applying its three
published interventions to TardigradeDB. All three already have
primitives in the codebase; this phase wires them together.

**Hypothesis:** Phase 1 alone closes most of the gap. Published
evidence supports an expectation of ~50-70% LongMemEval R@5 after
the three interventions, up from 3.14%. LoCoMo expected to climb
~5-10pp (the LoCoMo evidence-mode dataset is already at per-item
granularity, so the bigger interventions matter less).

If Phase 1 hits the targets, the foundational bet ("raw hidden
states as retrieval signal") is partially vindicated — adequate
*given* the right indexing.

### Slice 1A — Round-level chunking

**Source:** LongMemEval Finding 1 (round granularity, not session).

**What:** Replace `TextChunker`'s fixed-token chunks with a
**RoundBoundaryStrategy** — chunks at conversation-turn boundaries
(speaker change, paragraph break, sentence boundary for narrative
text). For LongMemEval, each session decomposes into N rounds;
each round becomes its own retrieval cell. Expected drop in cells
per item from ~80 to ~10-15.

**Pattern:** **Strategy** — new `BoundaryStrategy` implementation
plugged into `TextChunker` (the existing
`python/tardigrade_hooks/chunker.py:BoundaryStrategy` abstraction).
No engine-side changes required.

**AT-1A** (Python, `tests/python/test_round_boundary_chunker.py`):

```
GIVEN  a conversation transcript with N speaker turns
WHEN   chunked with RoundBoundaryStrategy
THEN   the chunker produces N chunks, one per turn (not a
       max-tokens-driven split)
AND    each chunk's text preserves the speaker prefix
       (so retrieval keys carry speaker identity)
```

**Constants:**
- `ROUND_MIN_TOKENS = 8` — merge very short turns (e.g. "Yes.")
  into the next-following turn.
- `ROUND_MAX_TOKENS = 512` — hard cap for runaway turns; falls back
  to sentence-boundary split if exceeded.

### Slice 1B — Fact-augmented keys via `add_view_keys`

**Source:** LongMemEval Finding (+4% retrieval, +5% downstream).
Mem0 uses the equivalent "entity boosting" pattern.

**What:** During or right after ingest, run a lightweight fact
extractor over each chunk's text. The extractor emits 1-3 atomic
fact strings per chunk (subject-predicate-object form, or a
question form that paraphrases the fact). Each extracted fact is
encoded as an additional retrieval key on the same cell via
`engine.add_view_keys`. The cell remains canonical (deduplication
at retrieval is unchanged by the parent-document pattern).

**Pattern:** **Strategy** for the fact extractor:
- v1: deterministic rule-based extractor (named-entity heuristics +
  sentence segmentation) — no LLM dependency, ships first.
- v2 (optional): LLM extractor — same interface, behind a feature
  flag; quality vs cost tradeoff measured separately.

**AT-1B** (Python,
`tests/python/test_fact_augmented_keys.py`):

```
GIVEN  a 100-turn conversation ingested via the production adapter
WITH   FactAugmentingExtractor enabled
WHEN   ingestion completes
THEN   each cell has >= 1 view key attached (verified via
       engine.view_count)
AND    a query that uses surface vocabulary present only in the
       extracted fact (not the raw chunk) retrieves the cell in
       top-K
```

**Constants:**
- `FACTS_PER_CHUNK_TARGET = 2` — soft target; extractor may emit
  1-3.
- `MAX_FACT_TOKENS = 64` — hard cap on a single fact's encoded
  length.

### Slice 1C — Time-aware query expansion

**Source:** LongMemEval Finding (+6.7 to +11.4% recall on temporal
queries).

**What:** A query classifier inspects the incoming query for
temporal cues (`yesterday`, `last week`, `before X`, ISO dates).
When found, the query is expanded with the *temporal scope*
resolved to a concrete date range, and that range is added to the
retrieval key. The mechanism reuses `add_view_keys`'s
infrastructure: queries carry an optional metadata vector matched
against stored metadata.

**Pattern:** **Strategy** for the temporal classifier (rule-based v1,
LLM-based v2 if needed).

**AT-1C** (Python,
`tests/python/test_temporal_query_expansion.py`):

```
GIVEN  a corpus with three cells from dates {2024-01-15,
       2025-06-01, 2026-05-10}
WHEN   queried with "what happened last year" on 2026-05-14
THEN   the 2025-06-01 cell ranks higher than the 2024-01-15 cell
       (which is two years old, not "last year")
```

**Constants:**
- `TEMPORAL_RECENCY_WINDOW_DAYS` per relative-time term
  (`yesterday=1`, `last week=7`, `last month=31`, `last year=365`).
- `TEMPORAL_SCORE_BOOST = 1.25` — multiplicative on the latent
  score for cells within the resolved window.

### Phase 1 gating

After all three slices: re-run the full clean-dataset bench under
the Phase 0 diagnostic harness.

**Target:** LongMemEval ≥ 50%, LoCoMo ≥ 35%.

- **≥ 50% LongMemEval** → ship Phase 1 to main; consider Phase 2
  only if LoCoMo is still mediocre vs published competitors.
- **30-50% LongMemEval** → ship Phase 1, proceed to Phase 2.
- **< 30% LongMemEval** → indexing wasn't the dominant lever for
  TardigradeDB. The bottleneck is in the retrieval signal itself;
  proceed directly to Phase 3 (projection head).

These thresholds are deliberately conservative; the LongMemEval
paper's interventions on BM25 + Contriever produce ~76% R@5, so
50% is a 25-point buffer for "TardigradeDB-specific signal weakness."

---

## Phase 2 — Add BM25 as a second retrieval signal

**Goal:** Test the hypothesis that the latent signal's failure
mode (lexical-overlap-blindness, surface-token drift) is
**complementary** to a sparse signal — i.e., BM25 catches what
hidden-state retrieval misses.

**Hypothesis:** Tuned convex combination of latent + BM25 lifts
recall meaningfully beyond Phase 1, especially on queries with
unique surface tokens (proper nouns, exact dates, technical terms).

**When to enter:** Phase 1 shipped, gap to baselines still > 10
points.

**Published support:**
- BEIR aggregate: hybrid +9pp nDCG@10 over BM25 alone.
- LongMemEval: BM25 alone is 68.3% R@5; the paper's intermediate
  configurations using fact-augmentation hit 76%. Hybrid is not
  ablated separately but expected to stack.
- Bruch et al. (TOIS 2023): tuned convex combination beats RRF.

### Architecture

A new `LexicalRetriever` stage in the pipeline, fused with the
existing latent retriever's output via **tuned convex combination**
(weighted score sum), then handed to the cross-encoder reranker
unchanged.

**Pattern:**
- **Strategy** — `LexicalRetriever` implements the same
  `Retriever` trait as `PerTokenRetriever`.
- **Bridge / Pipeline** — pipeline stages already chain via the
  existing `RetrieverPipeline`; the new stage slots in.
- **Decorator / Score-Fusion** — `ConvexCombinationFusion(
  latent_weight, sparse_weight)` wraps the two retrievers' outputs
  and produces a single ranked list.

### Slice 2A — `LexicalRetriever` over `TextStore`

The `TextStore` already exists in Rust; it holds cell text. A
BM25 index over its contents is straightforward — every Rust
search ecosystem has a BM25 implementation (e.g., `tantivy`,
`bm25` crate).

**AT-2A** (Rust, `crates/tdb-retrieval/tests/lexical_retriever.rs`):

```
GIVEN  a corpus of 1000 cells with text in the TextStore
WHEN   queried via LexicalRetriever with BM25 scoring
THEN   recall@5 for exact-token queries ≥ 95%
AND    BM25 scores are produced in <5ms for the 1000-cell corpus
```

### Slice 2B — Convex-combination fusion

Score normalisation is the subtlety. Latent scores and BM25
scores are on different ranges; min-max normalisation per query
before weighting is the standard recipe (Bruch et al.).

**Pattern:** **Template Method** — base `Fusion` trait, concrete
`ConvexCombinationFusion` and `ReciprocalRankFusion`
implementations (RRF kept as the zero-tuning fallback).

**AT-2B** (Python,
`tests/python/test_hybrid_retrieval_fusion.py`):

```
GIVEN  the bench adapter wired with LexicalRetriever + latent +
       ConvexCombinationFusion (alpha=0.5 default)
WHEN   the full clean LongMemEval bench is run
THEN   R@5 ≥ Phase 1 baseline + 5pp
AND    queries with unique proper-noun terms (verified separately)
       show large recall gains relative to latent-only
```

### Phase 2 gating

**Target:** combined LoCoMo ≥ 50%, LongMemEval ≥ 70%.

If hit → ship. If not → Phase 3.

---

## Phase 3 — Train a projection head on hidden states

**Goal:** Test whether retrieval-specific training of the latent
signal (the LLM2Vec / RepLLaMA / "Native Retrieval Embeddings"
pattern) closes the remaining gap, *without* abandoning the
KV-native architecture.

**Published support:**
- LLM2Vec (ICLR 2024): unsupervised contrastive on decoder LLM
  produces MTEB SOTA 56.8.
- "Native Retrieval Embeddings from LLM Agent Hidden States"
  (arXiv:2603.08429): explicitly demonstrates a lightweight
  projection head on agent hidden states achieving competitive
  retrieval.
- TardigradeDB's own cross-model results (Qwen3 → GPT-2 via MLP
  adapter, 76.7% R@5 on synthetic corpora) are a working prior on
  the same shape.

### Architecture

A small **PyTorch projection head** trained on a contrastive
objective: anchor = query hidden states, positive = correct cell's
hidden states, negatives = sampled wrong cells. Once trained, the
projection is applied **at retrieval-key build time** (engine-side,
not capture-side) — the engine stores raw hidden states still, but
the retrieval pipeline projects them before scoring. This keeps
the KV-native property intact for downstream injection.

**Pattern:**
- **Strategy** — new
  `RetrievalKeyStrategy::ProjectedHiddenStates(model_path)` in the
  existing pluggable retrieval-key strategy slot.
- **Adapter** — Python-side `ProjectionHeadTrainer` wraps the
  training loop, persistence, and integration with the
  `RetrievalKeyStrategy`.
- **Template Method** for the trainer's lifecycle
  (calibration → train → validate → persist).

### When to enter

Phase 2 shipped; LongMemEval still < 70%.

### Slice 3A — Training infrastructure

Build the training loop with a synthetic-corpus dry-run first
(the Sonia 100-fact corpus where TardigradeDB already gets 100%
specific recall) to validate the trainer is correct before
training on benchmark data.

**AT-3A:** projection trained on Sonia corpus maintains 100%
specific recall while improving vague-tier recall by ≥ 10pp.

### Slice 3B — Training on benchmark data

Train on a HELD-OUT split of LoCoMo + LongMemEval evidence
sentences (not the test queries). Evaluate on the test split.

**AT-3B:** LongMemEval R@5 ≥ 75% on the test split after projection
head is trained on the held-out split.

### Phase 3 gating

**Target:** LongMemEval ≥ 75%, LoCoMo ≥ 55%.

If hit → ship as `RetrievalKeyStrategy::ProjectedHiddenStates`,
make it the default. Foundational bet survives with retrieval-
specific training.

If not → Phase 4.

---

## Phase 4 — Trained dense retriever as primary signal

**Goal:** Last-resort architecture pivot. Accept that the hidden-
state signal — even with a projection head — is not competitive
on these benchmarks, and reposition TardigradeDB's KV-cache
machinery as the **injection** mechanism, not the **retrieval**
signal.

**When to enter:** Phase 3 didn't close the gap.

### Architecture

Add an off-the-shelf trained dense retriever (e5-small or
BGE-small, ~33-100M params, CPU-runnable) as a third retriever
in the pipeline. The latent hidden-state path becomes one signal
among three (latent + lexical + trained-dense), all fused via
convex combination, all reranked by the cross-encoder.

**Pattern:**
- **Strategy** — a third `Retriever` implementation
  `DenseEmbeddingRetriever(model_name)`.
- **Composite** — the pipeline now composes three retrievers
  cleanly via the existing fusion stage.

### Slice 4A — Wire `DenseEmbeddingRetriever`

Use `sentence-transformers` or HuggingFace `transformers` for the
embedding model. Cache embeddings at ingest time (one forward
pass per chunk's text, separate from the LM forward pass).

**AT-4A:** LongMemEval R@5 ≥ 75% with the dense retriever
contributing > 30% of the fused score on average.

### Phase 4 gating

**Target:** LongMemEval ≥ 75%, LoCoMo ≥ 60%. Field-comparable.

If hit → ship, **reframe TardigradeDB's product positioning** in
docs: the differentiator becomes the persistent KV-cache
injection mechanism + AKL governance + multi-view consolidation,
not the retrieval signal.

---

## Reliability checklist (per CLAUDE.md)

- **Durability contract:** unchanged in Phases 0-2; Phase 3 introduces
  a model weights sidecar (~10-100MB) that must be flushed before
  any cell using `ProjectedHiddenStates` is written; recovery test
  required.
- **Consistency mode:** unchanged.
- **Recovery contract:** projection-head weights are derived state
  (rebuildable from durable training data). Document the rebuild
  command. Fail-fast on corrupt sidecar.
- **Derived-state rebuildability:** BM25 index (Phase 2) and
  projection-head weights (Phase 3) are both derivable. Rebuild
  paths must be documented.
- **Fail-fast replay:** stays the same — replay inconsistencies
  are hard errors.

Mandatory metrics updates for each phase:
- Phase 0: per-query rank histogram, latency per stage.
- Phase 1: same + cells-per-item distribution shift after
  round-level chunking.
- Phase 2: same + per-query weights of fusion components (which
  signal dominated).
- Phase 3: same + projection-head training metrics (final
  contrastive loss, validation recall).
- Phase 4: same + dense retriever forward-pass latency.

---

## Out of scope (deferred)

- **ColBERTv2 centroid + residual quantization** —
  `docs/refs/external-references.md` §A3f deferral; appropriate at
  ≥5M cells, well beyond LoCoMo / LongMemEval scale.
- **Larger capture model (Qwen2.5-3B)** — orthogonal to retrieval
  architecture; should be evaluated separately after Phase 3 to
  test whether stronger hidden states amplify the projection head's
  signal.
- **Agent-driven retrieval (ByteRover-style Tier 3/4)** — a
  different architectural philosophy that trades retrieval signal
  for LLM compute; would require its own plan.
- **RLS / reformulation reactivation** — currently quarantined.
  Could be revived in a different form (retrieval-shape query
  rewriting via RankRAG pattern, not synonym expansion) only after
  Phase 1's time-aware expansion is shown to work.

---

## Risks

- **Phase 1 doesn't move recall.** Possible if our chunker boundary
  detection is wrong (TardigradeDB ingest doesn't preserve speaker
  metadata cleanly in evidence-only mode). Test on LongMemEval
  *first* since it has explicit session/round structure; LoCoMo
  evidence-mode chunks are already per-evidence-sentence.
- **Fact-augmented keys add noise instead of signal.** Possible if
  extracted facts paraphrase the chunk too literally — the
  retrieval key ends up near-identical to the original. Diversity
  filter (cosine threshold) on `add_view_keys` already exists in
  the codebase; reuse it.
- **BM25 dominates fusion and undoes the latent signal's value.**
  Possible if latent signal is fundamentally too weak. Convex
  combination weight is the safety lever; sweep `alpha` from 0.0
  (BM25 only) to 1.0 (latent only) on a held-out split before
  shipping.
- **Projection-head training is data-hungry.** Realistic for small
  benchmarks. Mitigation: train unsupervised first (LLM2Vec recipe
  doesn't require labels); supervised contrastive only as
  refinement.
- **Phase 4 is the "we lost the bet" scenario.** If we get there,
  the docs need a clean reframing — KV-native memory + adaptive
  governance + multi-view as the kernel, trained retriever as one
  signal among several. Not a failure, but a different product
  story than "hidden states ARE the retrieval signal."

---

## Cost estimate

| Phase | Engineering time | New code | Tests | New dependencies |
|---|---|---|---|---|
| 0 | 0.5-1 day | Diagnostic harness | AT-0 | none |
| 1A | 0.5 day | RoundBoundaryStrategy | AT-1A | none |
| 1B | 1-2 days | Fact extractor + `add_view_keys` wiring | AT-1B | none (rule-based) or LLM via existing path |
| 1C | 0.5-1 day | Temporal classifier + range scoring | AT-1C | none |
| 2A | 1 day | `LexicalRetriever` over `TextStore` | AT-2A | `bm25` crate or `tantivy` |
| 2B | 0.5-1 day | `ConvexCombinationFusion` | AT-2B | none |
| 3A | 2-3 days | Training infrastructure | AT-3A | `torch` (already present), retrieval data loader |
| 3B | 2-5 days | Training on held-out, evaluation | AT-3B | held-out split |
| 4A | 1-2 days | `DenseEmbeddingRetriever` | AT-4A | `sentence-transformers` or HF transformers (already present) |

Total: Phase 0+1 in 2-4 days. Phase 2 adds 1-2 days. Phase 3 adds
4-8 days. Phase 4 adds 1-2 days. The plan stops at the first
phase that delivers the target — most realistic shipping outcome
is Phase 1 or Phase 1+2.

---

## Order of work

1. Phase 0 — instrument first.
2. Phase 1A → 1B → 1C in order (each independently shippable).
3. Re-bench under Phase 0 harness. Gate against the Phase 1 target.
4. If Phase 2 needed: 2A → 2B → re-bench.
5. If Phase 3 needed: 3A on synthetic → 3B on held-out → re-bench.
6. If Phase 4 needed: 4A → re-bench → docs reframe.

Each phase ends with a forensic update to
`docs/experiments/2026-05-14-bench-audit.md` recording the
measured uplift (or lack thereof) and any architectural learnings.
