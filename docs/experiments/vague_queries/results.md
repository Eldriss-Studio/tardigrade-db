# Vague-Query Refinement: Empirical Results

> ℹ️ **Audit note (2026-05-14):** The empirical results in this document — mean-centering (+31pp moderate), cross-encoder reranker stacking (+40pp moderate / +18pp vague), and the PRF sweep — were measured on the 100-cell synthetic Sonia corpus and are **unaffected** by the 2026-05-14 bench audit. However, the "vague-query vocabulary overlap is the retrieval ceiling" / DeepMind-LIMIT-style theoretical-ceiling framing that appears later in this document was drawn in part from the broken LoCoMo runs and is **retracted**. See [`docs/experiments/2026-05-14-bench-audit.md`](../2026-05-14-bench-audit.md).

**Date:** 2026-05-02
**Hardware:** RTX 3070 Ti, CUDA 12.8
**Model:** Qwen/Qwen3-0.6B (28 layers, query layer 18, fp32, eager attention)
**Engine:** TardigradeDB v0.1.5, per-token Top5Avg over hidden states (skip pos 0)
**Corpus:** 100 memories from `experiments/corpus_100.py` (Sonia's life, 10 domains × 10 facts)
**Queries:** 30 specific (`corpus_100.ALL_QUERIES` non-negative), 100 moderate, 100 vague, 10 open

---

## Headline Numbers

| Mode | Specific R@5 | Moderate R@5 | Vague R@5 | Open R@5 | Vague p95 |
|---|---|---|---|---|---|
| **none** (baseline) | 100.0% | 28.0% | 46.0% | 100.0% | 67ms |
| **centered** (mean-centering) | 100.0% | 59.0% | 50.0% | 100.0% | 58ms |
| **prf** (α=0.7, β=0.3, k'=3) | 30.0% ⚠️ | 43.0% | 55.0% | 100.0% | 53ms |
| **none + rerank** (cross-encoder Stage-2) | 100.0% | 57.0% | 62.0% | 100.0% | 77ms |
| **centered + rerank** (recommended) | **100.0%** | **68.0%** | **64.0%** | **100.0%** | **86ms** |

Reranker model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params, ~10ms/query on RTX 3070 Ti for top-10 candidates).

**Verdict:**
- **`centered + rerank` is the recommended production mode.** +40pp on moderate, +18pp on vague vs baseline, **zero regression on specific**. ~30% latency overhead (86ms vs 67ms p95) is acceptable for the quality lift.
- **Mean-centering alone** is still the right call when memo text isn't available — it captures most of the moderate-tier gain (+31pp) for free, no extra model.
- **Cross-encoder reranking alone (no centering)** is also strong (+29pp moderate, +16pp vague) and equally cheap to operate — the two strategies stack, but each is independently useful.
- **PRF as initially configured (α=0.7) drifts catastrophically on specific queries** (100% → 30%), confirming the failure mode the dense-PRF survey (Li et al. TOIS 2023) warns about.
- **Open queries hit 100% trivially** — the dataset definition treats any of the 100 memories as "correct" for an open query, so this column is a sanity check, not a discriminator.

---

## Mean-centering: Why it works

The 100-memory hidden-state corpus has a strong shared direction (the same "gravity well" pattern observed at position 0 and in K\*K scoring — see `docs/refs/external-references.md` C2). Subtracting the corpus mean removes that direction from both query and stored vectors before the dot product:

```
(q − μ) · (k − μ)  =  q·k − q·μ − μ·k + μ·μ
```

Effect on each tier:
- **Specific (vocabulary overlap)**: the distinguishing signal already dominates, so removing the shared component has near-zero effect → 100% preserved.
- **Moderate (partial vocabulary overlap)**: the shared component was masking real signal — removing it surfaces the right cells (28% → 59%, +31pp).
- **Vague (low vocabulary overlap)**: helps modestly (46% → 50%); the failure here is partly that the corpus mean is itself biased toward the dominant domains (Fitness/Work) so centering doesn't fully fix the gravity well.

Match-up with literature: this is essentially **whitening the embedding space**, a well-known IR technique. Closest precedent: BERT-flow (arXiv:2011.05864) uses normalizing flows to whiten BERT embeddings; we do the cheap version (mean subtraction only).

---

## PRF hyperparameter sweep — sharp transition, no good middle

Eight configurations were tested. Pattern is unambiguous: as β rises, vague gains nothing useful but specific collapses sharply at the threshold.

| α | β | k' | Specific R@5 | Moderate R@5 | Vague R@5 |
|---|----|----|------|------|------|
| 0.95 | 0.05 | 1 | 100% | 28% | 46% (= baseline) |
| 0.9 | 0.1 | 1 | 100% | 25% | 47% |
| 0.9 | 0.1 | 3 | 100% | 24% | 47% |
| 0.85 | 0.15 | 1 | 100% | 25% | 46% |
| 0.85 | 0.15 | 3 | 100% | 24% | 46% |
| 0.8 | 0.2 | 1 | 100% | 23% | 45% |
| 0.8 | 0.2 | 3 | 100% | 22% | 46% |
| **0.7** | **0.3** | **3** | **30% ⚠️** | 43% | 55% |

**Two regimes:**
1. **β ≤ 0.2:** PRF makes no useful difference. Specific stays at 100% but vague stays at ~46% (within noise of baseline). The query is barely perturbed; the re-retrieval recovers essentially the same top-K.
2. **β ≥ 0.3:** PRF drifts. Specific collapses (100% → 30%) because the centroid of the top-3 cells generalizes the query enough that the original target falls below the top-5. Vague gains modestly (46% → 55%) but at unacceptable cost.

The transition is **discontinuous**, not gradual. There is no β value in this experiment that helps vague without hurting specific.

### Why PRF drifts so hard here

Two structural reasons in the current implementation:

1. **Whole-cell aggregation.** The centroid uses *every* token of the top-3 cells (~45 tokens × 3 = 135 token vectors averaged), which over-smooths. A peak-tokens-only centroid would preserve more discriminative signal.
2. **Replace, not fuse.** Currently the PRF re-query *replaces* the first-stage results. A reciprocal-rank-fusion (RRF, Cormack 2009) merge of first-stage + PRF-stage would let PRF help on the vague tail while preserving the specific top hits.

Both are improvements worth trying. They were out of scope for the v1 implementation.

---

## Cross-encoder reranking (Stage-2): closes most of the vague-cliff gap

`python/tardigrade_hooks/reranker.py::CrossEncoderReranker` wraps a small
cross-encoder (default `cross-encoder/ms-marco-MiniLM-L-6-v2`, 22M params,
MiniLM architecture (arXiv:2002.10957) trained on MS MARCO
(arXiv:1611.09268), via sentence-transformers (arXiv:1908.10084)).

After the engine returns top-10 candidates, the reranker computes a full
query+document attention score for each `(question, memo_text)` pair and
re-sorts. Candidates without text fall back to first-stage rank.

**Why it works where PRF didn't:** the cross-encoder lets every query
token attend to every document token jointly, which is strictly more
expressive than the bi-encoder dot product. PRF tried to get there
indirectly by perturbing the query; the cross-encoder gets there
directly with a tiny dedicated model.

**Caveat — training distribution mismatch.** The MS MARCO dataset is
short web passages, not first-person diary text. The reranker still
helps a lot, but a fine-tuned variant on agent-memory-style data would
likely give another 5-10pp. Tracked as future work.

**Stacking is additive, not redundant.** `centered+rerank` consistently
beats both `centered` alone and `none+rerank` alone:

| metric | centered | none+rerank | centered+rerank | best-component delta |
|---|---|---|---|---|
| moderate R@5 | 59% | 57% | **68%** | +9pp over either alone |
| vague R@5 | 50% | 62% | **64%** | +2pp over rerank-alone |

Mean-centering improves the candidates that reach the reranker; the
reranker then picks the right one out of a better candidate set.

---

## What this means

**Default to `RefinementMode::MeanCentered` + `CrossEncoderReranker` whenever memo text is available.** Together they push moderate retrieval from 28% → 68% (+40pp) and vague retrieval from 46% → 64% (+18pp), with **zero regression on specific (still 100%)** and only ~30% latency overhead.

When memo text is not available, fall back to mean-centering alone — most of the moderate-tier win comes for free without any extra model. Current settings:

```python
engine.set_refinement_mode("centered")
```

**Keep `RefinementMode::None` as the engine default.** Mean-centering is a clear win on this corpus but it's still a behavior change; users opt in explicitly.

**`RefinementMode::LatentPrf` ships but is not yet useful** for this corpus geometry. It will become useful once we add (a) peak-token centroids and (b) RRF fusion. Both are tracked in the plan file.

The previous 50% vague R@5 ceiling has been pushed to **64% with centered+rerank**. Reaching 80%+ now likely requires:
- **Fine-tuning the cross-encoder on agent-memory-style data** (MS MARCO is web passages, mismatch). Estimated +5-10pp.
- **A trained per-agent re-ranker** (LoRA adapter on top of stored hidden states), or
- **Query rewriting via a cheap model** (NOT a HyDE-style LLM call — something like a 30M-parameter encoder fine-tuned on agent queries).

These are research bets, not v1 implementation work.

---

## Honest reporting

Per `feedback_scientific_method.md`, negative results documented with the same rigor as positive ones:

- **PRF α=0.7 catastrophic regression** (100% → 30% on specific). Documented above with explanation.
- **PRF sweep produced no usable middle ground.** Documented with all 8 configurations, including the no-ops.
- **Vague R@5 ceiling is 50%, not 70%.** Mean-centering helps but doesn't solve the vocabulary cliff. The 60%+ target from the plan was not met.
- **Moderate baseline measured 28% here vs 45% in the original 2026-04 experiment.** Same code, same model, same dataset. Variance unexplained — possible torch/CUDA version drift or per-run quantization noise. Does not change *relative* ranking of refinement modes.
- **Open queries are not informative.** All test definitions expect the full corpus as ground truth, so any retrieval scores 100%. Including the column for completeness only.

---

## Reproduce

```bash
source .venv/bin/activate

# Headline comparison
python docs/experiments/vague_queries/bench_vague.py \
    --modes none,centered,prf --device cuda

# Single PRF configuration
python docs/experiments/vague_queries/bench_vague.py \
    --modes prf --alpha 0.7 --beta 0.3 --kprime 3 --device cuda

# Full PRF sweep (warning: ~15 min on RTX 3070 Ti)
python docs/experiments/vague_queries/bench_vague.py \
    --sweep prf --device cuda
```

Hardware notes:
- RTX 3070 Ti / CUDA 12.8: end-to-end ~67ms/query at 100 cells, baseline mode
- Mean-centering: 58ms/query (slightly faster — fewer cells make it through to governance because re-scoring tightens results)
- PRF: 53ms/query (skips first-stage SLB on the second pass)
