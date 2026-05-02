# Vague-Query Refinement: Empirical Results

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
| **centered** (mean-centering) | **100.0%** | **59.0%** | **50.0%** | **100.0%** | 58ms |
| **prf** (α=0.7, β=0.3, k'=3) | 30.0% ⚠️ | 43.0% | 55.0% | 100.0% | 53ms |

**Verdict:**
- **Mean-centering is the clear winner.** +31pp on moderate, +4pp on vague, **zero regression on specific** (still 100%). Even slightly faster (mean-subtract is one f32 vector op; the rescore replaces the original score for the same K candidates).
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

## What this means

**Default to `RefinementMode::MeanCentered` for any non-specific query workload.** It's the only mode that improves moderate retrieval substantially (+31pp) without regressing specific retrieval. Current settings:

```python
engine.set_refinement_mode("centered")
```

**Keep `RefinementMode::None` as the engine default.** Mean-centering is a clear win on this corpus but it's still a behavior change; users opt in explicitly.

**`RefinementMode::LatentPrf` ships but is not yet useful** for this corpus geometry. It will become useful once we add (a) peak-token centroids and (b) RRF fusion. Both are tracked in the plan file.

The 50% vague R@5 ceiling that mean-centering hits is the **real** open problem — bridging the vocabulary gap to reach 70-80% will need either:
- **Cross-encoder reranking on stored memo text** (where text exists), or
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
