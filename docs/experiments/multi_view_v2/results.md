# Multi-View Consolidation v2: Experiment Results

## Setup

- **Corpus:** 10 novel facts about fictional character "Sonia" (same as v1 diagnosis)
- **Model:** Qwen3-0.6B on MPS (Apple Silicon), float32, eager attention
- **Query layer:** 18/28 (67% depth)
- **Refinement:** centered (mean-centering)
- **Queries:** 5 specific + 5 moderate + 5 vague

## Architecture Under Test

Parent-document pattern (B) + LLM-generated views with diversity filter (C):
- Views stored as additional retrieval cells on canonical pack via `engine.add_view_keys()`
- `mem_read_pack` deduplicates by pack_id — canonical pack appears once regardless of how many views match
- LLM (`ViewGenerator(mode="llm")`) generates 3 question candidates per fact
- Cosine diversity filter (threshold=0.92) rejects near-duplicates
- 19 view keys attached across 10 facts (1.9 per fact avg)

## Results

| Tier | Baseline (centered) | + Multi-view v2 | Delta |
|------|-------------------|-----------------|-------|
| Specific | 100% (5/5) | 100% (5/5) | +0% |
| Moderate | 80% (4/5) | 80% (4/5) | +0% |
| Vague | 60% (3/5) | 60% (3/5) | +0% |

### Success Criteria Assessment

- Specific >= 100%: **PASS** — no regression
- Moderate >= 80%: **PASS** — no regression (v1 degraded to 20%, v2 holds at 80%)
- Vague > 60%: **FAIL** — no improvement

## Key Finding: Parent-Document Pattern Fixes Degradation

The v1 approach (views as separate packs) degraded moderate R@5 from 80% → 20%.
The v2 approach (views as retrieval cells on canonical pack) holds moderate at 80%.
The architectural fix works — views no longer compete with canonicals.

## Why No Vague Improvement

Two factors:

### 1. Qwen3-0.6B generates low-quality questions

Many generated questions are:
- **Blank/garbled:** `_____________?`, `_______.?` (appeared for facts 1, 2, 4, 5, 8)
- **Too narrow:** "What was the percentage reduction in emissions?" (fact 3 — all 3 questions asked the same thing)
- **Too generic:** "What is the name of the library?" (fact 7 — all 3 variants identical in semantics)

The 0.6B model lacks the generative capacity for diverse, specific question generation. A larger model (Qwen3-1.7B, Llama-3.2-3B) would likely produce higher-quality questions.

### 2. Diversity filter correctly rejects redundant views

The filter reduced views from 3→1 for 4 out of 10 facts (facts 3, 5, 7, 9). When all generated questions are semantically identical, only one survives — which is correct behavior but means those facts get minimal view augmentation.

## Comparison to v1

| Metric | v1 (rule-based, separate packs) | v2 (LLM, add_view_keys) |
|--------|-------------------------------|------------------------|
| Moderate R@5 | 20% (-60pp) | 80% (+0pp) |
| Vague R@5 | 60% (+0pp) | 60% (+0pp) |
| Architecture | Views compete as results | Views are retrieval-only |
| Generation | Rule-based (no model) | LLM (Qwen3-0.6B) |
| Failure mode | Index dilution | Low-quality generation |

## Next Steps

1. **Try a larger model for view generation** — Qwen3-1.7B or Llama-3.2-3B should produce higher-quality, more diverse questions. The architecture is model-agnostic; only the generation step changes.
2. **Measure with the 100-cell corpus** — The 10-fact corpus has thin margins. The 100-cell corpus (from `experiments/corpus_100.py`) would give more statistical power.
3. **Consider external question generation** — Use a dedicated question-generation model (e.g., `mrm8488/t5-base-finetuned-question-generation-ap`) instead of the same model used for KV capture.

## Honest Assessment

The parent-document architecture (B) is validated — it eliminates the v1 degradation. The LLM generation quality (C) is the bottleneck. The 0.6B model is too small for reliable question generation. The infrastructure is ready; the improvement will come from a better generator, not a better architecture.
