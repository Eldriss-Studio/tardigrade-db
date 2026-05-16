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

## Key Finding: Parent-Document Pattern Prevents Catastrophic Degradation

The v1 approach (views as separate packs) **destroyed** moderate R@5: 80% → 20%.
Four out of five queries that previously worked started returning wrong memories.
The v2 approach (views as retrieval cells on canonical pack) holds moderate at 80%.

However: holding at 80% means **zero net improvement**. The architecture prevents
the catastrophe but doesn't advance vague-query recall. We are back where we
started. The multi-view effort has not yet delivered measurable retrieval value.

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

## Follow-Up: Prompt Variations and Larger Model (Same Session)

Tested 3 prompt strategies with Qwen3-0.6B (original, vague-memory, structured-list)
and one paraphrase strategy. All failed: the model produces blank underscores,
repetitive narrow questions ("What was the percentage..."), or light rewording.
The 0.6B model lacks generative capacity for any text rewriting task.

Qwen2.5-3B (cached locally) was tested but takes >5 minutes per question on MPS
float32 — too slow for iterative experimentation.

### Root Cause: Wrong Model for the Job

The KV capture model (Qwen3-0.6B) runs forward passes to produce hidden states.
It never needs to *generate* text. Multi-view consolidation needs a model that
can produce diverse, specific questions — a fundamentally harder task that requires
a larger model or a specialized question-generation model.

The capture model and the generation model should be decoupled. The architecture
already supports this (ViewGenerator takes its own model/tokenizer).

## Next Steps

1. **Decouple capture and generation models** — Use Qwen3-0.6B for KV capture,
   a dedicated question-generation model (e.g., `mrm8488/t5-base-finetuned-question-generation-ap`,
   ~220M params, fast on CPU) for view text generation.
2. **Try Qwen2.5-3B in float16** — Halving precision would cut generation time
   and memory. MPS supports float16.
3. **Measure with the 100-cell corpus** — The 10-fact corpus has thin margins.
4. **Run LongMemEval/LoCoMo baseline** — The research memo recommended this
   before shipping features. We still don't know where TardigradeDB lands
   relative to the field.

> **UPDATE 2026-05-14:** Honest LoCoMo/LongMemEval baselines have since been
> measured on the clean dataset: ~36% R@1 on a 50-item clean LoCoMo subset
> (native engine, no RLS). Earlier "68.2% / 90.9%" headlines were measured
> against the lexical fallback adapter on a corrupted dataset and were
> retracted. Full-corpus clean-data re-runs are still pending. See
> [`docs/experiments/2026-05-14-bench-audit.md`](../2026-05-14-bench-audit.md).

## Honest Assessment

The parent-document architecture (B) is validated — it prevents the v1 catastrophe.
But the net result is zero improvement on vague recall. We spent significant effort
(Rust engine changes, PyO3 bindings, consolidator refactor, LLM view generation,
diversity filter) and the retrieval numbers are identical to before any of this work.

The bottleneck is now the generator: Qwen3-0.6B is too small for question generation.
The architecture is model-agnostic, so testing with a larger model (Qwen2.5-3B,
Qwen3-1.7B) requires no code changes. But until a better generator produces
measurable R@5 improvement, this feature has no proven value.
