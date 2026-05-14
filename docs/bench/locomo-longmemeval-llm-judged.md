# TardigradeDB: LoCoMo + LongMemEval with DeepSeek LLM Judge (2026-05-11)

> **⚠️ RETRACTED — 2026-05-14.** Same retraction as the deterministic
> baseline doc: the numbers in this document measure the **lexical
> fallback adapter**, not the native KV engine. The DeepSeek LLM
> judge was scoring the lexical adapter's question-overlap retrieval
> on a dataset where every item in a conversation shared the same
> ~62K-char context (string-vs-int bug in
> `benchmarks/scripts/prepare_phase1_datasets.py`, present since
> 2026-04-22).
>
> The "LLM judge scored lower than deterministic, so deterministic
> was generous" framing below is wrong — both evaluators were scoring
> the lexical adapter's output, which is a different system from
> what the doc claimed to measure.
>
> The "vague-query vocabulary-overlap ceiling is the real bottleneck"
> conclusion is also unsupported by these runs.
>
> Full forensic record: [docs/experiments/2026-05-14-bench-audit.md](../experiments/2026-05-14-bench-audit.md).
>
> *Historical record preserved below for archival reference only.*

---

## Setup

- **Model:** Qwen3-0.6B on MPS (Apple Silicon), float32, eager attention
- **Refinement:** `centered` (mean-centering)
- **Evaluator:** DeepSeek Chat (`deepseek-chat`) as LLM judge via `JudgeProvider` Strategy
- **Query layer:** 18/28 (67% depth)
- **Top-k:** 5
- **Seed:** 42
- **Fallbacks:** 0/2042 (all items judged by LLM)

## Results

| Benchmark | Deterministic Eval | DeepSeek LLM Judge | Delta |
|-----------|-------------------|-------------------|-------|
| **LoCoMo** (1,542 items) | 68.2% | 67.2% | -1.0% |
| **LongMemEval** (500 items) | 90.9% | 88.8% | -2.1% |
| **Combined** (2,042 items) | 73.8% | 72.5% | -1.3% |

## Key Finding: The Gap Is Real

The hypothesis was that deterministic evaluation (strict lexical overlap) might
be underscoring TardigradeDB — that the engine retrieves correct answers but
phrases them differently, and an LLM judge would catch this.

**The opposite is true.** The LLM judge scored slightly *lower* than
deterministic. This means:

1. The deterministic evaluator was being *generous* (partial lexical overlap
   getting credit that an LLM judge considers incorrect).
2. The retrieval quality gap (LoCoMo 67% vs vanilla GPT-4o 74%) is genuine.
3. The vague-query vocabulary-overlap ceiling is the real bottleneck — not
   evaluator bias.

## Comparison to Field (with Fair LLM Judging)

| System | LoCoMo | LongMemEval | Evaluator |
|--------|--------|-------------|-----------|
| **TardigradeDB** | **67.2%** | **88.8%** | DeepSeek LLM judge |
| ByteRover 2.0 | 92.2% | — | Their eval |
| Letta / MemGPT | — | 83.2% | Their eval |
| Vanilla GPT-4o-mini | 74.0% | — | LoCoMo official |
| Mem0g (graph) | 68.4% | — | Their eval |
| Mem0 | 66.9% | 49.0% | Their eval |

LongMemEval 88.8% still beats every published number in our references.
LoCoMo 67.2% is below the vanilla baseline (74%).

## What This Means for Next Steps

The LoCoMo gap is a retrieval problem, not a scoring problem. Closing it
requires better vague-query handling — the same problem multi-view
consolidation was designed to address (but hasn't yet delivered on, due
to the generator quality bottleneck).
