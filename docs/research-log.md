# Research Log

A chronological narrative of the experiments that shaped TardigradeDB's retrieval pipeline. The current product behaviour is documented in [`docs/architecture.md`](architecture.md) and [`docs/positioning.md`](positioning.md); this document is the *why we got here*.

For the dated, primary-source experiment writeups, see [`docs/experiments/`](experiments/). This log is the index + narrative.

---

## Retrieval: 100 % recall at 100 memories (April 23, 2026)

A series of experiments on Qwen3-0.6B tested different retrieval approaches. The progression from 31 % to 100 % recall revealed what works and what doesn't for latent-space retrieval:

| Method | Recall@5 (100 memories) | Notes |
|--------|-------------------------|-------|
| Hidden states mean-pool | 31 % | Gravity well — one memory dominates |
| K projections mean-pool | 63 % | Better, but signal lost by averaging |
| Q × K per-token max-sim | 40 % | K vectors share common component |
| Traditional RAG (e5-small-v2) | 100 % | Embedding baseline |
| **Hidden states + Top5Avg (engine pipeline)** | **100 %** | **Through Q4, full pipeline** |

What worked: storing **raw hidden states** per token (not K or Q projections) and scoring by **Top5Avg** — the mean of the top 5 highest dot products per cell. Hidden states contain all the information Q and K derive from, without the artifacts that make cross-sequence Q × K fail. Mean-pooling was the failure mode, not hidden states.

The retrieval pipeline chains: **SLB (mean-pooled hot cache) → PerTokenRetriever (Top5Avg) → BruteForceRetriever (fallback)**.

All 30 queries found at rank #1. No gravity well. 97 ms average latency. Q4 quantization preserved the signal. Vague queries (*"What have I been cooking?"*): 87 % latent vs 100 % RAG.

---

## KV Injection: byte-identical to text RAG (April 24, 2026)

Following the Knowledge Packs paper (arXiv 2604.03270), KV injection through the full TardigradeDB pipeline (Q4 quantized, persisted to disk, read back, reconstructed) produces **byte-identical output** to having the text in the prompt:

| Path | Correct (10 novel facts) | Prompt Tokens |
|------|--------------------------|---------------|
| Text RAG | 8 / 10 | 438 total (43 avg) |
| **KV Injection (through engine)** | **8 / 10** | **235 total (23 avg)** |

Same 2 misses on both paths. Identical responses character-for-character. **46 % fewer prompt tokens** with injection.

Storage trade-off: 730 KB per memory (Q4 quantized KV cache) vs 65 bytes per memory (text). KV injection trades disk space for context-window space — relevant when context windows are scarce or memories are numerous.

Pipeline fidelity verified stage-by-stage: Q4 round-trip cosine similarity = 0.999 on KV tensors.

---

## KV Pack API: atomic multi-layer storage (April 24, 2026)

The Rust engine provides first-class `mem_write_pack` and `mem_read_pack` APIs. A KV Pack stores a complete multi-layer KV cache (e.g. 28 layers for Qwen3-0.6B) as a single atomic unit — one fsync, grouped retrieval, pack-level governance.

The Python `KnowledgePackStore` wraps this API for end-to-end injection:

1. Wrap fact in chat template → compute KV cache → `engine.mem_write_pack()` (single fsync)
2. Compute query hidden states → `engine.mem_read_pack()` → reconstruct `DynamicCache`
3. Clone cache → inject into `model.generate()` → byte-identical output

This is the canonical path for using TardigradeDB with HuggingFace models. See [`docs/guide/knowledge-pack-store.md`](guide/knowledge-pack-store.md).

---

## Synthetic KV injection — unambiguous proof (April 27, 2026)

10 fully synthetic gibberish facts — nonsense proper nouns (*"Zyphlox-9"*, *"9-Quornth-44"*, *"Yombliquid-X"*), fake units (*"zennits"*, *"drazeks"*, *"plonks"*), invented entities (*"Vrenthar"*, *"Gorflax-12"*) — that cannot exist in any training corpus. `KnowledgePackStore` injected stored KV via `model.generate(past_key_values=...)` on Qwen3-0.6B. **Result: 9 / 10**, matching text RAG exactly (100 % recall ratio), with 236 prompt tokens saved.

Any correct recall is unambiguous proof — these gibberish strings can only come from the injected KV tensors. Full writeup: [`docs/experiments/synthetic-kv-injection.md`](experiments/synthetic-kv-injection.md).

---

## Multi-Memory: trace-linked retrieval (April 25, 2026)

For queries requiring information from multiple memories (*"What car does Lucia's swimming instructor drive?"* — needs to know who the instructor is AND what car they drive), TardigradeDB uses **trace-boosted retrieval**: link related facts at storage time, follow links at retrieval time.

| Approach | Accuracy (140 memories, 20 queries) |
|----------|--------------------------------------|
| No trace links | ~30 % |
| Trace-linked (`store_linked`) | 55 % |
| Trace-boosted (link-density scoring) | **70 %** |
| Text RAG baseline | 95 % |

The agent controls linking via `store_linked()` (batch) or `store_and_link()` (incremental). The engine records links and boosts retrieval scores for connected memories. No auto-linking — [the experiment](experiments/multi-memory-injection.md) showed that latent similarity can't distinguish "same event" from "same topic" without entity extraction.

---

## LoCoMo + LongMemEval — RETRACTED (2026-05-14)

> ⚠️ The previously published LoCoMo (68.2 %) and LongMemEval (90.9 %) headline numbers were **retracted** on 2026-05-14. A bench audit found those runs used the lexical fallback adapter on a corpus corrupted by a dataset-prep bug — they did not measure the native KV engine on the intended dataset.

**Honest native-engine numbers on clean data:**

| Benchmark | Honest measurement | Notes |
|-----------|--------------------|-------|
| LoCoMo Phase-1 (50-item subset, clean) | **~36 % R@1** | Native engine, no RLS |
| LoCoMo Phase-1 (full corpus, clean) | not yet re-measured | Pending |
| LongMemEval (full corpus, clean) | not yet re-measured | Pending |

All four RLS modes (keyword / multiphrasing / embedding / generative / agent) **underperform** the no-RLS baseline on clean data; the DeepSeek agent reformulator loses 12.7pp. The earlier *"vocabulary-gap is the retrieval ceiling"* framing is also retracted — it was drawn from the broken data.

Full record: [`docs/experiments/2026-05-14-bench-audit.md`](experiments/2026-05-14-bench-audit.md).

### Why this happened, and what changed

Six hours of LoCoMo-tuning after the audit (2026-05-16) produced four engineering-correct fixes (chunker boundary, reranker chunk-text, hardcoded k=5, Decorator widening, session-timestamp prep) and **zero LoCoMo score movement past ~7 %**. The conclusion: LoCoMo is not the right product target. We pivoted to:

- **Latency / footprint / KV-native positioning** where TardigradeDB legitimately wins — sub-millisecond p99 retrieval at 5K cells, 751 B per cell on disk. See [`docs/positioning/latency_first.md`](positioning/latency_first.md).
- **LoCoMo Judge as architecture work** (Qwen3-1.7B + full-conversation context + justify-then-judge evaluator), not hot-patching.

---

## Vague-query refinement (synthetic corpus — unaffected by the audit)

Measured on RTX 3070 Ti / Qwen3-0.6B (100-cell Sonia corpus, 230 queries):

| Refinement | Specific R@5 | Moderate R@5 | Vague R@5 |
|------------|--------------|--------------|-----------|
| `none` | 100 % | 28 % | 46 % |
| `centered` (mean-centering) | 100 % | 59 % (+31pp) | 50 % (+4pp) |
| `centered` + cross-encoder rerank | 100 % | **68 % (+40pp)** | **64 % (+18pp)** |

Cross-encoder default model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22 M params). Stage-2 reranker adds ~30 % latency (~86 ms vs ~67 ms p95).

API: `engine.set_refinement_mode("none" | "centered" | "prf", ...)`. Full results: [`docs/experiments/vague_queries/results.md`](experiments/vague_queries/results.md).

---

## Hybrid end-to-end recall — RecurrentGemma-2B-it (v0.3.4)

Full calibrate → store → retrieve → inject → generate pipeline on RecurrentGemma-2B-it: calibration sweep picks `k_vector @ layer 5` (the first attention layer in Griffin's 3:1 layout) at **19/20 top-1 / 20/20 top-5**. `KnowledgePackStore` wired via `CalibrationRegistry` lifts engine retrieval to **20/20 (100 %)** and generation-hit (answer substring in greedy output) to **18/20 (90 %)** on the bundled 20-fact paraphrased corpus.

Calibration persists to `~/.tardigrade/calibration.json`; second runs skip the sweep. See [`docs/guide/calibration.md`](guide/calibration.md).

---

## See also

- [`docs/experiments/`](experiments/) — every dated experiment writeup in full.
- [`docs/positioning/latency_first.md`](positioning/latency_first.md) — the authoritative positioning numbers post-audit.
- [`docs/experiments/2026-05-14-bench-audit.md`](experiments/2026-05-14-bench-audit.md) — the audit itself.
