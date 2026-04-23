# Experiments

Empirical tests validating TardigradeDB's core thesis: that persistent memory in latent space — using raw KV cache tensors, not text or embeddings — enables a fundamentally different kind of LLM memory.

## Completed Experiments

### [Two-Agent Memory Cycle](two-agent-memory-test.md)

**Date:** April 22, 2026  
**Status:** Complete

Two independent Claude Sonnet agents — one generating experiential memories, one retrieving them blind — communicate exclusively through TardigradeDB. Tests both a word-hash baseline and real GPT-2 KV cache tensors.

**Key findings:**
- 92% recall with word-hash vectors (weakest possible baseline)
- 80% recall with GPT-2 KV cache tensors, but with 100x stronger signal-to-noise separation
- ~1,600-point score gap between genuine memories and noise makes confidence thresholding practical
- Missed memories are model-quality limitations (GPT-2, 117M params), not architectural ones
- One missed memory (a peripheral observation about a coworker's keyboard) accidentally modeled realistic human memory decay

**Scripts:** `examples/sonnet_memory_test.py` (word-hash), `examples/kv_memory_test.py` (KV cache)

### [KV Injection Critique & Validation](kv-injection-critique.md)

**Date:** April 22, 2026  
**Status:** Complete — validated empirically

External critique questioning whether cross-context KV injection works. Three documents cover this:
- **[Critique](kv-injection-critique.md)** — The original peer review concerns and analysis
- **[Test design](kv-injection-validation-test.md)** — 6-condition experiment design with experiential memories
- **[Results](kv-injection-results.md)** — Measured outcomes across 5 prompt pairs (30 measurements)

Key findings from [validation results](kv-injection-results.md):
- **Full per-token KV injection works** — 26x to 829x improvement over baseline, matching or exceeding Text RAG
- **Mean-pooled injection is broken** — mathematical category error (hidden states ≠ K/V projection space, causes collapse to "?" token at 78-90%)
- **Q4 quantization preserves 89% of injection quality** — TardigradeDB's storage approach is viable
- **Irrelevant KV injection ≈ baseline** — confirms improvement is genuine semantic transfer, not noise
- **Selective token injection fails** — salience heuristic (hidden state norm) doesn't identify the tokens that matter for recall

**Automated test coverage:** 5 ATDD tests in `tests/python/test_kv_injector.py` validate the injection pipeline mechanism (reshape, cache extend, GPT-2 integration, multi-cell, round-trip). 5 ATDD tests in `tests/python/test_sweep.py` validate background governance sweep (Active Object pattern).

**Architectural implication:** Mean-pooled vectors work for **retrieval** (search index). Full per-token KV is necessary for **injection** (attention augmentation). The emerging architecture uses mean-pooled as the index key and full KV as the stored value.

### [Sonia Parallel Subagent Validation](sonia-subagent-parallel-test.md)

**Date:** April 23, 2026  
**Status:** Complete

Two parallel Codex subagents (different agent models) executed separate Sonia experiment scripts and reported recall/SNR outcomes independently.

**Key findings:**
- `sonia_real_kv_cache.py`: per-token real-KV beat mean-pool on recall in this run (`75.0%` vs `62.5%`, +12.5 points)
- `sonia_production_sim.py`: recall was equal between modes (`31.2%` each) but per-token showed much larger SNR separation
- Both runs completed successfully with no operational blockers

### [KV Cache Validation — Full Progression](kv-cache-validation.md)

**Date:** April 22-23, 2026
**Status:** Complete — three major discoveries

Systematic exploration of what to store and how to retrieve, tested on Sonia (16 diverse life memories) with GPT-2, Qwen3-0.6B, and Qwen2.5-3B.

**Three discoveries:**
1. **Store K projections, not hidden states** — hidden states produce gravity wells (31.2%), K projections doubled recall (62.5-75%)
2. **K*K per-token matching fails** — K vectors share a massive common component across all sequences (position-0 cross-sentence dot = 6281 for unrelated text). Per-token K*K got 25%, worse than mean-pool
3. **Query with Q, store K (Q*K)** — matches how attention actually works. The fixed Q*K per-token pipeline gets 68.8% recall with 8 unique top-1 memories, and now exercises encoded Q tokens against encoded K tokens through max-sim scoring

**Full progression:** 31.2% (hidden) → 25% (K*K per-token) → 62.5% (K*K mean-pool) → 75% (K*K per-token manual) → 68.8% (Q*K per-token pipeline, 16 memories)

**100-memory scale test:** Q*K recall dropped to **40%** at 100 memories. Gravity well returned (one memory dominated 7/30 queries). The gap between raw Q*K dot product and proper attention (with softmax normalization) is the current bottleneck. Retrieval quality degrades at scale — this is the main open problem.

**Traditional RAG baseline:** `intfloat/e5-small-v2` embedding RAG achieved **100% recall@1/@3/@5/@10** on the same 100-memory corpus. This is a retrieval-only baseline, not an architecture change, but it shows current Q*K retrieval is not competitive with standard embedding retrieval on this test.

**Scripts:** `experiments/scale_100_qk.py` (100-memory scale test), `experiments/scale_100_qk_diagnostics.py` (diagnostic scorer lab), `experiments/scale_100_rag_baseline.py` (traditional RAG baseline), `experiments/sonia_per_token_pipeline.py` (Q*K pipeline), `experiments/sonia_real_kv_cache.py` (K*K real KV)

## Planned Experiments

| Experiment | Goal | Status |
|-----------|------|--------|
| **100-memory scale test** | Does Q*K retrieval hold at realistic memory counts? | **Complete — 40% recall, needs scoring improvements** |
| **Traditional RAG baseline** | Compare current Q*K retrieval against standard embedding retrieval | **Complete — RAG got 100% recall on this corpus** |
| **Softmax-normalized scoring** | Replace raw dot product with softmax(Q*K^T/sqrt(d_k)) in retriever | **Next up — main open problem** |
| GQA K-expansion | Expand K heads to match Q dims for Q*K retrieval | Complete |
| Per-head scoring | Score per attention head instead of concatenating all heads | Planned |
| Multi-session memory | Cross-day retrieval ("what happened last week") | Planned |
| Adversarial retrieval | Contradictory memories, test which surfaces | Planned |
| Confidence thresholding | Calibrate "I don't remember" cutoff using SNR | Planned |
| [Cross-model retrieval](cross-model-memory-test.md) | Store with one model, retrieve with another | Designed |
| RoPE injection | Test KV injection with rotary position encoding | Planned |

## Running Experiments

### Prerequisites

```bash
cd ~/Dev/tardigrade-db
source .venv/bin/activate
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop -m crates/tdb-python/Cargo.toml

# For KV cache tests
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

### Word-hash test
```bash
python examples/sonnet_memory_test.py store "memory 1" "memory 2"
python examples/sonnet_memory_test.py query "related question"
python examples/sonnet_memory_test.py info
```

### KV cache tensor test
```bash
python examples/kv_memory_test.py store "memory 1" "memory 2"
python examples/kv_memory_test.py query "related question"
python examples/kv_memory_test.py info
```
