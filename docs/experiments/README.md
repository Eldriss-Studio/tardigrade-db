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

### [KV Cache Validation — Hidden States vs Real KV](kv-cache-validation.md)

**Date:** April 22-23, 2026
**Status:** Complete

Systematic comparison of what to store: raw hidden states vs actual K projections from the KV cache. Tested on Sonia (16 diverse life memories) with GPT-2 and Qwen3-0.6B.

**Key findings:**
- **Storing hidden states produces gravity wells** — one memory dominates all queries regardless of content (31.2% recall)
- **Storing real KV cache (K projections) doubled recall** — 62.5% mean-pool, 75.0% per-token on a 0.6B model
- **Per-token KV breaks the gravity well** — 7 unique memories in top-1 across 16 queries (vs 1 with hidden states)
- **Domain diversity helps** — memories across different life domains (cooking, legal, medical, social) separate naturally in K-projection space
- **The 4 misses require world knowledge** (Coco = Day of the Dead), not better retrieval — model size problem, not architecture problem
- Previous experiments that showed poor results were storing the wrong data

**Scripts:** `experiments/sonia_real_kv_cache.py` (real KV), `experiments/sonia_production_sim.py` (hidden states comparison), `experiments/maya_kv_tensors_comparison.py` (GPT-2 vs Qwen3)

## Planned Experiments

| Experiment | Goal | Status |
|-----------|------|--------|
| Larger model test (Llama 3.2:3b) | Validate that richer representations improve recall + injection quality | Planned — `examples/llama_memory_test.py` prepared |
| RoPE injection | Test KV injection with rotary position encoding (requires unrotate/re-rotate) | Planned — blocks Llama/Qwen support |
| Multi-session memory | Cross-day retrieval ("what happened last week") | Planned |
| Governance decay | Verify unused memories demote over simulated time | Validated via `test_sweep.py` |
| Adversarial retrieval | Contradictory memories, test which surfaces | Planned |
| Confidence thresholding | Calibrate "I don't remember" cutoff | Planned |
| [Cross-model retrieval](cross-model-memory-test.md) | Sonnet stores, Opus retrieves — test memory portability across models | Designed — **next up** |
| GQA head mismatch | Test injection with grouped query attention (Llama 2+) | Planned |
| Q4 injection at scale | Verify Q4 quality holds across hundreds of memories | Planned |

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
