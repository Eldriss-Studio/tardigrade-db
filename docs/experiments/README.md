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

### vLLM KV Connector — End-to-End Round-Trip

**Date:** April 26, 2026
**Status:** Complete — 5/5 GPU integration tests passing

First validation that TardigradeDB plugs into a production LLM serving framework. The `tardigrade_vllm.connector.TardigradeConnector` implements vLLM's KV Connector v1 API, captures KV during generation, persists packs to TardigradeDB, and runs the scheduler-side semantic match for incoming requests. Tested in WSL2 + Ubuntu 24.04 with an RTX 3070 Ti.

**Setup:** vLLM 0.19.1, PyTorch 2.10 (CUDA 12.8), Qwen3-0.6B (28 layers, 8 KV heads, head_dim=128, kv_dim=1024) loaded in bf16 with `enforce_eager=True` and `max_model_len=512`.

**Findings:**
- **Save path works on real generation.** A 20-token completion writes 20 packs (one per forward pass), each containing all 28 layers. Mean-pooled K of layer 0 is the retrieval key.
- **Semantic matching runs.** Scheduler-side `get_num_new_matched_tokens` computes a retrieval key from prompt token IDs via the model's embedding table (no GPU forward), queries `mem_read_pack_with_trace_boost`, and reports matched tokens. No crashes when `start_load_kv` runs.
- **Round-trip generation stays coherent.** Multi-prompt sessions accumulate packs monotonically and the model still produces valid text.
- **vLLM 0.19 contract drift surfaced four real bugs vs the 0.9-era code:** `build_connector_meta` must return non-None, `request_finished` must return `(bool, dict|None)`, `kv_layer` is now a single Tensor `[2, blocks, bs, h, d]` (K stacked with V), layer names are `"model.layers.N.self_attn.attn"`, and bf16 needs an explicit cast before `.numpy()`.
- **Cross-process engine state is NOT shared.** TardigradeDB caches engine state at `Engine::open()`. The connector lives in vLLM's `EngineCore` subprocess; observers must reopen the engine to see fresh writes. This was caught only because the GPU integration test asserted on `pack_count` from a different process.

**Known gap (not an architectural problem, just unfinished work):** `save_kv_layer` doesn't yet thread per-request `slot_mapping` from `attn_metadata`, so the save path captures block 0 of each layer as a placeholder. The pipe is proven; the cargo is a stub. See `docs/guide/vllm-setup.md` for the full status table and limitations.

**Tests:** `tests/python/test_vllm_format.py` (4 unit), `tests/python/test_vllm_connector.py` (4 engine-surface), `tests/python/test_vllm_load_path.py` (4 mock-context for `start_load_kv`), `tests/python/test_vllm_integration.py` (5 GPU acceptance with `-m gpu`).

### [KV Cache Validation — Full Progression](kv-cache-validation.md)

**Date:** April 22-23, 2026
**Status:** Complete — three major discoveries

Systematic exploration of what to store and how to retrieve, tested on Sonia (16 diverse life memories) with GPT-2, Qwen3-0.6B, and Qwen2.5-3B.

**Three discoveries:**
1. **Store K projections, not hidden states** — hidden states produce gravity wells (31.2%), K projections doubled recall (62.5-75%)
2. **K*K per-token matching fails** — K vectors share a massive common component across all sequences (position-0 cross-sentence dot = 6281 for unrelated text). Per-token K*K got 25%, worse than mean-pool
3. **Query with Q, store K (Q*K)** — matches how attention actually works. The fixed Q*K per-token pipeline gets 68.8% recall with 8 unique top-1 memories, and now exercises encoded Q tokens against encoded K tokens through max-sim scoring

**Full progression:** 31.2% (hidden) → 25% (K*K per-token) → 62.5% (K*K mean-pool) → 75% (K*K per-token manual) → 68.8% (Q*K per-token pipeline, 16 memories)

**100-memory scale test:** Q*K recall dropped to **40%** at 100 memories. Gravity well returned. Traditional RAG baseline achieved 100% on the same corpus.

**Signal audit verdict: `LAYER_OR_HEAD_PROBLEM`.** The correct memories ARE in the latent signal (R@100 = 100%). Hidden states + top5_pair_avg achieves 100% recall with 10% false positive rate.

**Engine pipeline validation: 100% recall.** Hidden states + Top5Avg scoring through the full engine pipeline (Q4 quantization, INT8 scoring, per-token retrieval) achieves **30/30 recall, all at rank #1, no gravity well, 97ms latency**. This matches traditional RAG using the model's own latent representations.

**Open question: vague queries.** All test queries use specific vocabulary. Real agent queries would be vaguer ("How is Lucia doing?"). This is the next test.

**Scripts:** `experiments/scale_100_hidden_top5.py` (100% result), `experiments/scale_100_qk_diagnostics.py` (diagnostic suite), `experiments/scale_100_rag_baseline.py` (RAG baseline)

## Planned Experiments

| Experiment | Goal | Status |
|-----------|------|--------|
| **100-memory scale test** | Does Q*K retrieval hold at realistic memory counts? | **Complete — 40% recall, needs scoring improvements** |
| **Traditional RAG baseline** | Compare current Q*K retrieval against standard embedding retrieval | **Complete — RAG got 100% recall on this corpus** |
| **Hidden states + top5_pair_avg validation** | Validate 100% recall path through engine pipeline | **Next up** |
| **False positive calibration** | Reduce 10% negFP rate via score thresholding | Planned |
| GQA K-expansion | Expand K heads to match Q dims for Q*K retrieval | Complete |
| Per-head scoring | Score per attention head instead of concatenating all heads | Planned |
| Multi-session memory | Cross-day retrieval ("what happened last week") | Planned |
| Adversarial retrieval | Contradictory memories, test which surfaces | Planned |
| Confidence thresholding | Calibrate "I don't remember" cutoff using SNR | Planned |
| [Cross-model retrieval](cross-model-memory-test.md) | Store with one model, retrieve with another | Designed |
| RoPE injection | Test KV injection with rotary position encoding | Planned |
| **vLLM connector — semantic save** | Thread per-request `slot_mapping` from `attn_metadata` so save captures the request's actual blocks (not placeholder block 0). Validate that re-querying the same prompt returns the stored KV with non-zero overlap. | Planned (next vLLM work) |
| **vLLM cross-session retrieval** | Save with vLLM run #1, restart, query with vLLM run #2. Confirm load path injects the prior session's KV and `start_load_kv` writes non-zero data into the allocated GPU block slots. | Planned |

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
