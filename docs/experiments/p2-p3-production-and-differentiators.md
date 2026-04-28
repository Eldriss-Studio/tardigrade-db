# P2+P3: Production Story & Differentiators

**Date:** April 28, 2026
**Status:** Complete — 263 Rust tests, 216 Python tests, both e2e demos verified

## Context

Following the P1 architectural unification (wiring disconnected layers), P2 closed credibility gaps in documentation and operational APIs, while P3 turned implemented-but-unused Rust features into real capabilities accessible from Python.

## P2: Close the Production Story

### 2.1+2.2: Honest Documentation

**Problem:** `spec.md` and `tdd.md` claimed GPU DMA offloading, MemArt 91-135× prefill reduction, Atlas Index with 3.09μs traversal, and RelayCaching 4.7× TTFT — none implemented.

**Fix:** Both documents rewritten to match implementation reality. Unimplemented features moved to explicit "Future Work" sections. README architecture diagram updated (Atlas → Vamana, MemArt → Per-token Top5Avg, GPU DMA → append-only segments). Test counts updated throughout. vLLM correctly described as "prefix-cache accelerator" everywhere.

### 2.3: Pluggable Retrieval Key Strategies (Strategy Pattern)

**Problem:** `LastTokenEmbeddingStrategy` was the only retrieval key strategy for the vLLM connector. It assumes `hidden_size == kv_dim`, which works for Qwen3-0.6B by coincidence but breaks on most models.

**Fix:** Added two strategies:
- **`MeanPoolEmbeddingStrategy`** — pools all token embeddings. More robust for variable-length prompts.
- **`ProjectedEmbeddingStrategy`** — linear projection from `hidden_size` to `kv_dim` via random orthogonal matrix. Handles GQA models where dimensions don't match.

Strategy names are module-level constants (`LAST_TOKEN_EMBEDDING`, `MEAN_POOL_EMBEDDING`, `PROJECTED_EMBEDDING`), not magic strings. Factory method looks up by constant. 14 tests (7 existing + 7 new).

### 2.4: Engine Status API

**Problem:** No single call to get a snapshot of engine health. Needed 5+ individual getters.

**Fix:** `Engine::status() → EngineStatus` returns: `cell_count`, `pack_count`, `segment_count`, `slb_occupancy`, `slb_capacity`, `vamana_active`, `pipeline_stages`, `governance_entries`, `trace_edges`. Exposed to Python as a dict. 1 Rust ATDD test.

### 2.5: Engine Configuration from Python

**Problem:** Python `Engine()` only accepted a path. No way to set segment size or Vamana threshold.

**Fix:** `Engine(path, segment_size=None, vamana_threshold=None)` — optional kwargs passed to Rust `open_with_options`.

### Additional P2 Fix: Tier Boost Constants

Extracted hardcoded tier boost multipliers (1.0, 1.1, 1.25) to named constants (`DRAFT_RETRIEVAL_BOOST`, `VALIDATED_RETRIEVAL_BOOST`, `CORE_RETRIEVAL_BOOST`) in `tdb-core/src/types.rs`.

---

## P3: Realize the Differentiators

### 3.1: Semantic Edge Types (Generalized Command Pattern)

**Problem:** `EdgeType::Supports` and `Contradicts` were defined in `tdb-index` but never written by any code path. Only `CausedBy` and `Follows` were used.

**Design:** Generalize `add_pack_link` into `add_pack_edge(pack_1, pack_2, edge_type)` — the edge type parameterizes the command. Agent decides what's Supports vs Contradicts (no auto-detection — explicit > magic).

**API added:**
- `Engine::add_pack_edge(pack_1, pack_2, EdgeType)` — generalized, bidirectional, WAL-logged
- `Engine::pack_links_by_type(pack_id, EdgeType)` — type-filtered query
- `Engine::pack_supports(pack_id)` / `pack_contradicts(pack_id)` — convenience delegates
- `add_pack_link` refactored to delegate to `add_pack_edge(..., Follows)` (backward compat)
- Python: `add_pack_edge(id1, id2, edge_type_u8)`, `pack_links_by_type`, `pack_supports`, `pack_contradicts`
- KPS: `store_supporting(fact, related_id)`, `store_contradicting(fact, related_id)` with named constants

**Tests:** 6 Rust ATDD (supports, contradicts, type filtering, WAL replay, backward compat, mixed boost) + 4 Python ATDD.

### 3.2: SynapticBank Python Exposure (Repository + Facade)

**Problem:** `SynapticBankEntry` (LoRA adapter pairs in FP16, owner-isolated, dimension-validated) was fully implemented in Rust with durable storage — but zero Python exposure. Dead code from the user's perspective.

**Design:** Repository pattern (already in Rust) exposed through PyO3 Facade. f32 numpy arrays in Python; f16 conversion at the binding boundary. Dimension validation at the boundary with `PyValueError` (not Rust panic).

**API added:**
- `engine.store_synapsis(id, owner, lora_a, lora_b, scale, rank, d_model, last_used=None, quality=None)`
- `engine.load_synapsis(owner)` → list of dicts with f32 numpy arrays

**Tests:** 6 Python ATDD (round-trip, owner isolation, persistence across reopen, dimension validation, metadata fields, large matrices rank=16 × d_model=512).

### 3.3: Multi-Agent Acceptance Tests (Fixture-Based Organization)

**Problem:** Owner isolation was implemented at every API level but only tested with 2 owners and 1-2 packs each. No test validated 3+ agents, cross-agent trace isolation, scoped eviction, or delete isolation at scale.

**Design:** Shared fixture creates a 3-agent engine (ALPHA=100, BETA=200, GAMMA=300) with 5 packs each. Named constants for all agent IDs and thresholds. Tests exercise every owner-scoped operation.

**Result:** All 12 tests pass immediately against existing code — the owner isolation was already correct. These tests serve as acceptance gates for future multi-agent work.

**Tests:** 7 Rust (pack isolation, trace link isolation, scoped eviction, delete isolation, retrieval isolation, trace boost owner filter, mixed independent tiers) + 5 Python mirrors.

---

## Verification

| Check | Result |
|-------|--------|
| `cargo test --workspace --exclude tdb-python` | 263 passed |
| `cargo clippy --workspace --all-targets -- -D warnings` | Clean |
| `pytest tests/python/ -m "not gpu"` | 216 passed |
| `examples/e2e_demo.py` (GPT-2) | SUCCESS |
| `examples/agent_memory.py` (Qwen3-0.6B) | SUCCESS |

## Test Counts After P2+P3

| Layer | Crate | Tests |
|-------|-------|-------|
| Core | tdb-core | 6 |
| Storage | tdb-storage | 33 |
| Retrieval | tdb-retrieval | 51 |
| Organization | tdb-index | 23 |
| Governance | tdb-governance | 26 |
| Engine | tdb-engine | 124 |
| Python | pytest | 216 |
| **Total** | | **479** |
