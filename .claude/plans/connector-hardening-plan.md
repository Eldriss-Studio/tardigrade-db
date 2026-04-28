# Connector Review Evaluation & Response Plan

## Context

External review of commit `6d54841` raised 3 major and 4 minor concerns about the vLLM connector. This plan evaluates each claim against the actual code, separates valid problem identification from proposed solutions, and designs architecturally principled fixes.

---

## Verdict Summary

| # | Claim | Valid? | Severity | Reviewer's Fix | Our Direction |
|---|-------|--------|----------|----------------|---------------|
| 1 | Connector metadata empty, scheduler→worker breaks in distributed | **Partially valid** | Medium — blocks distributed deployment, but single-process is the validated path | "Pass pack_id, block_ids, num_tokens in metadata" | Follow vLLM's ExampleConnector contract properly |
| 2 | Save fingerprint (`block_indices[0]`) is fragile | **Overstated** — fingerprint IS stable during request lifetime | Low-Medium — real issue is unbounded map growth, not fingerprint fragility | "Use request_id, not heuristic" | Bound the map; the fingerprint is sound |
| 3 | Retrieval key strategy is "conceptually weak" | **Valid core observation, wrong framing** | High — this is the retrieval quality ceiling | "Both sides same strategy or forward pass on load" | Unify on embedding-table lookup for both sides (cheap, consistent) |
| M1 | `_get_embed_weights` loads entire model | **Valid** | Medium | — | Use safetensors weight-only loading |
| M2 | `match_threshold=150` is magic | **Valid** — default is stale, tuned for old strategy | Low | — | Recalibrate after key alignment fix |
| M3 | Test count 453 vs 451 | **Stale** — CLAUDE.md now says 442, table sums correctly | Non-issue | — | Already fixed |
| M4 | Refresh test `>= 5` for 8 written | **Valid** | Low | — | Tighten assertion |

---

## Detailed Analysis

### Issue 1: Empty Connector Metadata

**What the reviewer sees:** `build_connector_meta()` returns empty `_TardigradeConnectorMetadata()`. `start_load_kv()` reads from `self._load_packs`/`self._load_meta` instance dicts. In distributed vLLM (separate scheduler/worker processes), the worker can't see scheduler state.

**What's actually true:** The code documents this limitation explicitly (lines 88-95, 322-326). It works in single-process mode. The reviewer is right that this violates the vLLM connector V1 contract — vLLM's own `ExampleConnector` demonstrates the proper pattern where metadata carries per-request `(token_ids, block_ids, is_store)` tuples through the IPC bridge.

**Is it urgent?** No. TardigradeDB's validated deployment is single-process vLLM on one GPU. Distributed serving (tensor-parallel, pipeline-parallel) is not yet a target. But the code should be *correct by contract* even if only one deployment mode is tested today.

**Our approach:** Follow vLLM's ExampleConnector pattern properly — not because the reviewer said to, but because violating the framework contract means any vLLM refactor could silently break us. The metadata should carry the load request data. This is the right engineering, regardless of current deployment mode.

**Files:** `python/tardigrade_vllm/connector.py` (lines 87-95, 316-328, 332-388)

### Issue 2: Fingerprint Stability

**What the reviewer claims:** `block_indices[0]` is "fragile as fuck" because blocks are "reusable physical resources" — could delete wrong pack or coalesce different requests.

**What's actually true:** The reviewer is *wrong about the fingerprint*. `block_indices[0]` is the first block allocated to a request by vLLM's scheduler. vLLM does NOT reassign blocks for a live request. The fingerprint IS stable for the request's lifetime. The code's own documentation (lines 440-442) is correct.

**What the reviewer missed:** The real problem is that `_pack_id_by_fingerprint` grows unboundedly. `request_finished()` fires on the scheduler side, but `_pack_id_by_fingerprint` lives on the worker side. After thousands of requests, the map leaks memory. Worse: after a block is freed and reallocated to a new request, a stale fingerprint entry could cause `delete_pack` on the wrong pack.

**Our approach:** Bound the map using `max_num_seqs` from vLLM config (natural concurrency limit). Use `collections.OrderedDict` with LRU eviction. This is correct because vLLM never has more than `max_num_seqs` concurrent requests — entries older than that are guaranteed stale.

**Files:** `python/tardigrade_vllm/connector.py` (lines 178-184, 447-473)

### Issue 3: Retrieval Key Space Mismatch

**What the reviewer sees:** Save side uses last-token K from last layer. Load side uses embedding table lookup. These aren't the same mathematical object, even when dimensions match.

**What's actually true:** This is the most valid observation. The reviewer is right that `embedding[token_id]` and `W_k @ hidden_state[-1]` are fundamentally different vectors. The code is honest about this (retrieval_key.py lines 6-8, lines 30-35), and `check_key_alignment` warns when dimensions don't even match.

**What the reviewer gets wrong:** The suggested fix ("both sides same strategy OR forward pass on load") frames it as a binary. A forward pass on the load (scheduler) side defeats the purpose — the scheduler doesn't have GPU access and shouldn't need it for coarse-grained matching.

**Our approach — and this is where we diverge from the reviewer:** The right solution isn't to make load match save (expensive) or to add a projection layer (per-model calibration burden). The right solution is to make save match load: **both sides use embedding table lookup**. This is:
- Cheap on both sides (array index, not forward pass)
- Identical computation = identical vector space = exact match possible  
- Sufficient for coarse pack retrieval (token identity is semantically informative)
- The save side already has access to token IDs via `attn_metadata` (or will, once metadata bridge is fixed)

The deeper retrieval quality comes from TardigradeDB's Rust engine (PerTokenRetriever, Top5Avg, SLB) — the connector's job is coarse retrieval to find the right pack. Embedding-table matching is appropriate for that granularity.

**Dependency:** This requires the metadata bridge (Issue 1) to deliver token IDs to the save side.

**Files:** `python/tardigrade_vllm/connector.py` (lines 500-516), `python/tardigrade_vllm/retrieval_key.py`

### Minor Issues

**M1: `_get_embed_weights` loads entire model** — Valid. `AutoModel.from_pretrained` instantiates the full model just to extract one weight matrix. Replace with targeted weight-only loading (e.g., `huggingface_hub.hf_hub_download` + safetensors selective tensor load). Falls back to current approach if not available.

**M2: `match_threshold=150`** — Valid but premature to fix. The threshold was tuned for mean-pooled keys. After Issue 3 fix (both sides use embedding lookup), the score distribution will change again. Calibrate AFTER the key alignment fix lands. Consider making it a required config parameter with no silent default.

**M3: Test count mismatch** — Already resolved in updated CLAUDE.md (now says 442, table sums to 442).

**M4: Refresh test `>= 5` for 8 written** — Valid. The assertion at `acceptance.rs:2621` is loose. Should assert `== 8` for all cells retrievable, since `cell_count()` already confirms 8 cells exist.

---

## Implementation Plan

---

### Phase 1: Bounded Fingerprint Lifecycle

**Design Pattern: Bounded Cache (LRU eviction with domain-invariant capacity)**

The fingerprint map is a write-through cache keyed by `block_indices[0]`. The capacity invariant comes from vLLM itself: `max_num_seqs` is the hard concurrency ceiling. Any entry older than `max_num_seqs` insertions is guaranteed stale — the request has finished and its blocks have been freed.

**SOLID breakdown:**
- **SRP:** `_pack_id_by_fingerprint` does one thing — maps live request fingerprints to their latest pack_id. Eviction is part of the map's contract, not the caller's.
- **OCP:** Changing from `dict` to `OrderedDict` with bounded eviction doesn't change the interface `wait_for_save` uses (same `get`/`__setitem__`/`pop` API).

**ATDD — write these FIRST, they define "done":**

```python
# test_vllm_load_path.py

def test_fingerprint_map_bounded_after_many_requests():
    """GIVEN a connector with max_num_seqs=4
    AND 20 sequential requests each completing save_kv_layer + wait_for_save
    WHEN all 20 have completed
    THEN len(_pack_id_by_fingerprint) <= 4"""

def test_stale_fingerprint_eviction_prevents_wrong_pack_deletion():
    """GIVEN request A used block_indices[0]=5, wrote pack_id=10
    AND request A is evicted from the fingerprint map (capacity exceeded)
    AND request B is allocated the same block_indices[0]=5
    WHEN request B calls wait_for_save
    THEN pack_id=10 is NOT deleted
    AND request B's new pack exists in the engine"""

def test_live_request_fingerprint_survives_eviction_of_others():
    """GIVEN requests A, B, C, D all active (max_num_seqs=4)
    AND request E arrives (would exceed capacity)
    WHEN wait_for_save processes E
    THEN request A's fingerprint is evicted (oldest)
    AND requests B, C, D fingerprints remain"""
```

**Implementation:**
1. `__init__`: read `max_num_seqs` from `vllm_config.scheduler_config.max_num_seqs`, default 256
2. Replace `self._pack_id_by_fingerprint: dict[int, int] = {}` with `self._pack_id_by_fingerprint = OrderedDict()` + store `self._max_fingerprints = max_num_seqs`
3. In `wait_for_save`, after `self._pack_id_by_fingerprint[fingerprint] = pack_id`, call `self._pack_id_by_fingerprint.move_to_end(fingerprint)` then evict: `while len(self._pack_id_by_fingerprint) > self._max_fingerprints: self._pack_id_by_fingerprint.popitem(last=False)`

**Files:** `python/tardigrade_vllm/connector.py`, `tests/python/test_vllm_load_path.py`

---

### Phase 2: Metadata Bridge

**Design Pattern: Data Transfer Object (DTO) — `_TardigradeConnectorMetadata` as a serializable DTO crossing the scheduler→worker process boundary**

vLLM's connector V1 contract requires `build_connector_meta` to produce a metadata object that travels through the `SchedulerOutput` IPC bridge. The current empty metadata works by accident (single-process). The DTO pattern makes the contract explicit: the metadata is a value object carrying everything the worker needs to execute the load, with no references to scheduler-side state.

**SOLID breakdown:**
- **SRP:** `_TardigradeConnectorMetadata` carries load request data across the IPC boundary — nothing else. It doesn't decide what to load; it transports what the scheduler decided.
- **ISP:** The worker only sees `load_requests: list[_LoadRequestMeta]`. It doesn't need to know about `_load_packs`/`_load_meta` internal scheduler dicts.
- **DIP:** `start_load_kv` depends on the `KVConnectorMetadata` abstraction (via `self._get_connector_metadata()`), not on scheduler instance state.

**ATDD — write these FIRST:**

```python
# test_vllm_load_path.py

def test_build_connector_meta_packages_matched_packs():
    """GIVEN a connector with 2 matched packs in _load_packs/_load_meta
    WHEN build_connector_meta is called
    THEN returned metadata has 2 load_requests
    AND each contains correct request_id, pack data, seq_len, block_ids, num_tokens"""

def test_build_connector_meta_clears_scheduler_state():
    """GIVEN a connector with matched packs
    WHEN build_connector_meta is called
    THEN _load_packs and _load_meta are empty dicts"""

def test_start_load_kv_reads_from_bound_metadata_not_instance_state():
    """GIVEN metadata bound via bind_connector_metadata (simulating IPC)
    AND _load_packs is empty (proving scheduler state isn't used)
    WHEN start_load_kv is called with valid forward_context
    THEN blocks are written to the correct GPU slots from metadata"""

def test_metadata_survives_serialization_round_trip():
    """GIVEN _TardigradeConnectorMetadata with load request data
    WHEN serialized and deserialized (simulating IPC transport)
    THEN all fields are preserved and usable"""

def test_build_connector_meta_with_no_matches_returns_empty():
    """GIVEN a connector with no matched packs
    WHEN build_connector_meta is called
    THEN returned metadata has empty load_requests list
    AND is still a valid KVConnectorMetadata instance"""
```

**Implementation:**
1. Define `@dataclass _LoadRequestMeta` with fields: `request_id: str`, `pack: dict`, `seq_len: int`, `block_ids: list[int]`, `num_tokens: int`
2. Extend `_TardigradeConnectorMetadata` with `load_requests: list[_LoadRequestMeta] = field(default_factory=list)`
3. `build_connector_meta`: iterate `_load_packs` joined with `_load_meta`, build `_LoadRequestMeta` for each, clear both dicts, return populated metadata
4. `start_load_kv`: call `meta = self._get_connector_metadata()` (inherited from `KVConnectorBase_V1`), iterate `meta.load_requests` instead of `self._load_packs`

**Files:** `python/tardigrade_vllm/connector.py`, `tests/python/test_vllm_load_path.py`

---

### Phase 3: Retrieval Key Unification

**Design Pattern: Strategy (already in place) — add `EmbeddingTableStrategy` as a unified save+load strategy, retiring the asymmetric `LastTokenEmbeddingStrategy` + raw-K-extraction split**

The current architecture has an asymmetric Strategy: the load side uses `LastTokenEmbeddingStrategy` (embedding table lookup), while the save side bypasses the strategy entirely and extracts raw K projections from the KV tensor. This means the two keys live in different vector spaces. The fix: the Strategy pattern should govern BOTH sides. A new `EmbeddingTableStrategy` computes `embedding_table[last_token_id]` for both save and load.

**SOLID breakdown:**
- **SRP:** `RetrievalKeyStrategy` has one job: compute a retrieval key. Currently the save side doesn't use it. After this change, both sides go through the same strategy.
- **LSP:** `EmbeddingTableStrategy` is substitutable for `LastTokenEmbeddingStrategy` — same interface, same contract, different (and correct) implementation.
- **OCP:** Adding a new strategy doesn't modify existing ones. `LastTokenEmbeddingStrategy` stays for backward compatibility; `EmbeddingTableStrategy` becomes the default.

**ATDD — write these FIRST:**

```python
# test_vllm_load_path.py

def test_save_and_load_produce_identical_retrieval_keys():
    """GIVEN a connector configured with EmbeddingTableStrategy
    AND a known prompt with token_ids [101, 202, 303]
    WHEN save computes a retrieval key for this prompt
    AND load computes a retrieval key for the same prompt
    THEN the two keys are identical (np.allclose)"""

def test_save_side_uses_strategy_not_raw_k_extraction():
    """GIVEN a connector with EmbeddingTableStrategy
    WHEN _write_pack_for_batch is called
    THEN the retrieval key equals embed_weights[last_token_id]
    AND NOT k_last_layer[-1] (the old raw-K path)"""

def test_retrieval_key_strategy_is_model_agnostic():
    """GIVEN a model where hidden_size=2048 != kv_dim=512
    AND EmbeddingTableStrategy is used on both sides
    WHEN save and load compute keys
    THEN keys match (both are hidden_size-dimensional, from embedding table)
    AND the dimension mismatch warning does NOT fire"""

def test_default_strategy_is_embedding_table():
    """GIVEN default connector config (no explicit retrieval_key_strategy)
    WHEN the strategy is resolved
    THEN it is EmbeddingTableStrategy"""
```

**Implementation:**
1. Add `EmbeddingTableStrategy` to `retrieval_key.py` — identical to `LastTokenEmbeddingStrategy` in compute logic
2. Add `compute_for_save(self, token_ids, embed_weights)` method to `RetrievalKeyStrategy` ABC — same signature as `compute`. Default impl delegates to `compute`. This lets strategies distinguish save vs load if needed in future
3. Change default strategy from `"last_token_embedding"` to `"embedding_table"` in connector `__init__`
4. In `_write_pack_for_batch`: replace raw K extraction (lines 500-516) with `self._retrieval_key_strategy.compute_for_save(token_ids, self._get_embed_weights())`. The `token_ids` come from `BatchSlice` (needs extension) or from saved metadata
5. **Token ID plumbing for save side:** `save_kv_layer` receives `attn_metadata` which has `query_start_loc` and the model runner's token list. Extend `BatchSlice` to carry `token_ids: tuple[int, ...]` (populated by `RequestSlotResolver`). This is the cleanest path — the resolver already slices per-request, adding token IDs is a natural extension

**Files:** `python/tardigrade_vllm/retrieval_key.py`, `python/tardigrade_vllm/connector.py`, `python/tardigrade_vllm/slot_resolver.py`, `tests/python/test_vllm_load_path.py`

---

### Phase 4: Cleanup

**M1: Embedding Weight Loading — Lightweight Factory**

**Design Pattern: Factory Method with Fallback Chain**

Replace the monolithic `AutoModel.from_pretrained` with a chain of progressively heavier loaders:
1. Try `safetensors` direct tensor load (fastest, no model instantiation)
2. Try `huggingface_hub.hf_hub_download` + targeted file (network but no model)
3. Fall back to current `AutoModel.from_pretrained` (last resort)

**ATDD:**
```python
def test_embed_weights_loaded_without_full_model():
    """GIVEN a model available locally as safetensors
    WHEN _get_embed_weights is called
    THEN embed_weights are loaded
    AND AutoModel.from_pretrained was NOT called"""
```

**M4: Refresh Test Tightening**

**ATDD (Rust):**
```rust
// acceptance.rs — tighten existing test
// GIVEN 8 cells written across two engines
// WHEN engine_a.refresh() then mem_read(&key, 8)
// THEN results.len() == 8
// AND all 8 original cell_ids appear in results
```

Change `assert!(results.len() >= 5, ...)` to `assert_eq!(results.len(), 8, ...)` at `acceptance.rs:2621`.

**Files:** `python/tardigrade_vllm/connector.py`, `crates/tdb-engine/tests/acceptance.rs`

---

### Verification

```bash
# Rust (including tightened refresh assertion)
cargo test --workspace --exclude tdb-python
cargo clippy --workspace --exclude tdb-python -- -D warnings

# Python CPU (all new ATDD tests)
pytest tests/python/ -v -m "not gpu"

# Python GPU (key alignment validation — if GPU available)
pytest tests/python/ -v -m gpu
```
