# Python → Rust Migration — 12 Boundary Fixes (ATDD Plan)

## Context

A five-agent audit on 2026-05-20 mapped the full Python/Rust boundary of TardigradeDB. The architecture is sound — heavy numerics already live in Rust — but the boundary has accumulated *plumbing* debt: tensor→numpy→PyO3 double-copies, N×FFI loops, write/read asymmetries, and one N+1 inside the Rust marshalling loop (`crates/tdb-python/src/lib.rs:925` — `pack_text` called per pack while iterating `list_packs`).

Twelve fixes identified (H1–H5 hot path; M1–M7 medium). User scoping:
- **All 12 in scope.**
- **Bench gates required** per phase (baseline + post-change measurement with stated pass criterion).
- **tch-rs in scope** (Phase 12, behind a `torch-bridge` Cargo feature).

Intended outcome: close the read/write encoding asymmetry left over from commit `c92d584`, eliminate per-layer FFI loops, and make `list_packs` proportional to what the caller actually needs — without breaking the published Python API for ≥ one minor version.

---

## Feature in one sentence

Migrate twelve Python-side computations into Rust so the store/read paths stop paying per-layer FFI overhead, `list_packs` stops paying N+1 text I/O, and the vLLM connector's per-request numerics run native — while preserving existing Python API signatures behind deprecation warnings.

---

## Acceptance Tests — behavioural only

Naming: Rust ATs in `crates/<crate>/tests/<phase>_acceptance.rs`; Python ATs in `tests/python/test_<phase>_<area>.py`. Each AT describes an *observable* behaviour through the public API. Call-count, AST-grep, and `tracemalloc` assertions are **not** ATs — they live in the per-phase **Observability Checks** section below, not the AT list.

### Phase 1 — H5 `list_packs` text-fetch split
1. `test_list_packs_metadata_returns_empty_list_when_owner_has_no_packs`
2. `test_list_packs_metadata_omits_text_field` — result dicts contain `{pack_id, owner, tier, importance}` only
3. `test_list_packs_with_fetch_text_false_yields_same_rows_as_metadata`
4. `test_list_packs_with_fetch_text_true_returns_stored_text_or_none`
5. `test_legacy_list_packs_emits_deprecation_warning_once_per_process`
6. `test_consolidate_one_pack_completes_under_p99_latency_at_10k_packs` — *behavioural* assertion (≤ X ms) replacing the call-count check
7. (Rust) `it_returns_metadata_sorted_by_importance_descending`
8. `test_list_packs_metadata_consistent_under_concurrent_writes`

### Phase 2 — H1 write-side raw-token API (`mem_write_pack_tokens`)
1. `test_round_trip_via_tokens_equals_round_trip_via_encoded` (score delta < 1e-5 between the two write paths)
2. `test_rejects_dim_mismatch_with_stable_error_code` (`tdb::write::dim_mismatch`)
3. `test_rejects_empty_token_matrix` (`EmptyCorpus` code)
4. `test_non_contiguous_numpy_view_is_accepted`
5. `test_single_token_matrix_is_legal`
6. `test_concurrent_writers_serialize_without_corruption` — 8 threads × 25 writes → exactly 200 packs, no truncated layers
7. (Rust) `it_records_dim_idx_in_header_round_trip`
8. `test_snapshot_created_pre_migration_restores_after`

### Phase 3 — M1 batch writes (`mem_write_batch_packs`)
1. `test_returns_pack_ids_in_input_order_strictly_increasing`
2. `test_batch_with_one_pack_equals_single_write_in_id_salience_and_tier`
3. `test_empty_batch_is_a_no_op_returning_empty_list`
4. `test_partial_failure_leaves_pack_count_unchanged` (bad pack at index 17/50 → 0 written, raises)
5. `test_batch_visible_immediately_without_explicit_flush`
6. `test_batch_packs_start_in_draft_tier`
7. (Rust) `it_recovers_full_batch_or_none_after_crash_during_fsync` — CLAUDE.md durability rule
8. `test_one_million_pack_batch_completes_without_oom_or_corruption`

### Phase 4 — M5 `salience_mode` parameter
1. `test_default_salience_mode_l2_produces_same_value_as_current_python_path` (within 1e-5)
2. `test_salience_mode_max_uses_inf_norm`
3. `test_salience_mode_none_with_explicit_value_stores_that_exact_value`
4. `test_unknown_salience_mode_raises_value_error_with_stable_code`
5. `test_salience_mode_clamps_at_100_for_extreme_payloads`

### Phase 5 — M3 chunker boundary search (`find_chunk_boundary`)
1. `test_whitespace_boundary_matches_python_baseline_on_1mb_corpus_at_max_pos_64_256_1024`
2. `test_sentence_boundary_returns_position_after_terminal_punctuation`
3. `test_paragraph_boundary_uses_double_newline`
4. `test_no_boundary_within_range_falls_back_to_max_pos`
5. `test_unicode_sentence_endings_handled` (CJK `。`)
6. `test_empty_text_returns_zero`
7. `test_max_pos_beyond_text_length_returns_text_length`
8. `test_unknown_strategy_raises_value_error`

### Phase 6 — M6 HTTP `pack_text` fallback removal
1. `test_mem_read_pack_response_includes_text_when_stored`
2. `test_mem_read_pack_response_text_is_null_when_not_stored`
3. `test_http_query_p95_latency_under_threshold_at_100_packs` — *behavioural* replacement for "zero secondary calls"
4. `test_old_client_pattern_still_works_emitting_deprecation_note`

### Phase 7 — H3 vLLM retrieval-key compute (`compute_retrieval_key`)
1. `test_last_token_strategy_matches_python_within_1e6_max_abs_delta`
2. `test_mean_pool_strategy_matches_python_within_1e6_max_abs_delta`
3. `test_filters_out_of_bounds_token_ids_matching_python_filter`
4. `test_empty_token_id_list_returns_none`
5. `test_projected_strategy_with_identity_matrix_equals_last_token_strategy`
6. `test_compute_retrieval_key_returns_none_when_embedding_table_unloaded`
7. (Rust) `it_dispatches_through_retrieval_key_strategy_trait`

### Phase 8 — H4 vLLM KV-block reshape (`reshape_kv_paged`)
1. `test_flat_to_blocks_round_trip_is_lossless` (seq_lens 1, 15, 16, 17, 100)
2. `test_seq_len_not_multiple_of_block_size_is_zero_padded_at_unused_slots`
3. `test_block_size_one_is_legal`
4. `test_seq_len_zero_returns_empty_blocks`
5. `test_blocks_to_flat_rejects_mismatched_seq_len_with_stable_error`
6. (Rust) `it_matches_python_byte_for_byte_on_qwen3_dims`

### Phase 9 — M7 vLLM fingerprint cache to engine
1. `test_lookup_after_max_num_seqs_plus_one_inserts_evicts_oldest`
2. `test_lookup_after_request_finished_returns_none`
3. `test_lookup_after_request_finished_on_error_path_also_returns_none`
4. `test_concurrent_inserts_yield_cache_size_at_most_max_num_seqs`
5. `test_refresh_does_not_evict_fingerprint_entries` (process-local invariant)
6. `test_block_id_reused_by_new_request_does_not_return_old_pack`

### Phase 10 — M2 batch reads with refinement (`mem_read_pack_batch`)
1. `test_batch_read_results_match_per_query_calls_one_for_one`
2. `test_empty_batch_returns_empty_list`
3. `test_heterogeneous_k_per_query_produces_rows_of_requested_length`
4. `test_owner_filter_applies_independently_per_query`
5. `test_batch_with_prf_refinement_top1_matches_per_query_with_prf_refinement` — behavioural equivalence

### Phase 11 — M4 multi-layer RRF in Rust (`mem_read_multi_layer`)
1. `test_rrf_output_matches_python_implementation_top_k_for_k_in_5_10_20`
2. `test_default_rrf_k_is_60`
3. `test_single_layer_input_returns_that_layer_ordering_unchanged`
4. `test_tied_rrf_scores_break_ties_by_pack_id_ascending`
5. `test_layer_ratios_outside_zero_to_one_raise_value_error`
6. `test_zero_layer_ratios_returns_empty_list`

### Phase 12 — H2 tch-rs torch tensor bindings (feature-gated)
1. `test_module_imports_when_feature_disabled_with_no_torch_symbol`
2. `test_torch_path_pack_contents_equal_numpy_path_pack_contents` (behavioural equivalence between the two write APIs)
3. `test_cuda_tensor_produces_same_pack_as_cpu_tensor` (CUDA-gated)
4. `test_bf16_tensor_accepted_with_bounded_rounding_delta`
5. `test_legacy_numpy_path_still_works_alongside_torch_path`
6. `test_torch_tensor_requiring_grad_is_detached_safely`

---

## Design Patterns — honestly named

Only patterns where the GoF / SOLID definition genuinely fits. Where the original draft over-claimed a label, I've replaced it with the actual structural decision.

| Decision | Pattern (or "not a pattern") | Why it fits |
|---|---|---|
| H5 split `list_packs` / `list_packs_metadata` | **Interface Segregation Principle** (SOLID) | Two callers shapes — metadata-only vs metadata+text — get two interfaces. ISP applied directly. |
| H1 raw-token write API | *Not a pattern* — **API symmetry** with `mem_read_tokens` | This was originally labelled Template Method, but Template Method needs a base class with hook points. It's structural symmetry, not Template Method. State that honestly. |
| H2 torch tensor bridge | **Adapter** | `tch::Tensor` → engine `&[f32]` is the canonical Adapter shape (wrap an incompatible interface). The `torch-bridge` Cargo feature is a build-system concern, not part of the pattern. |
| H3 retrieval-key strategies | **Strategy** (extending existing trait) | `tdb-retrieval` already has `RetrievalKeyStrategy`; Rust impls of `LastToken`, `MeanPool`, `Projected` are textbook Strategy variants behind one trait. |
| H3 embedding-table storage | *Not a pattern* — **engine-owned state with lazy load** | Originally claimed "Borrowed Singleton" — that's not a real pattern. It's just stateful engine ownership; load happens on first `load_embedding_table` call. |
| H4 KV reshape | *Not a pattern* — **stateless function on `Engine`** | Originally claimed "Pure Function" as if it were a pattern. It isn't; it's an implementation style. State that. |
| M1 batch writes | **Unit of Work** | Group of `KVPack`s commits or fails together; single fsync; explicitly distinct from streaming `WriteBuffer`. Canonical UoW shape. |
| M2 batch reads with refinement | **Memoization** (corpus stats per batch) | "Unit of Work for reads" is a stretch; the real reused pattern is memoising refinement corpus stats across the batch's queries. |
| M3 chunker boundary | **Strategy** (Rust enum) + **Adapter** (Python wrapper) | Rust `enum BoundaryStrategy { Whitespace, Sentence, Paragraph }` is genuine Strategy. The Python `BoundaryStrategy` ABC becomes an Adapter to the Rust enum. Tokenizer stays Python (HF dep). |
| M4 multi-layer RRF | **Facade** | One method (`mem_read_multi_layer`) hides N parallel queries + RRF fusion behind a simple interface. Originally claimed Composite — that's about uniform tree structures, which doesn't fit. Facade fits. |
| M5 `salience_mode` | **Strategy** (as a parameter) | `"l2" | "max" | "none"` is a Strategy enum, applied per call. |
| M6 HTTP fallback removal | *Not a pattern* — **redundant code elimination** | The Python fallback was dead defensiveness; removing it isn't a pattern, just cleanup. |
| M7 fingerprint cache | **LRU Cache** + **Observer-style lifecycle hook** | LRU eviction is a real cache discipline. `request_finished(id)` callback into the engine is Observer-shaped (vLLM publishes lifecycle events; engine subscribes). |
| Backward compatibility wrappers | **Adapter** + (deprecation warnings are a convention, not a pattern) | Old API shape adapts to new internal call. Honest framing. |

---

## Implementation Phases

Order: **P1 H5 → P2 H1 → P3 M1 → P4 M5 → P5 M3 → P6 M6 → P7 H3 → P8 H4 → P9 M7 → P10 M2 → P11 M4 → P12 H2.**

Two deviations from the audit's recommended ordering: M5 placed right after M1 (keeps numerics changes contiguous after H1); M6 placed after M3 (cleanup ships with low risk before the vLLM block).

Every phase has four sections: **Builds**, **Gates** (ATs from above that must pass), **Patterns**, **Bench gate**, and **Observability checks** (the call-count / AST / `tracemalloc` assertions that were wrongly in the AT list before — they validate the optimisation, but are not behavioural ATs).

### Phase 1 — `list_packs` text-fetch split (split into 1a + 1b)

**Discovery from execution (2026-05-20).** The audit framed Phase 1 as a PyO3-only fix: skip the per-pack `pack_text` call inside `list_packs` and reach a 3× speedup at 10K packs. Both assumptions turned out to be wrong:

1. *The Rust engine already returns metadata-only* (`engine.rs:1921`). The text fetch lived only in the PyO3 marshalling loop. So the surface-level fix is bridge-only — no new Rust engine method needed.
2. *The text fetch isn't the dominant cost.* Skipping all 10K `pack_text` calls saves ~17ms out of an 80ms call (1.27× speedup). The actual bottleneck is one level deeper: `Engine::list_packs` calls `pool.get(retrieval_cell_id)` per pack (`engine.rs:1930`, dispatching to `crates/tdb-storage/src/block_pool.rs:152`), which does a HashMap index lookup, a linear scan over segments, and **a full Q4 cell decompression** — all to read the `owner` u64 from the retrieval cell. Ten thousand full pack reads to answer a metadata query.
3. *Changing the PyO3 return shape doesn't help.* Switching from list-of-dicts to dict-of-numpy-arrays gave the same 1.30×. The dict allocation was a real but minor cost; the storage decompression dominates.

Phase 1 therefore splits into two sub-phases. The first ships the bridge cleanup (small perf, real API improvement, deprecation pathway). The second adds the storage-layer index that actually unlocks the speedup. They merge independently and the headline 8× target now sits on 1b, not 1a.

#### Phase 1a — PyO3 columnar metadata path (in flight)
- **Builds:** PyO3 `list_packs_metadata(owner)` returning `{"pack_ids": np.uint64[N], "owners": np.uint64[N], "tiers": np.uint8[N], "importances": np.float32[N]}` (parallel columns, no per-row dicts). PyO3 `list_packs(owner, fetch_text=True|None)` keeps the legacy list-of-dicts shape; omitted kwarg emits a `DeprecationWarning` matching the `sweep.py` convention. Update callers: `consolidator.py` (two sites, migrate to columnar), `client.pack_count` (columnar), `client.list_packs` / `prefix_builder.py` / `mcp/server.py` (pass `fetch_text=True` explicitly — they genuinely need text). **No Rust engine change**; PyO3 only.
- **Gates:** Phase 1a ATs (empty owner, four aligned arrays, importance-descending order, fetch_text=True returns text, deprecation warning fires at least once, concurrent reader safety, 10K-pack consolidate latency under `@pytest.mark.slow`).
- **Patterns:** Interface Segregation (metadata vs full); Adapter (Python deprecation wrapper).
- **Bench gate:** `experiments/phase1-list-packs-microbench.py` at 10K packs. **Pass:** ≥ 1.2× speedup. Documents the lower-than-promised wall-clock reduction and the discovery that the dominant cost is storage-layer, not PyO3.
- **Observability:** the latency AT replaces the original call-count check with a behavioural budget (`consolidate(target) < 500ms at 10K packs`).

#### Phase 1b — `PackDirectory` owner index (storage-layer)
- **Builds:** add `pack_owners: HashMap<PackId, OwnerId>` to `crates/tdb-engine/src/pack_directory.rs` (or equivalent persistent location). Maintained at write time: every `mem_write_pack` / `mem_write_pack_tokens` / `mem_write_batch_packs` inserts; every `delete_pack` removes. Rebuilt from segment scan on engine open (no snapshot-format change required if we rebuild rather than persist; document the recovery contract either way per CLAUDE.md's reliability rules). `Engine::list_packs` becomes a pure in-memory iteration: walk `pack_directory.pack_ids()`, read owner from `pack_owners`, read `(tier, importance)` from governance, sort by importance. Zero `pool.get` calls.
- **Gates (new ATs):**
  1. `it_returns_correct_owner_for_each_pack_after_writes`
  2. `it_excludes_deleted_packs_from_owner_index`
  3. `it_rebuilds_owner_index_after_engine_reopen_without_snapshot` — durability/recovery rule
  4. `it_handles_owner_index_under_concurrent_write_and_list`
  5. `it_returns_empty_metadata_when_directory_is_empty`
  6. `it_preserves_sort_order_through_owner_index_path`
- **Patterns:** **in-memory secondary index** (the same shape as `pack_directory.pack_ids()` already has — extends an existing pattern). No new structural pattern; this is just adding a column to an existing index.
- **Bench gate:** re-run `experiments/phase1-list-packs-microbench.py`. **Pass:** ≥ 8× speedup at 10K packs (the original Phase 1 target, now technically achievable).
- **Reliability rule (CLAUDE.md mandatory):** "Recovery contract required" — the owner index is derived state. Document that it is rebuilt from segment scan on `Engine::open`, that crashes leave the segment trail authoritative, and add a recovery AT (`it_rebuilds_owner_index_after_simulated_crash`).
- **Snapshot/restore:** if the index is *not* persisted, snapshot/restore is unchanged. If we choose to persist it (faster open at scale), the snapshot format version bumps and we need a regression AT (`it_restores_pre_1b_snapshot_correctly`). Default: don't persist; rebuild on open. Revisit if open latency becomes the next bottleneck.

**Storage-layer coordination.** Phase 1b is the first phase in this plan that genuinely modifies a storage-layer crate (`tdb-engine`'s pack directory). It does not change `tdb-storage`'s segment format or `BlockPool` API. Phase 3 (batch writes) is the only other phase that interacts with `pool.append_batch`; 1b must land before 3 because 3's write path also needs to maintain the owner index. The write-side phases that come after (Phase 2 raw-token writes, Phase 4 salience_mode) inherit the index automatically by routing through the existing write path.

### Phase 2 — H1 write-side raw-token API
- **Builds:** `Engine::mem_write_pack_tokens(owner, token_matrix_flat, n_tokens, dim, layers, salience, salience_mode, text) -> PackId` in `tdb-engine`; PyO3 binding shaped after `mem_read_tokens` (`crates/tdb-python/src/lib.rs:219-266`). Header construction moves into Rust.
- **Gates:** ATs 2.1–2.8.
- **Patterns:** API symmetry with `mem_read_tokens` (no GoF label claimed).
- **Bench gate:** new `bench_mem_write_pack_raw_tokens` group in `crates/tdb-engine/benches/engine.rs` vs `bench_mem_write_pack`. **Pass:** ≥ 25 % wall-time reduction at `(seq_len=256, dim=1024)`; numbers archived under `docs/perf/`.
- **Observability checks:** AST grep confirms the public Python store paths (`client.store`, `kp_injector.store`) no longer call `encoding.encode_per_token` once consumers migrate.

### Phase 3 — M1 batch writes
- **Builds:** `Engine::mem_write_batch_packs(&[KVPack]) -> Vec<PackId>` calling `pool.append_batch` once and updating governance/text-store/deletion-log in one critical section. PyO3 `mem_write_batch_packs(list[dict|tuple]) -> list[int]`.
- **Gates:** ATs 3.1–3.8.
- **Patterns:** Unit of Work; `///` docs contrast with streaming `WriteBuffer`.
- **Bench gate:** new `bench_mem_write_batch_packs_50` group vs 50 sequential `mem_write_pack`. **Pass:** ≥ 10× on rotational-disk-simulated fsync; ≥ 2× on SSD.
- **Observability checks:** instrumented `BlockPool` records exactly one `append_batch` invocation per `mem_write_batch_packs([…50])`.

### Phase 4 — M5 `salience_mode`
- **Builds:** `salience_mode: Option<&str>` parameter on `mem_write_pack` and `mem_write_pack_tokens` (`"l2" | "max" | "none"`). Engine computes salience from `retrieval_key` when set; preserves caller's explicit `salience` when `None`.
- **Gates:** ATs 4.1–4.5.
- **Patterns:** Strategy parameter.
- **Bench gate:** `python/tdb_bench/salience_microbench.py` times one `hf_kv_hook.on_generate` before/after. **Pass:** ≥ 5 % wall-time reduction on Qwen3-0.6B (40 layers).
- **Observability checks:** AST grep confirms `hf_kv_hook.py` no longer imports `numpy` purely for `np.linalg.norm` in the salience path.

### Phase 5 — M3 chunker boundary search in Rust
- **Builds:** `Engine::find_chunk_boundary(text: &str, max_pos: usize, strategy: &str) -> usize` in **tdb-engine** (PyO3 host; the chunker has no retrieval logic, so tdb-retrieval is the wrong crate). Strategy enum `{Whitespace, Sentence, Paragraph}`. Python `BoundaryStrategy` becomes an Adapter; `TextChunker` keeps its HF tokenizer dependency.
- **Gates:** ATs 5.1–5.8.
- **Patterns:** Strategy enum (Rust), Adapter (Python).
- **Bench gate:** new `bench_find_chunk_boundary` group in `crates/tdb-engine/benches/` on a 10 MB corpus. **Pass:** ≥ 20× speedup over Python's `str.rfind` loop.

### Phase 6 — M6 HTTP `pack_text` fallback removal
- **Builds:** delete fallback at `python/tardigrade_http/server.py:145`. Contract: `mem_read_pack` guarantees `text` in result.
- **Gates:** ATs 6.1–6.4.
- **Patterns:** none (cleanup).
- **Bench gate:** `python/tdb_bench/http_microbench.py` measures `/mem/query` p95 before/after. **Pass:** ≥ 10 % p95 reduction at 100 packs.

### Phase 7 — H3 vLLM retrieval-key compute in Rust
- **Builds:** PyO3 `engine.compute_retrieval_key(token_ids, strategy_name) -> Optional[np.ndarray]`. **Embedding table lives on `Engine`** (stateful engine ownership; not on the stateless retrieval crate), loaded via `engine.load_embedding_table(path | weights)`. Python `get_strategy()` factory becomes a thin wrapper.
- **Gates:** ATs 7.1–7.7.
- **Patterns:** Strategy (extended trait), engine-owned state with lazy load.
- **Bench gate:** new `bench_retrieval_key_compute` group. **Pass:** ≥ 30× speedup vs Python at prompt length 1024, embedding size 1024.
- **Observability checks:** load-counter on the embedding table records exactly one load across 1 000 `get_num_new_matched_tokens` calls.

### Phase 8 — H4 vLLM KV-block reshape in Rust
- **Builds:** `Engine::reshape_kv_paged(flat, num_kv_heads, head_dim, block_size, direction, seq_len) -> Vec<f32>`. Stateless function on `Engine`.
- **Gates:** ATs 8.1–8.6.
- **Patterns:** stateless function (no pattern claimed).
- **Bench gate:** new `bench_kv_reshape` group on Qwen3-0.6B dims. **Pass:** ≥ 15× per-layer speedup over numpy.

### Phase 9 — M7 vLLM fingerprint cache to engine
- **Builds:** move `_pack_id_by_fingerprint` (`python/tardigrade_vllm/connector.py:198`) into engine as LRU `HashMap<u64, PackId>` capped at `max_num_seqs`. APIs: `fingerprint_get`, `fingerprint_put`, `fingerprint_release(req_id)`. Connector calls `fingerprint_release` from `request_finished`.
- **Gates:** ATs 9.1–9.6.
- **Patterns:** LRU Cache + Observer-style lifecycle hook (engine subscribes to `request_finished` events published by vLLM connector).
- **Bench gate:** sanity microbench in `python/tdb_bench/fingerprint_microbench.py` — **Pass:** no regression vs Python OrderedDict.

### Phase 10 — M2 batch reads with refinement
- **Builds:** `engine.mem_read_pack_batch(queries, k: int | list, owner: int | list | None, refinement: bool) -> list[list[dict]]`. Refinement corpus statistics computed once per batch and reused across the batch's queries.
- **Gates:** ATs 10.1–10.5.
- **Patterns:** Memoization (corpus stats reused within batch).
- **Bench gate:** new `bench_mem_read_pack_batch_16` group vs 16 sequential `mem_read_pack`. **Pass:** ≥ 2× with PRF refinement; ≥ 1.3× without.
- **Observability checks:** instrumented refinement strategy records exactly one `prepare_corpus` call per batch.

### Phase 11 — M4 multi-layer RRF in Rust
- **Builds:** `engine.mem_read_multi_layer(query_keys_per_layer, layer_ratios, k, rrf_k, owner) -> list[dict]`. Python `MultiLayerQuery` thins down to model-forward → per-layer hidden states → one engine call.
- **Gates:** ATs 11.1–11.6.
- **Patterns:** Facade (one method hides N queries + RRF fusion).
- **Bench gate:** new `bench_multi_layer_rrf_3_layers` group at 5 K packs. **Pass:** ≥ 5× over Python `rrf_fuse`.

### Phase 12 — H2 tch-rs torch tensor bindings (feature-gated)
- **Builds:** Cargo feature `torch-bridge = ["dep:tch"]` on `tdb-python`. Under the feature: `mem_write_pack_tokens_torch(tensor, …)`, `extract_kv_payload_torch(...)`. tch-rs detaches, moves to CPU, dtype-converts, flattens — no numpy intermediary.
- **Gates:** ATs 12.1–12.6.
- **Patterns:** Adapter.
- **Bench gate:** new `bench_torch_to_pack` group at `(seq_len=256, dim=1024, layers=40)` comparing:
  1. baseline `tensor → numpy → mem_write_pack`,
  2. Phase 2 path `tensor → numpy → mem_write_pack_tokens`,
  3. Phase 12 path `tensor → mem_write_pack_tokens_torch`.
  **Pass:** (3) ≥ 1.5× over (2); (3) ≥ 3× over (1).
- **Observability checks:** `tracemalloc`-style hook confirms zero `numpy.ndarray.__array__` invocations on the torch path.

---

## SOLID Audit

- **S — Single Responsibility:** each new Rust function owns one concern (write-tokens, batch-write, reshape, RRF, chunk-boundary). Tradeoff: `mem_write_pack` and `mem_write_pack_tokens` both write packs; the legacy entry internally calls the new one during deprecation. Conscious.
- **O — Open/Closed:** Strategy enums (`BoundaryStrategy`, `RetrievalKeyStrategy`, `SalienceMode`) are additive. The `torch-bridge` Cargo feature extends `tdb-python` without modifying its core.
- **L — Liskov:** new Rust `RetrievalKeyStrategy` impls satisfy the Python ABC contract — same dim out for same dim in, same `None` on empty input (AT 7.4).
- **I — Interface Segregation:** H5 is the canonical case — callers that don't need text don't pay for it. M2 applies the same shape to refinement (opt-in per batch).
- **D — Dependency Inversion:** Engine consumers depend on traits, not concretions. Tradeoff: tch-rs introduces a concrete `tch::Tensor` dependency; mitigated by the feature gate so the default build stays dep-free.

---

## Gap Review

**Empty input:** H1 zero-token (AT 2.3); M1 empty batch (3.3); H3 empty token_ids (7.4); M3 empty text (5.6); M2 empty queries (10.2); M4 zero layer ratios (11.6).

**Boundary cases:** H1 single-token (2.5); M1 single-pack (3.2); H4 `seq_len=0`, `block_size=1`, non-multiple (8.2–8.4); M3 `max_pos=0`, `max_pos > text.len()` (5.6, 5.7); M4 single layer (11.3).

**Error paths:** H1 dim mismatch (2.2); H1 empty (2.3); M1 atomicity (3.4); M5 unknown mode (4.4); M3 unknown strategy (5.8); H3 unloaded table (7.6); M4 invalid ratios (11.5); H2 feature-disabled symbol (12.1).

**Concurrent access:** H1 concurrent writers (2.6); M7 concurrent inserts (9.4); H5 metadata under writes (1.8). All ride the existing `Arc<Mutex<>>` PyO3 Engine wrapper.

**State machine transitions:** M1 batch packs land in `Draft` tier (3.6); M7 fingerprint released even on error path (9.3).

**Hostile callers:** H1 non-contiguous view (2.4); H4 mismatched `seq_len` (8.5); H2 grad-tracking tensor (12.6); M1 one-million-pack batch (3.8).

**Ordering assumptions:** M1 ids in input order (3.1); M2 results in input-query order (10.1); M4 tie-break by `pack_id` (11.4); H5 metadata sort preserved (1.7).

**Backward compatibility:** Phase 1 emits deprecation warning (1.5); Phase 2 keeps `mem_write_pack` working with snapshot regression guard (2.8); Phase 4 keeps positional `salience`; Phase 6 keeps old client pattern (6.4); Phase 12 keeps numpy path (12.5).

**Durability/Recovery (CLAUDE.md canonical rules):** M1 crash-mid-fsync recovery is mandatory (AT 3.7 — `it_recovers_full_batch_or_none_after_crash_during_fsync`). All other phases note "does not touch durability contract" in PR description.

---

## Critical Files

- `/home/flagrare/Dev/ares-project/tardigrade-db/crates/tdb-python/src/lib.rs` — PyO3 spine; every phase touches this
- `/home/flagrare/Dev/ares-project/tardigrade-db/crates/tdb-engine/src/engine.rs` — Engine internal API
- `/home/flagrare/Dev/ares-project/tardigrade-db/crates/tdb-engine/benches/engine.rs` — Criterion gates
- `/home/flagrare/Dev/ares-project/tardigrade-db/crates/tdb-retrieval/src/lib.rs` — `RetrievalKeyStrategy` trait (extended in Phase 7)
- `/home/flagrare/Dev/ares-project/tardigrade-db/python/tardigrade_hooks/encoding.py` — stubbed in Phase 2
- `/home/flagrare/Dev/ares-project/tardigrade-db/python/tardigrade_hooks/hf_kv_hook.py` — Phases 2, 4, 12
- `/home/flagrare/Dev/ares-project/tardigrade-db/python/tardigrade_hooks/consolidator.py` — Phase 1 caller
- `/home/flagrare/Dev/ares-project/tardigrade-db/python/tardigrade_hooks/chunker.py` — Phase 5 Adapter
- `/home/flagrare/Dev/ares-project/tardigrade-db/python/tardigrade_hooks/multi_layer_query.py` — Phase 11
- `/home/flagrare/Dev/ares-project/tardigrade-db/python/tardigrade_http/server.py` — Phase 6
- `/home/flagrare/Dev/ares-project/tardigrade-db/python/tardigrade_vllm/connector.py` — Phases 7, 8, 9
- `/home/flagrare/Dev/ares-project/tardigrade-db/python/tardigrade_vllm/retrieval_key.py` — Phase 7
- `/home/flagrare/Dev/ares-project/tardigrade-db/python/tardigrade_vllm/format.py` — Phase 8
- New: `python/tdb_bench/` (Python microbench harnesses), `tests/python/observability/` (call-count / AST / tracemalloc checks — kept separate from ATs)

---

## Existing Patterns to Reuse (Do Not Re-Invent)

- **`mem_read_tokens`** (`crates/tdb-python/src/lib.rs:219-266`) — exact symmetry target for Phase 2.
- **`Arc<Mutex<>>` Monitor Object** wrapping Engine — gives every new API thread safety for free.
- **`CheckpointRepository`** (`lib.rs:1137-1218`) — Repository pattern reference for any persistence API.
- **Streaming `WriteBuffer`** (`Engine::open_with_write_buffer`) — Phase 3 is complementary, not a replacement. `///` docs must contrast.
- **`RetrievalKeyStrategy` trait** in `tdb-retrieval` — Phase 7 extends, doesn't replace.
- **Existing Criterion benches** in `crates/tdb-{retrieval,index,engine}/benches/` — extend; do not create new harnesses.
- **`miette` typed errors** (commit `5b0e230`) — every new error variant gets a `tdb::<area>::<name>` code + `#[help]`.

---

## Verification (end-to-end)

For each phase:

1. **Write the ATs first**, observe them fail. (`pytest tests/python/test_<phase>_*.py -v` and `cargo test -p <crate> --test <phase>_acceptance`.) A test that passes on first run is a smell.
2. **Implement until ATs pass.** Run `cargo test --workspace --exclude tdb-python` and `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop && pytest tests/python/ -m "not gpu"`.
3. **Bench gate:** `cargo bench -p <crate> -- --save-baseline <phase>-pre` before, rerun with `--baseline <phase>-pre` after; Python microbenches in `python/tdb_bench/` write JSON to `docs/perf/<phase>.json`. Phase's pass criterion must hold; if it does not, redesign or document the regression in the PR.
4. **Observability checks** (separate from ATs): the call-count / AST / `tracemalloc` files in `tests/python/observability/` confirm the optimisation actually changes the implementation path. These are *not* gates on correctness — they're regression guards on the optimisation itself.
5. **Refactor pass** — naming, duplication, SOLID adherence, `///` docs on every new public Rust API per CLAUDE.md.
6. **Gap review** — walk the gap-review list above for the phase; add ATs for anything uncovered.
7. **Update CLAUDE.md status block** with the new tag (`v0.4.2`, `v0.4.3`, …) and a one-line foundation-completion bullet.

GPU-gated ATs (12.3) run only in CI with CUDA; locally use `pytest -m gpu` (skipped when CUDA absent).

End-to-end smoke after Phases 1–3 (the read/write asymmetry close): `python examples/e2e_demo.py` and `python examples/multi_agent_demo.py` — both must complete without deprecation warnings firing for the new APIs.

---

## Refactor Pass Reminder

No phase is complete until naming, duplication, structure, and SOLID adherence have been reviewed. The mandatory gap-review and refactor passes from CLAUDE.md apply to every phase, not just the last one.
