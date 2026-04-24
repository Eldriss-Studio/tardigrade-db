# Plan: Rust Retrieval Key Adapter And Benchmark Contract

## Context

The previous Rust benchmark work validated the current encoded per-token retrieval path:

- encoded per-token keys
- `PerTokenRetriever(Top5Avg)`
- `Engine::mem_read`
- `Engine::mem_read_pack`
- Vamana activation after encoded writes

That work also exposed a real boundary bug. Encoded per-token keys leaked into Vamana as raw encoded slices instead of fixed-dimension pooled vectors. The fix mean-pools encoded keys before fixed-dimension retrieval stages and makes per-token decoding tolerate Q4 metadata rounding.

The conclusion is not that the architecture is finished. The conclusion is that the retrieval-key contract needs to be centralized so this class of bug cannot recur.

## Goal

Make the Rust retrieval path harder to misuse before adding more scoring or architecture changes.

This is still not a production retrieval pivot:

- No Python changes.
- No public API changes.
- No new scorer.
- No RAG or lexical hybrid work.
- No behavior change unless an ATDD test proves the current behavior is wrong.

## Design Patterns

| Pattern | Where | Purpose |
| --- | --- | --- |
| **Adapter** | Retrieval key view | Converts one stored key representation into the shape required by each retrieval stage. |
| **Value Object** | Parsed key representation | Holds either plain vector data or encoded per-token data with validated dimensions. |
| **Specification** | Acceptance tests | Encodes the rules for which stage may receive raw tokens vs pooled vectors. |
| **Template Method** | Benchmarks | Keeps benchmark flow consistent across 100, 1K, and 10K corpora. |
| **Fixture Builder / Object Mother** | Tests and benches | Creates broad-match, spike, malformed-header, Vamana-threshold, and pack fixtures. |

## Phase 1: Centralize Retrieval Key Adaptation

Add an internal Rust abstraction, tentatively named `RetrievalKeyView` or `RetrievalKeyAdapter`.

It should provide a single place to answer:

- Is this a plain vector or encoded per-token key?
- What raw token matrix should `PerTokenRetriever` use?
- What fixed-dimension vector should SLB, Vamana, and brute-force use?
- What happens when encoded metadata has been damaged by Q4 quantization?

Acceptance tests:

- `test_encoded_key_view_exposes_raw_tokens_for_per_token_retrieval`
- `test_encoded_key_view_exposes_pooled_vector_for_fixed_dim_retrievers`
- `test_plain_key_view_uses_original_vector_for_fixed_dim_retrievers`
- `test_malformed_encoded_key_fails_or_falls_back_predictably`
- `test_q4_rounded_encoded_metadata_still_builds_pooled_vector`

Decision rule:

- If a stage requires fixed dimensions, it must obtain keys through the adapter's pooled-vector API.
- Raw encoded slices must not be passed directly to SLB, Vamana, or brute-force.

## Phase 2: Lock Engine Retrieval Contract

Add one higher-level Rust contract suite that proves the full pipeline still behaves after the adapter extraction.

Acceptance tests:

- `test_engine_encoded_query_uses_per_token_before_fixed_dim_fallbacks`
- `test_engine_vamana_threshold_preserves_encoded_ranking`
- `test_engine_reopen_preserves_encoded_ranking_after_adapter_rebuild`
- `test_mem_read_pack_preserves_pack_deduplication_after_adapter_rebuild`

Specification:

```text
encoded query
  -> raw tokens are scored by PerTokenRetriever(Top5Avg)
  -> fixed-dim fallback stages receive pooled vectors only
  -> merged results deduplicate by cell or pack
```

## Phase 3: Add Rust Correctness Benchmarks

Current Criterion benches measure latency. Add a deterministic correctness benchmark/report for the same current path.

Fixture sizes:

- 100 cells
- 1,000 cells
- 10,000 cells

Report:

- recall@1
- recall@5
- worst top-1 concentration
- whether Vamana activation changed outcomes
- latency for `PerTokenRetriever`, `Engine::mem_read`, and `Engine::mem_read_pack`

Keep model-heavy or long-running work out of CI. Helper functions should still be unit-testable.

Acceptance tests:

- `test_rust_retrieval_metrics_match_known_rankings`
- `test_top1_concentration_flags_gravity_well`
- `test_synthetic_corpus_builder_assigns_one_target_per_query`
- `test_correctness_report_detects_vamana_regression`

## Phase 4: Decide Optimization Direction

Only after the adapter contract and correctness benchmark are stable, decide whether to optimize the current path.

Possible optimization experiments:

- pre-filter candidates before full `Top5Avg`
- apply owner/layer filtering before token scoring
- SIMD batch the token-dot loop
- use pooled-key candidate generation followed by per-token rerank

Decision gate:

- Correctness must not regress.
- Engine-level recall must remain stable across 100, 1K, and 10K deterministic fixtures.
- Any optimization must have an ATDD test proving the behavior it preserves.

## Done Criteria

- Rust acceptance tests pass.
- `cargo clippy -p tdb-retrieval -p tdb-engine --all-targets -- -D warnings` passes.
- `cargo fmt --all -- --check` passes.
- Benchmarks produce comparable latency and correctness reports for 100, 1K, and 10K.
- The adapter contract is documented in Rust docs/comments.
