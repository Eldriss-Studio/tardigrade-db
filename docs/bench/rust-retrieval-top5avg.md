# Rust Retrieval Top5Avg Benchmark

**Date:** April 24, 2026
**Scope:** Rust-only validation of the current encoded per-token retrieval path.

## What Was Tested

This benchmark covers the current Rust path:

- encoded per-token retrieval keys
- `PerTokenRetriever(Top5Avg)`
- `Engine::mem_read`
- `Engine::mem_read_pack`
- Vamana activation after encoded writes

No Python retrieval code, RAG baseline, or production architecture pivot is part of this benchmark.

## Adapter Contract Added

The Rust path now has a single retrieval-key adapter/value-object boundary:

- raw encoded per-token matrices go only to `PerTokenRetriever`
- SLB, brute-force, and Vamana receive fixed-dimension pooled vectors
- plain vectors stay plain for fixed-dimension stages
- malformed sentinel-looking keys are rejected instead of treated as normal vectors
- Q4-rounded metadata can still recover token count from data length

This is the clean-code fix for the earlier boundary bug: key-shape decisions are no longer spread across engine, brute-force, and per-token retrieval code.

## ATDD Coverage Added

The Rust acceptance tests now lock these behaviors:

- engine default retrieval uses `Top5Avg`
- reopening the engine preserves `Top5Avg` behavior
- `mem_read_pack` uses the per-token pipeline before SLB fallback
- `mem_read_pack` deduplicates layer cells by pack
- Vamana handles encoded per-token keys after activation
- per-token decoding tolerates Q4 rounding that crushes token-count metadata
- encoded-key adapter exposes raw tokens only to per-token retrieval
- encoded-key adapter exposes pooled vectors to fixed-dimension retrievers
- malformed encoded keys do not flow into fixed-dimension retrievers as raw vectors
- Vamana threshold crossing does not change encoded ranking
- correctness metric helpers catch recall and gravity-well regressions
- pack-read phase profile labels include layer count, payload dimension, and
  indexed cells per pack
- zero-layer packs remain retrievable through `mem_read_pack`
- pack payload dimensions round-trip through reconstruction
- direct `BlockPool::get` hydration fixtures preserve cell IDs and payload
  dimensions

## Bug Found

The benchmark path exposed a real boundary bug:

```text
encoded per-token key length: 1088
Vamana expected dimension:    128
```

The raw encoded key was leaking into Vamana, which is a fixed-dimension retrieval stage. The correct behavior is:

```text
PerTokenRetriever receives raw encoded token matrices.
SLB, Vamana, and brute-force receive mean-pooled vectors.
```

The fix mean-pools encoded keys before Vamana insertion/query and makes encoded-key decoding infer token count from data length when Q4 quantization rounds the metadata count to zero.

## Benchmark Results

Measured with:

```bash
RUSTFLAGS="-C target-cpu=native" PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo bench -p tdb-retrieval
RUSTFLAGS="-C target-cpu=native" PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo bench -p tdb-engine
```

### Adapter Baseline

Before candidate reduction, the exact per-token path had these 10K timings:

| Path | Size | Correctness label | Latency |
| --- | ---: | --- | ---: |
| `PerTokenRetriever(Top5Avg)` | 10,000 cells | R@1 100%, R@5 100%, worst top-1 1 | ~15.95 ms |
| `Engine::mem_read` encoded path | 10,000 cells | R@1 100%, R@5 100%, worst top-1 1, Vamana unchanged | ~16.22 ms |
| `Engine::mem_read_pack` encoded path | 10,000 packs | latency only | ~32.7 ms |

### Candidate Reduction

Candidate reduction keeps exact scoring for small corpora and switches to pooled latent
candidate selection above 512 cells. At `k = 5`, the candidate limit is 320 cells.
Those candidates are reranked with the existing `Top5Avg` scorer.

| Path | Size | Correctness label | Latency |
| --- | ---: | --- | ---: |
| `PerTokenRetriever(Top5Avg)` | 100 cells | R@1 100%, R@5 100%, worst top-1 1, 100 candidates | ~136 us |
| `PerTokenRetriever(Top5Avg)` | 1,000 cells | R@1 100%, R@5 100%, worst top-1 1, 320 candidates | ~547 us |
| `PerTokenRetriever(Top5Avg)` | 10,000 cells | R@1 100%, R@5 100%, worst top-1 1, 320 candidates | ~1.75 ms |
| `Engine::mem_read` encoded path | 100 cells | R@1 100%, R@5 100%, worst top-1 1, Vamana unchanged, 100 candidates | ~249 us |
| `Engine::mem_read` encoded path | 1,000 cells | R@1 100%, R@5 100%, worst top-1 1, Vamana unchanged, 320 candidates | ~1.82 ms |
| `Engine::mem_read` encoded path | 10,000 cells | R@1 100%, R@5 100%, worst top-1 1, Vamana unchanged, 320 candidates | ~3.90 ms |
| `Engine::mem_read_pack` encoded path | 100 packs | pack dedup preserved | ~399 us |
| `Engine::mem_read_pack` encoded path | 1,000 packs | pack dedup preserved | ~3.71 ms |
| `Engine::mem_read_pack` encoded path | 10,000 packs | pack dedup preserved | ~8.84 ms |

### Pack Reverse Index

The next pack-read hardening step replaced the linear `cell_id -> pack_id`
lookup with a private `PackDirectory` value object. The directory owns both
`pack_id -> cell_ids` and `cell_id -> pack_id`, and it is rebuilt from persisted
cells on `Engine::open`.

Measured with:

```bash
RUSTFLAGS="-C target-cpu=native" PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo bench -p tdb-engine "mem_read_pack"
```

| Path | Size | Layers per pack | Correctness label | Latency |
| --- | ---: | ---: | --- | ---: |
| `Engine::mem_read_pack` encoded path | 100 packs | 1 | target true, dedup true | ~371 us |
| `Engine::mem_read_pack` encoded path | 100 packs | 4 | target true, dedup true | ~1.00 ms |
| `Engine::mem_read_pack` encoded path | 1,000 packs | 1 | target true, dedup true | ~3.47 ms |
| `Engine::mem_read_pack` encoded path | 1,000 packs | 4 | target true, dedup true | ~4.81 ms |
| `Engine::mem_read_pack` encoded path | 10,000 packs | 1 | target true, dedup true | ~8.39 ms |
| `Engine::mem_read_pack` encoded path | 10,000 packs | 4 | target true, dedup true | ~17.06 ms |

## Validation

The latest Rust profile checks passed:

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo nextest run -p tdb-engine -p tdb-storage
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo clippy -p tdb-engine -p tdb-storage --all-targets -- -D warnings
cargo fmt --all -- --check
```

Result:

```text
82 tests run: 82 passed (1 leaky), 1 skipped
clippy passed
format check passed
```

## Conclusion

The retrieval-key contract is centralized and locked with ATDD. Correctness is clean on the deterministic Rust fixture: 100% recall@1/@5, no gravity well, and no ranking change when Vamana activates.

Candidate reduction is now the default encoded per-token path above 512 cells. It is still latent-only: no RAG, BM25, lexical index, or public API change. The 10K path is no longer dominated by scoring every stored token: `PerTokenRetriever` dropped from ~15.95 ms to ~1.75 ms, `Engine::mem_read` dropped from ~16.22 ms to ~3.90 ms, and `mem_read_pack` dropped from ~32.7 ms to ~8.84 ms.

The remaining problem is pack overhead and mid-size engine overhead. Criterion showed the 1K `mem_read_pack` benchmark regressed while 10K improved sharply, which means candidate reduction solved the large linear scan but pack reconstruction still needs its own profiling pass.

The reverse index fixed an obvious clean-code issue and gave a modest latency
improvement for 1-layer packs: 1K improved from ~3.71 ms to ~3.47 ms, and 10K
improved from ~8.84 ms to ~8.39 ms. It did not eliminate the pack-read overhead.
The 4-layer benchmark shows that pack reconstruction and duplicate pack-cell
retrieval are now the likely bottlenecks.

### Pack Key-Only Indexing

Pack key-only indexing keeps all pack cells persisted but indexes only the pack
retrieval cell. Layer payload cells remain available for reconstruction through
the pack directory, but they no longer create duplicate retrieval candidates.

Measured with:

```bash
RUSTFLAGS="-C target-cpu=native" PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo bench -p tdb-engine "mem_read_pack"
```

| Path | Size | Layers per pack | Correctness label | Latency |
| --- | ---: | ---: | --- | ---: |
| `Engine::mem_read_pack` encoded path | 100 packs | 1 | target true, dedup true, indexed 1 | ~237 us |
| `Engine::mem_read_pack` encoded path | 100 packs | 4 | target true, dedup true, indexed 1 | ~533 us |
| `Engine::mem_read_pack` encoded path | 1,000 packs | 1 | target true, dedup true, indexed 1 | ~1.64 ms |
| `Engine::mem_read_pack` encoded path | 1,000 packs | 4 | target true, dedup true, indexed 1 | ~2.06 ms |
| `Engine::mem_read_pack` encoded path | 10,000 packs | 1 | target true, dedup true, indexed 1 | ~6.11 ms |
| `Engine::mem_read_pack` encoded path | 10,000 packs | 4 | target true, dedup true, indexed 1 | ~6.60 ms |

Key-only indexing confirms duplicate pack-cell retrieval was a real bottleneck:
10K 4-layer reads improved from ~17.06 ms to ~6.60 ms. The remaining latency is
now much less sensitive to layer count, which means the next pack-read plan
should profile residual retrieval cost and storage hydration instead of dedup.

### Pack Read Phase Profile

The phase profile compares three costs without changing production behavior:

- direct retrieval-cell reads through `Engine::mem_read`
- complete pack reads through `Engine::mem_read_pack`
- direct payload hydration through `BlockPool::get`

Measured with:

```bash
RUSTFLAGS="-C target-cpu=native" PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo bench -p tdb-engine "mem_read_pack"
RUSTFLAGS="-C target-cpu=native" PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo bench -p tdb-storage "BlockPool get"
```

All `mem_read_pack` labels reported `target true`, `dedup true`, and `indexed 1`.

| Path | Size | Layers | Payload dim | Latency |
| --- | ---: | ---: | ---: | ---: |
| `Engine::mem_read_pack` | 100 packs | 0 | 128 | ~135 us |
| `Engine::mem_read_pack` | 100 packs | 1 | 128 | ~242 us |
| `Engine::mem_read_pack` | 100 packs | 4 | 128 | ~535 us |
| `Engine::mem_read_pack` | 100 packs | 16 | 128 | ~1.71 ms |
| `Engine::mem_read_pack` | 1,000 packs | 0 | 128 | ~1.62 ms |
| `Engine::mem_read_pack` | 1,000 packs | 1 | 128 | ~1.73 ms |
| `Engine::mem_read_pack` | 1,000 packs | 4 | 128 | ~2.03 ms |
| `Engine::mem_read_pack` | 1,000 packs | 16 | 128 | ~3.19 ms |
| `Engine::mem_read_pack` | 10,000 packs | 0 | 128 | ~6.16 ms |
| `Engine::mem_read_pack` | 10,000 packs | 1 | 128 | ~6.51 ms |
| `Engine::mem_read_pack` | 10,000 packs | 4 | 128 | ~6.62 ms |
| `Engine::mem_read_pack` | 10,000 packs | 16 | 128 | ~7.96 ms |

Payload dimension did not materially change the result in this profile. At
10K packs, 4-layer reads were ~6.62 ms at payload dim 128 and ~6.75 ms at
payload dim 256.

The retrieval-cell-only profile used `Engine::mem_read` against the same pack
fixtures:

| Path | Size | Latency |
| --- | ---: | ---: |
| `Engine::mem_read` on pack retrieval cells | 100 packs | ~232 us |
| `Engine::mem_read` on pack retrieval cells | 1,000 packs | ~1.68 ms |
| `Engine::mem_read` on pack retrieval cells | 10,000 packs | ~4.0 ms |

This is not a perfect subtraction model because `mem_read` and `mem_read_pack`
do different result handling, governance updates, and reconstruction work. It
does show that direct encoded retrieval-cell reads are still a large part of the
10K budget.

Direct storage hydration was much cheaper:

| Path | Cell count | Payload dim | Latency |
| --- | ---: | ---: | ---: |
| `BlockPool::get` | 100 cells | 128 | ~18.6 us |
| `BlockPool::get` | 100 cells | 256 | ~18.9 us |
| `BlockPool::get` | 1,000 cells | 128 | ~18.8 us |
| `BlockPool::get` | 1,000 cells | 256 | ~18.9 us |
| `BlockPool::get` | 10,000 cells | 128 | ~18.9 us |
| `BlockPool::get` | 10,000 cells | 256 | ~19.0 us |

#### Phase Profile Conclusion

The next bottleneck is not raw `BlockPool::get`. A single payload hydration is
only ~19 us, and doubling payload dimension from 128 to 256 barely moves either
the storage benchmark or the pack benchmark.

At 10K packs, zero-layer `mem_read_pack` is already ~6.1 ms. A 4-layer pack is
only about half a millisecond slower, while direct retrieval-cell `mem_read` is
about ~4.0 ms. That means the remaining 10K latency is mostly retrieval/base
pack-read overhead, with reconstruction becoming important only at high layer
counts such as 16 layers.

The next implementation plan should therefore profile and reduce the residual
pack-read path around result materialization, governance/SLB warming, and the
`mem_read_pack` reconstruction loop. Storage hydration should not be the first
target unless a later profile uses much larger payload tensors.

### Pack Read Materialization Refactoring

The previous phase profile showed that raw `BlockPool::get` was not the 10K
bottleneck. The next step was to split `mem_read_pack`'s internal overhead into
named phases, then apply only optimizations proven safe by ATDD.

#### Architectural Change (Template Method)

`mem_read_pack` was refactored from a single monolithic loop into five private
phases following the Template Method pattern:

```text
mem_read_pack(query_key, k, owner_filter)
  │
  ├─ collect_pack_candidates    — SLB/pipeline retrieval, dedup by cell_id
  ├─ deduplicate_pack_candidates — sort by score, pack-level dedup via PackDirectory
  ├─ hydrate_pack_layers        — BlockPool::get per layer cell, sort by layer_idx
  ├─ apply_pack_access_governance — single on_access per pack (not per layer)
  └─ build_pack_read_result     — assemble PackReadResult value object
```

Value objects extracted to `pack_materialization.rs`:

- `PackCandidate` — ranked retrieval result with pack membership
- `PackAccessSnapshot` — immutable governance snapshot (tier + decay)
- `PackMaterializationCounters` — diagnostic counter record (test-only)
- `PackMaterializationPhaseProfile` — phase timing record (test-only)

Free functions:

- `keep_first_ranked_pack_candidates` — pack-level dedup (keeps first by score)
- `build_pack_read_result` — assembles final `PackReadResult` from candidate + layers + governance

#### Optimizations Applied

All four allowed optimizations from the plan were structurally achieved by the
refactoring itself:

1. **Avoid repeated governance map lookups:** `apply_pack_access_governance` does
   exactly one `get_mut` per pack. The old code did `get_mut` + `get` in the same
   loop iteration.

2. **Pre-size layer vectors:** `hydrate_pack_layers` uses
   `Vec::with_capacity(cell_ids.len() - 1)` from pack directory membership.

3. **Avoid redundant PACK_RETRIEVAL_LAYER checks:** The old code checked
   `cell.layer != PACK_RETRIEVAL_LAYER` inside the layer loop. The refactored code
   uses `.skip(PACK_RETRIEVAL_CELL_COUNT)` to structurally skip the retrieval cell.

4. **Keep sorting only where needed:** The `layers.sort_by_key(|l| l.layer_idx)`
   in `hydrate_pack_layers` is load-bearing because `PackDirectory` sorts cell_ids
   numerically, not by layer_idx. ATDD test 35 proves out-of-order layers must be
   sorted on read.

No additional code changes were justified beyond the refactoring.

#### ATDD Coverage Added

- `test_pack_materialization_profile_sums_phase_counts` — counters add up correctly
- `test_pack_candidate_dedup_keeps_first_ranked_pack` — dedup preserves first-ranked
- `test_pack_layer_hydration_preserves_layer_order` — out-of-order layers sorted
- `test_pack_materialization_updates_governance_once_per_returned_pack` — exactly one
  `on_access` per pack, not per layer or candidate

#### Benchmark Labels Enriched

Benchmark IDs now include `returned_pack_count` and `hydrated_layer_count` in
addition to the existing `target`, `dedup`, `layers`, `payload`, and `indexed`
fields. This makes the phase profile self-documenting in Criterion output.

#### Benchmark Results (Post-Refactoring)

Measured with:

```bash
RUSTFLAGS="-C target-cpu=native" PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo bench -p tdb-engine "mem_read_pack"
```

All `mem_read_pack` labels reported `target true`, `dedup true`, `indexed 1`,
and `returned 5` (except 16-layer at 100 packs).

| Path | Size | Layers | Payload dim | Latency | vs. previous |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Engine::mem_read_pack` | 100 packs | 0 | 128 | ~130 us | ~135 us |
| `Engine::mem_read_pack` | 100 packs | 4 | 128 | ~517 us | ~535 us |
| `Engine::mem_read_pack` | 100 packs | 16 | 128 | ~1.68 ms | ~1.71 ms |
| `Engine::mem_read_pack` | 1,000 packs | 0 | 128 | ~1.45 ms | ~1.62 ms |
| `Engine::mem_read_pack` | 1,000 packs | 4 | 128 | ~1.81 ms | ~2.03 ms |
| `Engine::mem_read_pack` | 1,000 packs | 16 | 128 | ~2.97 ms | ~3.19 ms |
| `Engine::mem_read_pack` | 10,000 packs | 0 | 128 | ~5.11 ms | ~6.16 ms |
| `Engine::mem_read_pack` | 10,000 packs | 4 | 128 | ~5.36 ms | ~6.62 ms |
| `Engine::mem_read_pack` | 10,000 packs | 16 | 128 | ~6.59 ms | ~7.96 ms |

The retrieval-cell-only profile (`Engine::mem_read`) confirms that layer count
does not affect retrieval cost:

| Path | Size | Latency |
| --- | ---: | ---: |
| `Engine::mem_read` on pack retrieval cells | 100 packs | ~224 us |
| `Engine::mem_read` on pack retrieval cells | 1,000 packs | ~1.57 ms |
| `Engine::mem_read` on pack retrieval cells | 10,000 packs | ~3.43 ms |

#### Phase Profile Conclusion

The refactoring improved 10K 4-layer `mem_read_pack` from ~6.62 ms to ~5.36 ms
(~19% reduction). The improvement came from eliminating governance double-lookups,
pre-sizing vectors, and structurally skipping PACK_RETRIEVAL_LAYER checks.

The residual 10K `mem_read_pack` latency is dominated by **candidate retrieval**
(the `collect_pack_candidates` phase), not by materialization, governance, or
layer hydration. Evidence:

- Zero-layer `mem_read_pack` at 10K packs: ~5.1 ms (no layer hydration at all)
- Four-layer `mem_read_pack` at 10K packs: ~5.4 ms (only ~250 us more)
- Direct `BlockPool::get`: ~19 us per cell (negligible)
- `Engine::mem_read` on the same retrieval cells: ~3.4 ms

The ~1.7 ms gap between `mem_read` (~3.4 ms) and zero-layer `mem_read_pack`
(~5.1 ms) is the pack-read overhead: pack directory lookups, candidate dedup,
and result construction. The ~3.4 ms `mem_read` cost is the encoded per-token
retrieval pipeline itself.

The next bottleneck to address is the retrieval pipeline at scale. The
materialization path is clean and does not warrant further optimization at
current payload dimensions.
