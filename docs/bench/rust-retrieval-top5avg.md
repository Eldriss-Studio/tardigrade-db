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

## Validation

The relevant Rust checks passed:

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo nextest run -p tdb-retrieval -p tdb-engine
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo clippy -p tdb-retrieval -p tdb-engine --all-targets -- -D warnings
cargo fmt --all -- --check
```

Result:

```text
97 tests passed, 3 skipped
clippy passed
format check passed
```

## Conclusion

The retrieval-key contract is centralized and locked with ATDD. Correctness is clean on the deterministic Rust fixture: 100% recall@1/@5, no gravity well, and no ranking change when Vamana activates.

Candidate reduction is now the default encoded per-token path above 512 cells. It is still latent-only: no RAG, BM25, lexical index, or public API change. The 10K path is no longer dominated by scoring every stored token: `PerTokenRetriever` dropped from ~15.95 ms to ~1.75 ms, `Engine::mem_read` dropped from ~16.22 ms to ~3.90 ms, and `mem_read_pack` dropped from ~32.7 ms to ~8.84 ms.

The remaining problem is pack overhead and mid-size engine overhead. Criterion showed the 1K `mem_read_pack` benchmark regressed while 10K improved sharply, which means candidate reduction solved the large linear scan but pack reconstruction still needs its own profiling pass.
