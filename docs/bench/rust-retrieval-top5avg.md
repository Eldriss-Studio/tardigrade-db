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

| Path | Size | Correctness label | Latency |
| --- | ---: | --- | ---: |
| `PerTokenRetriever(Top5Avg)` | 100 cells | R@1 100%, R@5 100%, worst top-1 1 | ~134 us |
| `PerTokenRetriever(Top5Avg)` | 1,000 cells | R@1 100%, R@5 100%, worst top-1 1 | ~1.43 ms |
| `PerTokenRetriever(Top5Avg)` | 10,000 cells | R@1 100%, R@5 100%, worst top-1 1 | ~15.95 ms |
| `Engine::mem_read` encoded path | 100 cells | R@1 100%, R@5 100%, worst top-1 1, Vamana unchanged | ~234 us |
| `Engine::mem_read` encoded path | 1,000 cells | R@1 100%, R@5 100%, worst top-1 1, Vamana unchanged | ~1.53 ms |
| `Engine::mem_read` encoded path | 10,000 cells | R@1 100%, R@5 100%, worst top-1 1, Vamana unchanged | ~16.22 ms |
| `Engine::mem_read_pack` encoded path | 100 packs | latency only | ~372 us |
| `Engine::mem_read_pack` encoded path | 1,000 packs | latency only | ~2.97 ms |
| `Engine::mem_read_pack` encoded path | 10,000 packs | latency only | ~32.7 ms |

## Validation

The relevant Rust checks passed:

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo nextest run -p tdb-retrieval -p tdb-engine
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo clippy -p tdb-retrieval -p tdb-engine --all-targets -- -D warnings
cargo fmt --all -- --check
```

Result:

```text
89 tests passed, 3 skipped
clippy passed
format check passed
```

## Conclusion

The retrieval-key contract is now centralized and locked with ATDD. Correctness is clean on the deterministic Rust fixture: 100% recall@1/@5, no gravity well, and no ranking change when Vamana activates.

The remaining problem is speed. The current path is still linear at 10K scale, and `mem_read_pack` is the slowest path. The next Rust step is candidate reduction before full `Top5Avg` scoring, with pack reconstruction profiled separately.
