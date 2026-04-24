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

## ATDD Coverage Added

The Rust acceptance tests now lock these behaviors:

- engine default retrieval uses `Top5Avg`
- reopening the engine preserves `Top5Avg` behavior
- `mem_read_pack` uses the per-token pipeline before SLB fallback
- `mem_read_pack` deduplicates layer cells by pack
- Vamana handles encoded per-token keys after activation
- per-token decoding tolerates Q4 rounding that crushes token-count metadata

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

| Path | Size | Latency |
| --- | ---: | ---: |
| `PerTokenRetriever(Top5Avg)` | 100 cells | ~131 us |
| `PerTokenRetriever(Top5Avg)` | 1,000 cells | ~1.37 ms |
| `PerTokenRetriever(Top5Avg)` | 10,000 cells | ~13.85 ms |
| `Engine::mem_read` encoded path | 100 cells | ~238 us |
| `Engine::mem_read` encoded path | 1,000 cells | ~1.47 ms |
| `Engine::mem_read` encoded path | 10,000 cells | ~14.13 ms |
| `Engine::mem_read_pack` encoded path | 100 packs | ~379 us |
| `Engine::mem_read_pack` encoded path | 1,000 packs | ~2.92 ms |
| `Engine::mem_read_pack` encoded path | 10,000 packs | ~28.44 ms |

## Validation

The relevant Rust checks passed:

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo nextest run -p tdb-retrieval -p tdb-engine
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo clippy -p tdb-retrieval -p tdb-engine --all-targets -- -D warnings
cargo fmt --all -- --check
```

Result:

```text
76 tests passed, 3 skipped
clippy passed
format check passed
```

## Conclusion

The current Rust path is now correct enough to benchmark meaningfully, but it is still linear at 10K scale. The next Rust step is not a new scorer. It is to centralize the retrieval-key contract behind an adapter so raw encoded keys cannot accidentally reach fixed-dimension retrieval stages again.
