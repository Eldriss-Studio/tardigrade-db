# Warm-tier compression: the 2.66× number, measured

If you followed a link here from the README or the positioning doc, the short answer is yes — Validated and Core cells take **2.66× less disk space** than Draft cells on a like-for-like workload. At a realistic 1024-dim KV vector, that's 1372 bytes per cell on Draft (uniform Q4, what TardigradeDB has always done) versus 516 bytes per cell on Validated (uniform Q4 then zstd level 3, what this commit added). On 100 cells the savings come to 62% — 134 KB shrinks to 50 KB.

The rest of this page is the measurement methodology, the things this number doesn't promise, and how to rerun the experiment yourself.

## How it was measured

Two engine directories. The same 100 cells written to each. Same key vectors, same value vectors, same dim (1024 floats, matching Qwen3-0.6B's hidden width), same write path. The only difference is the tier each cell was tagged with on the way in — `Tier::Draft` for one engine, `Tier::Validated` for the other — which is the only thing the new codec dispatch reads. The KV vectors themselves are sin/cos with a small per-cell phase offset and a low-amplitude jitter term, so the zstd encoder is looking at activation-shaped entropy rather than the all-zero input that would let it cheat.

The number you read above isn't a microbench. It's the sum of `segment_*.tdb` file sizes in each engine directory after all 100 writes, measured by `fs::metadata().len()` — the same number `du` would give you. Everything that lives on disk counts: record framing, per-group scale factors, the new `codec_id` byte, the pos_encoding tail, all of it. Fixed-cost headers don't compress, and including them in the denominator is the honest way to count.

## What the numbers are

```
Codec footprint — 100 cells × 1024-dim KV
  Draft (uniform Q4):      137208 bytes  (133.99 KB)
  Validated (ZstdQ4):       51641 bytes  (50.43 KB)
  Saved:                    85567 bytes  (62.4%)
  Compression ratio:        2.66×

  Per-cell footprint:
    Draft:     1372 bytes
    Validated:  516 bytes
```

The acceptance test `zstd_q4_compresses_warm_tier_q4_payloads` gates at ≥1.5× as the bar we refuse to ship below; 2.66× is the measured number on this workload. CacheGen's published 3–4× target is the design ceiling that a future calibrated codec (Part A.MVP's follow-up) would aim for.

The per-cell number at this dim is bigger than the historical "~751 B/cell" line you may have seen on the positioning doc, because that figure was taken at a smaller working dim. Both numbers are honest; they just measure different shapes of cell. The compression *ratio* (2.66×) is what generalises across dims — the absolute bytes-per-cell scales with the KV width.

## What this proves and what it doesn't

It proves the codec activates correctly when a cell is written at Validated tier, that the bytes-on-disk for the codec-encoded payload are substantially smaller than the raw Q4 stream, and that the round-trip through `Q4::quantize` → zstd → segment → zstd → `Q4::dequantize` reconstructs the original cell within the same Q4 SNR tolerance the project has always held to. Those three things together are what makes the storage layer change safe to merge.

It does *not* prove the same ratio on a real-model KV corpus. Sin/cos vectors are smoother than transformer hidden states; real KV captures from Qwen3 or Llama may compress less. The follow-up validation is an `examples/e2e_demo.py`-captured corpus run through the same harness, and that's open work. It also doesn't measure the read-path latency cost of decode-on-read. zstd level 3 decode at single-core speeds is in the hundreds of MB/s — tens of microseconds per cell at this size — and the existing engine criterion benches still pass, but a dedicated p50/p99 measurement at 5K cells is a separate gate the Part A follow-up plan owns.

## Reproducing the number

The experiment is a single binary checked in under the storage crate's `examples/` directory:

```bash
cargo run -p tdb-storage --example codec_footprint --release
```

You'll need a working Rust toolchain and the project's workspace dependencies (`cargo build --workspace --exclude tdb-python` once will fetch everything). Output goes straight to stdout in the format above. The script writes into `tempfile::TempDir` paths and cleans up on exit, so it doesn't leave state behind.

If you want to tweak the corpus shape — different dim, more cells, a real-distribution input — the script lives at `crates/tdb-storage/examples/codec_footprint.rs` and is short enough to fork.

## Related work

- Plan: `~/.claude/plans/is-there-anything-dreamy-pike.md` — Part A.MVP, the umbrella plan this commit ships under.
- Originating research: [`../research/2026-05-21-production-kv-cache-learnings.md`](../research/2026-05-21-production-kv-cache-learnings.md) — the survey of LMCache, vLLM PagedAttention, SGLang RadixAttention, Mooncake, CacheGen, KVTC, and NVIDIA Dynamo that surfaced warm-tier compression as one of the top-3 actionable changes.
- Codec source: [`crates/tdb-storage/src/compression.rs`](../../crates/tdb-storage/src/compression.rs).
- Acceptance tests: [`crates/tdb-storage/tests/acceptance.rs`](../../crates/tdb-storage/tests/acceptance.rs) — `zstd_q4_validated_cell_round_trips_within_q4_tolerance`, `tier_gating_drives_codec_id_byte_on_disk`, `mixed_codec_segment_reads_back_every_cell`, `zstd_q4_compresses_warm_tier_q4_payloads`, `legacy_v1_segments_still_readable_as_uniform_q4`.
