//! Measure on-disk footprint for warm-tier compression at corpus scale.
//!
//! Writes the same 100-cell, hidden-dim-1024 corpus into two block pools —
//! one with every cell tagged `Draft` (uniform Q4 stays in effect), one with
//! every cell tagged `Validated` (zstd-over-Q4 activates) — and prints the
//! resulting segment sizes. Single-script proof that the codec is actually
//! producing the byte savings the acceptance test claims.
//!
//! Run with: `cargo run -p tdb-storage --example codec_footprint --release`

// Reason: this is a measurement script whose entire purpose is to print numbers
// to stdout. The workspace `print_stdout = "deny"` rule applies to library and
// engine code, not single-purpose `examples/` binaries.
#![allow(clippy::print_stdout)]

use std::fs;
use std::path::Path;

use tdb_core::Tier;
use tdb_core::memory_cell::MemoryCellBuilder;
use tdb_storage::block_pool::BlockPool;

const HIDDEN_DIM: usize = 1024;
const NUM_CELLS: u64 = 100;

fn write_corpus(dir: &Path, tier: Tier) -> u64 {
    let mut pool = BlockPool::open(dir).unwrap();
    for i in 0..NUM_CELLS {
        // A real-ish KV shape: sin/cos with small jitter so zstd sees
        // typical activation entropy, not all-zero pathological input.
        let phase = i as f32 * 0.013;
        let key: Vec<f32> = (0..HIDDEN_DIM)
            .map(|j| {
                ((j as f32 * 0.001 + phase).sin() * 0.7) + (j as f32 * 0.0003 + phase).cos() * 0.1
            })
            .collect();
        let value: Vec<f32> = (0..HIDDEN_DIM)
            .map(|j| {
                ((j as f32 * 0.002 + phase).cos() * 0.5) + (j as f32 * 0.0005 + phase).sin() * 0.1
            })
            .collect();
        let cell = MemoryCellBuilder::new(i, 1, 0, key, value).tier(tier).build();
        pool.append(&cell).unwrap();
    }
    drop(pool);
    fs::read_dir(dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("tdb"))
        .map(|p| fs::metadata(&p).unwrap().len())
        .sum()
}

fn main() {
    let draft_dir = tempfile::tempdir().unwrap();
    let validated_dir = tempfile::tempdir().unwrap();

    let draft_bytes = write_corpus(draft_dir.path(), Tier::Draft);
    let validated_bytes = write_corpus(validated_dir.path(), Tier::Validated);

    let ratio = draft_bytes as f64 / validated_bytes as f64;
    let saved = draft_bytes.saturating_sub(validated_bytes);
    let saved_pct = 100.0 * saved as f64 / draft_bytes as f64;

    println!("Codec footprint — {NUM_CELLS} cells × {HIDDEN_DIM}-dim KV");
    println!(
        "  Draft (uniform Q4):  {draft_bytes:>10} bytes  ({:.2} KB)",
        draft_bytes as f64 / 1024.0
    );
    println!(
        "  Validated (ZstdQ4):  {validated_bytes:>10} bytes  ({:.2} KB)",
        validated_bytes as f64 / 1024.0
    );
    println!("  Saved:               {saved:>10} bytes  ({saved_pct:.1}%)");
    println!("  Compression ratio:   {ratio:.2}×");
    println!();
    println!("  Per-cell footprint:");
    println!("    Draft:     {} bytes", draft_bytes / NUM_CELLS);
    println!("    Validated: {} bytes", validated_bytes / NUM_CELLS);
}
