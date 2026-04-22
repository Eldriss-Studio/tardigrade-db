//! Basic usage example for `TardigradeDB`.
//!
//! Demonstrates the full write→read cycle: open an engine, store two KV-cache
//! tensors from different transformer layers, then retrieve the most relevant
//! one via attention-score similarity.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example basic_usage -p tdb-engine
//! ```

// Examples are binary targets — println! is intentional output, not debug noise.
#![allow(clippy::print_stdout)]

use tdb_engine::engine::Engine;

fn main() {
    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let mut engine = Engine::open(dir.path()).expect("failed to open engine");

    // Simulate capturing KV tensors from two transformer layers (dim=64).
    // In a real inference pass these would be the actual key/value activations.
    let key_layer12: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
    let val_layer12: Vec<f32> = vec![0.5f32; 64];

    let key_layer24: Vec<f32> = (0..64).map(|i| -(i as f32) / 64.0).collect();
    let val_layer24: Vec<f32> = vec![-0.5f32; 64];

    let agent_id: u64 = 42;

    let cell_a = engine
        .mem_write(agent_id, 12, &key_layer12, val_layer12, 60.0, None)
        .expect("write layer 12 failed");

    let cell_b = engine
        .mem_write(agent_id, 24, &key_layer24, val_layer24, 40.0, None)
        .expect("write layer 24 failed");

    println!("Wrote cell_a (layer 12): id={cell_a}");
    println!("Wrote cell_b (layer 24): id={cell_b}");
    println!("Total cells: {}", engine.cell_count());

    // Query with a key close to layer 12's key — expect cell_a to rank first.
    let query: Vec<f32> = (0..64).map(|i| i as f32 / 64.0 + 0.01).collect();
    let results = engine.mem_read(&query, 2, Some(agent_id)).expect("read failed");

    assert!(!results.is_empty(), "expected at least one result");
    assert_eq!(results[0].cell.id, cell_a, "expected layer-12 cell to be most relevant");

    println!("\nTop result:");
    println!("  cell_id : {}", results[0].cell.id);
    println!("  layer   : {}", results[0].cell.layer);
    println!("  score   : {:.4}", results[0].score);
    println!("  tier    : {:?}", results[0].tier);

    println!("\nDone.");
}
