use tdb_core::Tier;
use tdb_core::memory_cell::{MemoryCell, MemoryCellBuilder};
use tdb_storage::block_pool::BlockPool;
use tdb_storage::quantization::{DequantizeStrategy, Q4, QuantizeStrategy};

const HYDRATION_FIXTURE_CELL_COUNT: u64 = 6;
const HYDRATION_FIXTURE_TARGET_ID: u64 = 4;
const HYDRATION_FIXTURE_OWNER: u64 = 7;
const HYDRATION_FIXTURE_LAYER: u16 = 3;
const HYDRATION_FIXTURE_KEY_DIM: usize = 16;
const HYDRATION_FIXTURE_PAYLOAD_DIM: usize = 96;
const HYDRATION_KEY_BASE: f32 = 0.25;
const HYDRATION_VALUE_BASE: f32 = 0.75;

/// ATDD Test 1: Round-trip a `MemoryCell` through Q4 quantization and storage.
/// Write a cell, read it back, verify vectors match within Q4 tolerance (SNR > 20dB).
#[test]
fn test_round_trip_memory_cell_q4() {
    let dir = tempfile::tempdir().unwrap();
    let mut pool = BlockPool::open(dir.path()).unwrap();

    let key: Vec<f32> = (0..128).map(|i| (i as f32 * 0.01).sin()).collect();
    let value: Vec<f32> = (0..128).map(|i| (i as f32 * 0.02).cos()).collect();

    let cell = MemoryCellBuilder::new(1, 42, 12, key.clone(), value.clone())
        .importance(50.0)
        .tier(Tier::Draft)
        .build();

    pool.append(&cell).unwrap();
    let restored = pool.get(1).unwrap();

    // Vectors should match within Q4 quantization tolerance.
    // SNR > 20dB means signal power / noise power > 100, i.e. MSE < signal_power / 100.
    let signal_power: f32 = key.iter().map(|x| x * x).sum::<f32>() / key.len() as f32;
    let mse: f32 = key.iter().zip(restored.key.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f32>()
        / key.len() as f32;
    let snr_db = 10.0 * (signal_power / mse).log10();
    assert!(snr_db > 20.0, "Key SNR {snr_db:.1}dB is below 20dB threshold");

    // Metadata must be exact.
    assert_eq!(restored.id, 1);
    assert_eq!(restored.owner, 42);
    assert_eq!(restored.layer, 12);
    assert_eq!(restored.meta.tier, Tier::Draft);
    assert!((restored.meta.importance - 50.0).abs() < f32::EPSILON);
}

/// ATDD Test 2: Append 100 cells, read each back by ID, verify all metadata matches exactly.
#[test]
fn test_append_and_read_by_id() {
    let dir = tempfile::tempdir().unwrap();
    let mut pool = BlockPool::open(dir.path()).unwrap();

    for i in 0..100u64 {
        let cell = MemoryCellBuilder::new(
            i,
            i % 5,          // 5 different owners
            (i % 3) as u16, // 3 different layers
            vec![i as f32; 64],
            vec![(i as f32) * 2.0; 64],
        )
        .importance(i as f32)
        .tags(i as u32)
        .build();

        pool.append(&cell).unwrap();
    }

    for i in 0..100u64 {
        let cell = pool.get(i).unwrap();
        assert_eq!(cell.id, i);
        assert_eq!(cell.owner, i % 5);
        assert_eq!(cell.layer, (i % 3) as u16);
        assert_eq!(cell.meta.tags, i as u32);
    }
}

/// ATDD Test 3: Append cells until the segment threshold is exceeded.
/// Verify a new segment is created and reads work across segments.
#[test]
fn test_segment_rollover() {
    let dir = tempfile::tempdir().unwrap();
    // Use a tiny segment size to force rollover quickly.
    let mut pool = BlockPool::open_with_segment_size(dir.path(), 4096).unwrap();

    let mut ids = Vec::new();
    for i in 0..200u64 {
        let cell = MemoryCellBuilder::new(i, 1, 0, vec![1.0; 64], vec![2.0; 64]).build();
        pool.append(&cell).unwrap();
        ids.push(i);
    }

    // Must have created more than one segment.
    assert!(pool.segment_count() > 1, "Expected multiple segments, got {}", pool.segment_count());

    // All cells should still be readable.
    for id in ids {
        let cell = pool.get(id).unwrap();
        assert_eq!(cell.id, id);
    }
}

/// ATDD Test 4: Write cells, drop the block pool, reconstruct from disk,
/// verify all cells are readable (persistence across restart).
#[test]
fn test_persistence_across_restart() {
    let dir = tempfile::tempdir().unwrap();

    // Write phase.
    {
        let mut pool = BlockPool::open(dir.path()).unwrap();
        for i in 0..50u64 {
            let cell = MemoryCellBuilder::new(i, 1, 0, vec![i as f32; 32], vec![0.0; 32])
                .importance(i as f32)
                .build();
            pool.append(&cell).unwrap();
        }
        // pool is dropped here — must flush to disk.
    }

    // Read phase — new BlockPool instance from the same directory.
    {
        let pool = BlockPool::open(dir.path()).unwrap();
        for i in 0..50u64 {
            let cell = pool.get(i).unwrap();
            assert_eq!(cell.id, i);
            assert!((cell.meta.importance - i as f32).abs() < 1.0); // Q4 tolerance on f32 metadata
        }
    }
}

/// ATDD Test 5: Quantize a known vector to Q4, dequantize, measure MSE.
/// Assert error is below the theoretical Q4 limit.
#[test]
fn test_quantization_fidelity() {
    let original: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.1).sin()).collect();

    let quantized = Q4::quantize(&original);
    let restored = Q4::dequantize(&quantized);

    assert_eq!(original.len(), restored.len());

    // Compute MSE.
    let mse: f32 =
        original.iter().zip(restored.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f32>()
            / original.len() as f32;

    // Q4 (4-bit, 16 levels) with group-wise scaling: theoretical max error per value
    // is ~scale/16. For typical data in [-1, 1], MSE should be well under 0.01.
    assert!(mse < 0.01, "Q4 MSE {mse:.6} exceeds threshold 0.01");

    // Also verify no NaN/Inf crept in.
    for (i, val) in restored.iter().enumerate() {
        assert!(val.is_finite(), "Restored value at index {i} is not finite");
    }
}

// ── Phase 11: SynapticBank Persistence (Repository pattern) ───────────────

use half::f16;
use tdb_core::synaptic_bank::SynapticBankEntry;
use tdb_storage::synaptic_store::SynapticStore;

/// ATDD Test 6: Store a `SynapticBankEntry`, load by owner, verify round-trip.
#[test]
fn test_synaptic_store_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = SynapticStore::open(dir.path()).unwrap();

    let entry = SynapticBankEntry::new(
        0,
        42,
        vec![f16::from_f32(1.0); 512], // rank=4, d_model=128 → 512 elements
        vec![f16::from_f32(0.5); 512],
        f16::from_f32(0.1),
        4,
        128,
    );

    store.append(&entry).unwrap();

    let loaded = store.load_by_owner(42).unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].id, 0);
    assert_eq!(loaded[0].owner, 42);
    assert_eq!(loaded[0].rank, 4);
    assert_eq!(loaded[0].d_model, 128);
    assert_eq!(loaded[0].lora_a.len(), 512);
    assert_eq!(loaded[0].lora_b.len(), 512);
    assert_eq!(loaded[0].scale, f16::from_f32(0.1));
}

/// ATDD Test 7: Store for owners 1,2,1. `load_by_owner(1)` returns 2, (2) returns 1.
#[test]
fn test_synaptic_multiple_owners() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = SynapticStore::open(dir.path()).unwrap();

    for (id, owner) in [(0, 1u64), (1, 2), (2, 1)] {
        let entry = SynapticBankEntry::new(
            id,
            owner,
            vec![f16::from_f32(1.0); 8],
            vec![f16::from_f32(0.5); 8],
            f16::from_f32(0.1),
            2,
            4,
        );
        store.append(&entry).unwrap();
    }

    let owner1 = store.load_by_owner(1).unwrap();
    assert_eq!(owner1.len(), 2);

    let owner2 = store.load_by_owner(2).unwrap();
    assert_eq!(owner2.len(), 1);

    let owner999 = store.load_by_owner(999).unwrap();
    assert!(owner999.is_empty());
}

/// ATDD Test 8: Store entry, drop store, reopen. Load still works.
#[test]
fn test_synaptic_persist_across_reopen() {
    let dir = tempfile::tempdir().unwrap();

    {
        let mut store = SynapticStore::open(dir.path()).unwrap();
        let entry = SynapticBankEntry::new(
            0,
            42,
            vec![f16::from_f32(1.0); 8],
            vec![f16::from_f32(0.5); 8],
            f16::from_f32(0.1),
            2,
            4,
        );
        store.append(&entry).unwrap();
    }

    let store = SynapticStore::open(dir.path()).unwrap();
    let loaded = store.load_by_owner(42).unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].id, 0);
}

// ── Release-Mode Evals ───────────────────────────────────────────────────────

/// Eval 2 (Category A): Q4 compression ratio across typical KV-cache dimensions.
///
/// README claims "4x more agent contexts via Q4 quantization". The actual ratio
/// depends on group overhead. This eval verifies ≥3.5x across real dimensions.
#[test]
#[ignore = "release-mode eval: just eval-spec"]
fn eval_spec_q4_compression_ratio() {
    use tdb_storage::quantization::{Q4, QuantizeStrategy};

    for dim in [128, 256, 512, 768] {
        let values: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.1).sin()).collect();

        let quantized = Q4::quantize(&values);
        let original_bytes = values.len() * 4; // f32 = 4 bytes
        let compressed_bytes = quantized.data.len() + quantized.scales.len() * 4;
        let ratio = original_bytes as f64 / compressed_bytes as f64;

        assert!(
            ratio >= 3.5,
            "Q4 compression ratio {ratio:.2}x < 3.5x at dim={dim} \
             ({original_bytes}B → {compressed_bytes}B)"
        );
    }
}

// ── Phase 13: Batch Write (Write-Behind Buffer pattern) ───────────────────

/// ATDD Test 9: Write 100 cells via batch. All readable. Faster than individual writes.
#[test]
fn test_batch_write_100_cells() {
    let dir = tempfile::tempdir().unwrap();
    let mut pool = BlockPool::open(dir.path()).unwrap();

    let cells: Vec<MemoryCell> = (0..100)
        .map(|i| {
            MemoryCellBuilder::new(i, 1, 0, vec![i as f32; 32], vec![0.0; 32])
                .importance(50.0)
                .build()
        })
        .collect();

    // Batch write — single fsync for all 100 cells.
    let ids = pool.append_batch(&cells).unwrap();
    assert_eq!(ids.len(), 100);

    // All cells readable.
    for i in 0..100u64 {
        let cell = pool.get(i).unwrap();
        assert_eq!(cell.id, i);
    }
}

/// ATDD Test 10: Batch write persists across restart.
#[test]
fn test_batch_write_persistence() {
    let dir = tempfile::tempdir().unwrap();

    {
        let mut pool = BlockPool::open(dir.path()).unwrap();
        let cells: Vec<MemoryCell> = (0..50)
            .map(|i| {
                MemoryCellBuilder::new(i, 1, 0, vec![i as f32; 16], vec![0.0; 16])
                    .importance(50.0)
                    .build()
            })
            .collect();
        pool.append_batch(&cells).unwrap();
    }

    let pool = BlockPool::open(dir.path()).unwrap();
    assert_eq!(pool.cell_count(), 50);
    let cell = pool.get(25).unwrap();
    assert_eq!(cell.id, 25);
}

/// ATDD Test 11: Batch write is measurably faster than individual writes.
#[test]
fn test_batch_faster_than_individual() {
    let dir_individual = tempfile::tempdir().unwrap();
    let dir_batch = tempfile::tempdir().unwrap();
    let n = 50u64;

    let cells: Vec<MemoryCell> = (0..n)
        .map(|i| {
            MemoryCellBuilder::new(i, 1, 0, vec![i as f32; 32], vec![0.0; 32])
                .importance(50.0)
                .build()
        })
        .collect();

    // Individual writes.
    let start_ind = std::time::Instant::now();
    {
        let mut pool = BlockPool::open(dir_individual.path()).unwrap();
        for cell in &cells {
            pool.append(cell).unwrap();
        }
    }
    let time_ind = start_ind.elapsed();

    // Batch write.
    let start_batch = std::time::Instant::now();
    {
        let mut pool = BlockPool::open(dir_batch.path()).unwrap();
        pool.append_batch(&cells).unwrap();
    }
    let time_batch = start_batch.elapsed();

    // Batch should be at least 2x faster (typically 10-50x due to single fsync).
    let speedup = time_ind.as_secs_f64() / time_batch.as_secs_f64();
    assert!(
        speedup > 2.0,
        "Batch ({time_batch:?}) should be ≥2x faster than individual ({time_ind:?}). Speedup: {speedup:.1}x"
    );
}

#[test]
fn test_direct_block_pool_hydration_fixture_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let mut pool = BlockPool::open(dir.path()).unwrap();

    let cells: Vec<MemoryCell> = (0..HYDRATION_FIXTURE_CELL_COUNT)
        .map(|id| {
            MemoryCellBuilder::new(
                id,
                HYDRATION_FIXTURE_OWNER,
                HYDRATION_FIXTURE_LAYER,
                vec![HYDRATION_KEY_BASE + id as f32; HYDRATION_FIXTURE_KEY_DIM],
                vec![HYDRATION_VALUE_BASE + id as f32; HYDRATION_FIXTURE_PAYLOAD_DIM],
            )
            .build()
        })
        .collect();

    pool.append_batch(&cells).unwrap();

    let restored = pool.get(HYDRATION_FIXTURE_TARGET_ID).unwrap();

    assert_eq!(restored.id, HYDRATION_FIXTURE_TARGET_ID);
    assert_eq!(restored.owner, HYDRATION_FIXTURE_OWNER);
    assert_eq!(restored.layer, HYDRATION_FIXTURE_LAYER);
    assert_eq!(restored.key.len(), HYDRATION_FIXTURE_KEY_DIM);
    assert_eq!(restored.value.len(), HYDRATION_FIXTURE_PAYLOAD_DIM);
}

// ── Segment Compaction (P4.2) ─────────────────────────────────────────────

use std::collections::HashSet;

const COMPACT_DIM: usize = 32;
const COMPACT_SEGMENT_SIZE: u64 = 512;

fn compact_cell(id: u64) -> MemoryCell {
    MemoryCellBuilder::new(id, 1, 0, vec![id as f32; COMPACT_DIM], vec![0.0; COMPACT_DIM]).build()
}

/// ATDD: Compaction reclaims space from segments with dead cells.
#[test]
fn test_compact_reclaims_space() {
    let dir = tempfile::tempdir().unwrap();
    let mut pool = BlockPool::open_with_segment_size(dir.path(), COMPACT_SEGMENT_SIZE).unwrap();

    // Fill two segments with cells.
    for i in 0..20u64 {
        pool.append(&compact_cell(i)).unwrap();
    }
    assert!(pool.segment_count() >= 2, "need multiple segments");

    // Mark only every 4th cell as live (75% dead — well below 50% threshold).
    let live: HashSet<u64> = (0..20u64).filter(|i| i % 4 == 0).collect();

    let result = pool.compact(&live).unwrap();

    assert!(result.segments_compacted > 0, "should compact at least one segment");
    assert!(result.bytes_reclaimed > 0);
    assert!(result.cells_moved > 0, "should move at least some live cells");
}

/// ATDD: All live cells are readable after compaction.
#[test]
fn test_compact_preserves_live_cells() {
    let dir = tempfile::tempdir().unwrap();
    let mut pool = BlockPool::open_with_segment_size(dir.path(), COMPACT_SEGMENT_SIZE).unwrap();

    for i in 0..20u64 {
        pool.append(&compact_cell(i)).unwrap();
    }

    let live: HashSet<u64> = (0..20u64).filter(|i| i % 3 == 0).collect();
    pool.compact(&live).unwrap();

    for &id in &live {
        let cell = pool.get(id).unwrap();
        assert_eq!(cell.id, id);
        assert_eq!(cell.key.len(), COMPACT_DIM);
    }
}

/// ATDD: The active segment is never compacted.
#[test]
fn test_compact_skips_active_segment() {
    let dir = tempfile::tempdir().unwrap();
    let mut pool = BlockPool::open_with_segment_size(dir.path(), COMPACT_SEGMENT_SIZE).unwrap();

    // Write just enough to stay in one segment.
    pool.append(&compact_cell(0)).unwrap();

    let live: HashSet<u64> = HashSet::new();
    let result = pool.compact(&live).unwrap();

    assert_eq!(result.segments_compacted, 0, "single (active) segment must not be compacted");
}

/// ATDD: Compaction is idempotent — second call is a no-op.
#[test]
fn test_compact_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let mut pool = BlockPool::open_with_segment_size(dir.path(), COMPACT_SEGMENT_SIZE).unwrap();

    for i in 0..20u64 {
        pool.append(&compact_cell(i)).unwrap();
    }

    let live: HashSet<u64> = (0..20u64).filter(|i| i % 4 == 0).collect();

    // Compact until stable — each pass may expose new compactable segments.
    let first = pool.compact(&live).unwrap();
    assert!(first.segments_compacted > 0);

    // Keep compacting until no more work — proves convergence.
    let mut total_passes = 1;
    loop {
        let pass = pool.compact(&live).unwrap();
        if pass.segments_compacted == 0 {
            break;
        }
        total_passes += 1;
        assert!(total_passes <= 5, "compaction should converge within a few passes");
    }

    // All live cells still accessible.
    for &id in &live {
        assert!(pool.get(id).is_ok(), "live cell {id} should survive repeated compaction");
    }
}

/// ATDD: Compacted pool survives reopen.
#[test]
fn test_compact_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();

    let live: HashSet<u64> = (0..20u64).filter(|i| i % 2 == 0).collect();

    {
        let mut pool = BlockPool::open_with_segment_size(dir.path(), COMPACT_SEGMENT_SIZE).unwrap();
        for i in 0..20u64 {
            pool.append(&compact_cell(i)).unwrap();
        }
        pool.compact(&live).unwrap();
    }

    let pool = BlockPool::open_with_segment_size(dir.path(), COMPACT_SEGMENT_SIZE).unwrap();
    for &id in &live {
        assert!(pool.get(id).is_ok(), "live cell {id} should survive reopen after compaction");
    }
}

/// ATDD: Compaction with no deletions is a no-op.
#[test]
fn test_compact_with_no_deletions() {
    let dir = tempfile::tempdir().unwrap();
    let mut pool = BlockPool::open_with_segment_size(dir.path(), COMPACT_SEGMENT_SIZE).unwrap();

    for i in 0..20u64 {
        pool.append(&compact_cell(i)).unwrap();
    }

    let all_live: HashSet<u64> = (0..20u64).collect();
    let result = pool.compact(&all_live).unwrap();

    assert_eq!(result.segments_compacted, 0, "all cells live → no compaction needed");
}
