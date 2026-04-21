use tdb_core::Tier;
use tdb_core::memory_cell::MemoryCellBuilder;
use tdb_storage::block_pool::BlockPool;
use tdb_storage::quantization::{DequantizeStrategy, Q4, QuantizeStrategy};

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
