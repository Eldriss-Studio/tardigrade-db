//! AT-1 — `BlockPoolCellSource` returns each cell's encoded key.
//!
//! Pins the `CellSource` contract at the storage→retrieval seam before
//! any retriever change. Slice 1 of the universal scalability fix.

use tdb_core::memory_cell::MemoryCellBuilder;
use tdb_core::{CellId, LayerId, OwnerId};
use tdb_engine::cell_source_pool::BlockPoolCellSource;
use tdb_retrieval::cell_source::CellSource;
use tdb_storage::block_pool::BlockPool;
use tempfile::TempDir;

const FIXTURE_OWNER: OwnerId = 1;
const FIXTURE_LAYER: LayerId = 0;
const SMALL_KEY_DIM: usize = 128;
const MEDIUM_KEY_DIM: usize = 256;
const TINY_KEY_DIM: usize = 64;
const SIN_PHASE_STEP: f32 = 0.01;
const COS_PHASE_STEP: f32 = 0.02;
const PARITY_POS_VALUE: f32 = 1.0;
const PARITY_NEG_VALUE: f32 = -1.0;
const DUMMY_VALUE: f32 = 0.0;
const UNKNOWN_LOW_ID: CellId = 0;
const UNKNOWN_HIGH_ID: CellId = u64::MAX;

fn make_cell(id: CellId, key: Vec<f32>) -> tdb_core::memory_cell::MemoryCell {
    MemoryCellBuilder::new(id, FIXTURE_OWNER, FIXTURE_LAYER, key, vec![DUMMY_VALUE]).build()
}

fn fixture_keys() -> Vec<Vec<f32>> {
    let sin_key = (0..SMALL_KEY_DIM).map(|i| (i as f32 * SIN_PHASE_STEP).sin()).collect();
    let cos_key = (0..MEDIUM_KEY_DIM).map(|i| (i as f32 * COS_PHASE_STEP).cos()).collect();
    let parity_key = (0..TINY_KEY_DIM)
        .map(|i| if i % 2 == 0 { PARITY_POS_VALUE } else { PARITY_NEG_VALUE })
        .collect();
    vec![sin_key, cos_key, parity_key]
}

#[test]
fn returns_buffer_with_correct_length_for_each_written_cell() {
    // Contract: source returns Some(buf) where buf.len() == cell.key.len()
    // for every written cell. Element-level Q4 fidelity is covered by
    // tests/per_token_q4_roundtrip.rs; here we only assert presence and
    // shape — exactly what callers depend on.
    let dir = TempDir::new().expect("tempdir");
    let mut pool = BlockPool::open(dir.path()).expect("open pool");

    let keys = fixture_keys();
    let ids: Vec<_> = keys
        .iter()
        .enumerate()
        .map(|(i, k)| pool.append(&make_cell(i as CellId, k.clone())).expect("append"))
        .collect();

    let source = BlockPoolCellSource::new(&pool);

    for (id, expected_key) in ids.iter().zip(keys.iter()) {
        let recovered = source.get_encoded_key(*id).expect("cell must be present");
        assert_eq!(
            recovered.len(),
            expected_key.len(),
            "key length must round-trip exactly (Q4 preserves shape)",
        );
    }
}

#[test]
fn returns_none_for_unknown_cell_id() {
    let dir = TempDir::new().expect("tempdir");
    let pool = BlockPool::open(dir.path()).expect("open pool");
    let source = BlockPoolCellSource::new(&pool);

    assert!(source.get_encoded_key(UNKNOWN_LOW_ID).is_none());
    assert!(source.get_encoded_key(UNKNOWN_HIGH_ID).is_none());
}
