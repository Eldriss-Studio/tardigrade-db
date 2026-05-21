//! Acceptance tests for `Engine::flat_to_paged` / `Engine::paged_to_flat`.

use tdb_engine::engine::Engine;

const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const BLOCK_SIZE: usize = 16;

fn deterministic_flat(seq_len: usize, seed: u64) -> Vec<f32> {
    let kv_dim = NUM_KV_HEADS * HEAD_DIM;
    let n = 2 * seq_len * kv_dim;
    (0..n).map(|i| ((i as u64).wrapping_mul(seed) % 9973) as f32 * 0.001).collect()
}

#[test]
fn it_round_trips_losslessly_for_assorted_seq_lens() {
    for &seq_len in &[1usize, 15, 16, 17, 100] {
        let flat = deterministic_flat(seq_len, seq_len as u64 + 7);
        let (k, v) = Engine::flat_to_paged(&flat, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE).unwrap();
        let restored = Engine::paged_to_flat(&k, &v, seq_len, NUM_KV_HEADS, HEAD_DIM).unwrap();
        assert_eq!(restored, flat, "round-trip failed at seq_len={seq_len}");
    }
}

#[test]
fn it_zero_pads_unused_slots_in_the_final_block() {
    let seq_len = 17; // 1 + 1/16 blocks
    let flat = deterministic_flat(seq_len, 42);
    let (k, _v) = Engine::flat_to_paged(&flat, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE).unwrap();
    let num_blocks = 2;
    let kv_dim = NUM_KV_HEADS * HEAD_DIM;
    assert_eq!(k.len(), num_blocks * BLOCK_SIZE * kv_dim);
    // Tail 15 token slots × kv_dim floats must be zero. f32 bit-pattern
    // comparison sidesteps the (deliberate) clippy::float_cmp gate —
    // padding has to be exactly +0.0, not "approximately zero".
    for &f in &k[seq_len * kv_dim..] {
        assert_eq!(f.to_bits(), 0u32);
    }
}

#[test]
fn it_accepts_block_size_one() {
    let flat = deterministic_flat(5, 11);
    let (k, v) = Engine::flat_to_paged(&flat, NUM_KV_HEADS, HEAD_DIM, 1).unwrap();
    let restored = Engine::paged_to_flat(&k, &v, 5, NUM_KV_HEADS, HEAD_DIM).unwrap();
    assert_eq!(restored, flat);
}

#[test]
fn it_returns_empty_blocks_for_seq_len_zero() {
    let (k, v) = Engine::flat_to_paged(&[], NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE).unwrap();
    assert!(k.is_empty());
    assert!(v.is_empty());
    let restored = Engine::paged_to_flat(&k, &v, 0, NUM_KV_HEADS, HEAD_DIM).unwrap();
    assert!(restored.is_empty());
}

#[test]
fn it_rejects_paged_to_flat_with_seq_len_exceeding_block_buffer() {
    let flat = deterministic_flat(5, 13);
    let (k, v) = Engine::flat_to_paged(&flat, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE).unwrap();
    let err = Engine::paged_to_flat(&k, &v, 9999, NUM_KV_HEADS, HEAD_DIM).unwrap_err();
    assert!(err.to_string().contains("seq_len"), "got: {err}");
}

#[test]
fn it_rejects_flat_buffer_not_a_multiple_of_two_kv_dim() {
    let kv_dim = NUM_KV_HEADS * HEAD_DIM;
    // 2 * kv_dim + 1 element — guarantees the split into K/V halves
    // cannot work cleanly.
    let bad = vec![0.0f32; 2 * kv_dim + 1];
    let err = Engine::flat_to_paged(&bad, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE).unwrap_err();
    assert!(err.to_string().contains("multiple of"), "got: {err}");
}

#[test]
fn it_rejects_zero_dimensions() {
    let err = Engine::flat_to_paged(&[], 0, HEAD_DIM, BLOCK_SIZE).unwrap_err();
    assert!(err.to_string().contains("> 0"), "got: {err}");
    let err = Engine::flat_to_paged(&[], NUM_KV_HEADS, 0, BLOCK_SIZE).unwrap_err();
    assert!(err.to_string().contains("> 0"), "got: {err}");
    let err = Engine::flat_to_paged(&[], NUM_KV_HEADS, HEAD_DIM, 0).unwrap_err();
    assert!(err.to_string().contains("> 0"), "got: {err}");
}
