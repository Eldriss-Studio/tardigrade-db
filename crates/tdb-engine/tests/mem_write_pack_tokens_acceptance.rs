//! ATDD for `Engine::mem_write_pack_tokens` — the symmetric write-side
//! counterpart to `Engine::mem_read_tokens`.
//!
//! These tests pin the wire-level contract: tokens passed in shape
//! `(n_tokens, dim)` produce a retrieval key whose Q4-surviving header
//! records the dimension, so subsequent reads can validate.

use tdb_core::OwnerId;
use tdb_core::kv_pack::KVLayerPayload;
use tdb_engine::engine::Engine;
use tdb_retrieval::per_token::encode_per_token_keys;

const OWNER: OwnerId = 1;

fn layer_payload(marker: f32) -> Vec<KVLayerPayload> {
    vec![KVLayerPayload { layer_idx: 0, data: vec![marker; 64] }]
}

#[test]
fn it_makes_token_path_pack_retrievable_with_same_tokens() {
    // The minimal behavioural property: a pack written via the token API
    // is retrievable via the read-side token API with a matching score.
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let n_tokens = 4_usize;
    let dim = 16_usize;
    let tokens: Vec<f32> = (0..(n_tokens * dim)).map(|i| (i as f32) * 0.01).collect();

    let pid = engine
        .mem_write_pack_tokens(OWNER, &tokens, n_tokens, dim, layer_payload(0.5), 70.0, None)
        .unwrap();

    let results = engine.mem_read_tokens(&tokens, n_tokens, dim, 5, Some(OWNER)).unwrap();
    assert!(!results.is_empty(), "pack {pid} not retrievable via mem_read_tokens");
}

#[test]
fn it_rejects_empty_token_matrix() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let result = engine.mem_write_pack_tokens(OWNER, &[], 0, 16, layer_payload(0.5), 70.0, None);
    assert!(result.is_err(), "expected error on empty token matrix");
}

#[test]
fn it_produces_equivalent_pack_to_pre_encoded_path() {
    // Writing the same content two ways must produce retrieval keys that
    // score identically to a query of the same shape.
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let n_tokens = 3_usize;
    let dim = 8_usize;
    let token_rows: Vec<Vec<f32>> = (0..n_tokens)
        .map(|t| (0..dim).map(|d| 0.1 * (t + 1) as f32 + 0.01 * d as f32).collect())
        .collect();

    // Pre-encoded path: use the canonical helper to build the key, hand it
    // to mem_write_pack.
    let row_refs: Vec<&[f32]> = token_rows.iter().map(Vec::as_slice).collect();
    let pre_encoded = encode_per_token_keys(&row_refs);
    let pack_a = tdb_core::kv_pack::KVPack {
        id: 0,
        owner: OWNER,
        retrieval_key: pre_encoded,
        layers: layer_payload(0.5),
        salience: 70.0,
        text: None,
    };
    let pid_a = engine.mem_write_pack(&pack_a).unwrap();

    // Token-matrix path: flat (n_tokens * dim) buffer, Rust builds the key.
    let flat: Vec<f32> = token_rows.iter().flatten().copied().collect();
    let pid_b = engine
        .mem_write_pack_tokens(OWNER, &flat, n_tokens, dim, layer_payload(0.5), 70.0, None)
        .unwrap();

    assert_ne!(pid_a, pid_b, "expected two distinct pack ids");

    // Both packs must come back when we query with the same shape.
    let results = engine.mem_read_tokens(&flat, n_tokens, dim, 5, Some(OWNER)).unwrap();
    let cell_ids: Vec<_> = results.iter().map(|r| r.cell.id).collect();
    assert!(
        results.len() >= 2,
        "expected both packs in top-5; got {} results ({:?})",
        results.len(),
        cell_ids
    );
    // The top two scores should match within float noise.
    let top1 = results[0].score;
    let top2 = results[1].score;
    assert!(
        (top1 - top2).abs() < 1e-3,
        "token-path and encoded-path scores diverge: {top1} vs {top2}"
    );
}
