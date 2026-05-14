//! Acceptance test: per-token encoded keys survive the Q4 round-trip.
//!
//! Contract under test: a buffer produced by `encode_per_token_keys` can be
//! quantized with `Q4`, dequantized, and passed back through
//! `decode_per_token_keys` while still recovering `(n, dim)` correctly and
//! reconstructing per-token data within Q4's precision envelope.
//!
//! This is the contract `engine::activate_vamana` silently relied on —
//! it pools every stored cell's key into a fixed-dim vector for index build,
//! and that pooling reads `(n, dim)` from the decoded header. When the header
//! does not survive Q4, pooling produces an empty vector and Vamana asserts.

use tdb_retrieval::per_token::{
    DIM_IDX, HEADER_SENTINEL, HEADER_SIZE, N_TOKENS_IDX, decode_per_token_keys,
    encode_per_token_keys,
};
use tdb_storage::quantization::{DequantizeStrategy, Q4, QuantizeStrategy};

/// Mirror of Qwen3-0.6B-shaped chunks: 127 tokens (128-chunk minus position-0
/// sink), 1024-dim hidden states.
const N: usize = 127;
const D: usize = 1024;

fn build_token_matrix() -> Vec<Vec<f32>> {
    let mut rows = Vec::with_capacity(N);
    for token_idx in 0..N {
        let mut row = Vec::with_capacity(D);
        for dim_idx in 0..D {
            let t = token_idx as f32 * 0.01;
            let d = dim_idx as f32 * 0.001;
            row.push((t + d).sin());
        }
        rows.push(row);
    }
    rows
}

#[test]
fn encoded_per_token_key_survives_q4_roundtrip() {
    let tokens = build_token_matrix();
    let refs: Vec<&[f32]> = tokens.iter().map(Vec::as_slice).collect();
    let encoded = encode_per_token_keys(&refs);

    assert_eq!(encoded.len(), HEADER_SIZE + N * D);
    assert_eq!(encoded[0], HEADER_SENTINEL);
    assert_eq!(encoded[DIM_IDX], D as f32);

    let quantized = Q4::quantize(&encoded);
    let dequantized = Q4::dequantize(&quantized);
    assert_eq!(dequantized.len(), encoded.len());

    let (recovered_n, recovered_d, recovered_data) = decode_per_token_keys(&dequantized).expect(
        "decoder must succeed after Q4 round-trip — this is the contract \
             activate_vamana relies on",
    );

    assert_eq!(recovered_n, N, "n must be recoverable from data.len()/d");
    assert_eq!(recovered_d, D);
    assert_eq!(recovered_data.len(), N * D);

    // Q4 envelope: per-group abs_max / 7 is the quantization step. We don't
    // assert exact equality on the data — just that values are in the right
    // ballpark and not catastrophically off. The point of the test is that
    // decode succeeds at all, not Q4 fidelity (which is exercised elsewhere).
    let original_first = tokens[0][0];
    let recovered_first = recovered_data[0];
    assert!(
        (recovered_first - original_first).abs() < 0.5,
        "first datum drift too large: {original_first} -> {recovered_first}",
    );
}

/// The decoded header's `n_tokens` slot is corrupted by Q4 — verify that
/// callers who read it directly would see a non-127 value (i.e., the test
/// above is genuinely exercising the corruption path, not a no-op).
#[test]
fn q4_corrupts_n_tokens_header_field() {
    let tokens = build_token_matrix();
    let refs: Vec<&[f32]> = tokens.iter().map(Vec::as_slice).collect();
    let encoded = encode_per_token_keys(&refs);
    let dequantized = Q4::dequantize(&Q4::quantize(&encoded));

    let stored_n = dequantized[N_TOKENS_IDX];
    let stored_d = dequantized[DIM_IDX];

    assert!(
        (stored_n.round() as usize) != N,
        "preconditions changed: Q4 now preserves n_tokens \
         (stored_n={stored_n}); the regression fix may no longer be necessary",
    );
    assert_eq!(
        stored_d.round() as usize,
        D,
        "dim must still survive Q4 (it is the group's abs_max)",
    );
}
