//! Retrieval key adapter — one contract for raw token and fixed-dimension views.
//!
//! Encoded per-token keys are useful only for token-aware retrievers such as
//! [`PerTokenRetriever`](crate::per_token::PerTokenRetriever). Fixed-dimension
//! retrievers such as SLB, brute force, and Vamana must receive a pooled vector.
//! This module is the Adapter + Value Object boundary that prevents raw encoded
//! headers from leaking into fixed-dimension stages.

use crate::per_token::decode_per_token_keys;

/// Parsed retrieval-key representation.
#[derive(Debug, Clone, Copy)]
pub enum RetrievalKeyView<'a> {
    /// A legacy fixed-dimension vector.
    Plain(&'a [f32]),
    /// A valid encoded per-token matrix.
    Encoded { token_count: usize, dim: usize, data: &'a [f32] },
}

/// Parse failure for sentinel-marked keys that do not satisfy the encoding contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalKeyError {
    /// The key looks encoded but has invalid metadata or mismatched data length.
    MalformedEncodedKey,
}

impl<'a> RetrievalKeyView<'a> {
    /// Parse a key into a view without allocating.
    ///
    /// Plain vectors are accepted as-is. Sentinel-marked keys must decode
    /// successfully; otherwise they are rejected instead of being treated as
    /// normal fixed-dimension vectors.
    pub fn parse(key: &'a [f32]) -> std::result::Result<Self, RetrievalKeyError> {
        if let Some((token_count, dim, data)) = decode_per_token_keys(key) {
            return Ok(Self::Encoded { token_count, dim, data });
        }

        if looks_encoded(key) {
            Err(RetrievalKeyError::MalformedEncodedKey)
        } else {
            Ok(Self::Plain(key))
        }
    }

    /// Whether this key is encoded per-token data.
    pub fn is_encoded_per_token(&self) -> bool {
        matches!(self, Self::Encoded { .. })
    }

    /// Return raw token-matrix data for token-aware retrieval.
    pub fn raw_tokens(&self) -> Option<(usize, usize, &'a [f32])> {
        match self {
            Self::Encoded { token_count, dim, data } => Some((*token_count, *dim, *data)),
            Self::Plain(_) => None,
        }
    }

    /// Return a fixed-dimension vector for SLB, brute-force, and Vamana.
    pub fn pooled_vector(&self) -> Vec<f32> {
        match self {
            Self::Plain(key) => key.to_vec(),
            Self::Encoded { token_count, dim, data } => {
                let mut mean = vec![0.0f32; *dim];
                for token_idx in 0..*token_count {
                    for dim_idx in 0..*dim {
                        mean[dim_idx] += data[token_idx * *dim + dim_idx];
                    }
                }
                let inv_n = 1.0 / *token_count as f32;
                for value in &mut mean {
                    *value *= inv_n;
                }
                mean
            }
        }
    }
}

/// Whether a raw key is a valid encoded per-token matrix.
pub fn is_encoded_per_token_key(key: &[f32]) -> bool {
    RetrievalKeyView::parse(key).is_ok_and(|view| view.is_encoded_per_token())
}

/// Build a fixed-dimension vector for a retrieval stage.
pub fn fixed_dim_key(key: &[f32]) -> Option<Vec<f32>> {
    RetrievalKeyView::parse(key).ok().map(|view| view.pooled_vector())
}

fn looks_encoded(key: &[f32]) -> bool {
    key.first().is_some_and(|sentinel| *sentinel <= -1.0e8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::per_token::encode_per_token_keys;

    #[test]
    fn test_encoded_key_view_exposes_raw_tokens_for_per_token_retrieval() {
        let first = [1.0f32, 0.0, 0.0];
        let second = [0.0f32, 1.0, 0.0];
        let encoded = encode_per_token_keys(&[&first, &second]);

        let view = RetrievalKeyView::parse(&encoded).expect("encoded key should parse");
        let (token_count, dim, data) =
            view.raw_tokens().expect("encoded key should expose raw tokens");

        assert_eq!(token_count, 2);
        assert_eq!(dim, 3);
        assert_eq!(data, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_encoded_key_view_exposes_pooled_vector_for_fixed_dim_retrievers() {
        let first = [1.0f32, 3.0, 5.0];
        let second = [3.0f32, 5.0, 7.0];
        let encoded = encode_per_token_keys(&[&first, &second]);

        let pooled = RetrievalKeyView::parse(&encoded).unwrap().pooled_vector();

        assert_eq!(pooled, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_plain_key_view_uses_original_vector_for_fixed_dim_retrievers() {
        let plain = [0.25f32, 0.5, 0.75];

        let view = RetrievalKeyView::parse(&plain).expect("plain key should parse");

        assert!(!view.is_encoded_per_token());
        assert_eq!(view.pooled_vector(), plain);
        assert!(view.raw_tokens().is_none());
    }

    #[test]
    fn test_malformed_encoded_key_fails_or_falls_back_predictably() {
        let malformed = [-1.0e9f32, 0.0, 0.0, 0.0];

        let parsed = RetrievalKeyView::parse(&malformed);

        assert_eq!(parsed.err(), Some(RetrievalKeyError::MalformedEncodedKey));
        assert!(
            fixed_dim_key(&malformed).is_none(),
            "malformed encoded keys must not flow into fixed-dimension retrievers as raw vectors"
        );
    }

    #[test]
    fn test_q4_rounded_encoded_metadata_still_builds_pooled_vector() {
        let mut encoded = vec![0.0f32; 64 + 2 * 3];
        encoded[0] = -1.0e9;
        encoded[32] = 0.0;
        encoded[33] = 3.0;
        encoded[64..].copy_from_slice(&[1.0, 3.0, 5.0, 3.0, 5.0, 7.0]);

        let view = RetrievalKeyView::parse(&encoded).expect("token count should be inferred");

        assert_eq!(view.pooled_vector(), vec![2.0, 4.0, 6.0]);
    }
}
