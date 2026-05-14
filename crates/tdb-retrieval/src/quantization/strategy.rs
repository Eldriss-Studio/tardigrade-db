//! `RetrievalQuantStrategy` — Strategy trait for per-token retrieval
//! quantization.
//!
//! Distinct from `tdb-storage`'s archival quantization (`QuantizeStrategy`
//! over `Q4_0`). Retrieval quantization governs the *scoring* precision
//! tier — INT8 (or finer) with per-channel scaling — and is independent
//! of how cells are durably stored on disk.

/// Opaque quantized representation of a single token vector.
///
/// `values` contains packed integers (the bit-width depends on the
/// strategy that produced it). `scales` contains one per-group scale.
/// `dim` is the original (pre-quantization) dimension, retained so
/// dequantization can truncate any group-alignment padding.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedToken {
    pub values: Vec<i8>,
    pub scales: Vec<f32>,
    pub dim: usize,
}

impl QuantizedToken {
    /// Construct an empty placeholder for the "wrong-dim input" or
    /// "all-zero" cases. The retriever treats empty quantized tokens
    /// as "skip."
    pub fn empty() -> Self {
        Self { values: Vec::new(), scales: Vec::new(), dim: 0 }
    }

    /// True when this token has no resolvable values.
    pub fn is_empty(&self) -> bool {
        self.dim == 0 || self.values.is_empty()
    }
}

/// Strategy: how the retriever quantizes and dequantizes a single
/// token vector for the scoring path.
///
/// Implementations must be deterministic — the same input always
/// yields the same output — and must be `Send + Sync` because the
/// engine holds them behind a shared reference and dispatches from
/// concurrent query contexts.
pub trait RetrievalQuantStrategy: Send + Sync {
    /// Per-token dimension this strategy operates on.
    fn dim(&self) -> usize;

    /// Quantize one token vector.
    ///
    /// Returns [`QuantizedToken::empty()`] on input that doesn't match
    /// `self.dim()` — callers treat this as "skip" rather than
    /// failing the surrounding batch.
    fn quantize_token(&self, token: &[f32]) -> QuantizedToken;

    /// Dequantize one token back to f32.
    ///
    /// Returns an empty `Vec` if `quantized.dim` does not match
    /// `self.dim()`. The recovered vector has length `self.dim()`.
    fn dequantize_token(&self, quantized: &QuantizedToken) -> Vec<f32>;
}
