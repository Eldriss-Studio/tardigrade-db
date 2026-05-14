//! `PerChannelScaled<Q>` — Decorator that pre-scales tokens by a
//! calibrated σ vector before delegating quantization to an inner
//! strategy, and inverse-scales on dequantization.
//!
//! This is the `SmoothQuant` pattern applied to *storage* rather than
//! inference. After dividing each dim by σ[d], every channel has
//! roughly unit range — the activation-outlier channel no longer
//! dominates its quantization group, and non-outlier dims retain
//! their resolution.
//!
//! # Pattern: Decorator
//!
//! The wrapper exposes the same [`RetrievalQuantStrategy`] interface
//! as its inner. New quantizers (`Q8Group32`, `NfFour`, future PQ
//! impls) can be composed with per-channel scaling without changing
//! either side — Open-Closed.

use super::strategy::{QuantizedToken, RetrievalQuantStrategy};

/// Decorator: pre-scale by σ, delegate to inner, inverse-scale on
/// dequant.
#[derive(Debug, Clone)]
pub struct PerChannelScaled<Q: RetrievalQuantStrategy> {
    inner: Q,
    sigma: Vec<f32>,
}

impl<Q: RetrievalQuantStrategy> PerChannelScaled<Q> {
    /// Wrap an inner strategy with a calibrated σ.
    ///
    /// `sigma.len()` must equal `inner.dim()`. Constructing with a
    /// mismatched σ length yields a strategy that returns empty
    /// quantized tokens for every input — fail-fast at score time.
    pub fn new(inner: Q, sigma: Vec<f32>) -> Self {
        Self { inner, sigma }
    }

    fn dims_match(&self) -> bool {
        self.sigma.len() == self.inner.dim()
    }
}

impl<Q: RetrievalQuantStrategy> RetrievalQuantStrategy for PerChannelScaled<Q> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn quantize_token(&self, token: &[f32]) -> QuantizedToken {
        if !self.dims_match() || token.len() != self.inner.dim() {
            return QuantizedToken::empty();
        }
        let scaled: Vec<f32> =
            token.iter().zip(self.sigma.iter()).map(|(value, sigma)| value / sigma).collect();
        self.inner.quantize_token(&scaled)
    }

    fn dequantize_token(&self, quantized: &QuantizedToken) -> Vec<f32> {
        if !self.dims_match() {
            return Vec::new();
        }
        let mut out = self.inner.dequantize_token(quantized);
        if out.len() != self.sigma.len() {
            return Vec::new();
        }
        for (value, sigma) in out.iter_mut().zip(self.sigma.iter()) {
            *value *= *sigma;
        }
        out
    }
}
