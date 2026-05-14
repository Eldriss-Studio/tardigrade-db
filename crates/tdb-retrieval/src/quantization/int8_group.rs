//! `Int8Group32` — plain group-of-32 INT8 quantization for retrieval.
//!
//! Mirrors the `Q4_0` group structure used by `tdb-storage` for archival
//! (one f32 scale per 32 values) but with 8-bit values instead of 4-bit.
//! Suitable for retrieval scoring as the inner quantizer; pair with a
//! [`PerChannelScaled`](super::per_channel_scaled::PerChannelScaled)
//! decorator to neutralize activation outliers before encoding.
//!
//! # Why INT8 not Q4
//!
//! `Q4_0` has 16 levels per group (4-bit signed [-7, 7]). At Qwen3-style
//! hidden-state magnitudes, the half-step is large enough to round
//! non-outlier values to zero — destroying retrieval precision. `INT8`
//! has 256 levels (signed [-127, 127]); the half-step is 16× smaller,
//! and even on outlier-heavy groups the non-outlier dims remain
//! resolvable. Production retrieval systems (`Faiss` SQ8, `Qdrant`,
//! Sentence-Transformers) standardize on `INT8` as their precision
//! floor for the same reason.

use super::strategy::{QuantizedToken, RetrievalQuantStrategy};

/// Group size for INT8 quantization. Matches the storage-layer Q4
/// group size for alignment.
pub const INT8_GROUP_SIZE: usize = 32;

/// 8-bit signed integer range — half-width used when scaling
/// `abs_max` into `[-INT8_LEVELS, INT8_LEVELS]` before clamping.
/// `127` rather than `128` keeps quantization symmetric around
/// zero.
pub const INT8_LEVELS: f32 = 127.0;

/// Floor applied to per-group `abs_max` to avoid divide-by-zero in
/// all-zero groups. Mirrors
/// [`PER_CHANNEL_SCALE_FLOOR`](super::calibrator::PER_CHANNEL_SCALE_FLOOR)
/// but per-group rather than per-channel.
pub const INT8_GROUP_SCALE_FLOOR: f32 = 1e-12;

/// Plain INT8 group-of-32 quantizer. No outlier handling — see
/// [`PerChannelScaled`](super::per_channel_scaled::PerChannelScaled)
/// for the wrapper that pairs this with a calibrated σ to neutralize
/// activation outliers.
#[derive(Debug, Clone, Copy)]
pub struct Int8Group32 {
    dim: usize,
}

impl Int8Group32 {
    /// Construct for a fixed per-token dimension.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl RetrievalQuantStrategy for Int8Group32 {
    fn dim(&self) -> usize {
        self.dim
    }

    fn quantize_token(&self, token: &[f32]) -> QuantizedToken {
        if token.len() != self.dim {
            return QuantizedToken::empty();
        }
        let num_groups = self.dim.div_ceil(INT8_GROUP_SIZE);
        let mut packed_values = Vec::with_capacity(num_groups * INT8_GROUP_SIZE);
        let mut group_scales = Vec::with_capacity(num_groups);
        for chunk in token.chunks(INT8_GROUP_SIZE) {
            let abs_max =
                chunk.iter().map(|value| value.abs()).fold(INT8_GROUP_SCALE_FLOOR, f32::max);
            let scale = abs_max / INT8_LEVELS;
            group_scales.push(scale);
            for value in chunk {
                let scaled = (value / scale).round().clamp(-INT8_LEVELS, INT8_LEVELS);
                packed_values.push(scaled as i8);
            }
            // Pad to group boundary so dequant length matches dim only
            // after truncation. (Original length tracked via dim().)
            let padding_needed = INT8_GROUP_SIZE.saturating_sub(chunk.len());
            packed_values.extend(std::iter::repeat_n(0_i8, padding_needed));
        }
        QuantizedToken { values: packed_values, scales: group_scales, dim: self.dim }
    }

    fn dequantize_token(&self, quantized: &QuantizedToken) -> Vec<f32> {
        if quantized.dim != self.dim {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(self.dim);
        for (group_index, scale) in quantized.scales.iter().enumerate() {
            let start = group_index * INT8_GROUP_SIZE;
            let end = (start + INT8_GROUP_SIZE).min(quantized.values.len());
            for value in &quantized.values[start..end] {
                out.push(f32::from(*value) * scale);
            }
            if out.len() >= self.dim {
                break;
            }
        }
        out.truncate(self.dim);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_DIM: usize = 64;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm_a * norm_b).max(f32::MIN_POSITIVE)
    }

    const UNIFORM_RECALL_FLOOR: f32 = 0.999;

    #[test]
    fn uniform_token_round_trips_within_int8_envelope() {
        let token: Vec<f32> = (0..TEST_DIM).map(|i| (i as f32 * 0.05).sin()).collect();
        let q = Int8Group32::new(TEST_DIM);
        let recovered = q.dequantize_token(&q.quantize_token(&token));
        assert_eq!(recovered.len(), TEST_DIM);
        let cos = cosine(&token, &recovered);
        assert!(
            cos >= UNIFORM_RECALL_FLOOR,
            "uniform token cosine after INT8 round-trip {cos} below floor {UNIFORM_RECALL_FLOOR}",
        );
    }
}
