//! Quantization strategies for KV cache tensors.
//!
//! Implements the Strategy pattern: `QuantizeStrategy` and `DequantizeStrategy` traits.
//! Supported formats: Q4 (group-wise 4-bit), Q8 (symmetric INT8).

/// Group size for block quantization. Each group shares one scale factor.
const Q4_GROUP_SIZE: usize = 32;

/// Opaque container for quantized tensor data.
/// Includes the quantized bytes plus per-group scale factors needed for dequantization.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Packed quantized values. For Q4: two 4-bit values per byte.
    pub data: Vec<u8>,
    /// One f32 scale factor per group.
    pub scales: Vec<f32>,
    /// Number of original f32 elements.
    pub original_len: usize,
}

/// Quantize f32 vectors into a compressed representation.
pub trait QuantizeStrategy {
    fn quantize(values: &[f32]) -> QuantizedTensor;
}

/// Dequantize compressed data back to f32 vectors.
pub trait DequantizeStrategy {
    fn dequantize(tensor: &QuantizedTensor) -> Vec<f32>;
}

/// Group-wise 4-bit quantization (`Q4_0`).
///
/// For each group of 32 values:
/// 1. Find `abs_max` — the maximum absolute value in the group.
/// 2. Compute `scale = abs_max / 7.0` (mapping to signed range [-7, 7]).
/// 3. Quantize each value: `q = round(value / scale) + 8` clamped to [0, 15].
/// 4. Pack two 4-bit values per byte (low nibble first).
///
/// This matches the GGML `Q4_0` scheme used in llama.cpp.
#[derive(Debug)]
pub struct Q4;

impl QuantizeStrategy for Q4 {
    fn quantize(values: &[f32]) -> QuantizedTensor {
        let num_groups = values.len().div_ceil(Q4_GROUP_SIZE);
        let packed_bytes_per_group = Q4_GROUP_SIZE / 2; // 2 values per byte
        let mut data = Vec::with_capacity(num_groups * packed_bytes_per_group);
        let mut scales = Vec::with_capacity(num_groups);

        for group in values.chunks(Q4_GROUP_SIZE) {
            let abs_max = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

            let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 7.0 };
            scales.push(scale);

            // Quantize and pack two values per byte.
            let mut i = 0;
            while i < group.len() {
                let q_low = quantize_value(group[i], scale);
                let q_high = if i + 1 < group.len() {
                    quantize_value(group[i + 1], scale)
                } else {
                    8 // zero-point for padding
                };
                data.push((q_high << 4) | q_low);
                i += 2;
            }

            // Pad if group was smaller than Q4_GROUP_SIZE (last group).
            let packed_count = group.len().div_ceil(2);
            let pad_count = packed_bytes_per_group - packed_count;
            data.extend(std::iter::repeat_n(0x88u8, pad_count));
        }

        QuantizedTensor { data, scales, original_len: values.len() }
    }
}

impl DequantizeStrategy for Q4 {
    fn dequantize(tensor: &QuantizedTensor) -> Vec<f32> {
        let mut result = Vec::with_capacity(tensor.original_len);
        let packed_bytes_per_group = Q4_GROUP_SIZE / 2;

        for (group_idx, &scale) in tensor.scales.iter().enumerate() {
            let byte_offset = group_idx * packed_bytes_per_group;
            let group_start = group_idx * Q4_GROUP_SIZE;

            for j in 0..packed_bytes_per_group {
                if group_start + j * 2 >= tensor.original_len {
                    break;
                }
                let byte = tensor.data[byte_offset + j];
                let q_low = i16::from(byte & 0x0F) - 8;
                result.push(q_low as f32 * scale);

                if group_start + j * 2 + 1 >= tensor.original_len {
                    break;
                }
                let q_high = i16::from((byte >> 4) & 0x0F) - 8;
                result.push(q_high as f32 * scale);
            }
        }

        result
    }
}

/// Map a single f32 value to a 4-bit unsigned integer [0, 15] with zero-point at 8.
#[inline]
fn quantize_value(value: f32, scale: f32) -> u8 {
    let q = (value / scale).round() as i8;
    (q + 8).clamp(0, 15) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_round_trip_zeros() {
        let values = vec![0.0; 64];
        let quantized = Q4::quantize(&values);
        let restored = Q4::dequantize(&quantized);
        assert_eq!(restored.len(), values.len());
        for v in &restored {
            assert!(v.abs() < f32::EPSILON, "Expected 0.0, got {v}");
        }
    }

    #[test]
    fn test_q4_round_trip_preserves_length() {
        for len in [1, 15, 31, 32, 33, 63, 64, 100, 128, 255, 256] {
            let values: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1).collect();
            let quantized = Q4::quantize(&values);
            let restored = Q4::dequantize(&quantized);
            assert_eq!(restored.len(), values.len(), "Length mismatch for input size {len}");
        }
    }

    #[test]
    fn test_q4_mse_within_tolerance() {
        let values: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.1).sin()).collect();
        let quantized = Q4::quantize(&values);
        let restored = Q4::dequantize(&quantized);

        let mse: f32 =
            values.iter().zip(restored.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f32>()
                / values.len() as f32;

        assert!(mse < 0.01, "MSE {mse:.6} exceeds 0.01");
    }

    #[test]
    fn test_q4_compression_ratio() {
        let values = vec![1.0f32; 128];
        let quantized = Q4::quantize(&values);
        let original_bytes = values.len() * 4; // f32 = 4 bytes
        let compressed_bytes = quantized.data.len() + quantized.scales.len() * 4;
        let ratio = original_bytes as f32 / compressed_bytes as f32;
        // Q4 should achieve ~4x compression (4 bits vs 32 bits, plus scale overhead).
        assert!(ratio > 3.0, "Compression ratio {ratio:.1}x is below 3x minimum");
    }
}
