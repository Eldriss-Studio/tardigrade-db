//! Symmetric INT8 scalar quantization for the SLB.
//!
//! Quantize: `scale = max(abs(vec)) / 127`, `q[i] = round(vec[i] / scale)`
//! Dequantize to FP32 on cache insertion to preserve L1-resident lookup performance.

/// A quantized INT8 vector with its scale factor.
#[derive(Debug, Clone)]
pub struct QuantizedInt8Vec {
    /// Quantized values in [-127, 127].
    pub values: Vec<i8>,
    /// Scale factor: `original[i] ≈ values[i] * scale`.
    pub scale: f32,
}

/// Symmetric INT8 quantizer.
pub struct Int8Quantizer;

impl Int8Quantizer {
    /// Quantize an f32 vector to symmetric INT8.
    ///
    /// The scale is chosen so that `max(abs(vec))` maps to ±127.
    pub fn quantize(values: &[f32]) -> QuantizedInt8Vec {
        let abs_max = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };
        let inv_scale = 1.0 / scale;

        let quantized: Vec<i8> = values
            .iter()
            .map(|&v| (v * inv_scale).round().clamp(-127.0, 127.0) as i8)
            .collect();

        QuantizedInt8Vec {
            values: quantized,
            scale,
        }
    }

    /// Dequantize back to f32.
    pub fn dequantize(qvec: &QuantizedInt8Vec) -> Vec<f32> {
        qvec.values.iter().map(|&v| v as f32 * qvec.scale).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_trip_preserves_length() {
        let values: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let q = Int8Quantizer::quantize(&values);
        let restored = Int8Quantizer::dequantize(&q);
        assert_eq!(values.len(), restored.len());
    }

    #[test]
    fn test_round_trip_low_mse() {
        let values: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin()).collect();
        let q = Int8Quantizer::quantize(&values);
        let restored = Int8Quantizer::dequantize(&q);

        let mse: f32 = values
            .iter()
            .zip(restored.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / values.len() as f32;

        // INT8 with 254 levels should have very low quantization error.
        assert!(mse < 0.001, "INT8 MSE {mse:.6} exceeds 0.001");
    }

    #[test]
    fn test_zeros() {
        let q = Int8Quantizer::quantize(&[0.0; 64]);
        assert!(q.values.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_max_value_maps_to_127() {
        let values = vec![1.0, -1.0, 0.5, -0.5];
        let q = Int8Quantizer::quantize(&values);
        assert_eq!(q.values[0], 127);
        assert_eq!(q.values[1], -127);
    }
}
