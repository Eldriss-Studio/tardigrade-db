//! SIMD-accelerated distance/dot-product functions.
//!
//! Strategy pattern: scalar fallback, with platform-specific optimizations.
//! On aarch64 (Apple Silicon, ARM servers), uses NEON intrinsics.
//! On x86_64, uses auto-vectorization-friendly loops.

use crate::int8_quant::QuantizedInt8Vec;

/// Dot product implementations for retrieval scoring.
pub struct DotProduct;

impl DotProduct {
    /// FP32 dot product — reference implementation.
    #[inline]
    pub fn f32_dot(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        Self::f32_dot_scalar(a, b)
    }

    /// INT8 dot product with scale correction.
    /// Returns an approximate FP32 result: `(scale_a * scale_b) * sum(a_q[i] * b_q[i])`.
    #[inline]
    pub fn int8_dot(a: &QuantizedInt8Vec, b: &QuantizedInt8Vec) -> f32 {
        debug_assert_eq!(a.values.len(), b.values.len());
        let raw = Self::int8_dot_raw(&a.values, &b.values);
        raw as f32 * a.scale * b.scale
    }

    /// Scalar FP32 dot product. Written to auto-vectorize well:
    /// simple loop, no branches, no dependencies between iterations.
    #[inline]
    fn f32_dot_scalar(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Raw INT8 dot product returning i32 accumulator.
    /// On aarch64 with NEON, processes 16 elements per iteration.
    #[cfg(target_arch = "aarch64")]
    #[inline]
    fn int8_dot_raw(a: &[i8], b: &[i8]) -> i32 {
        use std::arch::aarch64::*;

        let len = a.len();
        let chunks = len / 16;

        let mut acc: i32 = unsafe {
            let mut vacc = vdupq_n_s32(0);

            for i in 0..chunks {
                let offset = i * 16;
                let va = vld1q_s8(a.as_ptr().add(offset));
                let vb = vld1q_s8(b.as_ptr().add(offset));

                // Multiply and pairwise widen: i8×i8 → i16, then accumulate to i32.
                let low_a = vget_low_s8(va);
                let high_a = vget_high_s8(va);
                let low_b = vget_low_s8(vb);
                let high_b = vget_high_s8(vb);

                let prod_low = vmull_s8(low_a, low_b);
                let prod_high = vmull_s8(high_a, high_b);

                vacc = vpadalq_s16(vacc, prod_low);
                vacc = vpadalq_s16(vacc, prod_high);
            }

            // Horizontal sum of 4×i32 lanes.
            vaddvq_s32(vacc)
        };

        // Handle remaining elements.
        let remainder_start = chunks * 16;
        for i in remainder_start..len {
            acc += a[i] as i32 * b[i] as i32;
        }

        acc
    }

    /// Scalar fallback for non-aarch64 platforms.
    #[cfg(not(target_arch = "aarch64"))]
    #[inline]
    fn int8_dot_raw(a: &[i8], b: &[i8]) -> i32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| x as i32 * y as i32)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_dot_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = DotProduct::f32_dot(&a, &b);
        assert!((result - 32.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_f32_dot_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((DotProduct::f32_dot(&a, &b)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_int8_dot_raw_basic() {
        let a: Vec<i8> = vec![1, 2, 3, 4];
        let b: Vec<i8> = vec![5, 6, 7, 8];
        let result = DotProduct::int8_dot_raw(&a, &b);
        assert_eq!(result, 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8);
    }

    #[test]
    fn test_int8_dot_raw_large_vector() {
        let len = 128;
        let a: Vec<i8> = (0..len).map(|i| (i % 10) as i8).collect();
        let b: Vec<i8> = (0..len).map(|i| ((i + 3) % 10) as i8).collect();
        let expected: i32 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x as i32 * y as i32)
            .sum();
        let result = DotProduct::int8_dot_raw(&a, &b);
        assert_eq!(result, expected);
    }
}
