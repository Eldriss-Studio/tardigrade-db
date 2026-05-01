//! SIMD-accelerated distance/dot-product functions.
//!
//! Strategy pattern: scalar fallback, with platform-specific optimizations.
//! On aarch64 (Apple Silicon, ARM servers), uses NEON intrinsics.
//! On `x86_64`, uses auto-vectorization-friendly loops.

use crate::int8_quant::QuantizedInt8Vec;

/// Dot product implementations for retrieval scoring.
#[derive(Debug)]
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
        use std::arch::aarch64::{
            vaddvq_s32, vdupq_n_s32, vget_high_s8, vget_low_s8, vld1q_s8, vmull_s8, vpadalq_s16,
        };

        let len = a.len();
        let chunks = len / 16;

        // SAFETY: NEON intrinsics require `unsafe`. Pointer arithmetic is bounded by
        // `chunks * 16 <= len`, so `a.as_ptr().add(offset)` and `b.as_ptr().add(offset)`
        // always point within the slice. The `#[cfg(target_arch = "aarch64")]` gate
        // ensures NEON is available.
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

    /// x86_64: AVX2 when available, scalar fallback otherwise.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn int8_dot_raw(a: &[i8], b: &[i8]) -> i32 {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature detection ensures AVX2 is available.
            unsafe { Self::int8_dot_avx2(a, b) }
        } else {
            Self::int8_dot_scalar(a, b)
        }
    }

    /// AVX2 INT8 dot product. Widens i8→i16 then uses `vpmaddwd` (i16×i16→i32
    /// pairwise add). Processes 32 i8 elements per iteration (two 16-element
    /// halves widened to i16, then accumulated to i32).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn int8_dot_avx2(a: &[i8], b: &[i8]) -> i32 {
        use std::arch::x86_64::{
            __m256i, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi8_epi16,
            _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16,
            _mm256_setzero_si256, _mm_add_epi32, _mm_cvtsi128_si32, _mm_srli_si128,
        };

        let len = a.len();
        let chunks = len / 32;
        let mut acc = _mm256_setzero_si256();

        for i in 0..chunks {
            let offset = i * 32;
            let va = _mm256_loadu_si256(a.as_ptr().add(offset).cast::<__m256i>());
            let vb = _mm256_loadu_si256(b.as_ptr().add(offset).cast::<__m256i>());

            // Widen lower 16 bytes: i8 → i16 (via sign extension)
            let va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
            let vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
            // Widen upper 16 bytes
            let va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(va));
            let vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(vb));

            // i16 × i16 → i32 with pairwise horizontal add
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va_lo, vb_lo));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va_hi, vb_hi));
        }

        // Horizontal sum: 8 × i32 → scalar
        let hi128 = _mm256_extracti128_si256::<1>(acc);
        let lo128 = _mm256_castsi256_si128(acc);
        let sum128 = _mm_add_epi32(lo128, hi128);
        let sum64 = _mm_add_epi32(sum128, _mm_srli_si128::<8>(sum128));
        let sum32 = _mm_add_epi32(sum64, _mm_srli_si128::<4>(sum64));
        let mut result = _mm_cvtsi128_si32(sum32);

        // Remainder
        let tail = chunks * 32;
        for i in tail..len {
            result += a[i] as i32 * b[i] as i32;
        }

        result
    }

    /// Scalar fallback.
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    #[inline]
    fn int8_dot_raw(a: &[i8], b: &[i8]) -> i32 {
        Self::int8_dot_scalar(a, b)
    }

    #[inline]
    fn int8_dot_scalar(a: &[i8], b: &[i8]) -> i32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x as i32 * y as i32).sum()
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
        assert_eq!(result, 5 + 2 * 6 + 3 * 7 + 4 * 8);
    }

    #[test]
    fn test_int8_dot_raw_large_vector() {
        let len = 128;
        let a: Vec<i8> = (0..len).map(|i| (i % 10) as i8).collect();
        let b: Vec<i8> = (0..len).map(|i| ((i + 3) % 10) as i8).collect();
        let expected: i32 = a.iter().zip(b.iter()).map(|(&x, &y)| x as i32 * y as i32).sum();
        let result = DotProduct::int8_dot_raw(&a, &b);
        assert_eq!(result, expected);
    }
}
