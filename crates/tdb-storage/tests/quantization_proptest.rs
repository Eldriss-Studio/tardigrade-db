//! Property-based tests for the Q4 quantization codec.
//!
//! The existing example-based tests in `src/quantization.rs` pin a
//! handful of specific inputs. These properties cover the space of
//! *all* finite f32 vectors up to a bounded length, catching the
//! class of invariant violations that example-based tests can't —
//! group-boundary off-by-ones, sign-bit handling at the wrapping
//! edge, zero-vector edge cases, padding bugs in the last group.
//!
//! The contract Q4 makes is bounded-error lossy compression. The
//! properties express each invariant as a universal claim:
//!
//! 1. Round-trip preserves length exactly (lossless on shape).
//! 2. Round-trip is bounded — every element is within half the
//!    group's scale of the original (the quantization step).
//! 3. The zero vector round-trips to itself exactly.
//! 4. Quantization is deterministic — the same input produces the
//!    same bytes, scales, and dequantized output.
//! 5. Re-quantizing a dequantized vector produces (almost) the same
//!    bytes — the codec is approximately idempotent past one round.

use proptest::prelude::*;
use tdb_storage::quantization::{DequantizeStrategy, Q4, QuantizeStrategy};

/// Vectors of finite f32 in a realistic range — KV cache values are
/// dominated by softmax-normalized weights that rarely exceed ±10.
/// Excluding NaN / ±inf because Q4 makes no claim about them; the
/// existing example tests cover those cases explicitly.
fn finite_f32_vec() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        prop::num::f32::POSITIVE | prop::num::f32::NEGATIVE | prop::num::f32::ZERO,
        0..=256,
    )
}

/// Compute the maximum allowed per-element error after a round-trip.
///
/// Q4's contract: each element is quantized to one of 15 buckets
/// spanning `[-abs_max, +abs_max]`. The quantization step is
/// `abs_max / 7.0`; with `round()`, the worst-case per-element
/// error is half a step — `abs_max / 14`. Padding the bound by a
/// small epsilon absorbs f32 rounding inside the bound calculation
/// itself.
fn max_per_element_error(group_max_abs: f32) -> f32 {
    let step = group_max_abs / 7.0;
    step / 2.0 + 1e-6
}

proptest! {
    /// AT property #1: dequantize(quantize(v)).len() == v.len()
    #[test]
    fn round_trip_preserves_length(v in finite_f32_vec()) {
        let tensor = Q4::quantize(&v);
        let restored = Q4::dequantize(&tensor);
        prop_assert_eq!(restored.len(), v.len(),
            "Q4 round-trip changed length from {} to {}", v.len(), restored.len());
    }

    /// AT property #2: every element is within half a group's
    /// quantization step of the original.
    ///
    /// The bound is per-group — different groups have different
    /// scales, so we compare against the group's own max.
    #[test]
    fn round_trip_error_is_bounded(v in finite_f32_vec()) {
        const GROUP_SIZE: usize = 32;
        let tensor = Q4::quantize(&v);
        let restored = Q4::dequantize(&tensor);
        for (group_idx, original_group) in v.chunks(GROUP_SIZE).enumerate() {
            let group_max_abs = original_group.iter()
                .map(|x| x.abs())
                .fold(0.0f32, f32::max);
            let bound = max_per_element_error(group_max_abs);
            let restored_group_start = group_idx * GROUP_SIZE;
            for (i, &orig) in original_group.iter().enumerate() {
                let restored_val = restored[restored_group_start + i];
                let err = (restored_val - orig).abs();
                prop_assert!(err <= bound,
                    "element {} of group {} restored to {} from {} (err {:e}, bound {:e})",
                    i, group_idx, restored_val, orig, err, bound);
            }
        }
    }

    /// AT property #3: the zero vector round-trips exactly.
    ///
    /// Quantizing zeros produces scale=1.0 (handled specially when
    /// abs_max == 0); dequantizing the zero-point bytes (8) yields
    /// 0.0 exactly. No rounding error possible.
    #[test]
    fn zero_vector_round_trips_exactly(len in 0usize..=256) {
        let zeros = vec![0.0f32; len];
        let tensor = Q4::quantize(&zeros);
        let restored = Q4::dequantize(&tensor);
        prop_assert_eq!(restored, zeros);
    }

    /// AT property #4: quantization is deterministic.
    ///
    /// Two independent quantizations of the same input produce
    /// identical bytes, scales, and length. Catches any accidental
    /// reliance on uninitialised memory, hash seed, or allocation
    /// addresses.
    #[test]
    fn quantization_is_deterministic(v in finite_f32_vec()) {
        let a = Q4::quantize(&v);
        let b = Q4::quantize(&v);
        prop_assert_eq!(&a.data, &b.data);
        prop_assert_eq!(&a.scales, &b.scales);
        prop_assert_eq!(a.original_len, b.original_len);
    }

    /// AT property #5: round-trip is (approximately) idempotent.
    ///
    /// q1 = quantize(v); v1 = dequantize(q1); q2 = quantize(v1).
    /// `q2` should equal `q1` byte-for-byte — dequantization already
    /// snapped to the quantization grid, so re-quantizing must hit
    /// the same buckets. Catches any drift in the bucket-selection
    /// rounding rule.
    ///
    /// Edge: when `abs_max == 0`, the code uses `scale = 1.0` to
    /// avoid div-by-zero; the dequantized vector is all zeros, and
    /// re-quantizing zeros also picks `scale = 1.0`. Stable.
    #[test]
    fn round_trip_is_idempotent_past_one_cycle(v in finite_f32_vec()) {
        let q1 = Q4::quantize(&v);
        let v1 = Q4::dequantize(&q1);
        let q2 = Q4::quantize(&v1);
        prop_assert_eq!(&q1.data, &q2.data);
        prop_assert_eq!(&q1.scales, &q2.scales);
    }
}
