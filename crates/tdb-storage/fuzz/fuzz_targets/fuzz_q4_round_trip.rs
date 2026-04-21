#![no_main]

use libfuzzer_sys::fuzz_target;
use tdb_storage::quantization::{DequantizeStrategy, Q4, QuantizeStrategy};

fuzz_target!(|data: &[u8]| {
    // Interpret arbitrary bytes as f32 values.
    if data.len() < 4 || data.len() % 4 != 0 {
        return;
    }

    let floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    // Skip inputs with non-finite values — quantization assumes finite input.
    if floats.iter().any(|v| !v.is_finite()) {
        return;
    }

    let quantized = Q4::quantize(&floats);
    let restored = Q4::dequantize(&quantized);

    // Invariant: output length must match input length.
    assert_eq!(restored.len(), floats.len());

    // Invariant: no NaN or Inf in output.
    for (i, v) in restored.iter().enumerate() {
        assert!(v.is_finite(), "NaN/Inf at index {i}");
    }
});
