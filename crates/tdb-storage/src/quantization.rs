//! Quantization strategies for KV cache tensors.
//!
//! Implements the Strategy pattern via the `Quantizer` trait.
//! Supported formats: Q4 (group-wise 4-bit), Q8 (symmetric INT8), FP16.
