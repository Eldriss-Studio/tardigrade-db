//! Symmetric INT8 scalar quantization for the SLB.
//!
//! Quantize: `scale = max(abs(vec)) / 127`, `q[i] = round(vec[i] / scale)`
//! Dequantize to FP32 on cache insertion to preserve L1-resident lookup performance.
