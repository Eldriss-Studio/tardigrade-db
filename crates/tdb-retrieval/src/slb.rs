//! Semantic Lookaside Buffer — fixed-size LRU cache of recently accessed cells.
//!
//! Stores cells in symmetric INT8 quantized format for maximum SIMD throughput.
//! Target: sub-5μs retrieval latency using NEON SDOT (ARM) / AVX2 VNNI (x86).
