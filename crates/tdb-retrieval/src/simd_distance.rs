//! SIMD-accelerated distance/dot-product functions.
//!
//! Strategy pattern: scalar fallback, NEON SDOT (ARM), AVX2 (x86).
//! Compile-time feature gates with runtime dispatch where needed.
