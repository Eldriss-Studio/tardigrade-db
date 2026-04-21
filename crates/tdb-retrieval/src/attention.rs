//! Latent-space attention score computation.
//!
//! Computes `score = softmax(q · K^T / √d_k)` directly against stored key vectors.
//! Uses brute-force SIMD matmul (MemArt-style) — faster than ANN at per-agent scale (<10K blocks).
