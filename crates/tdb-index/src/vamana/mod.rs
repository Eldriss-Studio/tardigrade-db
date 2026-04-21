//! Vamana graph index (DiskANN-style, NOT HNSW).
//!
//! Single-layer graph, natively disk-aware. PQ-compressed vectors in RAM
//! for navigation, full vectors on NVMe. Page-node alignment (PageANN-inspired)
//! so graph traversal maps to sequential page reads.
