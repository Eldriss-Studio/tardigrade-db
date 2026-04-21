//! Block pool — the Repository abstraction over segmented storage.
//!
//! Provides `append` and `get` operations over a collection of segment files,
//! with an in-memory B-tree index mapping CellId → (segment, offset).
