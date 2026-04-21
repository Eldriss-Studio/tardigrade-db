//! mmap-backed fixed-record arena for O(1) random access to memory cells.
//!
//! Each record occupies a fixed slot at `header_size + id * record_size`.
//! CPU reads use mmap; GPU DMA uses O_DIRECT + GDS (future).
