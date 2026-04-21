//! Persistent quantized storage for KV-cache tensors — the storage layer of TardigradeDB.
//!
//! `tdb-storage` owns the write path: it receives [`MemoryCell`] values, compresses
//! their key and value vectors to Q4 (4-bit), and appends them to an mmap-backed
//! segment file. Reads dequantize on-the-fly. The entire layer is crash-safe via
//! append-only writes and a monotone index rebuild on open.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │                     BlockPool                        │
//! │                                                      │
//! │  in-memory:  BTreeMap<CellId → (segment_id, offset)>│
//! │                         │                            │
//! │  on-disk:   ┌──────────┐ ┌──────────┐ ┌──────────┐  │
//! │             │ seg_000  │ │ seg_001  │ │ seg_002  │  │  ← append-only files
//! │             │  256 MB  │ │  256 MB  │ │ (active) │  │
//! │             └──────────┘ └──────────┘ └──────────┘  │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! Each segment file is a sequential log of binary records. New records are always
//! appended; existing records are never mutated. On open, [`BlockPool`] scans all
//! segments to rebuild the `CellId → location` index — this is the recovery path.
//!
//! # Quantization
//!
//! Key and value vectors are stored in **Q4 group-wise 4-bit** format, matching the
//! `Q4_0` scheme from [llama.cpp]. Each group of 32 floats shares one `f32` scale
//! factor; the 32 values are quantized to 4-bit unsigned integers and packed two-per-byte.
//!
//! Compression characteristics vs. raw `f32`:
//! - **Size**: ~4× smaller (4 bits vs 32 bits per element, plus scale overhead)
//! - **Error**: mean squared error < 0.01 on typical activation distributions
//! - **Speed**: dequantize is a simple multiply-and-shift, negligible vs. I/O
//!
//! Position encodings are stored **unquantized** (`f32`) because they must be
//! reproduced exactly for safe historical KV block reuse.
//!
//! # Segment File Format
//!
//! ```text
//! ┌─────────────────────────┐
//! │ magic: "TDBS" (4 bytes) │
//! │ version: u32 (4 bytes)  │
//! ├─────────────────────────┤  ← FILE_HEADER_SIZE = 8
//! │ record_len: u32         │  ← total bytes of this record (excludes this field)
//! │ cell_id: u64            │
//! │ owner: u64              │
//! │ layer: u16              │
//! │ … fixed metadata …      │
//! │ key (Q4 packed)         │
//! │ value (Q4 packed)       │
//! │ pos_encoding (f32[])    │
//! ├─────────────────────────┤
//! │ … next record …         │
//! └─────────────────────────┘
//! ```
//!
//! All integers are **little-endian**. The `record_len` prefix enables the recovery
//! scanner to skip partial records at EOF without corrupting the index.
//!
//! # Recovery
//!
//! [`block_pool::BlockPool::open`] rebuilds the entire in-memory index by calling [`scan_segment`]
//! on every segment file. `scan_segment` reads only the `record_len` and `cell_id`
//! prefix of each record — it does not deserialize the full tensor — making recovery
//! proportional to the number of cells, not their size. Partial records at EOF
//! (from a crash mid-write) are silently discarded.
//!
//! # Usage
//!
//! ```rust,no_run
//! use tdb_storage::block_pool::BlockPool;
//! use tdb_core::memory_cell::MemoryCellBuilder;
//!
//! let dir = std::path::Path::new("/tmp/tdb-data");
//! let mut pool = BlockPool::open(dir).unwrap();
//!
//! // Write a cell (key/value are compressed to Q4 on disk).
//! let cell = MemoryCellBuilder::new(0, 1, 12, vec![0.5f32; 64], vec![0.3f32; 64]).build();
//! let cell_id = pool.append(&cell).unwrap();
//!
//! // Read it back (dequantized from Q4).
//! let retrieved = pool.get(cell_id).unwrap();
//! assert_eq!(retrieved.id, 0);
//! ```
//!
//! [`MemoryCell`]: tdb_core::memory_cell::MemoryCell
//! [`BlockPool`]: block_pool::BlockPool
//! [`scan_segment`]: segment::scan_segment
//! [llama.cpp]: https://github.com/ggerganov/llama.cpp

#![deny(unsafe_code)]

pub mod arena;
pub mod block_pool;
pub mod quantization;
pub mod segment;
