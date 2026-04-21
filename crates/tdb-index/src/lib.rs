//! Organization layer for `TardigradeDB`: ANN graph, causal graph, and write-ahead log.
//!
//! `tdb-index` implements the **neuro-symbolic dual topology** described in the Aeon
//! architecture: two complementary data structures that organize memory cells, plus a
//! WAL for crash recovery of the causal graph.
//!
//! # Vamana Graph — Vector Index
//!
//! [`VamanaIndex`] is a **DiskANN-style single-layer graph** for approximate nearest
//! neighbor (ANN) search. It is the cold-path index used when the per-agent cell count
//! grows beyond the brute-force threshold (~10K) handled by `tdb-retrieval`.
//!
//! ```text
//! Insert phase:                         Query phase:
//! ┌───────┐  insert(id, vec)            query(q, k=5)
//! │       │ ──────────────────►         │
//! │ nodes │   add node to list          ▼ greedy beam search
//! │       │                        seed at medoid
//! └───────┘  build()               expand best candidates
//!             │                    explore neighbor edges
//!             ▼                    return top-k by score
//!    connect each node to
//!    its R nearest neighbors
//! ```
//!
//! **Key design choices vs. HNSW:**
//! - Single layer (no multi-layer hierarchy) → simpler build, faster cold-start
//! - Medoid seeding → avoids pathological cases when queries are far from the centroid
//! - Page-clustered insertions (future) → `DiskANN`'s disk-locality trick for `NVMe` I/O
//!
//! Configuration:
//! - `max_degree` (R): out-degree per node. Higher = better recall, more memory.
//! - `search_list_size` (L) is set automatically to `max(k, R) * 2` during queries.
//!
//! # Trace Graph — Causal Index
//!
//! [`TraceGraph`] is a **directed episodic graph** where nodes are [`CellId`]s and
//! edges represent causal relationships between stored memories:
//!
//! ```text
//!   cell_A ──[CausedBy]──► cell_B ──[CausedBy]──► cell_C
//!   cell_A ──[Supports]──► cell_D
//!   cell_E ──[Contradicts]► cell_A
//! ```
//!
//! Edge types: `CausedBy`, `Follows`, `Supports`, `Contradicts`.
//!
//! The graph supports transitive ancestor queries (`ancestors(node, EdgeType)`),
//! which an agent can use to reconstruct the causal chain that led to any stored memory.
//!
//! # Write-Ahead Log — Crash Recovery
//!
//! [`Wal`] is an append-only binary log that records every [`TraceGraph`] mutation
//! before it is applied. On restart, mutations are replayed from the WAL before the
//! in-memory graph is used.
//!
//! ```text
//! Graph mutation requested
//!        │
//!        ▼
//!   WAL.append(entry)   ← fsync to disk FIRST
//!        │
//!        ▼
//!   TraceGraph.add_edge(...)   ← then apply in memory
//!        │
//!        ▼
//!   (periodic) WAL.checkpoint()  ← truncate WAL after snapshot
//! ```
//!
//! WAL record format (little-endian): `type(1) | src(8) | dst(8) | edge_type(1) | timestamp(8)`
//! = 26 bytes per record. Fixed-size records allow O(1) position arithmetic during replay.
//!
//! # Usage
//!
//! ```rust,no_run
//! use tdb_index::vamana::VamanaIndex;
//! use tdb_index::trace::{TraceGraph, EdgeType};
//! use tdb_index::wal::Wal;
//!
//! // Build a vector index for cold-path ANN retrieval.
//! let mut idx = VamanaIndex::new(/* dim */ 128, /* max_degree */ 32);
//! idx.insert(0, &[0.5f32; 128]);
//! idx.insert(1, &[0.1f32; 128]);
//! idx.build(); // O(n²) — call after all inserts
//! let results = idx.query(&[0.5f32; 128], /* k */ 1);
//! assert_eq!(results[0].0, 0);
//!
//! // Record a causal edge between two cells (WAL-backed).
//! let dir = std::path::Path::new("/tmp/tdb-wal");
//! let mut wal = Wal::open(dir).unwrap();
//! let mut graph = TraceGraph::new();
//! let entry = tdb_index::wal::WalEntry::AddEdge {
//!     src: 0, dst: 1, edge_type: EdgeType::CausedBy as u8, timestamp: 1000,
//! };
//! wal.append(&entry).unwrap();        // durable FIRST
//! graph.add_edge(0, 1, EdgeType::CausedBy, 1000); // then in memory
//! ```
//!
//! [`VamanaIndex`]: vamana::VamanaIndex
//! [`TraceGraph`]: trace::TraceGraph
//! [`Wal`]: wal::Wal
//! [`CellId`]: tdb_core::CellId

#![deny(unsafe_code)]

pub mod trace;
pub mod vamana;
pub mod wal;
