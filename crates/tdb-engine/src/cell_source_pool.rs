//! `BlockPoolCellSource` — Repository Adapter bridging `BlockPool` to
//! [`CellSource`].
//!
//! The retrieval crate defines [`CellSource`] (one method:
//! `get_encoded_key`). The storage crate defines `BlockPool` (a
//! repository with `get(cell_id) -> Result<MemoryCell>`). This adapter
//! is the thin glue that lets a lazy retriever (e.g.
//! `PerTokenRetriever`) score cells stored in `BlockPool` without the
//! retriever knowing anything about segments, fsync, or recovery.
//!
//! Held by borrowed reference (no `Arc`), constructed at query time
//! by the engine.

use tdb_core::CellId;
use tdb_retrieval::cell_source::CellSource;
use tdb_storage::block_pool::BlockPool;

/// Repository Adapter: `&BlockPool` → `dyn CellSource`.
#[derive(Debug)]
pub struct BlockPoolCellSource<'a> {
    pool: &'a BlockPool,
}

impl<'a> BlockPoolCellSource<'a> {
    /// Wrap a borrowed `BlockPool` reference as a [`CellSource`].
    pub fn new(pool: &'a BlockPool) -> Self {
        Self { pool }
    }
}

impl CellSource for BlockPoolCellSource<'_> {
    fn get_encoded_key(&self, id: CellId) -> Option<Vec<f32>> {
        self.pool.get(id).ok().map(|cell| cell.key)
    }
}
