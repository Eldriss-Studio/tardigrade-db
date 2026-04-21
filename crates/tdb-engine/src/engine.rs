//! Top-level engine: Facade over storage, retrieval, and governance layers.
//!
//! Coordinates `BlockPool` + `BruteForceRetriever` + `ImportanceScorer` + `TierStateMachine`
//! behind a unified `mem_write` / `mem_read` API.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tdb_core::error::{Result, TardigradeError};
use tdb_core::memory_cell::{MemoryCell, MemoryCellBuilder};
use tdb_core::{CellId, OwnerId, Tier};
use tdb_governance::decay::recency_decay;
use tdb_governance::scoring::ImportanceScorer;
use tdb_governance::tiers::TierStateMachine;
use tdb_retrieval::attention::BruteForceRetriever;
use tdb_storage::block_pool::BlockPool;

/// Per-cell governance state tracked by the engine.
#[derive(Debug)]
struct CellGovernance {
    scorer: ImportanceScorer,
    tier_sm: TierStateMachine,
    /// Days since last update (for recency decay during retrieval).
    days_since_update: f32,
}

/// A result from `mem_read` including the cell data and its retrieval score.
#[derive(Debug)]
pub struct ReadResult {
    pub cell: MemoryCell,
    pub score: f32,
    pub tier: Tier,
}

/// Top-level `TardigradeDB` engine.
///
/// Facade pattern: single entry point coordinating storage, retrieval, and governance.
#[derive(Debug)]
pub struct Engine {
    pool: BlockPool,
    retriever: BruteForceRetriever,
    governance: HashMap<CellId, CellGovernance>,
    next_id: CellId,
    dir: PathBuf,
}

impl Engine {
    /// Open or create an engine at the given directory path.
    ///
    /// If the directory contains persisted cells from a prior session, the engine
    /// rebuilds all in-memory state (retriever index, governance scores/tiers, ID counter)
    /// by reading each cell from disk — the **Memento pattern**.
    pub fn open(dir: &Path) -> Result<Self> {
        let pool = BlockPool::open(dir)?;
        Self::rebuild_from_pool(pool, dir)
    }

    /// Open with a custom segment size threshold (for testing segment rollover).
    pub fn open_with_segment_size(dir: &Path, segment_size: u64) -> Result<Self> {
        let pool = BlockPool::open_with_segment_size(dir, segment_size)?;
        Self::rebuild_from_pool(pool, dir)
    }

    /// Rebuild all derived in-memory state from the persisted `BlockPool`.
    ///
    /// For each cell on disk: populate the retriever index, reconstruct the
    /// governance scorer/tier from persisted metadata, and track the max ID.
    fn rebuild_from_pool(pool: BlockPool, dir: &Path) -> Result<Self> {
        let mut retriever = BruteForceRetriever::new();
        let mut governance = HashMap::new();
        let mut max_id: CellId = 0;

        // Collect IDs first to avoid borrowing pool during iteration + get.
        let cell_ids: Vec<CellId> = pool.iter_cell_ids().collect();

        for cell_id in &cell_ids {
            let cell = pool.get(*cell_id)?;

            // Rebuild retriever index from persisted key vectors.
            retriever.insert(cell.id, cell.owner, cell.layer, &cell.key);

            // Reconstruct governance state from persisted metadata (Memento).
            let scorer = ImportanceScorer::new(cell.meta.importance);
            let tier_sm = TierStateMachine::with_tier(cell.meta.tier);

            governance.insert(cell.id, CellGovernance { scorer, tier_sm, days_since_update: 0.0 });

            if cell.id >= max_id {
                max_id = cell.id + 1;
            }
        }

        Ok(Self { pool, retriever, governance, next_id: max_id, dir: dir.to_path_buf() })
    }

    /// Write key/value vectors to the engine. Returns the assigned cell ID.
    ///
    /// This is the `MEM.WRITE` operation: persist a KV pair from inference,
    /// index it for retrieval, and initialize governance metadata.
    pub fn mem_write(
        &mut self,
        owner: OwnerId,
        layer: u16,
        key: &[f32],
        value: Vec<f32>,
        salience: f32,
    ) -> Result<CellId> {
        let id = self.next_id;
        self.next_id += 1;

        let now_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);

        // Compute governance state BEFORE persisting so on-disk metadata
        // matches in-memory state (required for Memento rebuild on reopen).
        let mut scorer = ImportanceScorer::new(salience);
        scorer.on_update(); // +5 for the initial write
        let mut tier_sm = TierStateMachine::new();
        tier_sm.evaluate(scorer.importance());

        let cell = MemoryCellBuilder::new(id, owner, layer, key.to_vec(), value)
            .importance(scorer.importance())
            .tier(tier_sm.current())
            .created_at(now_nanos)
            .updated_at(now_nanos)
            .build();

        // Persist to block pool (Q4 quantized on disk).
        self.pool.append(&cell)?;

        // Index the key for brute-force retrieval.
        self.retriever.insert(id, owner, layer, key);

        self.governance.insert(id, CellGovernance { scorer, tier_sm, days_since_update: 0.0 });

        Ok(id)
    }

    /// Read the top-k most relevant cells for a query key.
    ///
    /// This is the `MEM.READ` operation: retrieve by latent-space attention scoring,
    /// apply recency decay, boost importance on access, and return results.
    pub fn mem_read(
        &mut self,
        query_key: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
    ) -> Result<Vec<ReadResult>> {
        if query_key.is_empty() {
            return Ok(Vec::new());
        }

        let raw_results = self.retriever.query(query_key, k * 2, owner_filter);

        let mut results = Vec::with_capacity(k);
        for rr in raw_results {
            let decay_factor = self
                .governance
                .get(&rr.cell_id)
                .map_or(1.0, |g| recency_decay(g.days_since_update));

            let adjusted_score = rr.score * decay_factor;

            let tier =
                self.governance.get(&rr.cell_id).map_or(Tier::Draft, |g| g.tier_sm.current());

            // Read full cell from storage. Only boost importance if the cell
            // is successfully retrieved and will be returned to the caller.
            match self.pool.get(rr.cell_id) {
                Ok(cell) => {
                    if let Some(gov) = self.governance.get_mut(&rr.cell_id) {
                        gov.scorer.on_access();
                        gov.tier_sm.evaluate(gov.scorer.importance());
                    }
                    results.push(ReadResult { cell, score: adjusted_score, tier });
                }
                Err(TardigradeError::CellNotFound(_)) => continue,
                Err(e) => return Err(e),
            }

            if results.len() >= k {
                break;
            }
        }

        // Sort by adjusted score descending.
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    /// Get the current tier of a cell.
    pub fn cell_tier(&self, cell_id: CellId) -> Option<Tier> {
        self.governance.get(&cell_id).map(|g| g.tier_sm.current())
    }

    /// Get the current importance score of a cell.
    pub fn cell_importance(&self, cell_id: CellId) -> Option<f32> {
        self.governance.get(&cell_id).map(|g| g.scorer.importance())
    }

    /// Total number of cells in the engine.
    pub fn cell_count(&self) -> usize {
        self.pool.cell_count()
    }

    /// Directory path of this engine.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Simulate passage of time for governance decay.
    ///
    /// In production this would be called by a background sweep;
    /// for testing we expose it directly.
    pub fn advance_days(&mut self, days: f32) {
        for gov in self.governance.values_mut() {
            let old_whole = gov.days_since_update as u32;
            gov.days_since_update += days;
            let new_whole = gov.days_since_update as u32;
            let elapsed = new_whole.saturating_sub(old_whole);
            if elapsed > 0 {
                gov.scorer.apply_daily_decay(elapsed);
                gov.tier_sm.evaluate(gov.scorer.importance());
            }
        }
    }
}
