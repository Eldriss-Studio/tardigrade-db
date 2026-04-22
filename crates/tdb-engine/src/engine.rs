//! Top-level engine: Facade over storage, retrieval, governance, and organization layers.
//!
//! Coordinates `BlockPool` + `BruteForceRetriever` + SLB + `VamanaIndex` + `TraceGraph` + WAL
//! behind a unified `mem_write` / `mem_read` API.
//!
//! ## Retrieval Pipeline (Chain of Responsibility)
//!
//! ```text
//! query → SLB (INT8, sub-5μs) → Vamana (graph ANN) → BruteForce (exact)
//! ```
//!
//! Each stage returns results or passes to the next. Results are merged by `CellId`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use tdb_core::error::{Result, TardigradeError};
use tdb_core::memory_cell::{MemoryCell, MemoryCellBuilder};
use tdb_core::synaptic_bank::SynapticBankEntry;
use tdb_core::{CellId, OwnerId, Tier};
use tdb_governance::decay::recency_decay;
use tdb_governance::scoring::ImportanceScorer;
use tdb_governance::tiers::TierStateMachine;
use tdb_index::trace::{EdgeType, TraceGraph};
use tdb_index::vamana::VamanaIndex;
use tdb_index::wal::{Wal, WalEntry};
use tdb_retrieval::attention::{BruteForceRetriever, RetrievalResult};
use tdb_retrieval::slb::SemanticLookasideBuffer;
use tdb_storage::block_pool::BlockPool;
use tdb_storage::synaptic_store::SynapticStore;

/// Default SLB capacity.
const DEFAULT_SLB_CAPACITY: usize = 4096;
/// Default Vamana max degree.
const DEFAULT_VAMANA_MAX_DEGREE: usize = 16;
/// Default cell count threshold before activating Vamana.
const DEFAULT_VAMANA_THRESHOLD: usize = 10_000;

/// Per-cell governance state tracked by the engine.
#[derive(Debug)]
struct CellGovernance {
    scorer: ImportanceScorer,
    tier_sm: TierStateMachine,
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
/// Facade pattern: single entry point coordinating storage, retrieval,
/// governance, and organization (trace + WAL).
pub struct Engine {
    pool: BlockPool,
    retriever: BruteForceRetriever,
    slb: SemanticLookasideBuffer,
    vamana: Option<VamanaIndex>,
    trace: TraceGraph,
    wal: Wal,
    synaptic_store: SynapticStore,
    governance: HashMap<CellId, CellGovernance>,
    next_id: CellId,
    dir: PathBuf,
    vamana_threshold: usize,
    /// Key dimension (detected from first write, used for SLB/Vamana init).
    key_dim: Option<usize>,
}

impl Engine {
    /// Open or create an engine at the given directory path.
    pub fn open(dir: &Path) -> Result<Self> {
        Self::open_with_options(dir, DEFAULT_VAMANA_THRESHOLD, None)
    }

    /// Open with a custom segment size threshold (for testing segment rollover).
    pub fn open_with_segment_size(dir: &Path, segment_size: u64) -> Result<Self> {
        Self::open_with_options(dir, DEFAULT_VAMANA_THRESHOLD, Some(segment_size))
    }

    /// Open with a custom Vamana activation threshold.
    pub fn open_with_vamana_threshold(dir: &Path, threshold: usize) -> Result<Self> {
        Self::open_with_options(dir, threshold, None)
    }

    fn open_with_options(
        dir: &Path,
        vamana_threshold: usize,
        segment_size: Option<u64>,
    ) -> Result<Self> {
        let pool = match segment_size {
            Some(size) => BlockPool::open_with_segment_size(dir, size)?,
            None => BlockPool::open(dir)?,
        };

        // Open or create the SynapticStore (Repository for LoRA adapters).
        let synaptic_store =
            SynapticStore::open(dir).map_err(|e| TardigradeError::Io { source: e })?;

        // Open or create the WAL for trace graph durability.
        let wal = Wal::open(dir).map_err(|e| TardigradeError::WalRecovery(e.to_string()))?;

        // Replay WAL to rebuild TraceGraph (Observer pattern recovery).
        let mut trace = TraceGraph::new();
        let wal_entries = wal.replay().map_err(|e| TardigradeError::WalRecovery(e.to_string()))?;
        for entry in &wal_entries {
            match entry {
                WalEntry::AddEdge { src, dst, edge_type, timestamp } => {
                    if let Some(et) = EdgeType::from_u8(*edge_type) {
                        trace.add_edge(*src, *dst, et, *timestamp);
                    }
                }
            }
        }

        // Rebuild derived state from persisted cells (Memento pattern).
        let mut retriever = BruteForceRetriever::new();
        let mut governance = HashMap::new();
        let mut next_id: CellId = 0;
        let mut key_dim: Option<usize> = None;

        let cell_ids: Vec<CellId> = pool.iter_cell_ids().collect();

        for cell_id in &cell_ids {
            let cell = pool.get(*cell_id)?;

            if key_dim.is_none() {
                key_dim = Some(cell.key.len());
            }

            retriever.insert(cell.id, cell.owner, cell.layer, &cell.key);

            let scorer = ImportanceScorer::new(cell.meta.importance);
            let tier_sm = TierStateMachine::with_tier(cell.meta.tier);
            governance.insert(cell.id, CellGovernance { scorer, tier_sm, days_since_update: 0.0 });

            if cell.id >= next_id {
                next_id = cell.id + 1;
            }
        }

        let dim = key_dim.unwrap_or(128);
        let slb = SemanticLookasideBuffer::new(DEFAULT_SLB_CAPACITY, dim);

        Ok(Self {
            pool,
            retriever,
            slb,
            vamana: None,
            trace,
            wal,
            synaptic_store,
            governance,
            next_id,
            dir: dir.to_path_buf(),
            vamana_threshold,
            key_dim: if cell_ids.is_empty() { None } else { key_dim },
        })
    }

    /// Write key/value vectors to the engine. Returns the assigned cell ID.
    ///
    /// `parent_cell_id`: optional causal parent for Trace graph edges.
    pub fn mem_write(
        &mut self,
        owner: OwnerId,
        layer: u16,
        key: &[f32],
        value: Vec<f32>,
        salience: f32,
        parent_cell_id: Option<CellId>,
    ) -> Result<CellId> {
        let id = self.next_id;
        self.next_id += 1;

        // Detect key dimension on first write.
        if self.key_dim.is_none() {
            self.key_dim = Some(key.len());
            // Reinitialize SLB with correct dimension.
            self.slb = SemanticLookasideBuffer::new(DEFAULT_SLB_CAPACITY, key.len());
        }

        let now_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);

        // Compute governance before persisting (Memento rebuild requirement).
        let mut scorer = ImportanceScorer::new(salience);
        scorer.on_update();
        let mut tier_sm = TierStateMachine::new();
        tier_sm.evaluate(scorer.importance());

        let cell = MemoryCellBuilder::new(id, owner, layer, key.to_vec(), value)
            .importance(scorer.importance())
            .tier(tier_sm.current())
            .created_at(now_nanos)
            .updated_at(now_nanos)
            .build();

        // Persist to block pool.
        self.pool.append(&cell)?;

        // Index for retrieval (BruteForce + SLB).
        self.retriever.insert(id, owner, layer, key);
        self.slb.insert(id, owner, key);

        // If Vamana is active, insert online.
        if let Some(ref mut vamana) = self.vamana {
            vamana.insert(id, key);
            // Rebuild needed after insert — for now, defer to threshold activation.
        }

        self.governance.insert(id, CellGovernance { scorer, tier_sm, days_since_update: 0.0 });

        // Trace graph: log causal edge via WAL (Observer pattern).
        if let Some(parent_id) = parent_cell_id {
            let wal_entry = WalEntry::AddEdge {
                src: id,
                dst: parent_id,
                edge_type: EdgeType::CausedBy as u8,
                timestamp: now_nanos,
            };
            self.wal.append(&wal_entry).map_err(|e| TardigradeError::WalRecovery(e.to_string()))?;
            self.trace.add_edge(id, parent_id, EdgeType::CausedBy, now_nanos);
        }

        // Lazy Vamana activation: build when cell count crosses threshold.
        if self.vamana.is_none() && self.pool.cell_count() >= self.vamana_threshold {
            self.activate_vamana()?;
        }

        Ok(id)
    }

    /// Read the top-k most relevant cells for a query key.
    ///
    /// Chain of Responsibility: SLB → Vamana → `BruteForce`.
    pub fn mem_read(
        &mut self,
        query_key: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
    ) -> Result<Vec<ReadResult>> {
        if query_key.is_empty() {
            return Ok(Vec::new());
        }

        let mut seen = HashSet::new();
        let mut candidates: Vec<RetrievalResult> = Vec::new();

        // Stage 1: SLB (hot path, INT8 quantized).
        if !self.slb.is_empty() {
            let slb_results = self.slb.query(query_key, k * 2);
            for r in slb_results {
                if owner_filter.is_none_or(|o| o == r.owner) && seen.insert(r.cell_id) {
                    candidates.push(r);
                }
            }
        }

        // Stage 2: Vamana (warm path, graph ANN) if active and SLB didn't fill k.
        if candidates.len() < k {
            if let Some(ref vamana) = self.vamana {
                let vamana_results = vamana.query(query_key, k * 2);
                for (cell_id, score) in vamana_results {
                    if seen.insert(cell_id) {
                        let owner = self.governance.get(&cell_id).map_or(0, |_| 0); // Owner not stored in Vamana; filter later.
                        candidates.push(RetrievalResult { cell_id, owner, score });
                    }
                }
            }
        }

        // Stage 3: BruteForce (cold path, exact).
        if candidates.len() < k {
            let brute_results = self.retriever.query(query_key, k * 2, owner_filter);
            for r in brute_results {
                if seen.insert(r.cell_id) {
                    candidates.push(r);
                }
            }
        }

        // Score adjustment + governance boost + cell retrieval.
        candidates
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        let mut results = Vec::with_capacity(k);
        for rr in candidates {
            // Owner filter (needed for Vamana results which don't filter internally).
            if let Some(filter_owner) = owner_filter {
                if let Some(gov) = self.governance.get(&rr.cell_id) {
                    let _ = gov; // governance exists
                }
                // Read cell to check owner.
                match self.pool.get(rr.cell_id) {
                    Ok(ref cell) if cell.owner != filter_owner => continue,
                    Err(TardigradeError::CellNotFound(_)) => continue,
                    Err(e) => return Err(e),
                    _ => {}
                }
            }

            let decay_factor = self
                .governance
                .get(&rr.cell_id)
                .map_or(1.0, |g| recency_decay(g.days_since_update));

            let adjusted_score = rr.score * decay_factor;

            let tier =
                self.governance.get(&rr.cell_id).map_or(Tier::Draft, |g| g.tier_sm.current());

            match self.pool.get(rr.cell_id) {
                Ok(cell) => {
                    if let Some(gov) = self.governance.get_mut(&rr.cell_id) {
                        gov.scorer.on_access();
                        gov.tier_sm.evaluate(gov.scorer.importance());
                    }
                    // Populate SLB with accessed cells for future hot-path hits.
                    self.slb.insert(rr.cell_id, cell.owner, &cell.key);

                    results.push(ReadResult { cell, score: adjusted_score, tier });
                }
                Err(TardigradeError::CellNotFound(_)) => continue,
                Err(e) => return Err(e),
            }

            if results.len() >= k {
                break;
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    /// Get transitive ancestors of a cell following `CausedBy` edges.
    pub fn trace_ancestors(&self, cell_id: CellId) -> Vec<CellId> {
        self.trace.ancestors(cell_id, EdgeType::CausedBy)
    }

    /// Whether the Vamana index is currently active.
    pub fn has_vamana(&self) -> bool {
        self.vamana.is_some()
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

    /// Store a `SynapticBankEntry` (`LoRA` adapter) for an agent/user.
    pub fn store_synapsis(&mut self, entry: &SynapticBankEntry) -> Result<()> {
        self.synaptic_store.append(entry).map_err(|e| TardigradeError::Io { source: e })
    }

    /// Load all `SynapticBankEntry` records for a given owner.
    pub fn load_synapsis(&self, owner: OwnerId) -> Result<Vec<SynapticBankEntry>> {
        self.synaptic_store.load_by_owner(owner).map_err(|e| TardigradeError::Io { source: e })
    }

    /// Build and activate the Vamana index from all existing keys.
    fn activate_vamana(&mut self) -> Result<()> {
        let dim = self.key_dim.unwrap_or(128);
        let mut vamana = VamanaIndex::new(dim, DEFAULT_VAMANA_MAX_DEGREE);

        let cell_ids: Vec<CellId> = self.pool.iter_cell_ids().collect();
        for cell_id in &cell_ids {
            let cell = self.pool.get(*cell_id)?;
            vamana.insert(cell.id, &cell.key);
        }
        vamana.build();

        self.vamana = Some(vamana);
        Ok(())
    }
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("dir", &self.dir)
            .field("cell_count", &self.pool.cell_count())
            .field("has_vamana", &self.vamana.is_some())
            .field("trace_edges", &self.trace.edge_count())
            .finish_non_exhaustive()
    }
}
