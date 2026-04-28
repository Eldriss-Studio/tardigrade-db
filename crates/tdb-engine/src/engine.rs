//! Top-level engine: Facade over storage, retrieval, governance, and organization layers.
//!
//! Coordinates `BlockPool` + `BruteForceRetriever` + SLB + `VamanaIndex` + `TraceGraph` + WAL
//! behind a unified `mem_write` / `mem_read` API.
//!
//! ## Retrieval Pipeline (Chain of Responsibility)
//!
//! ```text
//! encoded query → PerTokenRetriever(Top5Avg) → SLB fallback → Vamana/BruteForce fallback
//! plain query   → SLB hot cache → RetrieverPipeline fallback
//! ```
//!
//! Encoded per-token keys are mean-pooled before they enter fixed-dimension
//! stages such as SLB, Vamana, and brute-force fallback.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tdb_core::error::{Result, TardigradeError};
use tdb_core::kv_pack::{KVLayerPayload, KVPack, PackId, PackReadResult};
use tdb_core::memory_cell::{MemoryCell, MemoryCellBuilder};
use tdb_core::synaptic_bank::SynapticBankEntry;
use tdb_core::{CellId, OwnerId, Tier};
use tdb_governance::decay::recency_decay;
use tdb_governance::scoring::ImportanceScorer;
use tdb_governance::tiers::TierStateMachine;
use tdb_index::trace::{EdgeType, TraceGraph};
use tdb_index::vamana::VamanaIndex;
use tdb_index::wal::{Wal, WalEntry};
use tdb_retrieval::attention::RetrievalResult;
use tdb_retrieval::key_view::{fixed_dim_key, is_encoded_per_token_key};
use tdb_retrieval::per_token::{PerTokenConfig, ScoringMode};
use tdb_retrieval::pipeline::RetrieverPipeline;
use tdb_retrieval::retriever::Retriever;
use tdb_retrieval::slb::SemanticLookasideBuffer;
use tdb_storage::block_pool::BlockPool;
use tdb_storage::deletion_log::DeletionLog;
use tdb_storage::synaptic_store::SynapticStore;
use tdb_storage::text_store::TextStore;

use crate::pack_directory::PackDirectory;
use crate::pack_materialization::{
    PackAccessSnapshot, PackCandidate, build_pack_read_result, keep_first_ranked_pack_candidates,
};

/// Default SLB capacity.
const DEFAULT_SLB_CAPACITY: usize = 4096;

/// Compute a mean-pooled key from a potentially per-token encoded key.
///
/// If the key is per-token encoded (has header), averages all token vectors.
/// If it's already a plain vector, returns it unchanged.
fn mean_pool_key(key: &[f32]) -> Vec<f32> {
    fixed_dim_key(key).unwrap_or_default()
}

fn should_index_for_retrieval(cell: &MemoryCell) -> bool {
    cell.token_span.0 == 0 || cell.layer == PACK_RETRIEVAL_LAYER
}

/// Default Vamana max degree.
const DEFAULT_VAMANA_MAX_DEGREE: usize = 16;
/// Default cell count threshold before activating Vamana.
const DEFAULT_VAMANA_THRESHOLD: usize = 10_000;
const PACK_RETRIEVAL_CELL_COUNT: usize = 1;

/// Adapter: wraps `VamanaIndex` (from `tdb-index`) to implement `Retriever` (from `tdb-retrieval`).
///
/// Bridges the interface gap: Vamana returns `(CellId, f32)` tuples with no owner info,
/// while `Retriever` returns `RetrievalResult` with owner. The adapter sets owner to 0
/// — the pipeline caller is responsible for final owner filtering.
struct VamanaAdapter {
    inner: VamanaIndex,
}

impl Retriever for VamanaAdapter {
    fn query(
        &mut self,
        query_key: &[f32],
        k: usize,
        _owner_filter: Option<OwnerId>,
    ) -> Vec<RetrievalResult> {
        let query = mean_pool_key(query_key);
        self.inner
            .query(&query, k)
            .into_iter()
            .map(|(cell_id, score)| RetrievalResult { cell_id, owner: 0, score })
            .collect()
    }

    fn insert(&mut self, cell_id: CellId, _owner: OwnerId, key: &[f32]) {
        let key = mean_pool_key(key);
        self.inner.insert(cell_id, &key);
    }

    fn remove(&mut self, _cell_id: CellId) {
        // Vamana doesn't support removal — deleted cells are filtered at the
        // PackDirectory level. This is a no-op.
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

/// Score multiplier applied during retrieval based on maturity tier.
///
/// Core memories have proven their value through repeated access;
/// they deserve a retrieval advantage over untested Draft memories.
fn tier_boost(tier: Tier) -> f32 {
    match tier {
        Tier::Draft => 1.0,
        Tier::Validated => 1.1,
        Tier::Core => 1.25,
    }
}

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

/// A single write request for batch operations (Batch Command pattern).
///
/// Groups the parameters that `mem_write` accepts into a reusable struct
/// so multiple writes can be collected and committed with a single fsync.
#[derive(Debug, Clone)]
pub struct WriteRequest {
    pub owner: OwnerId,
    pub layer: u16,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub salience: f32,
    pub parent_cell_id: Option<CellId>,
}

/// Top-level `TardigradeDB` engine.
///
/// Facade pattern: single entry point coordinating storage, retrieval,
/// governance, and organization (trace + WAL).
///
/// ## Retrieval Architecture
///
/// ```text
/// query → SLB (cache, separate) → Pipeline [BruteForce, Vamana*] → merge
/// ```
///
/// The SLB is a separate LRU cache (needs direct access for warming on reads).
/// The [`RetrieverPipeline`] chains cold-path retrievers via the [`Retriever`] trait.
/// Vamana is lazily activated when cell count crosses the threshold.
/// Sentinel layer index for KV Pack retrieval key cells.
const PACK_RETRIEVAL_LAYER: u16 = 0xFFFE;

pub struct Engine {
    pool: BlockPool,
    /// Cold-path retrieval: `BruteForce` + Vamana (when active).
    pipeline: RetrieverPipeline,
    /// Hot-path cache: separate from pipeline for direct LRU warming on reads/writes.
    slb: SemanticLookasideBuffer,
    vamana: Option<VamanaIndex>,
    trace: TraceGraph,
    wal: Wal,
    synaptic_store: SynapticStore,
    governance: HashMap<CellId, CellGovernance>,
    next_id: CellId,
    dir: PathBuf,
    vamana_threshold: usize,
    /// Private pack membership repository with forward and reverse lookup.
    pack_directory: PackDirectory,
    next_pack_id: PackId,
    /// Key dimension (detected from first write, used for SLB/Vamana init).
    key_dim: Option<usize>,
    /// Per-token retrieval config (stored for pipeline rebuild on refresh).
    per_token_config: PerTokenConfig,
    /// Durable text store for KV pack fact text.
    text_store: TextStore,
    /// Durable deletion log for pack deletions.
    deletion_log: DeletionLog,
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
        // Open durable backing stores (files + WAL handle).
        let pool = match segment_size {
            Some(size) => BlockPool::open_with_segment_size(dir, size)?,
            None => BlockPool::open(dir)?,
        };
        let synaptic_store =
            SynapticStore::open(dir).map_err(|e| TardigradeError::Io { source: e })?;
        let wal = Wal::open(dir).map_err(|e| TardigradeError::WalRecovery(e.to_string()))?;
        let text_store = TextStore::open(dir).map_err(|e| TardigradeError::Io { source: e })?;
        let deletion_log = DeletionLog::open(dir).map_err(|e| TardigradeError::Io { source: e })?;

        let per_token_config = PerTokenConfig::default();
        let pipeline = Self::build_default_pipeline(&per_token_config);

        // Construct with empty derived state, then rebuild via refresh().
        // DRY: open and refresh share the same Memento rebuild path
        // (reindex_cell, pack_directory scan, WAL replay, deletion filter).
        let mut engine = Self {
            pool,
            pipeline,
            slb: SemanticLookasideBuffer::new(DEFAULT_SLB_CAPACITY, 128),
            vamana: None,
            trace: TraceGraph::new(),
            wal,
            synaptic_store,
            governance: HashMap::new(),
            next_id: 0,
            dir: dir.to_path_buf(),
            vamana_threshold,
            pack_directory: PackDirectory::new(),
            next_pack_id: 0,
            key_dim: None,
            per_token_config,
            text_store,
            deletion_log,
        };

        engine.refresh()?;

        Ok(engine)
    }

    /// Re-sync in-memory state from disk without dropping the instance.
    ///
    /// Full **Memento** rebuild: rescans segments, rebuilds the retrieval
    /// pipeline from scratch (`PerToken` + `BruteForce`), re-populates all
    /// cells into both pipeline and SLB, replays the WAL, refreshes
    /// governance, rebuilds `pack_directory`, and re-syncs backing stores.
    /// Vamana is reset and re-activated lazily if cell count exceeds
    /// threshold. After this returns, the engine reflects writes performed
    /// by other [`Engine`] handles at the same path — same guarantee as
    /// a fresh [`open`](Self::open).
    ///
    /// **Idempotency:** `refresh().refresh()` is identical to a single
    /// `refresh()`.
    ///
    /// **Concurrency:** takes `&mut self`. Caller must ensure no other
    /// reader holds a borrow.
    pub fn refresh(&mut self) -> Result<()> {
        // 1. Rescan segments — picks up new cells written by another handle.
        self.pool.refresh_index()?;

        // 2. Detect whether the SLB needs rebuilding at a different dimension.
        //    If the engine was opened empty, the SLB defaults to dim=128. When
        //    refresh discovers cells written by another handle whose mean-pooled
        //    key has a different dim, the SLB's INT8 dot product would panic.
        //    Detect the dim from the first new indexable cell and rebuild SLB
        //    in place if needed.
        let known: std::collections::HashSet<CellId> = self.governance.keys().copied().collect();
        let all_cells: Vec<CellId> = self.pool.iter_cell_ids().collect();

        let mut observed_dim: Option<usize> = None;
        for cell_id in &all_cells {
            if known.contains(cell_id) {
                continue;
            }
            let cell = self.pool.get(*cell_id)?;
            if should_index_for_retrieval(&cell) {
                observed_dim = Some(mean_pool_key(&cell.key).len());
                break;
            }
        }
        if let Some(new_dim) = observed_dim
            && new_dim != self.slb.dim()
        {
            let cap = self.slb.capacity();
            self.slb = SemanticLookasideBuffer::new(cap, new_dim);
            self.key_dim = Some(new_dim);
        }

        // 3. Rebuild retrieval pipeline from scratch (Memento pattern).
        //    PerToken/BruteForce have no duplicate-insert protection, so we
        //    must start with empty stages and re-populate all cells below.
        self.pipeline = Self::build_default_pipeline(&self.per_token_config);
        self.vamana = None;

        // 4. Re-derive governance for new cells; re-populate pipeline for ALL cells.
        for cell_id in &all_cells {
            if known.contains(cell_id) {
                continue;
            }
            let cell = self.pool.get(*cell_id)?;
            self.reindex_cell(&cell);
            if cell.id >= self.next_id {
                self.next_id = cell.id + 1;
            }
        }

        // Re-populate rebuilt pipeline + SLB with previously-known cells.
        // Governance stays intact (accumulated importance scores preserved).
        for &cell_id in &known {
            if let Ok(cell) = self.pool.get(cell_id)
                && should_index_for_retrieval(&cell)
            {
                self.pipeline.insert(cell.id, cell.owner, &cell.key);
                let slb_key = mean_pool_key(&cell.key);
                self.slb.insert(cell.id, cell.owner, &slb_key);
            }
        }

        // 3. Rebuild pack_directory from current pool state. Idempotent —
        //    same cell set yields same directory.
        let mut pack_cells = Vec::new();
        for cell_id in &all_cells {
            if let Ok(cell) = self.pool.get(*cell_id) {
                let stored_pack_id = cell.token_span.0;
                if stored_pack_id > 0 || cell.layer == PACK_RETRIEVAL_LAYER {
                    pack_cells.push((stored_pack_id, cell.id));
                }
            }
        }
        let mut pack_directory = PackDirectory::from_cells(pack_cells);
        let next_pack_id = pack_directory.next_pack_id();

        // 4. Replay WAL again. TraceGraph.add_edge dedups so re-apply is safe.
        let entries = self.wal.replay().map_err(|e| TardigradeError::WalRecovery(e.to_string()))?;
        for entry in &entries {
            match entry {
                WalEntry::AddEdge { src, dst, edge_type, timestamp } => {
                    if let Some(et) = EdgeType::from_u8(*edge_type) {
                        self.trace.add_edge(*src, *dst, et, *timestamp);
                    }
                }
            }
        }

        // 5. Refresh persistent backing stores.
        self.text_store.refresh().map_err(|e| TardigradeError::Io { source: e })?;
        self.deletion_log.refresh().map_err(|e| TardigradeError::Io { source: e })?;

        // 6. Apply deletions to the rebuilt directory and text_store.
        for &pack_id in self.deletion_log.deleted_set() {
            pack_directory.remove_pack(pack_id);
            self.text_store.remove(pack_id);
        }

        // Commit the rebuilt pack directory + counter.
        self.pack_directory = pack_directory;
        self.next_pack_id = self.next_pack_id.max(next_pack_id);

        // 7. Re-check Vamana activation after full reindex.
        if self.vamana.is_none() && self.pool.cell_count() >= self.vamana_threshold {
            self.activate_vamana()?;
        }

        // 8. Checkpoint WAL — all entries have been replayed into the in-memory
        //    TraceGraph. Truncating is safe: a crash during truncation means the
        //    next refresh re-applies edges that are deduplicated by add_edge.
        if !entries.is_empty() {
            self.wal.checkpoint().map_err(|e| TardigradeError::WalRecovery(e.to_string()))?;
        }

        Ok(())
    }

    /// Re-index a single cell into pipeline + SLB + governance.
    ///
    /// **DRY hook** shared by `open_with_options` and `refresh()` — both
    /// rebuild paths must compute identical state for the same input cell.
    /// Idempotent at the call-site level: callers ensure they don't pass
    /// the same cell twice.
    fn reindex_cell(&mut self, cell: &MemoryCell) {
        if should_index_for_retrieval(cell) {
            let slb_key = mean_pool_key(&cell.key);
            if self.key_dim.is_none() {
                self.key_dim = Some(slb_key.len());
            }
            self.pipeline.insert(cell.id, cell.owner, &cell.key);
            self.slb.insert(cell.id, cell.owner, &slb_key);
        }
        let scorer = ImportanceScorer::new(cell.meta.importance);
        let tier_sm = TierStateMachine::with_tier(cell.meta.tier);
        self.governance.insert(cell.id, CellGovernance { scorer, tier_sm, days_since_update: 0.0 });
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

        // Detect key dimension on first write (use mean-pooled dim for SLB).
        if self.key_dim.is_none() {
            let mean_dim = mean_pool_key(key).len();
            self.key_dim = Some(mean_dim);
            self.slb = SemanticLookasideBuffer::new(DEFAULT_SLB_CAPACITY, mean_dim);
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

        // Index for retrieval.
        // Pipeline gets the raw key (PerTokenRetriever decodes per-token encoding).
        self.pipeline.insert(id, owner, key);

        // SLB gets a mean-pooled summary (it needs fixed-dim vectors for INT8 scoring).
        let slb_key = mean_pool_key(key);
        self.slb.insert(id, owner, &slb_key);

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

    /// Write multiple cells in a single batch with one fsync (Batch Command pattern).
    ///
    /// All cells are persisted to the block pool in one operation, then indexed
    /// into the retrieval pipeline, SLB, and governance. Causal edges (if any)
    /// are written to the WAL after the batch is persisted.
    ///
    /// Throughput: ~80us/cell amortized vs ~8ms/cell for individual writes.
    pub fn mem_write_batch(&mut self, requests: &[WriteRequest]) -> Result<Vec<CellId>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        // Detect key dimension on first write (use mean-pooled dim for SLB).
        if self.key_dim.is_none() {
            let mean_dim = mean_pool_key(&requests[0].key).len();
            self.key_dim = Some(mean_dim);
            self.slb = SemanticLookasideBuffer::new(DEFAULT_SLB_CAPACITY, mean_dim);
        }

        let now_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);

        // Phase 1: Build all cells and compute governance (in memory, no I/O).
        let mut cells = Vec::with_capacity(requests.len());
        let mut gov_entries = Vec::with_capacity(requests.len());

        for req in requests {
            let id = self.next_id;
            self.next_id += 1;

            let mut scorer = ImportanceScorer::new(req.salience);
            scorer.on_update();
            let mut tier_sm = TierStateMachine::new();
            tier_sm.evaluate(scorer.importance());

            let cell = MemoryCellBuilder::new(
                id,
                req.owner,
                req.layer,
                req.key.clone(),
                req.value.clone(),
            )
            .importance(scorer.importance())
            .tier(tier_sm.current())
            .created_at(now_nanos)
            .updated_at(now_nanos)
            .build();

            gov_entries.push((
                id,
                req.owner,
                req.key.clone(),
                req.parent_cell_id,
                CellGovernance { scorer, tier_sm, days_since_update: 0.0 },
            ));
            cells.push(cell);
        }

        // Phase 2: Persist all cells with single fsync.
        let ids = self.pool.append_batch(&cells)?;

        // Phase 3: Index all cells into retrieval + governance (in memory, fast).
        for (id, owner, key, parent_cell_id, gov) in gov_entries {
            self.pipeline.insert(id, owner, &key);
            let slb_key = mean_pool_key(&key);
            self.slb.insert(id, owner, &slb_key);
            self.governance.insert(id, gov);

            // Causal edges via WAL.
            if let Some(parent_id) = parent_cell_id {
                let wal_entry = WalEntry::AddEdge {
                    src: id,
                    dst: parent_id,
                    edge_type: EdgeType::CausedBy as u8,
                    timestamp: now_nanos,
                };
                self.wal
                    .append(&wal_entry)
                    .map_err(|e| TardigradeError::WalRecovery(e.to_string()))?;
                self.trace.add_edge(id, parent_id, EdgeType::CausedBy, now_nanos);
            }
        }

        // Lazy Vamana activation.
        if self.vamana.is_none() && self.pool.cell_count() >= self.vamana_threshold {
            self.activate_vamana()?;
        }

        Ok(ids)
    }

    /// Read the top-k most relevant cells for a query key.
    ///
    /// Encoded per-token queries run the [`RetrieverPipeline`] first so
    /// `PerTokenRetriever(Top5Avg)` scores the raw token matrix. The SLB uses
    /// mean-pooled summaries only as a fallback/hot cache because it requires
    /// fixed-size vectors. Results are merged, deduplicated, scored with
    /// recency decay, and filtered by owner.
    pub fn mem_read(
        &mut self,
        query_key: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
    ) -> Result<Vec<ReadResult>> {
        if query_key.is_empty() {
            return Ok(Vec::new());
        }

        let is_per_token_query = is_encoded_per_token_key(query_key);

        // Phase 1: SLB hot cache (uses mean-pooled query for fixed-dim INT8 scoring).
        let slb_query = mean_pool_key(query_key);
        let mut candidates = Vec::new();

        if is_per_token_query {
            // Encoded per-token queries require max-sim scoring. SLB can fill gaps,
            // but it must not suppress the cold per-token retriever.
            let cold_results = self.pipeline.query(query_key, k * 2, owner_filter);
            let mut seen: std::collections::HashSet<CellId> = std::collections::HashSet::new();
            for r in cold_results {
                if seen.insert(r.cell_id) {
                    candidates.push(r);
                }
            }

            if candidates.len() < k && !self.slb.is_empty() {
                for r in self.slb.query(&slb_query, k * 2) {
                    if seen.insert(r.cell_id) {
                        candidates.push(r);
                    }
                }
            }
        } else {
            candidates =
                if self.slb.is_empty() { Vec::new() } else { self.slb.query(&slb_query, k * 2) };

            // Phase 2: Cold-path pipeline (PerToken + BruteForce) — gets raw query.
            if candidates.len() < k {
                let cold_results = self.pipeline.query(query_key, k * 2, owner_filter);
                // Deduplicate against SLB results.
                let seen: std::collections::HashSet<CellId> =
                    candidates.iter().map(|r| r.cell_id).collect();
                for r in cold_results {
                    if !seen.contains(&r.cell_id) {
                        candidates.push(r);
                    }
                }
            }
        }

        // Score adjustment: recency decay × tier boost, then governance update.
        candidates
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        let mut results = Vec::with_capacity(k);
        for rr in candidates {
            let (decay_factor, tier) =
                self.governance.get(&rr.cell_id).map_or((1.0, Tier::Draft), |g| {
                    (recency_decay(g.days_since_update), g.tier_sm.current())
                });

            let adjusted_score = rr.score * decay_factor * tier_boost(tier);

            match self.pool.get(rr.cell_id) {
                Ok(cell) => {
                    // Owner filter (pipeline may not filter internally for all stages).
                    if let Some(filter_owner) = owner_filter
                        && cell.owner != filter_owner
                    {
                        continue;
                    }

                    if let Some(gov) = self.governance.get_mut(&rr.cell_id) {
                        gov.scorer.on_access();
                        gov.tier_sm.evaluate(gov.scorer.importance());
                    }
                    // Warm the SLB with accessed cells for future hot-path hits.
                    let slb_key = mean_pool_key(&cell.key);
                    self.slb.insert(rr.cell_id, cell.owner, &slb_key);

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

    /// Number of stages in the retrieval pipeline.
    pub fn pipeline_stage_count(&self) -> usize {
        self.pipeline.stage_count()
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

    /// Evict Draft-tier packs whose importance falls below the threshold.
    ///
    /// Only Draft-tier packs are eligible for eviction. Validated and Core
    /// packs are never evicted regardless of importance. Returns the number
    /// of packs evicted.
    pub fn evict_draft_packs(&mut self, importance_threshold: f32) -> Result<usize> {
        let to_evict: Vec<PackId> = self
            .list_packs(None)
            .into_iter()
            .filter(|&(_, _, tier, importance)| {
                tier == Tier::Draft && importance < importance_threshold
            })
            .map(|(pack_id, _, _, _)| pack_id)
            .collect();

        let count = to_evict.len();
        for pack_id in to_evict {
            self.delete_pack(pack_id)?;
        }
        Ok(count)
    }

    /// Store a `SynapticBankEntry` (`LoRA` adapter) for an agent/user.
    pub fn store_synapsis(&mut self, entry: &SynapticBankEntry) -> Result<()> {
        self.synaptic_store.append(entry).map_err(|e| TardigradeError::Io { source: e })
    }

    /// Load all `SynapticBankEntry` records for a given owner.
    pub fn load_synapsis(&self, owner: OwnerId) -> Result<Vec<SynapticBankEntry>> {
        self.synaptic_store.load_by_owner(owner).map_err(|e| TardigradeError::Io { source: e })
    }

    // ── KV Pack API (Repository pattern) ──────────────────────────────────

    /// Store a complete multi-layer KV cache as a single atomic pack.
    ///
    /// All layers are persisted with a single fsync. The retrieval key is
    /// indexed for search. Returns the assigned pack ID.
    pub fn mem_write_pack(&mut self, pack: &KVPack) -> Result<PackId> {
        let pack_id = self.next_pack_id;
        self.next_pack_id += 1;

        let now_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);

        // Detect key dimension from retrieval key.
        let retrieval_key_pooled = mean_pool_key(&pack.retrieval_key);
        if self.key_dim.is_none() {
            let dim = retrieval_key_pooled.len();
            self.key_dim = Some(dim);
            self.slb = SemanticLookasideBuffer::new(DEFAULT_SLB_CAPACITY, dim);
        }

        // Build cells: one for retrieval key + one per layer payload.
        let mut cells = Vec::with_capacity(1 + pack.layers.len());
        let mut cell_ids = Vec::with_capacity(1 + pack.layers.len());

        // Retrieval key cell (sentinel layer = PACK_RETRIEVAL_LAYER).
        let retrieval_cell_id = self.next_id;
        self.next_id += 1;

        let mut scorer = ImportanceScorer::new(pack.salience);
        scorer.on_update();
        let mut tier_sm = TierStateMachine::new();
        tier_sm.evaluate(scorer.importance());

        // Store full per-token encoded key in the cell (for PerTokenRetriever scoring).
        cells.push(
            MemoryCellBuilder::new(
                retrieval_cell_id,
                pack.owner,
                PACK_RETRIEVAL_LAYER,
                pack.retrieval_key.clone(),
                vec![], // no value for retrieval cell
            )
            .importance(scorer.importance())
            .tier(tier_sm.current())
            .token_span(pack_id, 0) // pack_id stored for cross-session rebuild
            .created_at(now_nanos)
            .updated_at(now_nanos)
            .build(),
        );
        cell_ids.push(retrieval_cell_id);

        // Layer payload cells.
        for layer in &pack.layers {
            let cell_id = self.next_id;
            self.next_id += 1;

            cells.push(
                MemoryCellBuilder::new(
                    cell_id,
                    pack.owner,
                    layer.layer_idx,
                    pack.retrieval_key.clone(),
                    layer.data.clone(),
                )
                .importance(scorer.importance())
                .tier(tier_sm.current())
                .token_span(pack_id, 0)
                .created_at(now_nanos)
                .updated_at(now_nanos)
                .build(),
            );
            cell_ids.push(cell_id);
        }

        // Persist atomically (single fsync).
        self.pool.append_batch(&cells)?;

        // Persist fact text if provided.
        if let Some(ref text) = pack.text {
            self.text_store.store(pack_id, text).map_err(|e| TardigradeError::Io { source: e })?;
        }

        // Index only the retrieval cell. Layer payload cells are persisted for
        // reconstruction, not search signal.
        let slb_key = mean_pool_key(&pack.retrieval_key);
        self.pipeline.insert(retrieval_cell_id, pack.owner, &pack.retrieval_key);
        self.slb.insert(retrieval_cell_id, pack.owner, &slb_key);

        // Governance for the pack (tracked on retrieval cell).
        self.governance
            .insert(retrieval_cell_id, CellGovernance { scorer, tier_sm, days_since_update: 0.0 });

        // Pack index.
        self.pack_directory.insert_pack(pack_id, cell_ids);

        Ok(pack_id)
    }

    /// Retrieve the top-k KV Packs matching a query key.
    ///
    /// Returns complete packs with all layer payloads reconstructed.
    pub fn mem_read_pack(
        &mut self,
        query_key: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
    ) -> Result<Vec<PackReadResult>> {
        let candidates = self.collect_pack_candidates(query_key, k, owner_filter);
        let candidates = self.deduplicate_pack_candidates(candidates, k, owner_filter);
        let mut results = Vec::with_capacity(k);

        for candidate in &candidates {
            let layers = self.hydrate_pack_layers(&candidate.cell_ids)?;
            let access = self.apply_pack_access_governance(candidate.retrieval_cell_id);
            let mut result = build_pack_read_result(candidate, layers, access);
            result.pack.text = self.text_store.get(candidate.pack_id).map(str::to_owned);
            results.push(result);
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    /// Retrieve packs with trace-boosted scoring.
    ///
    /// Retrieves an expanded candidate set, boosts scores by trace link
    /// count (discovery hubs rank higher), then returns the top k.
    pub fn mem_read_pack_with_trace_boost(
        &mut self,
        query_key: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
        boost_factor: f32,
    ) -> Result<Vec<PackReadResult>> {
        let k_expanded = k.saturating_mul(5).max(10);
        let candidates = self.collect_pack_candidates(query_key, k_expanded, owner_filter);
        let candidates = self.deduplicate_pack_candidates(candidates, k_expanded, owner_filter);

        // Score and boost by trace link count
        let mut scored: Vec<(PackCandidate, f32)> = candidates
            .into_iter()
            .map(|c| {
                let link_count = self.pack_links(c.pack_id).len() as f32;
                let boosted_score = c.score * (1.0 + link_count * boost_factor);
                (c, boosted_score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        let mut results = Vec::with_capacity(k);
        for (mut candidate, boosted_score) in scored {
            candidate.score = boosted_score;
            let layers = self.hydrate_pack_layers(&candidate.cell_ids)?;
            let access = self.apply_pack_access_governance(candidate.retrieval_cell_id);
            let mut result = build_pack_read_result(&candidate, layers, access);
            result.pack.text = self.text_store.get(candidate.pack_id).map(str::to_owned);
            results.push(result);
        }

        Ok(results)
    }

    fn collect_pack_candidates(
        &mut self,
        query_key: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
    ) -> Vec<RetrievalResult> {
        // Per-token queries need pipeline first (same logic as mem_read).
        let is_per_token = is_encoded_per_token_key(query_key);
        let slb_query = mean_pool_key(query_key);
        let mut candidates = Vec::new();

        if is_per_token {
            let cold = self.pipeline.query(query_key, k * 4, owner_filter);
            let mut seen: std::collections::HashSet<CellId> = std::collections::HashSet::new();
            for r in cold {
                if seen.insert(r.cell_id) {
                    candidates.push(r);
                }
            }
            if candidates.len() < k && !self.slb.is_empty() {
                for r in self.slb.query(&slb_query, k * 2) {
                    if seen.insert(r.cell_id) {
                        candidates.push(r);
                    }
                }
            }
        } else {
            candidates =
                if self.slb.is_empty() { Vec::new() } else { self.slb.query(&slb_query, k * 2) };
            if candidates.len() < k {
                let cold = self.pipeline.query(query_key, k * 2, owner_filter);
                let seen: std::collections::HashSet<CellId> =
                    candidates.iter().map(|r| r.cell_id).collect();
                for r in cold {
                    if !seen.contains(&r.cell_id) {
                        candidates.push(r);
                    }
                }
            }
        }

        candidates
    }

    fn deduplicate_pack_candidates(
        &self,
        mut retrieval_results: Vec<RetrievalResult>,
        k: usize,
        owner_filter: Option<OwnerId>,
    ) -> Vec<PackCandidate> {
        retrieval_results
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        let candidates = retrieval_results
            .into_iter()
            .filter_map(|rr| {
                let pack_id = self.pack_directory.pack_for_cell(rr.cell_id)?;
                let cell_ids = self.pack_directory.cell_ids(pack_id)?;
                if owner_filter.is_some_and(|filter_owner| filter_owner != rr.owner) {
                    return None;
                }
                Some(PackCandidate::new(pack_id, rr.cell_id, rr.owner, rr.score, cell_ids.to_vec()))
            })
            .collect();

        keep_first_ranked_pack_candidates(candidates, k)
    }

    fn hydrate_pack_layers(&self, cell_ids: &[CellId]) -> Result<Vec<KVLayerPayload>> {
        let layer_cell_count = cell_ids.len().saturating_sub(PACK_RETRIEVAL_CELL_COUNT);
        let mut layers = Vec::with_capacity(layer_cell_count);

        for &cell_id in cell_ids.iter().skip(PACK_RETRIEVAL_CELL_COUNT) {
            match self.pool.get(cell_id) {
                Ok(cell) => layers.push(KVLayerPayload { layer_idx: cell.layer, data: cell.value }),
                Err(TardigradeError::CellNotFound(_)) => {}
                Err(e) => return Err(e),
            }
        }

        layers.sort_by_key(|layer| layer.layer_idx);
        Ok(layers)
    }

    fn apply_pack_access_governance(&mut self, retrieval_cell_id: CellId) -> PackAccessSnapshot {
        let Some(gov) = self.governance.get_mut(&retrieval_cell_id) else {
            return PackAccessSnapshot { tier: Tier::Draft, decay_factor: 1.0, tier_boost: 1.0 };
        };

        gov.scorer.on_access();
        gov.tier_sm.evaluate(gov.scorer.importance());

        let tier = gov.tier_sm.current();
        PackAccessSnapshot {
            tier,
            decay_factor: recency_decay(gov.days_since_update),
            tier_boost: tier_boost(tier),
        }
    }

    /// Number of KV Packs stored.
    pub fn pack_count(&self) -> usize {
        self.pack_directory.len()
    }

    /// Get the stored text for a pack, if any.
    pub fn pack_text(&self, pack_id: PackId) -> Option<&str> {
        self.text_store.get(pack_id)
    }

    /// Whether a pack with the given ID exists (and has not been deleted).
    ///
    /// Distinguishes "pack exists but has no text" from "pack doesn't exist"
    /// — both cases return `None` from [`Self::pack_text`].
    pub fn pack_exists(&self, pack_id: PackId) -> bool {
        self.pack_directory.cell_ids(pack_id).is_some()
    }

    /// Set or update the stored text for an existing pack.
    ///
    /// Thin wrapper over [`Self::set_pack_texts`] for the single-entry case.
    /// Last-writer-wins semantics — the latest call's text is what reads return.
    ///
    /// # Errors
    ///
    /// Returns `CellNotFound` if the pack does not exist.
    pub fn set_pack_text(&mut self, pack_id: PackId, text: &str) -> Result<()> {
        self.set_pack_texts(&[(pack_id, text)])
    }

    /// Set or update text for many packs in a single batched fsync.
    ///
    /// Validates all `pack_id`s exist before any write — fail-fast: if any
    /// pack is missing, returns `CellNotFound` and **no entries are written**.
    /// This matches the existing batch convention from [`Self::mem_write_batch`]
    /// and avoids partial-write states.
    ///
    /// # When to use
    ///
    /// - Migrating a legacy text sidecar into the durable text store
    /// - Bulk-editing many memories in one operation
    ///
    /// For a single entry, prefer [`Self::set_pack_text`] for readability —
    /// it delegates here, so semantics are identical.
    ///
    /// # Errors
    ///
    /// Returns `CellNotFound` for the first missing pack ID encountered, with
    /// no on-disk side effects.
    pub fn set_pack_texts(&mut self, entries: &[(PackId, &str)]) -> Result<()> {
        for (pack_id, _) in entries {
            if !self.pack_exists(*pack_id) {
                return Err(TardigradeError::CellNotFound(*pack_id));
            }
        }
        self.text_store.store_batch(entries).map_err(|e| TardigradeError::Io { source: e })
    }

    /// Delete a pack permanently.
    ///
    /// Writes to the durable deletion log (fsynced) before updating in-memory
    /// state. Cells remain on disk in the `BlockPool` but become inaccessible.
    /// Trace edges are not removed — orphaned edges are filtered by
    /// `pack_directory` absence on query.
    ///
    /// # Errors
    ///
    /// Returns `CellNotFound` if the pack does not exist.
    pub fn delete_pack(&mut self, pack_id: PackId) -> Result<()> {
        // Verify pack exists.
        let cell_ids = self
            .pack_directory
            .cell_ids(pack_id)
            .ok_or(TardigradeError::CellNotFound(pack_id))?
            .to_vec();

        // Durable: write to deletion log (fsync before in-memory changes).
        self.deletion_log.mark_deleted(pack_id).map_err(|e| TardigradeError::Io { source: e })?;

        // Remove cells from retrieval pipeline and SLB.
        for &cell_id in &cell_ids {
            self.pipeline.remove(cell_id);
            self.slb.remove(cell_id);
            self.governance.remove(&cell_id);
        }

        // Remove from pack directory (in-memory).
        self.pack_directory.remove_pack(pack_id);

        // Remove text from in-memory text store.
        self.text_store.remove(pack_id);

        Ok(())
    }

    /// Get the importance score of a pack.
    pub fn pack_importance(&self, pack_id: PackId) -> Option<f32> {
        let cell_ids = self.pack_directory.cell_ids(pack_id)?;
        let retrieval_cell_id = *cell_ids.first()?;
        self.governance.get(&retrieval_cell_id).map(|g| g.scorer.importance())
    }

    /// Enumerate all packs, optionally filtered by owner.
    ///
    /// Returns `(pack_id, owner, tier, importance)` sorted by importance
    /// descending. Packs without governance default to `(Draft, 0.0)`.
    pub fn list_packs(&self, owner_filter: Option<OwnerId>) -> Vec<(PackId, OwnerId, Tier, f32)> {
        let mut results = Vec::new();
        for &pack_id in self.pack_directory.pack_ids() {
            let Some(cell_ids) = self.pack_directory.cell_ids(pack_id) else {
                continue;
            };
            let Some(&retrieval_cell_id) = cell_ids.first() else {
                continue;
            };
            let owner = match self.pool.get(retrieval_cell_id) {
                Ok(cell) => cell.owner,
                Err(_) => continue,
            };
            if let Some(filter) = owner_filter
                && owner != filter
            {
                continue;
            }
            let (tier, importance) = self
                .governance
                .get(&retrieval_cell_id)
                .map_or((Tier::Draft, 0.0), |g| (g.tier_sm.current(), g.scorer.importance()));
            results.push((pack_id, owner, tier, importance));
        }
        results.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Load a pack by ID without retrieval scoring.
    ///
    /// Returns the complete pack with all layer payloads. Applies access
    /// governance (importance boost + recency decay).
    pub fn load_pack_by_id(&mut self, pack_id: PackId) -> Result<PackReadResult> {
        let cell_ids = self
            .pack_directory
            .cell_ids(pack_id)
            .ok_or(TardigradeError::CellNotFound(pack_id))?
            .to_vec();

        let retrieval_cell_id = *cell_ids.first().ok_or(TardigradeError::CellNotFound(pack_id))?;

        let owner = self.pool.get(retrieval_cell_id).map_or(0, |cell| cell.owner);

        let layers = self.hydrate_pack_layers(&cell_ids)?;
        let access = self.apply_pack_access_governance(retrieval_cell_id);

        let candidate = PackCandidate::new(pack_id, retrieval_cell_id, owner, 0.0, cell_ids);
        let mut result = build_pack_read_result(&candidate, layers, access);
        result.pack.text = self.text_store.get(pack_id).map(str::to_owned);
        Ok(result)
    }

    /// Create a durable trace link between two packs.
    ///
    /// Links the retrieval cells of both packs via the Trace graph and
    /// logs the edge to WAL for crash recovery. Bidirectional: creates
    /// edges in both directions using `Follows` edge type.
    pub fn add_pack_link(&mut self, pack_id_1: PackId, pack_id_2: PackId) -> Result<()> {
        let cell_1 = self
            .pack_directory
            .cell_ids(pack_id_1)
            .and_then(|ids| ids.first().copied())
            .ok_or(TardigradeError::CellNotFound(pack_id_1))?;

        let cell_2 = self
            .pack_directory
            .cell_ids(pack_id_2)
            .and_then(|ids| ids.first().copied())
            .ok_or(TardigradeError::CellNotFound(pack_id_2))?;

        let now_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);

        // Forward edge: pack_1 → pack_2
        let wal_fwd = WalEntry::AddEdge {
            src: cell_1,
            dst: cell_2,
            edge_type: EdgeType::Follows as u8,
            timestamp: now_nanos,
        };
        self.wal.append(&wal_fwd).map_err(|e| TardigradeError::WalRecovery(e.to_string()))?;
        self.trace.add_edge(cell_1, cell_2, EdgeType::Follows, now_nanos);

        // Reverse edge: pack_2 → pack_1
        let wal_rev = WalEntry::AddEdge {
            src: cell_2,
            dst: cell_1,
            edge_type: EdgeType::Follows as u8,
            timestamp: now_nanos,
        };
        self.wal.append(&wal_rev).map_err(|e| TardigradeError::WalRecovery(e.to_string()))?;
        self.trace.add_edge(cell_2, cell_1, EdgeType::Follows, now_nanos);

        Ok(())
    }

    /// Get all packs linked to a given pack via trace edges.
    ///
    /// Returns pack IDs of all directly connected packs (both directions).
    pub fn pack_links(&self, pack_id: PackId) -> Vec<PackId> {
        let Some(cell_ids) = self.pack_directory.cell_ids(pack_id) else {
            return Vec::new();
        };
        let Some(&retrieval_cell) = cell_ids.first() else {
            return Vec::new();
        };

        let mut linked_packs = std::collections::HashSet::new();

        // Outgoing edges
        for edge in self.trace.outgoing(retrieval_cell, None) {
            if let Some(linked_pack) = self.pack_directory.pack_for_cell(edge.dst)
                && linked_pack != pack_id
            {
                linked_packs.insert(linked_pack);
            }
        }

        // Incoming edges
        for edge in self.trace.incoming(retrieval_cell, None) {
            if let Some(linked_pack) = self.pack_directory.pack_for_cell(edge.src)
                && linked_pack != pack_id
            {
                linked_packs.insert(linked_pack);
            }
        }

        linked_packs.into_iter().collect()
    }

    /// Build the default retrieval pipeline: `PerTokenRetriever` (`Top5Avg`) → `BruteForce`.
    ///
    /// DRY helper shared by `open_with_options` and `refresh()` (Memento rebuild).
    fn build_default_pipeline(config: &PerTokenConfig) -> RetrieverPipeline {
        let mut pipeline = RetrieverPipeline::new();
        pipeline.add_stage(Box::new(tdb_retrieval::per_token::PerTokenRetriever::with_config(
            ScoringMode::Top5Avg,
            config.clone(),
        )));
        pipeline.add_stage(Box::new(tdb_retrieval::attention::BruteForceRetriever::new()));
        pipeline
    }

    /// Build and activate the Vamana index, adding it as a pipeline stage.
    fn activate_vamana(&mut self) -> Result<()> {
        let dim = self.key_dim.unwrap_or(128);
        let mut vamana = VamanaIndex::new(dim, DEFAULT_VAMANA_MAX_DEGREE);

        let cell_ids: Vec<CellId> = self.pool.iter_cell_ids().collect();
        for cell_id in &cell_ids {
            let cell = self.pool.get(*cell_id)?;
            let key = mean_pool_key(&cell.key);
            vamana.insert(cell.id, &key);
        }
        vamana.build();

        // Add Vamana as a pipeline stage via the adapter.
        self.pipeline.add_stage(Box::new(VamanaAdapter { inner: vamana }));
        self.vamana = Some(VamanaIndex::new(dim, DEFAULT_VAMANA_MAX_DEGREE)); // Marker only.
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
