//! Python bindings for `TardigradeDB` via `PyO3`.
//!
//! Exposes the `Engine` as a Python class with `mem_write` / `mem_read` methods.
//! Key/value vectors are exchanged as numpy arrays for zero-copy where possible.
//!
//! Thread safety: Engine is wrapped in `Arc<Mutex<>>` (Monitor Object pattern).
//! Hot-path methods release the GIL via `py.detach()` so other Python threads
//! can run while Rust computes.

use std::sync::{Arc, Mutex};

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use tdb_engine::engine::Engine as RustEngine;

/// A single retrieval result returned from `mem_read`.
#[pyclass]
#[derive(Debug)]
struct ReadResult {
    #[pyo3(get)]
    cell_id: u64,
    #[pyo3(get)]
    owner: u64,
    #[pyo3(get)]
    layer: u16,
    #[pyo3(get)]
    score: f32,
    #[pyo3(get)]
    tier: u8,
    #[pyo3(get)]
    importance: f32,
    key_data: Vec<f32>,
    value_data: Vec<f32>,
}

#[pymethods]
impl ReadResult {
    /// Get the key vector as a list of floats.
    fn key(&self) -> Vec<f32> {
        self.key_data.clone()
    }

    /// Get the value vector as a list of floats.
    fn value(&self) -> Vec<f32> {
        self.value_data.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ReadResult(cell_id={}, owner={}, score={:.4}, tier={})",
            self.cell_id, self.owner, self.score, self.tier
        )
    }
}

/// `TardigradeDB` engine — persistent KV cache memory for LLM agents.
///
/// Thread-safe: wrapped in `Arc<Mutex<>>` so the GIL can be released
/// during engine operations and multiple Python threads can share one
/// instance.
#[pyclass]
struct Engine {
    inner: Arc<Mutex<RustEngine>>,
    maintenance_worker: Option<tdb_engine::maintenance::MaintenanceWorker>,
}

fn lock_engine(inner: &Mutex<RustEngine>) -> PyResult<std::sync::MutexGuard<'_, RustEngine>> {
    inner.lock().map_err(|_| PyRuntimeError::new_err("engine lock poisoned"))
}

#[pymethods]
impl Engine {
    /// Open or create a `TardigradeDB` engine at the given directory path.
    ///
    /// Optional configuration:
    ///   - `segment_size`: Segment file size threshold in bytes (default: 256MB).
    ///   - `vamana_threshold`: Cell count before activating Vamana ANN index (default: 10000).
    #[new]
    #[pyo3(signature = (path, segment_size=None, vamana_threshold=None))]
    fn new(
        path: &str,
        segment_size: Option<u64>,
        vamana_threshold: Option<usize>,
    ) -> PyResult<Self> {
        let dir = std::path::Path::new(path);
        let inner = match (segment_size, vamana_threshold) {
            (Some(seg), _) => RustEngine::open_with_segment_size(dir, seg),
            (None, Some(vt)) => RustEngine::open_with_vamana_threshold(dir, vt),
            (None, None) => RustEngine::open(dir),
        }
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(Mutex::new(inner)), maintenance_worker: None })
    }

    /// Write key/value vectors to the engine (cell-level API).
    ///
    /// **Deprecated:** Use `mem_write_pack` for new code. The Pack API is
    /// the canonical interface for storing multi-layer KV caches.
    fn mem_write(
        &self,
        py: Python<'_>,
        owner: u64,
        layer: u16,
        key: PyReadonlyArray1<'_, f32>,
        value: PyReadonlyArray1<'_, f32>,
        salience: f32,
        parent_cell_id: Option<u64>,
    ) -> PyResult<u64> {
        let key_vec = key.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?.to_vec();
        let value_vec =
            value.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?.to_vec();

        let engine = Arc::clone(&self.inner);
        py.detach(move || {
            lock_engine(&engine)?
                .mem_write(owner, layer, &key_vec, value_vec, salience, parent_cell_id)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Write multiple cells in a single batch with one fsync (Batch Command).
    ///
    /// Each item is a tuple of (owner, layer, key, value, salience, parent).
    /// Returns a list of assigned cell IDs.
    fn mem_write_batch(
        &self,
        requests: Vec<pyo3::Py<pyo3::PyAny>>,
        py: Python<'_>,
    ) -> PyResult<Vec<u64>> {
        use tdb_engine::engine::WriteRequest;

        let reqs: Vec<WriteRequest> = requests
            .iter()
            .map(|item| {
                let tuple = item.bind(py).cast::<pyo3::types::PyTuple>()?;
                let owner: u64 = tuple.get_item(0)?.extract()?;
                let layer: u16 = tuple.get_item(1)?.extract()?;
                let key: PyReadonlyArray1<'_, f32> = tuple.get_item(2)?.extract()?;
                let value: PyReadonlyArray1<'_, f32> = tuple.get_item(3)?.extract()?;
                let salience: f32 = tuple.get_item(4)?.extract()?;
                let parent: Option<u64> = tuple.get_item(5)?.extract()?;
                Ok(WriteRequest {
                    owner,
                    layer,
                    key: key
                        .as_slice()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .to_vec(),
                    value: value
                        .as_slice()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .to_vec(),
                    salience,
                    parent_cell_id: parent,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;

        let engine = Arc::clone(&self.inner);
        py.detach(move || {
            lock_engine(&engine)?
                .mem_write_batch(&reqs)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Read the top-k most relevant cells for a query key (cell-level API).
    ///
    /// **Deprecated:** Use `mem_read_pack` for new code. The Pack API returns
    /// complete multi-layer KV caches ready for injection.
    fn mem_read(
        &self,
        py: Python<'_>,
        query_key: PyReadonlyArray1<'_, f32>,
        k: usize,
        owner: Option<u64>,
    ) -> PyResult<Vec<ReadResult>> {
        let query_vec =
            query_key.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?.to_vec();

        let engine = Arc::clone(&self.inner);
        let raw_results = py.detach(move || -> PyResult<Vec<_>> {
            let mut eng = lock_engine(&engine)?;
            let results = eng
                .mem_read(&query_vec, k, owner)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(results
                .into_iter()
                .map(|r| {
                    let importance = eng.cell_importance(r.cell.id).unwrap_or(0.0);
                    (r, importance)
                })
                .collect())
        })?;

        Ok(raw_results
            .into_iter()
            .map(|(r, importance)| ReadResult {
                cell_id: r.cell.id,
                owner: r.cell.owner,
                layer: r.cell.layer,
                score: r.score,
                tier: r.tier as u8,
                importance,
                key_data: r.cell.key,
                value_data: r.cell.value,
            })
            .collect())
    }

    /// Read top-k cells using a raw per-token query matrix (Direct Token Query API).
    ///
    /// Accepts a 2D `NumPy` array of shape `(n_tokens, dim)` directly, avoiding
    /// the Python-side `encode_per_token` round-trip (header build + flatten +
    /// `PyO3` copy + Rust header parse). The Rust engine builds the encoded key
    /// in one allocation and delegates to `mem_read`.
    fn mem_read_tokens(
        &self,
        py: Python<'_>,
        query_tokens: PyReadonlyArray2<'_, f32>,
        k: usize,
        owner: Option<u64>,
    ) -> PyResult<Vec<ReadResult>> {
        let query_arr = query_tokens.as_array();
        let shape = query_arr.shape();
        let n_tokens = shape[0];
        let dim = shape[1];

        // numpy may give a non-contiguous view; ensure C-order before passing on.
        let flat: Vec<f32> = if let Some(slice) = query_arr.as_slice() {
            slice.to_vec()
        } else {
            query_arr.iter().copied().collect()
        };

        let engine = Arc::clone(&self.inner);
        let raw_results = py.detach(move || -> PyResult<Vec<_>> {
            let mut eng = lock_engine(&engine)?;
            let results = eng
                .mem_read_tokens(&flat, n_tokens, dim, k, owner)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(results
                .into_iter()
                .map(|r| {
                    let importance = eng.cell_importance(r.cell.id).unwrap_or(0.0);
                    (r, importance)
                })
                .collect())
        })?;

        Ok(raw_results
            .into_iter()
            .map(|(r, importance)| ReadResult {
                cell_id: r.cell.id,
                owner: r.cell.owner,
                layer: r.cell.layer,
                score: r.score,
                tier: r.tier as u8,
                importance,
                key_data: r.cell.key,
                value_data: r.cell.value,
            })
            .collect())
    }

    /// Configure post-retrieval refinement (vague-query rescue).
    ///
    /// `mode` selects the strategy: `"none"`, `"centered"`, `"whitened"`, `"prf"`.
    /// With `"prf"`, optional `alpha` (default 0.7), `beta` (default 0.3), and
    /// `k_prime` (default 3) tune the Rocchio expansion.
    #[pyo3(signature = (mode, alpha=None, beta=None, k_prime=None))]
    fn set_refinement_mode(
        &self,
        mode: &str,
        alpha: Option<f32>,
        beta: Option<f32>,
        k_prime: Option<usize>,
    ) -> PyResult<()> {
        let strategy = tdb_retrieval::refinement::strategy_from_name(mode, alpha, beta, k_prime)
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "unknown refinement mode '{mode}' (expected 'none', 'centered', 'prf', or 'whitened')"
                ))
            })?;
        lock_engine(&self.inner)?.set_refinement_strategy(strategy);
        Ok(())
    }

    /// Currently configured refinement mode as `"none" | "centered" | "whitened" | "prf"`.
    fn refinement_mode(&self) -> PyResult<String> {
        Ok(lock_engine(&self.inner)?.refinement_mode_name().to_string())
    }

    /// Enable or disable corpus-mean distance token importance reweighting.
    ///
    /// When enabled, each per-token score is scaled by
    /// `1 - cosine(token, corpus_mean)` before aggregation — distinctive tokens
    /// carry more retrieval signal than tokens aligned with the mean activation.
    fn set_token_reweighting(&self, enabled: bool) -> PyResult<()> {
        lock_engine(&self.inner)?.set_token_reweighting(enabled);
        Ok(())
    }

    /// Whether corpus-mean distance token importance reweighting is enabled.
    fn token_reweighting(&self) -> PyResult<bool> {
        Ok(lock_engine(&self.inner)?.token_reweighting())
    }

    /// Get the current importance score of a cell.
    fn cell_importance(&self, cell_id: u64) -> PyResult<Option<f32>> {
        Ok(lock_engine(&self.inner)?.cell_importance(cell_id))
    }

    /// Get the current tier of a cell (0=Draft, 1=Validated, 2=Core).
    fn cell_tier(&self, cell_id: u64) -> PyResult<Option<u8>> {
        Ok(lock_engine(&self.inner)?.cell_tier(cell_id).map(|t| t as u8))
    }

    /// Total number of cells in the engine.
    fn cell_count(&self) -> PyResult<usize> {
        Ok(lock_engine(&self.inner)?.cell_count())
    }

    /// Get transitive ancestors of a cell following causal edges.
    fn trace_ancestors(&self, cell_id: u64) -> PyResult<Vec<u64>> {
        Ok(lock_engine(&self.inner)?.trace_ancestors(cell_id))
    }

    /// Whether the Vamana ANN index is active.
    fn has_vamana(&self) -> PyResult<bool> {
        Ok(lock_engine(&self.inner)?.has_vamana())
    }

    /// Simulate passage of time for governance decay (testing utility).
    fn advance_days(&self, days: f32) -> PyResult<()> {
        lock_engine(&self.inner)?.advance_days(days);
        Ok(())
    }

    /// Evict Draft-tier packs below the importance threshold.
    #[pyo3(signature = (importance_threshold, owner=None))]
    fn evict_draft_packs(&self, importance_threshold: f32, owner: Option<u64>) -> PyResult<usize> {
        lock_engine(&self.inner)?
            .evict_draft_packs(importance_threshold, owner)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Number of stages in the retrieval pipeline.
    fn pipeline_stage_count(&self) -> PyResult<usize> {
        Ok(lock_engine(&self.inner)?.pipeline_stage_count())
    }

    // ── SynapticBank (`LoRA` adapter persistence) ─────────────────────────

    /// Store a `LoRA` adapter entry. Arrays are f32 in Python, converted to f16 internally.
    #[pyo3(signature = (id, owner, lora_a, lora_b, scale, rank, d_model, last_used=None, quality=None))]
    #[expect(clippy::too_many_arguments)]
    fn store_synapsis(
        &self,
        id: u64,
        owner: u64,
        lora_a: PyReadonlyArray1<'_, f32>,
        lora_b: PyReadonlyArray1<'_, f32>,
        scale: f32,
        rank: u32,
        d_model: u32,
        last_used: Option<u64>,
        quality: Option<f32>,
    ) -> PyResult<()> {
        use half::f16;
        use pyo3::exceptions::PyValueError;
        use tdb_core::synaptic_bank::SynapticBankEntry;

        let expected_len = (rank * d_model) as usize;
        let a_slice = lora_a.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let b_slice = lora_b.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        if a_slice.len() != expected_len {
            return Err(PyValueError::new_err(format!(
                "lora_a length {} != rank({}) * d_model({}) = {}",
                a_slice.len(),
                rank,
                d_model,
                expected_len
            )));
        }
        if b_slice.len() != expected_len {
            return Err(PyValueError::new_err(format!(
                "lora_b length {} != rank({}) * d_model({}) = {}",
                b_slice.len(),
                rank,
                d_model,
                expected_len
            )));
        }

        let mut entry = SynapticBankEntry::new(
            id,
            owner,
            a_slice.iter().map(|v| f16::from_f32(*v)).collect(),
            b_slice.iter().map(|v| f16::from_f32(*v)).collect(),
            f16::from_f32(scale),
            rank,
            d_model,
        );
        entry.last_used = last_used.unwrap_or(0);
        entry.quality = quality.unwrap_or(0.0);
        lock_engine(&self.inner)?
            .store_synapsis(&entry)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Load all `LoRA` adapter entries for an owner.
    ///
    /// Returns list of dicts with f32 numpy arrays (f16 converted to f32 on return).
    fn load_synapsis(&self, py: Python<'_>, owner: u64) -> PyResult<Vec<pyo3::Py<pyo3::PyAny>>> {
        let entries = lock_engine(&self.inner)?
            .load_synapsis(owner)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let mut results = Vec::with_capacity(entries.len());
        for entry in entries {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("id", entry.id)?;
            dict.set_item("owner", entry.owner)?;
            dict.set_item("rank", entry.rank)?;
            dict.set_item("d_model", entry.d_model)?;
            dict.set_item("scale", entry.scale.to_f32())?;
            dict.set_item("last_used", entry.last_used)?;
            dict.set_item("quality", entry.quality)?;

            let a_f32: Vec<f32> = entry.lora_a.iter().map(|v| v.to_f32()).collect();
            let b_f32: Vec<f32> = entry.lora_b.iter().map(|v| v.to_f32()).collect();
            dict.set_item("lora_a", numpy::PyArray1::from_vec(py, a_f32))?;
            dict.set_item("lora_b", numpy::PyArray1::from_vec(py, b_f32))?;

            results.push(dict.into_any().unbind());
        }
        Ok(results)
    }

    /// Snapshot of engine state for monitoring and diagnostics.
    ///
    /// Returns a dict with keys: `cell_count`, `pack_count`, `segment_count`,
    /// `slb_occupancy`, `slb_capacity`, `vamana_active`, `pipeline_stages`,
    /// `governance_entries`, `trace_edges`.
    fn status(&self, py: Python<'_>) -> PyResult<pyo3::Py<pyo3::PyAny>> {
        let s = lock_engine(&self.inner)?.status();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("cell_count", s.cell_count)?;
        dict.set_item("pack_count", s.pack_count)?;
        dict.set_item("segment_count", s.segment_count)?;
        dict.set_item("slb_occupancy", s.slb_occupancy)?;
        dict.set_item("slb_capacity", s.slb_capacity)?;
        dict.set_item("vamana_active", s.vamana_active)?;
        dict.set_item("pipeline_stages", s.pipeline_stages)?;
        dict.set_item("governance_entries", s.governance_entries)?;
        dict.set_item("trace_edges", s.trace_edges)?;
        Ok(dict.into_any().unbind())
    }

    /// Compact storage by rewriting segments with dead cells.
    ///
    /// Returns a dict with `segments_compacted`, `cells_moved`, `bytes_reclaimed`.
    fn compact(&self, py: Python<'_>) -> PyResult<pyo3::Py<pyo3::PyAny>> {
        let engine = Arc::clone(&self.inner);
        let result = py.detach(move || {
            lock_engine(&engine)?.compact().map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("segments_compacted", result.segments_compacted)?;
        dict.set_item("cells_moved", result.cells_moved)?;
        dict.set_item("bytes_reclaimed", result.bytes_reclaimed)?;
        Ok(dict.into_any().unbind())
    }

    // ── Maintenance (background governance + compaction) ────────────────

    /// Start background maintenance (governance sweep + auto-compaction).
    #[pyo3(signature = (sweep_interval_secs=3600.0, compaction_interval_secs=21600.0, eviction_threshold=15.0, hours_per_tick=1.0))]
    fn start_maintenance(
        &mut self,
        sweep_interval_secs: f64,
        compaction_interval_secs: f64,
        eviction_threshold: f32,
        hours_per_tick: f32,
    ) -> PyResult<()> {
        use tdb_engine::maintenance::{MaintenanceConfig, MaintenanceWorker};

        if self.maintenance_worker.as_ref().is_some_and(|w| w.is_running()) {
            return Err(PyRuntimeError::new_err("maintenance already running"));
        }

        let config = MaintenanceConfig {
            sweep_interval: std::time::Duration::from_secs_f64(sweep_interval_secs),
            compaction_interval: std::time::Duration::from_secs_f64(compaction_interval_secs),
            eviction_threshold,
            hours_per_tick,
            enabled: true,
        };

        let mut worker = MaintenanceWorker::new(config);
        worker.start(Arc::clone(&self.inner));
        self.maintenance_worker = Some(worker);
        Ok(())
    }

    /// Stop background maintenance.
    fn stop_maintenance(&mut self) {
        if let Some(ref mut worker) = self.maintenance_worker {
            worker.stop();
        }
    }

    /// Whether background maintenance is currently running.
    fn is_maintenance_running(&self) -> bool {
        self.maintenance_worker.as_ref().is_some_and(|w| w.is_running())
    }

    /// Get maintenance statistics.
    fn maintenance_status(&self, py: Python<'_>) -> PyResult<pyo3::Py<pyo3::PyAny>> {
        let status = self
            .maintenance_worker
            .as_ref()
            .map_or_else(tdb_engine::maintenance::MaintenanceStatus::default, |w| w.status());

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("sweep_count", status.sweep_count)?;
        dict.set_item("compaction_count", status.compaction_count)?;
        dict.set_item("total_packs_evicted", status.total_packs_evicted)?;
        dict.set_item("total_bytes_reclaimed", status.total_bytes_reclaimed)?;
        dict.set_item("last_sweep_epoch_secs", status.last_sweep_epoch_secs)?;
        dict.set_item("last_compaction_epoch_secs", status.last_compaction_epoch_secs)?;
        Ok(dict.into_any().unbind())
    }

    /// Store a complete multi-layer KV cache as a single atomic pack.
    ///
    /// Returns the assigned pack ID.
    #[pyo3(signature = (owner, retrieval_key, layer_payloads, salience, text=None))]
    fn mem_write_pack(
        &self,
        py: Python<'_>,
        owner: u64,
        retrieval_key: PyReadonlyArray1<'_, f32>,
        layer_payloads: Vec<(u16, PyReadonlyArray1<'_, f32>)>,
        salience: f32,
        text: Option<String>,
    ) -> PyResult<u64> {
        use tdb_core::kv_pack::{KVLayerPayload, KVPack};

        let key =
            retrieval_key.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?.to_vec();

        let layers: Vec<KVLayerPayload> = layer_payloads
            .iter()
            .map(|(idx, data)| {
                Ok(KVLayerPayload {
                    layer_idx: *idx,
                    data: data
                        .as_slice()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .to_vec(),
                })
            })
            .collect::<PyResult<Vec<_>>>()?;

        let pack = KVPack { id: 0, owner, retrieval_key: key, layers, salience, text };

        let engine = Arc::clone(&self.inner);
        py.detach(move || {
            lock_engine(&engine)?
                .mem_write_pack(&pack)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Register one or more view keys for an existing pack.
    ///
    /// View keys are additional retrieval keys — queries matching any view key
    /// will surface this pack during `mem_read_pack`. Returns the total number
    /// of view keys registered for this pack after the call.
    #[pyo3(signature = (pack_id, view_keys))]
    fn add_view_keys(
        &self,
        py: Python<'_>,
        pack_id: u64,
        view_keys: Vec<PyReadonlyArray1<'_, f32>>,
    ) -> PyResult<usize> {
        let keys: Vec<Vec<f32>> =
            view_keys.iter().map(|k| k.as_slice().map(|s| s.to_vec())).collect::<Result<_, _>>()?;
        let engine = Arc::clone(&self.inner);
        py.detach(move || {
            lock_engine(&engine)?
                .add_view_keys(pack_id, &keys)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Return the number of view keys registered for a pack.
    ///
    /// Raises `RuntimeError` if the pack does not exist.
    fn view_count(&self, pack_id: u64) -> PyResult<usize> {
        lock_engine(&self.inner)?
            .view_count(pack_id)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Store a pack with automatic discovery and linking of similar existing packs.
    ///
    /// Returns a dict with `pack_id` and `linked_pack_ids`.
    /// Threshold defaults to [`DEFAULT_AUTO_LINK_THRESHOLD`] (250.0) when omitted.
    #[pyo3(signature = (owner, retrieval_key, layer_payloads, salience, auto_link_threshold=None, text=None))]
    fn mem_write_pack_with_auto_link(
        &self,
        py: Python<'_>,
        owner: u64,
        retrieval_key: PyReadonlyArray1<'_, f32>,
        layer_payloads: Vec<(u16, PyReadonlyArray1<'_, f32>)>,
        salience: f32,
        auto_link_threshold: Option<f32>,
        text: Option<String>,
    ) -> PyResult<pyo3::Py<pyo3::PyAny>> {
        use tdb_core::kv_pack::{KVLayerPayload, KVPack};
        use tdb_engine::engine::DEFAULT_AUTO_LINK_THRESHOLD;

        let threshold = auto_link_threshold.unwrap_or(DEFAULT_AUTO_LINK_THRESHOLD);

        let key =
            retrieval_key.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?.to_vec();

        let layers: Vec<KVLayerPayload> = layer_payloads
            .iter()
            .map(|(idx, data)| {
                Ok(KVLayerPayload {
                    layer_idx: *idx,
                    data: data
                        .as_slice()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .to_vec(),
                })
            })
            .collect::<PyResult<Vec<_>>>()?;

        let pack = KVPack { id: 0, owner, retrieval_key: key, layers, salience, text };

        let engine = Arc::clone(&self.inner);
        let result = py.detach(move || {
            lock_engine(&engine)?
                .mem_write_pack_with_auto_link(&pack, threshold)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("pack_id", result.pack_id)?;
        dict.set_item("linked_pack_ids", &result.linked_pack_ids)?;
        Ok(dict.into())
    }

    /// Retrieve the top-k KV Packs matching a query key.
    ///
    /// Returns list of dicts with pack ID, owner, score, tier, and layers.
    fn mem_read_pack(
        &self,
        py: Python<'_>,
        query_key: PyReadonlyArray1<'_, f32>,
        k: usize,
        owner: Option<u64>,
    ) -> PyResult<Vec<pyo3::Py<pyo3::PyAny>>> {
        let query_vec =
            query_key.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?.to_vec();

        let engine = Arc::clone(&self.inner);
        let results = py.detach(move || {
            lock_engine(&engine)?
                .mem_read_pack(&query_vec, k, owner)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;

        let mut py_results = Vec::with_capacity(results.len());
        for r in results {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("pack_id", r.pack.id)?;
            dict.set_item("owner", r.pack.owner)?;
            dict.set_item("score", r.score)?;
            dict.set_item("tier", r.tier as u8)?;

            let layers_list = pyo3::types::PyList::empty(py);
            for layer in &r.pack.layers {
                let layer_dict = pyo3::types::PyDict::new(py);
                layer_dict.set_item("layer_idx", layer.layer_idx)?;
                let data_arr = numpy::PyArray1::from_slice(py, &layer.data);
                layer_dict.set_item("data", data_arr)?;
                layers_list.append(layer_dict)?;
            }
            dict.set_item("layers", layers_list)?;
            dict.set_item("text", r.pack.text.as_deref())?;

            py_results.push(dict.into_any().unbind());
        }

        Ok(py_results)
    }

    /// Number of KV Packs stored.
    fn pack_count(&self) -> PyResult<usize> {
        Ok(lock_engine(&self.inner)?.pack_count())
    }

    /// Re-sync in-memory state from disk.
    ///
    /// Use when another process or `Engine` handle has written to the same
    /// directory and this handle needs to see those writes. Re-applies the
    /// Memento pattern from `open()`: rescans segments, replays the WAL,
    /// refreshes governance / `pack_directory` / `text_store` / `deletion_log`.
    /// Idempotent and cheap when nothing changed on disk.
    fn refresh(&self) -> PyResult<()> {
        lock_engine(&self.inner)?.refresh().map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Get the importance score of a pack.
    fn pack_importance(&self, pack_id: u64) -> PyResult<Option<f32>> {
        Ok(lock_engine(&self.inner)?.pack_importance(pack_id))
    }

    /// Enumerate all packs, optionally filtered by owner.
    ///
    /// Returns a list of dicts with keys: `pack_id`, owner, tier, importance, text.
    /// Sorted by importance descending. Draft packs included (caller filters).
    #[pyo3(signature = (owner=None))]
    fn list_packs(&self, py: Python<'_>, owner: Option<u64>) -> PyResult<pyo3::Py<pyo3::PyAny>> {
        let eng = lock_engine(&self.inner)?;
        let packs = eng.list_packs(owner);
        let py_list = pyo3::types::PyList::empty(py);
        for (pack_id, pack_owner, tier, importance) in packs {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("pack_id", pack_id)?;
            dict.set_item("owner", pack_owner)?;
            dict.set_item("tier", tier as u8)?;
            dict.set_item("importance", importance)?;
            dict.set_item("text", eng.pack_text(pack_id))?;
            py_list.append(dict)?;
        }
        Ok(py_list.into_any().unbind())
    }

    /// Load a pack by ID without retrieval scoring.
    fn load_pack_by_id(&self, py: Python<'_>, pack_id: u64) -> PyResult<pyo3::Py<pyo3::PyAny>> {
        let engine = Arc::clone(&self.inner);
        let result = py.detach(move || {
            lock_engine(&engine)?
                .load_pack_by_id(pack_id)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;

        let layers_list = pyo3::types::PyList::empty(py);
        for layer in &result.pack.layers {
            let layer_dict = pyo3::types::PyDict::new(py);
            layer_dict.set_item("layer_idx", layer.layer_idx)?;
            let data_array = numpy::PyArray1::from_slice(py, &layer.data).into_any().unbind();
            layer_dict.set_item("data", data_array)?;
            layers_list.append(layer_dict)?;
        }

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("pack_id", result.pack.id)?;
        dict.set_item("owner", result.pack.owner)?;
        dict.set_item("score", result.score)?;
        dict.set_item("tier", result.tier as u8)?;
        dict.set_item("layers", layers_list)?;
        dict.set_item("text", result.pack.text.as_deref())?;
        Ok(dict.into_any().unbind())
    }

    /// Create a durable typed trace edge between two packs.
    ///
    /// `edge_type`: 0=CausedBy, 1=Follows, 2=Contradicts, 3=Supports.
    fn add_pack_edge(&self, pack_id_1: u64, pack_id_2: u64, edge_type: u8) -> PyResult<()> {
        use tdb_engine::EdgeType;
        let et = EdgeType::from_u8(edge_type).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "Invalid edge_type {edge_type}. Valid: 0=CausedBy, 1=Follows, 2=Contradicts, 3=Supports"
            ))
        })?;
        lock_engine(&self.inner)?
            .add_pack_edge(pack_id_1, pack_id_2, et)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Create a durable Follows link between two packs.
    fn add_pack_link(&self, pack_id_1: u64, pack_id_2: u64) -> PyResult<()> {
        lock_engine(&self.inner)?
            .add_pack_link(pack_id_1, pack_id_2)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Get all packs linked to a given pack via trace edges (any type).
    fn pack_links(&self, pack_id: u64) -> PyResult<Vec<u64>> {
        Ok(lock_engine(&self.inner)?.pack_links(pack_id))
    }

    /// Get packs linked via a specific edge type.
    fn pack_links_by_type(&self, pack_id: u64, edge_type: u8) -> PyResult<Vec<u64>> {
        use tdb_engine::EdgeType;
        let et = EdgeType::from_u8(edge_type)
            .ok_or_else(|| PyRuntimeError::new_err(format!("Invalid edge_type {edge_type}")))?;
        Ok(lock_engine(&self.inner)?.pack_links_by_type(pack_id, et))
    }

    /// Get packs that support a given pack.
    fn pack_supports(&self, pack_id: u64) -> PyResult<Vec<u64>> {
        Ok(lock_engine(&self.inner)?.pack_supports(pack_id))
    }

    /// Get packs that contradict a given pack.
    fn pack_contradicts(&self, pack_id: u64) -> PyResult<Vec<u64>> {
        Ok(lock_engine(&self.inner)?.pack_contradicts(pack_id))
    }

    /// Retrieve packs with trace-boosted scoring.
    fn mem_read_pack_with_trace_boost(
        &self,
        py: Python<'_>,
        query_key: PyReadonlyArray1<'_, f32>,
        k: usize,
        owner: Option<u64>,
        boost_factor: f32,
    ) -> PyResult<Vec<pyo3::Py<pyo3::PyAny>>> {
        let query_vec =
            query_key.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?.to_vec();

        let engine = Arc::clone(&self.inner);
        let results = py.detach(move || {
            lock_engine(&engine)?
                .mem_read_pack_with_trace_boost(&query_vec, k, owner, boost_factor)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;

        let mut py_results = Vec::with_capacity(results.len());
        for r in results {
            let layers_list = pyo3::types::PyList::empty(py);
            for layer in &r.pack.layers {
                let layer_dict = pyo3::types::PyDict::new(py);
                layer_dict.set_item("layer_idx", layer.layer_idx)?;
                let data_array = numpy::PyArray1::from_slice(py, &layer.data).into_any().unbind();
                layer_dict.set_item("data", data_array)?;
                layers_list.append(layer_dict)?;
            }

            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("pack_id", r.pack.id)?;
            dict.set_item("owner", r.pack.owner)?;
            dict.set_item("score", r.score)?;
            dict.set_item("tier", r.tier as u8)?;
            dict.set_item("layers", layers_list)?;
            dict.set_item("text", r.pack.text.as_deref())?;
            py_results.push(dict.into_any().unbind());
        }
        Ok(py_results)
    }

    /// Retrieve packs with trace-boosted scoring and automatic link traversal.
    fn mem_read_pack_with_trace_boost_and_follow(
        &self,
        py: Python<'_>,
        query_key: PyReadonlyArray1<'_, f32>,
        k: usize,
        owner: Option<u64>,
        boost_factor: f32,
    ) -> PyResult<Vec<pyo3::Py<pyo3::PyAny>>> {
        let query_vec =
            query_key.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?.to_vec();

        let engine = Arc::clone(&self.inner);
        let results = py.detach(move || {
            lock_engine(&engine)?
                .mem_read_pack_with_trace_boost_and_follow(&query_vec, k, owner, boost_factor)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;

        let mut py_results = Vec::with_capacity(results.len());
        for r in results {
            let layers_list = pyo3::types::PyList::empty(py);
            for layer in &r.pack.layers {
                let layer_dict = pyo3::types::PyDict::new(py);
                layer_dict.set_item("layer_idx", layer.layer_idx)?;
                let data_array = numpy::PyArray1::from_slice(py, &layer.data).into_any().unbind();
                layer_dict.set_item("data", data_array)?;
                layers_list.append(layer_dict)?;
            }

            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("pack_id", r.pack.id)?;
            dict.set_item("owner", r.pack.owner)?;
            dict.set_item("score", r.score)?;
            dict.set_item("tier", r.tier as u8)?;
            dict.set_item("layers", layers_list)?;
            dict.set_item("text", r.pack.text.as_deref())?;
            py_results.push(dict.into_any().unbind());
        }
        Ok(py_results)
    }

    /// Get the stored text for a pack, if any.
    fn pack_text(&self, pack_id: u64) -> PyResult<Option<String>> {
        Ok(lock_engine(&self.inner)?.pack_text(pack_id).map(str::to_owned))
    }

    /// Set or update the stored text for an existing pack.
    fn set_pack_text(&self, pack_id: u64, text: &str) -> PyResult<()> {
        lock_engine(&self.inner)?
            .set_pack_text(pack_id, text)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Set or update text for many packs in a single batched fsync.
    fn set_pack_texts(&self, entries: Vec<(u64, String)>) -> PyResult<()> {
        let borrowed: Vec<(u64, &str)> = entries.iter().map(|(id, t)| (*id, t.as_str())).collect();
        lock_engine(&self.inner)?
            .set_pack_texts(&borrowed)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Whether a pack with the given ID exists (and has not been deleted).
    fn pack_exists(&self, pack_id: u64) -> PyResult<bool> {
        Ok(lock_engine(&self.inner)?.pack_exists(pack_id))
    }

    /// Delete a pack permanently. Irreversible.
    fn delete_pack(&self, pack_id: u64) -> PyResult<()> {
        lock_engine(&self.inner)?
            .delete_pack(pack_id)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Explicit durability checkpoint — ensures all components have fsynced.
    fn flush(&self) -> PyResult<()> {
        lock_engine(&self.inner)?.flush().map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> PyResult<String> {
        let eng = lock_engine(&self.inner)?;
        Ok(format!("Engine(path='{}', cells={})", eng.dir().display(), eng.cell_count()))
    }
}

/// `TardigradeDB` — LLM-native database kernel for persistent KV cache memory.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Engine>()?;
    m.add_class::<ReadResult>()?;

    m.add("ENCODING_HEADER_SIZE", tdb_engine::encoding::HEADER_SIZE)?;
    m.add("ENCODING_SENTINEL", tdb_engine::encoding::HEADER_SENTINEL)?;
    m.add("ENCODING_N_TOKENS_IDX", tdb_engine::encoding::N_TOKENS_IDX)?;
    m.add("ENCODING_DIM_IDX", tdb_engine::encoding::DIM_IDX)?;

    Ok(())
}
