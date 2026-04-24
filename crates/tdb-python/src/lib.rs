//! Python bindings for `TardigradeDB` via `PyO3`.
//!
//! Exposes the `Engine` as a Python class with `mem_write` / `mem_read` methods.
//! Key/value vectors are exchanged as numpy arrays for zero-copy where possible.

use numpy::PyReadonlyArray1;
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
#[pyclass]
#[derive(Debug)]
struct Engine {
    inner: RustEngine,
}

#[pymethods]
impl Engine {
    /// Open or create a `TardigradeDB` engine at the given directory path.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let inner = RustEngine::open(std::path::Path::new(path))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Write key/value vectors to the engine.
    ///
    /// Args:
    ///     owner: Agent/user ID.
    ///     layer: Transformer layer index.
    ///     key: Key vector (numpy float32 array).
    ///     value: Value vector (numpy float32 array).
    ///     salience: Initial importance score hint (0-100).
    ///
    /// Returns:
    ///     The assigned cell ID.
    /// Write key/value vectors to the engine.
    ///
    /// Args:
    ///     owner: Agent/user ID.
    ///     layer: Transformer layer index.
    ///     key: Key vector (numpy float32 array).
    ///     value: Value vector (numpy float32 array).
    ///     salience: Initial importance score hint (0-100).
    ///     `parent_cell_id`: Optional causal parent cell ID for trace graph.
    ///
    /// Returns:
    ///     The assigned cell ID.
    fn mem_write(
        &mut self,
        owner: u64,
        layer: u16,
        key: PyReadonlyArray1<'_, f32>,
        value: PyReadonlyArray1<'_, f32>,
        salience: f32,
        parent_cell_id: Option<u64>,
    ) -> PyResult<u64> {
        let key_slice = key.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let value_vec =
            value.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?.to_vec();

        self.inner
            .mem_write(owner, layer, key_slice, value_vec, salience, parent_cell_id)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Write multiple cells in a single batch with one fsync (Batch Command).
    ///
    /// Each item is a tuple of (owner, layer, key, value, salience, parent).
    /// Returns a list of assigned cell IDs.
    fn mem_write_batch(
        &mut self,
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

        self.inner.mem_write_batch(&reqs).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Read the top-k most relevant cells for a query key.
    ///
    /// Args:
    ///     `query_key`: Query vector (numpy float32 array).
    ///     k: Number of results to return.
    ///     owner: Optional owner filter.
    ///
    /// Returns:
    ///     List of `ReadResult` objects sorted by score descending.
    fn mem_read(
        &mut self,
        query_key: PyReadonlyArray1<'_, f32>,
        k: usize,
        owner: Option<u64>,
    ) -> PyResult<Vec<ReadResult>> {
        let query_slice =
            query_key.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let results = self
            .inner
            .mem_read(query_slice, k, owner)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(results
            .into_iter()
            .map(|r| {
                let importance = self.inner.cell_importance(r.cell.id).unwrap_or(0.0);
                ReadResult {
                    cell_id: r.cell.id,
                    owner: r.cell.owner,
                    layer: r.cell.layer,
                    score: r.score,
                    tier: r.tier as u8,
                    importance,
                    key_data: r.cell.key,
                    value_data: r.cell.value,
                }
            })
            .collect())
    }

    /// Get the current importance score of a cell.
    fn cell_importance(&self, cell_id: u64) -> Option<f32> {
        self.inner.cell_importance(cell_id)
    }

    /// Get the current tier of a cell (0=Draft, 1=Validated, 2=Core).
    fn cell_tier(&self, cell_id: u64) -> Option<u8> {
        self.inner.cell_tier(cell_id).map(|t| t as u8)
    }

    /// Total number of cells in the engine.
    fn cell_count(&self) -> usize {
        self.inner.cell_count()
    }

    /// Get transitive ancestors of a cell following causal edges.
    fn trace_ancestors(&self, cell_id: u64) -> Vec<u64> {
        self.inner.trace_ancestors(cell_id)
    }

    /// Whether the Vamana ANN index is active.
    fn has_vamana(&self) -> bool {
        self.inner.has_vamana()
    }

    /// Simulate passage of time for governance decay (testing utility).
    fn advance_days(&mut self, days: f32) {
        self.inner.advance_days(days);
    }

    /// Store a complete multi-layer KV cache as a single atomic pack.
    ///
    /// Returns the assigned pack ID.
    fn mem_write_pack(
        &mut self,
        owner: u64,
        retrieval_key: PyReadonlyArray1<'_, f32>,
        layer_payloads: Vec<(u16, PyReadonlyArray1<'_, f32>)>,
        salience: f32,
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

        let pack = KVPack { id: 0, owner, retrieval_key: key, layers, salience };

        self.inner.mem_write_pack(&pack).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Retrieve the top-k KV Packs matching a query key.
    ///
    /// Returns list of dicts with pack ID, owner, score, tier, and layers.
    fn mem_read_pack(
        &mut self,
        py: Python<'_>,
        query_key: PyReadonlyArray1<'_, f32>,
        k: usize,
        owner: Option<u64>,
    ) -> PyResult<Vec<pyo3::Py<pyo3::PyAny>>> {
        let query = query_key.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let results = self
            .inner
            .mem_read_pack(query, k, owner)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

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

            py_results.push(dict.into_any().unbind());
        }

        Ok(py_results)
    }

    /// Number of KV Packs stored.
    fn pack_count(&self) -> usize {
        self.inner.pack_count()
    }

    /// Get the importance score of a pack.
    fn pack_importance(&self, pack_id: u64) -> Option<f32> {
        self.inner.pack_importance(pack_id)
    }

    fn __repr__(&self) -> String {
        format!("Engine(path='{}', cells={})", self.inner.dir().display(), self.inner.cell_count())
    }
}

/// `TardigradeDB` — LLM-native database kernel for persistent KV cache memory.
#[pymodule]
fn tardigrade_db(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Engine>()?;
    m.add_class::<ReadResult>()?;
    Ok(())
}
