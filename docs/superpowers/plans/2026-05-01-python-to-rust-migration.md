# Python→Rust Engine Logic Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move 7 pieces of engine logic from Python to Rust, eliminating round-trips, consolidating the encoding format spec, and making auto-linking transactionally safe.

**Architecture:** Each migration follows the same pattern: (1) add Rust method to Engine, (2) expose via PyO3, (3) update Python callers to use the new API, (4) keep old Python code as fallback until deprecated. ATDD-first: failing test → implementation → green.

**Tech Stack:** Rust (tdb-engine, tdb-retrieval, tdb-python via PyO3), Python (tardigrade_hooks, tardigrade_vllm)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/tdb-engine/src/engine.rs` | Tasks 1, 2: `mem_write_pack_with_auto_link`, extended `mem_read_pack_with_trace_boost` |
| `crates/tdb-retrieval/src/per_token.rs` | Task 3: export encoding constants |
| `crates/tdb-retrieval/src/lib.rs` | Task 3: re-export encoding API |
| `crates/tdb-python/src/lib.rs` | Tasks 1, 2, 3: PyO3 bindings |
| `python/tardigrade_hooks/kp_injector.py` | Tasks 1, 2: simplify to use new Rust APIs |
| `python/tardigrade_hooks/encoding.py` | Task 3: import constants from Rust |
| `python/tardigrade_hooks/sweep.py` | Task 7: deprecation notice |
| `crates/tdb-engine/tests/acceptance.rs` | Tasks 1, 2: Rust ATDD tests |
| `tests/python/test_auto_link.py` | Task 1: Python ATDD tests |

---

### Task 1: Auto-Link in Rust (`mem_write_pack_with_auto_link`)

**Files:**
- Modify: `crates/tdb-engine/src/engine.rs`
- Modify: `crates/tdb-engine/tests/acceptance.rs`
- Modify: `crates/tdb-python/src/lib.rs`
- Modify: `python/tardigrade_hooks/kp_injector.py`
- Create: `tests/python/test_auto_link.py`

The auto-link threshold score default:
```rust
const DEFAULT_AUTO_LINK_THRESHOLD: f32 = 250.0;
```

- [ ] **Step 1: Write failing Rust test**

In `crates/tdb-engine/tests/acceptance.rs`, append:

```rust
const AUTO_LINK_THRESHOLD: f32 = 0.0; // link everything for testing

/// ATDD: `mem_write_pack_with_auto_link` links to similar existing packs.
#[test]
fn test_mem_write_pack_with_auto_link() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key_a = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    let pack_a = engine
        .mem_write_pack(&KVPack {
            id: 0, owner: 1, retrieval_key: key_a.clone(),
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
            salience: 80.0, text: Some("First fact".to_owned()),
        })
        .unwrap();

    // Write second pack with auto-link — similar key should link to pack_a
    let key_b = encode_per_token_keys(&[&[0.95f32, 0.05, 0.0, 0.0]]);
    let pack_b = engine
        .mem_write_pack_with_auto_link(
            &KVPack {
                id: 0, owner: 1, retrieval_key: key_b,
                layers: vec![KVLayerPayload { layer_idx: 0, data: vec![2.0; 16] }],
                salience: 80.0, text: Some("Related fact".to_owned()),
            },
            AUTO_LINK_THRESHOLD,
        )
        .unwrap();

    assert!(engine.pack_links(pack_b.pack_id).contains(&pack_a));
}

/// ATDD: Auto-link returns the linked pack IDs.
#[test]
fn test_auto_link_returns_linked_ids() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    engine.mem_write_pack(&KVPack {
        id: 0, owner: 1, retrieval_key: key.clone(),
        layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
        salience: 80.0, text: None,
    }).unwrap();

    let result = engine.mem_write_pack_with_auto_link(
        &KVPack {
            id: 0, owner: 1, retrieval_key: key,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![2.0; 16] }],
            salience: 80.0, text: None,
        },
        AUTO_LINK_THRESHOLD,
    ).unwrap();

    assert!(!result.linked_pack_ids.is_empty());
}

/// ATDD: High threshold means no auto-links created.
#[test]
fn test_auto_link_high_threshold_no_links() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    engine.mem_write_pack(&KVPack {
        id: 0, owner: 1, retrieval_key: key.clone(),
        layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
        salience: 80.0, text: None,
    }).unwrap();

    let result = engine.mem_write_pack_with_auto_link(
        &KVPack {
            id: 0, owner: 1, retrieval_key: key,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![2.0; 16] }],
            salience: 80.0, text: None,
        },
        999_999.0, // impossibly high threshold
    ).unwrap();

    assert!(result.linked_pack_ids.is_empty());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p tdb-engine --test acceptance -- test_mem_write_pack_with_auto_link test_auto_link`
Expected: Compilation error — `mem_write_pack_with_auto_link` does not exist

- [ ] **Step 3: Add return type and method to Engine**

In `crates/tdb-engine/src/engine.rs`, add after `EngineStatus`:

```rust
const DEFAULT_AUTO_LINK_THRESHOLD: f32 = 250.0;

/// Result of a pack write with auto-linking.
#[derive(Debug, Clone)]
pub struct PackWriteResult {
    pub pack_id: PackId,
    pub linked_pack_ids: Vec<PackId>,
}
```

Add method after `mem_write_pack`:

```rust
/// Write a pack and auto-link to existing similar packs (Transactional Command).
///
/// Queries existing packs BEFORE writing (to avoid self-scoring),
/// writes the pack, then creates Follows links to any packs above
/// the score threshold.
pub fn mem_write_pack_with_auto_link(
    &mut self,
    pack: &KVPack,
    auto_link_threshold: f32,
) -> Result<PackWriteResult> {
    let mut linked_pack_ids = Vec::new();

    if self.pack_count() > 0 {
        let candidates = self.mem_read_pack(&pack.retrieval_key, 1, Some(pack.owner))?;
        for candidate in &candidates {
            if candidate.score >= auto_link_threshold {
                linked_pack_ids.push(candidate.pack.id);
            }
        }
    }

    let pack_id = self.mem_write_pack(pack)?;

    for &linked_id in &linked_pack_ids {
        self.add_pack_link(pack_id, linked_id)?;
    }

    Ok(PackWriteResult { pack_id, linked_pack_ids })
}
```

- [ ] **Step 4: Run Rust tests — all 3 pass**

Run: `cargo test -p tdb-engine --test acceptance -- test_mem_write_pack_with_auto_link test_auto_link`
Expected: 3 passed

- [ ] **Step 5: Add PyO3 binding**

In `crates/tdb-python/src/lib.rs`, add after `mem_write_pack`:

```rust
/// Write a pack with auto-linking to similar existing packs.
///
/// Returns dict with `pack_id` and `linked_pack_ids`.
#[pyo3(signature = (owner, retrieval_key, layer_payloads, salience, auto_link_threshold=250.0, text=None))]
fn mem_write_pack_with_auto_link(
    &self,
    py: Python<'_>,
    owner: u64,
    retrieval_key: PyReadonlyArray1<'_, f32>,
    layer_payloads: Vec<(u16, PyReadonlyArray1<'_, f32>)>,
    salience: f32,
    auto_link_threshold: f32,
    text: Option<String>,
) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    use tdb_core::kv_pack::{KVLayerPayload, KVPack};

    let key = retrieval_key
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        .to_vec();
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
            .mem_write_pack_with_auto_link(&pack, auto_link_threshold)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })?;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("pack_id", result.pack_id)?;
    dict.set_item("linked_pack_ids", result.linked_pack_ids)?;
    Ok(dict.into_any().unbind())
}
```

- [ ] **Step 6: Update `kp_injector.py` to use new Rust API**

In `python/tardigrade_hooks/kp_injector.py`, replace the `store()` method's auto-link block and write call (lines 90-107) with:

```python
if auto_link and self.engine.pack_count() > 0:
    threshold = auto_link_threshold if auto_link_threshold is not None else 250.0
    result = self.engine.mem_write_pack_with_auto_link(
        self.owner, retrieval_key, layer_payloads, salience,
        auto_link_threshold=threshold, text=fact_text
    )
    return result["pack_id"]

# No auto-link — standard write
pack_id = self.engine.mem_write_pack(
    self.owner, retrieval_key, layer_payloads, salience, text=fact_text
)
return pack_id
```

Remove the old `auto_link_matches` variable and `for match_id in auto_link_matches` loop.

- [ ] **Step 7: Build Python and run existing tests**

```bash
source .venv/bin/activate
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop -m crates/tdb-python/Cargo.toml
pytest tests/python/test_mcp_tools.py tests/python/test_kp_injector.py -v
```
Expected: All existing tests pass

- [ ] **Step 8: Commit**

```bash
git add crates/tdb-engine/ crates/tdb-python/ python/tardigrade_hooks/kp_injector.py
git commit -m "✨ feat(engine): mem_write_pack_with_auto_link — transactional auto-linking in Rust"
```

---

### Task 2: Trace-Link Traversal in Rust

**Files:**
- Modify: `crates/tdb-engine/src/engine.rs`
- Modify: `crates/tdb-engine/tests/acceptance.rs`
- Modify: `crates/tdb-python/src/lib.rs`
- Modify: `python/tardigrade_hooks/kp_injector.py`

- [ ] **Step 1: Write failing Rust test**

```rust
/// ATDD: Trace-boost with follow_links returns linked packs directly.
#[test]
fn test_trace_boost_follow_links_returns_linked_packs() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key_a = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    let pack_a = engine.mem_write_pack(&KVPack {
        id: 0, owner: 1, retrieval_key: key_a,
        layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
        salience: 80.0, text: Some("Fact A".to_owned()),
    }).unwrap();

    let key_b = encode_per_token_keys(&[&[0.0f32, 1.0, 0.0, 0.0]]);
    let pack_b = engine.mem_write_pack(&KVPack {
        id: 0, owner: 1, retrieval_key: key_b,
        layers: vec![KVLayerPayload { layer_idx: 0, data: vec![2.0; 16] }],
        salience: 80.0, text: Some("Fact B (linked to A)".to_owned()),
    }).unwrap();

    engine.add_pack_link(pack_a, pack_b).unwrap();

    let query = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    let results = engine.mem_read_pack_with_trace_boost_and_follow(
        &query, 1, Some(1), 0.3
    ).unwrap();

    // Should return pack_a (direct match) AND pack_b (followed link)
    let ids: Vec<u64> = results.iter().map(|r| r.pack.id).collect();
    assert!(ids.contains(&pack_a));
    assert!(ids.contains(&pack_b));
}
```

- [ ] **Step 2: Run test — expect compilation failure**

- [ ] **Step 3: Implement `mem_read_pack_with_trace_boost_and_follow`**

In `crates/tdb-engine/src/engine.rs`, add after `mem_read_pack_with_trace_boost`:

```rust
/// Trace-boosted retrieval with automatic link traversal.
///
/// Returns top-k packs with trace boost, plus all transitively
/// linked packs not already in the result set.
pub fn mem_read_pack_with_trace_boost_and_follow(
    &mut self,
    query_key: &[f32],
    k: usize,
    owner_filter: Option<OwnerId>,
    boost_factor: f32,
) -> Result<Vec<PackReadResult>> {
    let mut results = self.mem_read_pack_with_trace_boost(
        query_key, k, owner_filter, boost_factor,
    )?;

    let retrieved_ids: std::collections::HashSet<PackId> =
        results.iter().map(|r| r.pack.id).collect();

    let mut linked_ids = std::collections::HashSet::new();
    for r in &results {
        for linked in self.pack_links(r.pack.id) {
            if !retrieved_ids.contains(&linked) {
                linked_ids.insert(linked);
            }
        }
    }

    for linked_id in linked_ids {
        match self.load_pack_by_id(linked_id) {
            Ok(pack) => results.push(pack),
            Err(_) => {} // deleted pack — skip
        }
    }

    Ok(results)
}
```

- [ ] **Step 4: Run Rust test — passes**

- [ ] **Step 5: Add PyO3 binding and update `kp_injector.py`**

Add PyO3 binding `mem_read_pack_with_trace_boost_and_follow` mirroring `mem_read_pack_with_trace_boost` but calling the new method.

Update `retrieve_with_trace()` in `kp_injector.py` to replace the manual link-following loop with a single call:

```python
packs = self.engine.mem_read_pack_with_trace_boost_and_follow(
    query_key, k, self.owner, boost_factor
)
```

Remove the `retrieved_ids`, `linked_ids`, and `load_pack_by_id` loop (lines 320-329).

- [ ] **Step 6: Run existing Python tests**

```bash
pytest tests/python/ -v -m "not gpu" -x
```

- [ ] **Step 7: Commit**

```bash
git commit -m "✨ feat(engine): trace-boost with follow_links — single Rust call replaces Python loop"
```

---

### Task 3: Encoding Format Constants — Single Source of Truth

**Files:**
- Modify: `crates/tdb-retrieval/src/per_token.rs` — make constants `pub`
- Modify: `crates/tdb-retrieval/src/lib.rs` — re-export
- Modify: `crates/tdb-python/src/lib.rs` — expose as module constants
- Modify: `python/tardigrade_hooks/encoding.py` — import from Rust

- [ ] **Step 1: Make Rust constants public**

In `crates/tdb-retrieval/src/per_token.rs`, change:
```rust
const HEADER_SIZE: usize = 64;
const HEADER_SENTINEL: f32 = -1.0e9;
```
to:
```rust
pub const HEADER_SIZE: usize = 64;
pub const HEADER_SENTINEL: f32 = -1.0e9;
pub const N_TOKENS_IDX: usize = 32;
pub const DIM_IDX: usize = 33;
```

- [ ] **Step 2: Re-export from tdb-retrieval and expose to Python**

In `crates/tdb-retrieval/src/lib.rs`:
```rust
pub use per_token::{HEADER_SIZE, HEADER_SENTINEL, N_TOKENS_IDX, DIM_IDX};
```

In the Python module init (`crates/tdb-python/src/lib.rs`):
```rust
fn tardigrade_db(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ENCODING_HEADER_SIZE", tdb_retrieval::HEADER_SIZE)?;
    m.add("ENCODING_SENTINEL", tdb_retrieval::HEADER_SENTINEL)?;
    m.add("ENCODING_N_TOKENS_IDX", tdb_retrieval::N_TOKENS_IDX)?;
    m.add("ENCODING_DIM_IDX", tdb_retrieval::DIM_IDX)?;
    // ... existing class registrations
}
```

- [ ] **Step 3: Update Python encoding.py to use Rust constants**

```python
import tardigrade_db

HEADER_SIZE = tardigrade_db.ENCODING_HEADER_SIZE
SENTINEL_VALUE = tardigrade_db.ENCODING_SENTINEL
N_TOKENS_IDX = tardigrade_db.ENCODING_N_TOKENS_IDX
DIM_IDX = tardigrade_db.ENCODING_DIM_IDX
```

- [ ] **Step 4: Run encoding tests**

```bash
pytest tests/python/test_real_kv_hook.py tests/python/test_kp_injector.py -v
```

- [ ] **Step 5: Commit**

```bash
git commit -m "♻️ refactor: encoding constants from Rust — single source of truth"
```

---

### Task 4: Deprecate `sweep.py`

**Files:**
- Modify: `python/tardigrade_hooks/sweep.py`

- [ ] **Step 1: Add deprecation notice**

At the top of `sweep.py`, add:

```python
"""GovernanceSweepThread — DEPRECATED.

Use the Rust-native MaintenanceWorker instead:

    engine.start_maintenance(
        sweep_interval_secs=3600,
        compaction_interval_secs=21600,
        eviction_threshold=15.0,
    )

The Rust implementation runs governance sweep (decay + eviction)
AND segment compaction in a single background thread with no GIL
contention. This Python thread only runs decay.
"""

import warnings
```

Add to `__init__`:
```python
def __init__(self, engine, interval_secs=3600, hours_per_tick=1.0):
    warnings.warn(
        "GovernanceSweepThread is deprecated. Use engine.start_maintenance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # ... rest of init
```

- [ ] **Step 2: Run sweep tests still pass**

```bash
pytest tests/python/test_sweep.py -v
```

- [ ] **Step 3: Commit**

```bash
git commit -m "♻️ refactor: deprecate GovernanceSweepThread — use Rust MaintenanceWorker"
```

---

### Task 5: Documentation

**Files:**
- Modify: `docs/experiments/README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add migration entry to experiments README**

- [ ] **Step 2: Update CLAUDE.md status line with new APIs**

- [ ] **Step 3: Commit**

```bash
git commit -m "📝 docs: document Python→Rust migrations"
```

---

## Verification

After all tasks:

```bash
cargo test --workspace --exclude tdb-python
cargo clippy --workspace --all-targets -- -D warnings
source .venv/bin/activate
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop -m crates/tdb-python/Cargo.toml
pytest tests/python/ -v -m "not gpu"
python examples/agent_memory.py
```
