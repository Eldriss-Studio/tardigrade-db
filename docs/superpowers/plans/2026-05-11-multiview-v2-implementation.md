# Multi-View Consolidation v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attach LLM-generated view retrieval keys to existing packs so vague queries can discover memories through alternative phrasings, without views competing as independent results.

**Architecture:** New Rust method `add_view_keys` creates additional retrieval cells on an existing pack. Existing pack dedup in `mem_read_pack` ensures only the best-scoring cell per pack surfaces. Python-side `ViewGenerator` gains an LLM mode with diversity filtering. Consolidation triggers on tier promotion.

**Tech Stack:** Rust (tdb-engine, tdb-python/PyO3), Python (tardigrade_hooks), Qwen3-0.6B on MPS for LLM view generation.

---

### Task 1: Rust Engine — `add_view_keys` + `view_count`

**Files:**
- Modify: `crates/tdb-engine/src/engine.rs` (add methods after `mem_write_pack` ~line 982)
- Modify: `crates/tdb-engine/src/pack_directory.rs` (make `add_cell` pub(crate), line 100)
- Test: `crates/tdb-engine/tests/acceptance.rs`

- [ ] **Step 1: Write failing test — add_view_keys creates cells**

In `crates/tdb-engine/tests/acceptance.rs`:

```rust
#[test]
fn test_add_view_keys_creates_retrieval_cells() {
    let dir = tempdir().unwrap();
    let mut engine = Engine::open(dir.path(), 9999);

    let key = vec![1.0_f32; 8];
    let value = vec![0.0_f32; 8];
    let pack = KVPack {
        id: 0, owner: 1,
        retrieval_key: key.clone(),
        layers: vec![KVLayerPayload { layer_idx: 0, data: value }],
        salience: 80.0, text: Some("Test fact".into()),
    };
    let pack_id = engine.mem_write_pack(&pack).unwrap();

    let view1 = vec![2.0_f32; 8];
    let view2 = vec![3.0_f32; 8];
    let count = engine.add_view_keys(pack_id, &[view1, view2]).unwrap();

    assert_eq!(count, 2);
    assert_eq!(engine.view_count(pack_id).unwrap(), 2);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p tdb-engine --test acceptance test_add_view_keys_creates_retrieval_cells`
Expected: FAIL — method does not exist.

- [ ] **Step 3: Make `add_cell` pub(crate)**

In `crates/tdb-engine/src/pack_directory.rs` line 100, change `fn add_cell` to `pub(crate) fn add_cell`.

- [ ] **Step 4: Add constant + implement both methods**

In `crates/tdb-engine/src/engine.rs`, near line 67 add:

```rust
const VIEW_CELL_MARKER: u64 = u64::MAX;
```

After `mem_write_pack` (~line 982), add `add_view_keys`:

```rust
pub fn add_view_keys(
    &mut self,
    pack_id: PackId,
    view_keys: &[Vec<f32>],
) -> Result<usize> {
    let cell_ids = self
        .pack_directory
        .cell_ids(pack_id)
        .ok_or_else(|| TardigradeError::PackNotFound(pack_id))?;

    let canonical_cell_id = *cell_ids.first()
        .ok_or_else(|| TardigradeError::PackNotFound(pack_id))?;
    let canonical = self.pool.get(canonical_cell_id)?;
    let owner = canonical.owner;

    let now_nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);

    let mut count = 0;
    for vk in view_keys {
        let cell_id = self.next_id;
        self.next_id += 1;

        let cell = MemoryCellBuilder::new(
            cell_id, owner, PACK_RETRIEVAL_LAYER, vk.clone(), vec![],
        )
        .token_span(pack_id, VIEW_CELL_MARKER)
        .created_at(now_nanos)
        .updated_at(now_nanos)
        .build();

        self.pool.append(&cell)?;
        self.pipeline.insert(cell_id, owner, vk);

        let slb_key = mean_pool_key(vk);
        self.slb.insert(cell_id, owner, &slb_key);

        self.pack_directory.add_cell(pack_id, cell_id);
        count += 1;
    }
    Ok(count)
}

pub fn view_count(&self, pack_id: PackId) -> Result<usize> {
    let cell_ids = self
        .pack_directory
        .cell_ids(pack_id)
        .ok_or_else(|| TardigradeError::PackNotFound(pack_id))?;

    let retrieval_cells = cell_ids
        .iter()
        .filter(|&&cid| {
            self.pool.get(cid)
                .map(|c| c.layer == PACK_RETRIEVAL_LAYER)
                .unwrap_or(false)
        })
        .count();

    Ok(retrieval_cells.saturating_sub(1))
}
```

- [ ] **Step 5: Fix `hydrate_pack_layers` — filter by layer, not fixed skip**

Replace `hydrate_pack_layers` (~line 1177):

```rust
fn hydrate_pack_layers(&self, cell_ids: &[CellId]) -> Result<Vec<KVLayerPayload>> {
    let mut layers = Vec::with_capacity(cell_ids.len());
    for &cell_id in cell_ids {
        match self.pool.get(cell_id) {
            Ok(cell) if cell.layer != PACK_RETRIEVAL_LAYER => {
                layers.push(KVLayerPayload { layer_idx: cell.layer, data: cell.value });
            }
            Ok(_) => {}
            Err(TardigradeError::CellNotFound(_)) => {}
            Err(e) => return Err(e),
        }
    }
    layers.sort_by_key(|layer| layer.layer_idx);
    Ok(layers)
}
```

- [ ] **Step 6: Run test — should pass**

Run: `cargo test -p tdb-engine --test acceptance test_add_view_keys_creates_retrieval_cells`

- [ ] **Step 7: Write test — view key boosts pack in mem_read_pack**

```rust
#[test]
fn test_view_key_boosts_pack_in_read() {
    let dir = tempdir().unwrap();
    let mut engine = Engine::open(dir.path(), 9999);

    let key_a = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0_f32];
    let pack_a = KVPack {
        id: 0, owner: 1, retrieval_key: key_a,
        layers: vec![KVLayerPayload { layer_idx: 0, data: vec![0.0; 8] }],
        salience: 80.0, text: Some("Fact A".into()),
    };
    let pid_a = engine.mem_write_pack(&pack_a).unwrap();

    let key_b = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0_f32];
    let pack_b = KVPack {
        id: 0, owner: 1, retrieval_key: key_b,
        layers: vec![KVLayerPayload { layer_idx: 0, data: vec![0.0; 8] }],
        salience: 80.0, text: Some("Fact B".into()),
    };
    let pid_b = engine.mem_write_pack(&pack_b).unwrap();

    let view = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0_f32];
    engine.add_view_keys(pid_a, &[view]).unwrap();

    let query = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0_f32];
    let results = engine.mem_read_pack(&query, 5, Some(1)).unwrap();
    let packs: Vec<u64> = results.iter().map(|r| r.pack.id).collect();

    assert!(packs.contains(&pid_a), "Pack A found via view key");
    assert!(packs.contains(&pid_b), "Pack B found via canonical");
    assert_eq!(packs.iter().filter(|&&p| p == pid_a).count(), 1, "No duplicates");
}
```

- [ ] **Step 8: Write test — view_count edge cases**

```rust
#[test]
fn test_view_count_zero_for_fresh_pack() {
    let dir = tempdir().unwrap();
    let mut engine = Engine::open(dir.path(), 9999);
    let pack = KVPack {
        id: 0, owner: 1, retrieval_key: vec![1.0; 8],
        layers: vec![KVLayerPayload { layer_idx: 0, data: vec![0.0; 8] }],
        salience: 80.0, text: None,
    };
    let pid = engine.mem_write_pack(&pack).unwrap();
    assert_eq!(engine.view_count(pid).unwrap(), 0);
    assert!(engine.view_count(9999).is_err());
}
```

- [ ] **Step 9: Write test — views survive engine refresh (crash recovery)**

```rust
#[test]
fn test_view_keys_survive_refresh() {
    let dir = tempdir().unwrap();
    let pid;
    {
        let mut engine = Engine::open(dir.path(), 9999);
        let pack = KVPack {
            id: 0, owner: 1, retrieval_key: vec![1.0; 8],
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![0.0; 8] }],
            salience: 80.0, text: Some("Fact".into()),
        };
        pid = engine.mem_write_pack(&pack).unwrap();
        engine.add_view_keys(pid, &[vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).unwrap();
        assert_eq!(engine.view_count(pid).unwrap(), 1);
    }
    // Reopen — simulates crash recovery
    let engine = Engine::open(dir.path(), 9999);
    assert_eq!(engine.view_count(pid).unwrap(), 1);
    let query = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0_f32];
    let results = engine.mem_read_pack(&query, 5, Some(1)).unwrap();
    let packs: Vec<u64> = results.iter().map(|r| r.pack.id).collect();
    assert!(packs.contains(&pid), "View key should survive refresh");
}
```

- [ ] **Step 10: Run full engine test suite**

Run: `cargo test -p tdb-engine`
Expected: All existing + 4 new tests pass.

- [ ] **Step 11: Commit**

```bash
git add crates/tdb-engine/src/engine.rs crates/tdb-engine/src/pack_directory.rs crates/tdb-engine/tests/acceptance.rs
git commit -m "feat(engine): add_view_keys — attach view retrieval cells to existing packs"
```

---

### Task 2: PyO3 Bindings

**Files:**
- Modify: `crates/tdb-python/src/lib.rs`
- Test: `tests/python/test_consolidator.py`

- [ ] **Step 1: Write failing Python tests**

Add to `tests/python/test_consolidator.py`:

```python
class TestEngineViewKeys:
    def test_add_view_keys_returns_count(self, tmp_path):
        engine = tardigrade_db.Engine(str(tmp_path), vamana_threshold=9999)
        key = np.random.randn(8).astype(np.float32)
        val = np.random.randn(8).astype(np.float32)
        pid = engine.mem_write_pack(OWNER, key, [(0, val)], 80.0, text="Test")
        v1 = np.random.randn(8).astype(np.float32)
        v2 = np.random.randn(8).astype(np.float32)
        assert engine.add_view_keys(pid, [v1, v2]) == 2

    def test_view_count(self, tmp_path):
        engine = tardigrade_db.Engine(str(tmp_path), vamana_threshold=9999)
        key = np.random.randn(8).astype(np.float32)
        val = np.random.randn(8).astype(np.float32)
        pid = engine.mem_write_pack(OWNER, key, [(0, val)], 80.0, text="Test")
        assert engine.view_count(pid) == 0
        engine.add_view_keys(pid, [np.random.randn(8).astype(np.float32)])
        assert engine.view_count(pid) == 1

    def test_view_count_errors_on_missing(self, tmp_path):
        engine = tardigrade_db.Engine(str(tmp_path), vamana_threshold=9999)
        with pytest.raises(RuntimeError):
            engine.view_count(9999)

    def test_view_match_returns_canonical(self, tmp_path):
        engine = tardigrade_db.Engine(str(tmp_path), vamana_threshold=9999)
        ckey = np.array([1,0,0,0,0,0,0,0], dtype=np.float32)
        pid = engine.mem_write_pack(OWNER, ckey, [(0, np.zeros(8, dtype=np.float32))], 80.0, text="Fact")
        vkey = np.array([0,1,0,0,0,0,0,0], dtype=np.float32)
        engine.add_view_keys(pid, [vkey])
        results = engine.mem_read_pack(vkey, 5, OWNER)
        pids = [r["pack_id"] for r in results]
        assert pid in pids
        assert pids.count(pid) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/python/test_consolidator.py::TestEngineViewKeys -v`

- [ ] **Step 3: Add PyO3 bindings**

In `crates/tdb-python/src/lib.rs`, add after `mem_write_pack` binding:

```rust
#[pyo3(signature = (pack_id, view_keys))]
fn add_view_keys(
    &self,
    py: Python<'_>,
    pack_id: u64,
    view_keys: Vec<PyReadonlyArray1<'_, f32>>,
) -> PyResult<usize> {
    let keys: Vec<Vec<f32>> = view_keys
        .iter()
        .map(|k| k.as_slice().map(|s| s.to_vec()))
        .collect::<Result<_, _>>()?;
    let engine = Arc::clone(&self.inner);
    py.detach(move || {
        lock_engine(&engine)?
            .add_view_keys(pack_id, &keys)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}

fn view_count(&self, pack_id: u64) -> PyResult<usize> {
    lock_engine(&self.inner)?
        .view_count(pack_id)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}
```

- [ ] **Step 4: Build and test**

Run: `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop && pytest tests/python/test_consolidator.py::TestEngineViewKeys -v`

- [ ] **Step 5: Commit**

```bash
git add crates/tdb-python/src/lib.rs tests/python/test_consolidator.py
git commit -m "feat(python): PyO3 bindings for add_view_keys + view_count"
```

---

### Task 3: LLM View Generation + Diversity Filter

**Files:**
- Modify: `python/tardigrade_hooks/constants.py`
- Modify: `python/tardigrade_hooks/view_generator.py`
- Test: `tests/python/test_view_generator.py`

- [ ] **Step 1: Add constants**

Append to `python/tardigrade_hooks/constants.py`:

```python
VIEW_DIVERSITY_THRESHOLD: float = 0.92
MAX_VIEW_CANDIDATES: int = 5
MAX_VIEWS_KEPT: int = 3
LLM_VIEW_PROMPT_TEMPLATE: str = (
    "Write one specific question that the following fact can answer. "
    "Use different words than the original fact.\n\n"
    "Fact: {fact_text}\n\nQuestion:"
)
```

- [ ] **Step 2: Write failing test for filter_diverse**

Add to `tests/python/test_view_generator.py`:

```python
class TestDiversityFilter:
    def test_filters_near_duplicates(self):
        from tardigrade_hooks.view_generator import filter_diverse
        rng = np.random.default_rng(42)
        v1 = rng.standard_normal(64).astype(np.float32)
        v2 = rng.standard_normal(64).astype(np.float32)
        v3 = rng.standard_normal(64).astype(np.float32)
        v4 = v1 + rng.standard_normal(64).astype(np.float32) * 0.01
        v5 = v2 + rng.standard_normal(64).astype(np.float32) * 0.01
        kept = filter_diverse([v1, v2, v3, v4, v5], threshold=0.92, max_kept=3)
        assert 2 <= len(kept) <= 3

    def test_keeps_all_if_diverse(self):
        from tardigrade_hooks.view_generator import filter_diverse
        rng = np.random.default_rng(42)
        candidates = [rng.standard_normal(64).astype(np.float32) for _ in range(3)]
        assert len(filter_diverse(candidates, threshold=0.92, max_kept=3)) == 3

    def test_empty_input(self):
        from tardigrade_hooks.view_generator import filter_diverse
        assert filter_diverse([], threshold=0.92, max_kept=3) == []
```

Add `import numpy as np` to test file imports if missing.

- [ ] **Step 3: Run to verify failure**

Run: `pytest tests/python/test_view_generator.py::TestDiversityFilter -v`

- [ ] **Step 4: Implement filter_diverse + LLMQuestionFraming**

In `python/tardigrade_hooks/view_generator.py`, update imports and add:

```python
from .constants import (
    DEFAULT_VIEW_FRAMINGS,
    LLM_VIEW_PROMPT_TEMPLATE,
    MAX_VIEW_CANDIDATES,
    MAX_VIEWS_KEPT,
    VIEW_DIVERSITY_THRESHOLD,
)
```

Add after existing strategy classes:

```python
class LLMQuestionFraming(FramingStrategy):
    """LLM-powered question generation (HyPE pattern)."""

    def reframe(self, text: str, model=None, tokenizer=None) -> str:
        if model is None or tokenizer is None:
            raise ValueError("LLMQuestionFraming requires model and tokenizer")
        import torch
        prompt = LLM_VIEW_PROMPT_TEMPLATE.format(fact_text=text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=60,
                do_sample=True, temperature=0.7, top_p=0.9,
            )
        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        question = generated.strip().split("\n")[0].strip()
        if not question.endswith("?"):
            question += "?"
        return question
```

Register: `_FRAMINGS["llm_question"] = LLMQuestionFraming`

Add filter function:

```python
def filter_diverse(
    candidates: list,
    threshold: float = VIEW_DIVERSITY_THRESHOLD,
    max_kept: int = MAX_VIEWS_KEPT,
) -> list:
    if not candidates:
        return []
    import numpy as np

    def _mean(v):
        return v.mean(axis=0) if v.ndim > 1 else v

    def _cosine(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

    kept, kept_means = [], []
    for c in candidates:
        c_mean = _mean(c)
        if all(_cosine(c_mean, km) < threshold for km in kept_means):
            kept.append(c)
            kept_means.append(c_mean)
        if len(kept) >= max_kept:
            break
    return kept
```

Update `ViewGenerator` to support `mode="llm"`:

```python
class ViewGenerator:
    def __init__(self, *, model=None, tokenizer=None,
                 framings: Sequence[str] = DEFAULT_VIEW_FRAMINGS, mode: str = "rule"):
        self.model = model
        self.tokenizer = tokenizer
        self.mode = mode
        self._framings: dict[str, FramingStrategy] = {}
        if mode == "llm":
            self._framings["llm_question"] = LLMQuestionFraming()
        else:
            for name in framings:
                cls = _FRAMINGS.get(name)
                if cls is None:
                    raise ValueError(f"Unknown framing '{name}'. Available: {', '.join(sorted(_FRAMINGS))}")
                self._framings[name] = cls()

    def generate(self, text: str | None) -> list[str]:
        if not text or not text.strip():
            return []
        if self.mode == "llm":
            return self._generate_llm(text)
        return [s.reframe(text) for s in self._framings.values()]

    def _generate_llm(self, text: str) -> list[str]:
        strategy = self._framings["llm_question"]
        candidates = []
        for _ in range(MAX_VIEW_CANDIDATES):
            q = strategy.reframe(text, model=self.model, tokenizer=self.tokenizer)
            candidates.append(q)
        return candidates[:MAX_VIEWS_KEPT]
```

- [ ] **Step 5: Run diversity filter + existing tests**

Run: `pip install -e . && pytest tests/python/test_view_generator.py -v`
Expected: All pass (existing 18 + 3 new).

- [ ] **Step 6: Commit**

```bash
git add python/tardigrade_hooks/constants.py python/tardigrade_hooks/view_generator.py tests/python/test_view_generator.py
git commit -m "feat(hooks): LLMQuestionFraming + diversity filter for multi-view v2"
```

---

### Task 4: Refactor Consolidator

**Files:**
- Modify: `python/tardigrade_hooks/consolidator.py`
- Modify: `tests/python/test_consolidator.py`

- [ ] **Step 1: Rewrite consolidator tests for new API**

Replace test classes in `tests/python/test_consolidator.py`:

`TestConsolidationBasics` — `consolidate()` returns int, uses `view_count()`
`TestIdempotency` — second call returns 0, view_count stable
`TestConsolidateAll` — returns `dict[int, int]`

(See Task 4 in full plan above for exact test code.)

- [ ] **Step 2: Refactor consolidator.py**

Replace `consolidate()`: call `add_view_keys` instead of `mem_write_pack` + `add_pack_edge`.
Replace `_already_consolidated()`: check `view_count > 0` instead of `pack_supports`.
Replace `consolidate_all()` return type: `dict[int, int]`.
Remove `_store_view_pack()`.

- [ ] **Step 3: Run tests**

Run: `pytest tests/python/test_consolidator.py -v`

- [ ] **Step 4: Commit**

```bash
git add python/tardigrade_hooks/consolidator.py tests/python/test_consolidator.py
git commit -m "refactor(hooks): consolidator uses add_view_keys (parent-document pattern)"
```

---

### Task 5: Client + Sweep + Docs Update

**Files:**
- Modify: `python/tardigrade_hooks/client.py`, `consolidation_sweep.py`
- Modify: `tests/python/test_client.py`, `test_consolidation_sweep.py`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update client + sweep return types and field names**
- [ ] **Step 2: Update corresponding tests**
- [ ] **Step 3: Run full Python test suite**
- [ ] **Step 4: Update CLAUDE.md test counts + status**
- [ ] **Step 5: Commit**

---

### Task 6: End-to-End Experiment

**Files:**
- Create: `experiments/multiview_v2_experiment.py`
- Create: `docs/experiments/multi_view_v2/results.md`

- [ ] **Step 1: Write experiment script** (same 10 Sonia facts, real Qwen3-0.6B, LLM view generation + add_view_keys)
- [ ] **Step 2: Run experiment, record R@5 per tier**
- [ ] **Step 3: Document results**
- [ ] **Step 4: Commit**

Success criteria:
- Specific R@5 >= 100%
- Moderate R@5 >= 80%
- Vague R@5 > 60%
