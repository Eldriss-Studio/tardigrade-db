# Multi-View Consolidation v2: Parent-Document Pattern with LLM-Generated Views

## Problem

Multi-view consolidation v1 (rule-based framings stored as separate packs) degraded moderate R@5 from 80% → 20%. Root cause: low-diversity views crowded out canonicals in the shared retrieval index. Diagnosed in `experiments/multiview_diagnosis.py`.

Research consensus (Doc2Query--, HyPE, ENGRAM, SimpleMem) points to a combined fix:
- **Parent-document pattern (B):** Views are retrieval keys only — they resolve to their parent pack, never appear as independent results.
- **LLM generation + diversity filter (C):** Model-generated questions produce discriminative views; a cosine-similarity filter rejects near-duplicates.

## Design

### 1. Rust Engine: `add_view_keys`

New method on `Engine`:

```rust
pub fn add_view_keys(
    &mut self,
    pack_id: PackId,
    view_keys: &[Vec<f32>],
) -> Result<usize>
```

**Behavior:**
- Validates `pack_id` exists in PackDirectory
- For each view key, creates a retrieval cell (`PACK_RETRIEVAL_LAYER`) owned by the pack's owner
- Indexes each cell in the retrieval pipeline (PerTokenRetriever + SLB)
- Maps each cell to `pack_id` in PackDirectory
- Does NOT create governance entries (views ride the canonical's tier/importance)
- Does NOT persist text or layer payloads (retrieval-only)
- Marks view cells with a distinct `token_span` marker (e.g., `(pack_id, VIEW_CELL_MARKER)` where `VIEW_CELL_MARKER = u64::MAX`) so `refresh()` can distinguish them from the canonical retrieval cell during rebuild
- Returns count of view keys added

**Companion accessor:**

```rust
pub fn view_count(&self, pack_id: PackId) -> Result<usize>
```

Returns the number of retrieval cells for `pack_id` minus 1 (the canonical).

**PyO3 bindings:**

```python
engine.add_view_keys(pack_id: int, view_keys: list[np.ndarray]) -> int
engine.view_count(pack_id: int) -> int
```

**Why this works without query-path changes:** `mem_read_pack` calls `deduplicate_pack_candidates()` which groups retrieval cells by pack_id and keeps the highest-scoring cell. If a view key scores higher than the canonical for a given query, the pack gets that better score — but appears only once in results.

### 2. LLM View Generation

**New framing strategy** in `python/tardigrade_hooks/view_generator.py`:

```python
class LLMQuestionFraming(FramingStrategy):
    def reframe(self, text: str, model=None, tokenizer=None) -> str
```

Prompts the model to generate a specific question the fact answers. Each call produces one question. Called 5 times per fact, results fed into diversity filter.

**Prompt template:**

```
Write one specific question that the following fact can answer.
Use different words than the original fact.

Fact: {fact_text}

Question:
```

**ViewGenerator changes:**
- New `mode` parameter: `"rule"` (existing, default) or `"llm"` (new)
- In LLM mode, generates 5 candidates via `LLMQuestionFraming`, applies diversity filter, keeps ≤3

**Diversity filter** (Doc2Query-- pattern):

```python
VIEW_DIVERSITY_THRESHOLD = 0.92  # named constant in constants.py
MAX_VIEW_CANDIDATES = 5
MAX_VIEWS_KEPT = 3

def filter_diverse(candidates_hidden, threshold):
    kept = []
    for h in candidates_hidden:
        h_mean = mean_pool(h)
        if all(cosine_sim(h_mean, k) < threshold for k in kept_means):
            kept.append(h)
        if len(kept) >= MAX_VIEWS_KEPT:
            break
    return kept
```

### 3. Consolidation Flow

**Trigger:** On promotion from Draft → Validated (ι ≥ 65).

**Detection:** `engine.view_count(pack_id) == 0` means unconsolidated.

**Processing pipeline:**
1. `text = engine.pack_text(pack_id)` — get the canonical's source text
2. `view_texts = ViewGenerator(mode="llm").generate(text)` — 5 candidates → filter → ≤3
3. For each surviving view text: `capture_hidden(view_text)` → `encode_per_token()` → view key
4. `engine.add_view_keys(pack_id, view_keys)` — attach to canonical

**Refactored `MemoryConsolidator`:**
- `consolidate(pack_id)` now calls `add_view_keys` instead of creating separate packs
- `ConsolidationPolicy.should_consolidate()` checks tier ≥ Validated AND view_count == 0
- Returns `int` — count of views attached (not pack_ids, since no new packs are created)

**Refactored `ConsolidationSweepThread`:**
- Periodic sweep finds Validated+ packs with `view_count == 0`
- Processes them through the consolidation pipeline
- Status reports `packs_consolidated` and `views_attached` (not `views_created`)

### 4. What Gets Refactored (Existing Modules)

| Module | Change |
|--------|--------|
| `constants.py` | Add `VIEW_DIVERSITY_THRESHOLD = 0.92`, `MAX_VIEW_CANDIDATES = 5`, `MAX_VIEWS_KEPT = 3` |
| `view_generator.py` | Add `LLMQuestionFraming`, `mode` parameter, diversity filter |
| `consolidator.py` | Replace `mem_write_pack` + `add_pack_edge` with `add_view_keys`. Change return type from `list[int]` (pack_ids) to `int` (views attached count) |
| `consolidation_sweep.py` | Trigger change: process only packs with `view_count == 0` at Validated+ tier |
| `client.py` | `consolidate()` uses the refactored consolidator |

### 5. Testing

**Level 1 — Rust acceptance tests:**
- `add_view_keys` creates cells mapped to correct pack
- `view_count` returns 0/N correctly
- `mem_read_pack` deduplicates: view match returns canonical pack, once
- Error on nonexistent pack_id
- Views survive engine refresh (crash recovery)

**Level 2 — Python binding tests:**
- `engine.add_view_keys(pid, [k1, k2, k3])` returns 3
- `engine.view_count(pid)` returns 3
- Idempotency: consolidator skips packs where `view_count > 0`

**Level 3 — LLM integration tests:**
- `LLMQuestionFraming.reframe()` produces a question (contains "?")
- Diversity filter reduces 5 candidates to ≤3
- Filtered views have pairwise cosine < `VIEW_DIVERSITY_THRESHOLD`

**Level 4 — End-to-end experiment:**
- 10 facts, Qwen3-0.6B on MPS, centered refinement
- Consolidate all Validated+ packs with LLM views
- **Success criteria:**
  - Specific R@5 ≥ 100% (no regression)
  - Moderate R@5 ≥ 80% (no regression from baseline)
  - Vague R@5 > 60% (any measurable improvement)

### 6. Named Constants

```python
VIEW_DIVERSITY_THRESHOLD = 0.92
MAX_VIEW_CANDIDATES = 5
MAX_VIEWS_KEPT = 3
LLM_VIEW_PROMPT_TEMPLATE = (
    "Write one specific question that the following fact can answer. "
    "Use different words than the original fact.\n\n"
    "Fact: {fact_text}\n\nQuestion:"
)
```

### 7. What Does NOT Change

- `mem_read_pack` query path (dedup already handles multi-cell packs)
- Mean-centering refinement (corpus mean unaffected — view cells use per-token encoding like canonical)
- Cross-encoder reranker (operates on pack text, not retrieval keys)
- Governance scoring, AKL tiers, importance decay
- WAL, segment storage, TextStore
- File ingest pipeline (uses `mem_write_pack`, not affected)
- Rule-based framings (stay in registry, available as `mode="rule"`)

### 8. Research References

| Pattern | Source | How it applies |
|---------|--------|----------------|
| Parent-Document Retriever | LangChain, HyPE (Vake et al. 2025) | Views as retrieval keys resolving to parent |
| Quality Filter | Doc2Query-- (Gospodinov & MacAvaney 2023) | Cosine diversity filter on generated views |
| Dual-Index Fusion | Doc2Query++ (Oct 2025) | Inspiration; we achieve separation via cell-level dedup instead of separate indexes |
| Typed Partitioning | ENGRAM (2025) | Per-type retrieval prevents cross-type competition |
| Recursive Consolidation | SimpleMem (Liu et al. 2026) | Consolidation reduces redundancy, doesn't add noise |
| Cache Augmentation | Deliberation in Latent Space (DeepMind 2025) | Augmenting KV cache with computed embeddings |
