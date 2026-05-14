# Latent-Space Vague Query Refinement: Whitening + Token Reweighting + Multi-Layer Fusion

> **⚠️ Note — 2026-05-14.** The premise — "vague-query R@5 stuck at 60% / LoCoMo 67.2% due to vocabulary overlap" — is retracted. The 67.2% baseline measured the lexical fallback adapter; the vague/moderate vague-query recall numbers were drawn from a corpus where every item shared the same context, so the vocabulary-mismatch framing is unsupported by data. The `RefinementStrategy` refactor (Strategy + Decorator) is sound infrastructure; the target metrics and motivation are invalidated. See [`../../experiments/2026-05-14-bench-audit.md`](../../experiments/2026-05-14-bench-audit.md). Preserved below as a historical design artifact.

## Problem

TardigradeDB's vague-query recall is stuck at 60% R@5 (LoCoMo 67.2%). The
vocabulary-overlap ceiling means queries phrased differently from stored
memories fail in latent-space dot-product retrieval. Mean-centering lifted
moderate R@5 by +31pp but vague queries barely moved (+4pp).

All three features must preserve TardigradeDB's tensor-native premise: retrieval
operates entirely on the model's own hidden-state representations. No text
retrieval, no external embedding model, no BM25.

## Design Patterns

### Strategy Pattern — `RefinementStrategy` trait

The existing `RefinementMode` enum dispatches via match arms in `apply()` — a
closed design that requires editing the enum and dispatcher for every new mode.
Replace with the **Strategy pattern**: a `RefinementStrategy` trait with
concrete implementations. Each strategy transforms query tokens and/or
rescores candidates independently.

```
RefinementStrategy (trait)
├── NoOpStrategy          (existing: pass-through)
├── MeanCenteredStrategy  (existing: subtract corpus mean)
├── WhiteningStrategy     (new: ZCA whitening)
└── LatentPrfStrategy     (existing: Rocchio expansion)
```

This gives OCP — new strategies added without modifying existing code.

### Decorator Pattern — `TokenWeighter`

Token importance reweighting wraps the existing scoring function. Rather than
modifying `top_n_avg` directly, a `TokenWeighter` decorator applies per-token
weights before the aggregation step. The scorer doesn't know about weighting;
the weighter doesn't know about scoring. SRP preserved.

### Composite Pattern — Multi-Layer Fusion

Multiple query representations (from different layers) are each scored
independently, then fused via RRF. The `MultiLayerFusion` composite holds
N query keys and delegates retrieval to the engine N times, merging results.
This lives in Python (client-side orchestration), not in the Rust engine.

### Template Method — Corpus Statistics

`PerTokenRetriever` already uses Template Method for corpus mean tracking
(accumulate on insert, compute on demand). Whitening extends this with
covariance tracking — same pattern, same lifecycle hooks.

## ATDD Acceptance Criteria

### Whitening:
1. `RefinementMode::Whitened` produces different scores than `MeanCentered` for the same query
2. Whitened mode does not regress specific R@5 below 100% on the 10-fact corpus
3. When corpus has fewer tokens than dimensions, whitening falls back to mean-centering (graceful degradation)
4. Whitening matrix is computed once and cached until corpus changes
5. `set_refinement_mode("whitened")` from Python works

### Token Reweighting:
6. With reweighting enabled, same query produces different top-k ordering
7. Weight = `1 - cosine(token, corpus_mean)` — tokens near the mean get low weight
8. Reweighting is orthogonal to refinement mode (can stack with any strategy)
9. `engine.set_token_reweighting(True/False)` from Python works
10. When disabled (default), scoring is identical to current behavior

### Multi-Layer Fusion:
11. 3-layer query returns results that include memories not found by single-layer
12. RRF fusion ranks memories found by multiple layers higher
13. Single-layer mode produces identical results to current behavior
14. Configurable layer ratios (default: 50%, 67%, 83% depth)

### Stacking:
15. All three features can be enabled simultaneously without interference
16. Stacked configuration does not regress specific R@5 below 100%

## Components

### 1. `WhiteningStrategy` (Rust — `crates/tdb-retrieval/src/refinement.rs`)

Extends the refinement module with a new variant and implementation.

**Corpus statistics extension** in `PerTokenRetriever`:
- New field: `corpus_sq_sum: Vec<f32>` — running sum of `token_i * token_i^T` per dimension pair, stored as flattened `dim × dim` matrix. Updated incrementally on each insert: `corpus_sq_sum[j*dim+k] += token[j] * token[k]` for each token. This avoids recomputing outer products from scratch.
- New method: `corpus_covariance() -> Option<Vec<f32>>` — computes `(corpus_sq_sum / count) - (mean * mean^T)` (the standard incremental covariance formula)
- New method: `whitening_matrix() -> Option<Vec<f32>>` — returns `Σ^{-1/2}` via eigendecomposition, cached in `whitening_cache: Option<Vec<f32>>`
- Cache invalidated when `corpus_token_count` changes

**Whitening rescore** (same shape as `mean_centered_rescore`):
1. Get whitening matrix `W` from retriever (lazy compute + cache)
2. For each query token: `q_w = W @ (q - μ)`
3. For each candidate's stored tokens: `s_w = W @ (s - μ)`
4. Rescore via `top_n_avg` on whitened tokens
5. Sort and truncate to k

**Fallback:** If eigendecomposition fails or corpus too small (< dim tokens), fall back to mean-centering.

**Complexity:** Eigendecomposition is O(dim³) ≈ O(10⁹) for dim=1024. Done once, cached. At 1024-dim this takes ~1-2 seconds on modern CPU. The per-query cost is O(n_tokens × dim²) for the matrix multiply — ~1ms per query token at dim=1024.

### 2. `TokenWeighter` (Rust — `crates/tdb-retrieval/src/token_weighter.rs`)

New module implementing the Decorator pattern.

```rust
pub struct TokenWeighter {
    enabled: bool,
}

impl TokenWeighter {
    pub fn weight(&self, token: &[f32], corpus_mean: &[f32]) -> f32
    pub fn apply_weights(&self, dot_products: &mut [(f32, CellId)], query_tokens: &[f32], corpus_mean: &[f32])
}
```

**Weight formula:** `w_i = 1.0 - cosine_sim(token_i, corpus_mean)`.
- Tokens aligned with the corpus mean (common direction) → weight near 0
- Tokens orthogonal to the mean (distinctive) → weight near 1

**Integration point:** Applied in `top_n_avg` scoring before the top-5 selection. The dot product for each query-token × stored-token pair is multiplied by the query token's weight.

**Engine integration:**
- `Engine` gains a `token_reweighting: bool` field (default `false`)
- `set_token_reweighting(enabled: bool)` setter
- Passed to the refinement/scoring path alongside `refinement_mode`

### 3. Multi-Layer Fusion (Python — `python/tardigrade_hooks/multi_layer_query.py`)

New module implementing the Composite pattern.

```python
class MultiLayerQuery:
    def __init__(self, engine, layer_ratios=(0.50, 0.67, 0.83), rrf_k=60):
        ...
    
    def query(self, model, tokenizer, query_text, k=5, owner=None):
        """Run retrieval at each layer, fuse via RRF."""
        ...
```

**Flow:**
1. Run forward pass once, capturing hidden states at all requested layers
2. For each layer: `encode_per_token(hidden_states[layer], dim)` → query key
3. For each query key: `engine.mem_read_pack(key, k * 2, owner)` → ranked list
4. Fuse ranked lists via RRF: `score(pack) = Σ 1/(rrf_k + rank_i(pack))`
5. Sort by fused score, return top k

**Integration:** `TardigradeClient.query()` gains an optional `multi_layer=True` parameter that activates this path. Default `False` (single-layer, backward compatible).

**Cost:** 1 forward pass (same as now — hidden states from all layers are already computed, we just extract multiple). N engine queries (one per layer). RRF fusion is O(k × N) — negligible.

### 4. RefinementMode Enum Extension (Rust)

```rust
pub enum RefinementMode {
    None,
    MeanCentered,
    Whitened,           // NEW
    LatentPrf { alpha: f32, beta: f32, k_prime: usize },
}
```

The `apply()` dispatcher gains one match arm for `Whitened`. The function signature is unchanged — OCP at the call site.

### 5. Python API Extensions

```python
engine.set_refinement_mode("whitened")          # New mode
engine.set_token_reweighting(True)              # New flag
client.query(text, multi_layer=True)            # New option
```

All three are independently toggleable. Stacking: `engine.set_refinement_mode("whitened")` + `engine.set_token_reweighting(True)` + `client.query(text, multi_layer=True)`.

## Files

| File | Action | Pattern | What |
|------|--------|---------|------|
| `crates/tdb-retrieval/src/refinement.rs` | Modify | Strategy | Add `Whitened` variant + `whitened_rescore` function |
| `crates/tdb-retrieval/src/per_token.rs` | Modify | Template Method | Add covariance tracking + whitening matrix cache |
| `crates/tdb-retrieval/src/token_weighter.rs` | Create | Decorator | Per-token importance weighting |
| `crates/tdb-retrieval/src/lib.rs` | Modify | — | Export `token_weighter` module |
| `crates/tdb-engine/src/engine.rs` | Modify | — | Add `token_reweighting` field + setter, pass to scoring |
| `crates/tdb-python/src/lib.rs` | Modify | — | PyO3 bindings: `set_token_reweighting`, `"whitened"` mode |
| `python/tardigrade_hooks/multi_layer_query.py` | Create | Composite | Multi-layer query + RRF fusion |
| `python/tardigrade_hooks/client.py` | Modify | Facade | Add `multi_layer` param to `query()` |
| `crates/tdb-engine/tests/acceptance.rs` | Modify | — | Whitening + reweighting acceptance tests |
| `tests/python/test_multi_layer_query.py` | Create | — | Multi-layer fusion tests |
| `experiments/vague_refinement_v2_experiment.py` | Create | — | Stacking experiment |

## SOLID Analysis

- **SRP:** WhiteningStrategy rescores. TokenWeighter weights. MultiLayerQuery fuses. Each does one thing.
- **OCP:** New refinement strategies added to the enum without modifying existing strategies. TokenWeighter is orthogonal — enable/disable without touching the scorer.
- **LSP:** All refinement strategies produce `Vec<RetrievalResult>` — substitutable.
- **ISP:** TokenWeighter has a minimal interface (weight + apply_weights). MultiLayerQuery has one method (query).
- **DIP:** Engine depends on `RefinementMode` enum (value type), not concrete strategy internals. MultiLayerQuery depends on the engine's `mem_read_pack` interface, not internals.

## What Does NOT Change

- Memory storage format (packs, cells, Q4 quantization)
- `mem_write_pack`, `add_view_keys`, `mem_read_tokens` APIs
- Governance (AKL tiers, importance, decay)
- TextStore, WAL, segment compaction
- File ingestion pipeline
- Cross-encoder reranker (can stack on top as Stage-3)
- Default behavior (all features off by default)

## Verification

1. `cargo test -p tdb-retrieval` — whitening + reweighting unit tests pass
2. `cargo test -p tdb-engine` — acceptance tests pass (including existing refinement tests)
3. `pytest tests/python/test_multi_layer_query.py` — fusion tests pass
4. Stacking experiment on 10-fact corpus with Qwen3-0.6B:
   - Specific R@5 ≥ 100% (no regression)
   - Moderate R@5 ≥ 80% (no regression)
   - Vague R@5 > 60% (any measurable improvement)
5. If vague improves: re-run full LoCoMo benchmark to measure impact on 67.2%

## Research References

| Paper | How it applies |
|-------|----------------|
| WhiteningBERT (Huang & Tang, 2021) | Whitening sentence embeddings for better retrieval — our approach applied to KV hidden states |
| Soft-ZCA (ESANN 2025) | Modified ZCA controlling isotropy levels — fallback strategy when full whitening is unstable |
| SToRI (EMNLP 2024) | Semantic token reweighting via attention — our IDF-like approach is training-free version |
| LaSER (March 2026) | Latent reasoning in retriever — validates that latent-space transforms improve retrieval without text |
| BERT-flow (Li et al., EMNLP 2020) | Mean-centering as a cheap whitening approximation — what we already implemented, this spec extends it |
