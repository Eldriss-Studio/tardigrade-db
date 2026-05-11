# Latent-Space Vague Query Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve vague-query R@5 (currently 60%) with three training-free latent-space techniques: ZCA whitening, token importance reweighting, and multi-layer query fusion.

**Architecture:** Refactor the closed `RefinementMode` enum into a `RefinementStrategy` trait (Strategy pattern) first. Then add `WhiteningStrategy` as a new implementation. Token reweighting decorates the scoring path. Multi-layer fusion is Python-side RRF over multiple engine queries.

**Tech Stack:** Rust (tdb-retrieval with `faer` crate for linear algebra), Python (tardigrade_hooks).

---

### Task 1: Refactor RefinementMode enum → RefinementStrategy trait (Strategy Pattern)

This task changes NO behavior — pure refactor. All existing tests must continue passing with identical results.

**Files:**
- Modify: `crates/tdb-retrieval/src/refinement.rs`
- Modify: `crates/tdb-engine/src/engine.rs` (update field type + call site)
- Modify: `crates/tdb-python/src/lib.rs` (update mode construction)
- Test: existing tests in `crates/tdb-retrieval/src/refinement.rs` + `crates/tdb-engine/tests/acceptance.rs`

- [ ] **Step 1: Write acceptance test that proves the refactor is behavioral no-op**

In `crates/tdb-retrieval/src/refinement.rs`, add to the existing `#[cfg(test)] mod tests`:

```rust
#[test]
fn strategy_trait_produces_same_results_as_enum() {
    // Verify behavioral equivalence: Strategy trait dispatch matches old enum dispatch.
    let mut r = populated_retriever();
    let q = encode_per_token_keys(&[&[0.7_f32, 0.3, 0.1, 0.0][..]]);
    let first_stage = r.query(&q, 3, Some(OWNER));

    // MeanCentered via trait should produce identical scores to direct function call.
    let direct = mean_centered_rescore(&r, &q, first_stage.clone(), 3);
    let via_trait = MeanCenteredStrategy.refine(&mut r, &q, first_stage, 3, Some(OWNER));

    assert_eq!(direct.len(), via_trait.len());
    for (d, t) in direct.iter().zip(via_trait.iter()) {
        assert_eq!(d.cell_id, t.cell_id);
        assert!((d.score - t.score).abs() < 1e-6);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p tdb-retrieval strategy_trait_produces`
Expected: FAIL — `MeanCenteredStrategy` does not exist.

- [ ] **Step 3: Define the RefinementStrategy trait and implementations**

In `crates/tdb-retrieval/src/refinement.rs`, add after the imports:

```rust
/// Strategy trait for post-retrieval refinement (replaces enum dispatch).
///
/// Each implementation transforms first-stage results in latent space.
/// The engine holds a `Box<dyn RefinementStrategy>` instead of an enum.
pub trait RefinementStrategy: Send + Sync {
    fn name(&self) -> &str;
    fn refine(
        &self,
        retriever: &mut PerTokenRetriever,
        query_key: &[f32],
        first_stage: Vec<RetrievalResult>,
        k: usize,
        owner_filter: Option<OwnerId>,
    ) -> Vec<RetrievalResult>;
}

/// Pass-through — no refinement.
pub struct NoOpStrategy;

impl RefinementStrategy for NoOpStrategy {
    fn name(&self) -> &str { "none" }
    fn refine(&self, _r: &mut PerTokenRetriever, _q: &[f32],
              first_stage: Vec<RetrievalResult>, _k: usize, _o: Option<OwnerId>,
    ) -> Vec<RetrievalResult> {
        first_stage
    }
}

/// Subtract corpus mean from query + stored tokens, re-score.
pub struct MeanCenteredStrategy;

impl RefinementStrategy for MeanCenteredStrategy {
    fn name(&self) -> &str { "centered" }
    fn refine(&self, retriever: &mut PerTokenRetriever, query_key: &[f32],
              first_stage: Vec<RetrievalResult>, k: usize, _o: Option<OwnerId>,
    ) -> Vec<RetrievalResult> {
        mean_centered_rescore(retriever, query_key, first_stage, k)
    }
}

/// Rocchio expansion in latent K-space.
pub struct LatentPrfStrategy {
    pub alpha: f32,
    pub beta: f32,
    pub k_prime: usize,
}

impl RefinementStrategy for LatentPrfStrategy {
    fn name(&self) -> &str { "prf" }
    fn refine(&self, retriever: &mut PerTokenRetriever, query_key: &[f32],
              first_stage: Vec<RetrievalResult>, k: usize, owner_filter: Option<OwnerId>,
    ) -> Vec<RetrievalResult> {
        latent_prf(retriever, query_key, first_stage, self.alpha, self.beta, self.k_prime, k, owner_filter)
    }
}
```

Keep the existing `RefinementMode` enum AND the `apply()` function — they'll be removed in a follow-up once the engine is migrated. This way existing tests pass during the transition.

Add a factory function:

```rust
/// Build a strategy from a mode name (Python API compatibility).
pub fn strategy_from_name(name: &str, alpha: f32, beta: f32, k_prime: usize) -> Box<dyn RefinementStrategy> {
    match name {
        "none" => Box::new(NoOpStrategy),
        "centered" | "mean_centered" => Box::new(MeanCenteredStrategy),
        "prf" | "latent_prf" => Box::new(LatentPrfStrategy { alpha, beta, k_prime }),
        _ => Box::new(NoOpStrategy),
    }
}
```

- [ ] **Step 4: Run the new test + all existing tests**

Run: `cargo test -p tdb-retrieval`
Expected: All existing tests pass + new test passes.

- [ ] **Step 5: Migrate Engine to use `Box<dyn RefinementStrategy>`**

In `crates/tdb-engine/src/engine.rs`:

Change the field (line 213):
```rust
    refinement_strategy: Box<dyn tdb_retrieval::refinement::RefinementStrategy>,
```

Update `open_with_options` initialization (line 287):
```rust
    refinement_strategy: Box::new(tdb_retrieval::refinement::NoOpStrategy),
```

Replace the setter (line 240-243):
```rust
    pub fn set_refinement_mode(&mut self, mode: tdb_retrieval::refinement::RefinementMode) {
        self.refinement_strategy = match mode {
            tdb_retrieval::refinement::RefinementMode::None => {
                Box::new(tdb_retrieval::refinement::NoOpStrategy)
            }
            tdb_retrieval::refinement::RefinementMode::MeanCentered => {
                Box::new(tdb_retrieval::refinement::MeanCenteredStrategy)
            }
            tdb_retrieval::refinement::RefinementMode::LatentPrf { alpha, beta, k_prime } => {
                Box::new(tdb_retrieval::refinement::LatentPrfStrategy { alpha, beta, k_prime })
            }
        };
    }
```

Replace the getter (line 245-246):
```rust
    pub fn refinement_mode_name(&self) -> &str {
        self.refinement_strategy.name()
    }
```

Replace the refinement call site (lines 678-691):
```rust
        if self.refinement_strategy.name() != "none"
            && is_per_token_query
            && let Some(per_token) =
                self.pipeline.first_stage_as_mut::<tdb_retrieval::per_token::PerTokenRetriever>()
        {
            candidates = self.refinement_strategy.refine(
                per_token, query_key, candidates, k * 2, owner_filter,
            );
        }
```

- [ ] **Step 6: Run full engine + retrieval test suites**

Run: `cargo test --workspace --exclude tdb-python`
Expected: All tests pass — behavioral no-op.

- [ ] **Step 7: Update PyO3 bindings**

In `crates/tdb-python/src/lib.rs`, update `set_refinement_mode` to use `strategy_from_name`:

```rust
    fn set_refinement_mode(/* existing params */) -> PyResult<()> {
        // Keep existing param parsing...
        // Replace the final line with:
        lock_engine(&engine)?.set_refinement_strategy(
            tdb_retrieval::refinement::strategy_from_name(&mode_str, alpha, beta, k_prime)
        );
    }
```

And add the corresponding engine method:
```rust
    pub fn set_refinement_strategy(&mut self, strategy: Box<dyn tdb_retrieval::refinement::RefinementStrategy>) {
        self.refinement_strategy = strategy;
    }
```

- [ ] **Step 8: Run Python tests**

Run: `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop && pytest tests/python/test_vague_refinement.py -v`
Expected: All 11 refinement tests pass.

- [ ] **Step 9: Commit**

```bash
git add crates/tdb-retrieval/src/refinement.rs crates/tdb-engine/src/engine.rs crates/tdb-python/src/lib.rs
git commit -m "♻️ refactor(retrieval): RefinementStrategy trait replaces enum dispatch (Strategy pattern)"
```

---

### Task 2: Corpus Covariance + ZCA Whitening via `faer`

**Files:**
- Modify: `crates/tdb-retrieval/Cargo.toml` (add `faer`)
- Modify: `crates/tdb-retrieval/src/per_token.rs` (covariance tracking + whitening matrix)
- Modify: `crates/tdb-retrieval/src/refinement.rs` (add `WhiteningStrategy`)
- Modify: `crates/tdb-python/src/lib.rs` (add `"whitened"` to factory)
- Test: inline in `per_token.rs` and `refinement.rs`

- [ ] **Step 1: Write ATDD tests FIRST — covariance and whitening**

In `crates/tdb-retrieval/src/per_token.rs` tests:

```rust
#[test]
fn test_corpus_covariance_none_when_empty() {
    let r = PerTokenRetriever::with_scoring_mode(ScoringMode::Top5Avg);
    assert!(r.corpus_covariance().is_none());
}

#[test]
fn test_corpus_covariance_computed_correctly() {
    let mut r = PerTokenRetriever::with_scoring_mode(ScoringMode::Top5Avg);
    r.insert(0, 1, &encode_per_token_keys(&[&[2.0_f32, 0.0]]));
    r.insert(1, 1, &encode_per_token_keys(&[&[0.0_f32, 2.0]]));
    let cov = r.corpus_covariance().unwrap();
    assert_eq!(cov.len(), 4);
    assert!((cov[0] - 1.0).abs() < 0.01);
    assert!((cov[3] - 1.0).abs() < 0.01);
}
```

In `crates/tdb-retrieval/src/refinement.rs` tests:

```rust
#[test]
fn whitened_produces_different_scores_than_mean_centered() {
    let mut r = populated_retriever();
    let q = encode_per_token_keys(&[&[0.7_f32, 0.3, 0.1, 0.0][..]]);
    let first = r.query(&q, 3, Some(OWNER));
    let mc = MeanCenteredStrategy.refine(&mut r, &q, first.clone(), 3, Some(OWNER));
    let wh = WhiteningStrategy.refine(&mut r, &q, first, 3, Some(OWNER));
    let mc_scores: Vec<f32> = mc.iter().map(|r| r.score).collect();
    let wh_scores: Vec<f32> = wh.iter().map(|r| r.score).collect();
    assert_ne!(mc_scores, wh_scores);
}

#[test]
fn whitened_falls_back_when_corpus_too_small() {
    let mut r = PerTokenRetriever::with_scoring_mode(ScoringMode::Top5Avg);
    r.insert(0, OWNER, &encode_per_token_keys(&[&[1.0_f32, 0.0, 0.0, 0.0]]));
    let q = encode_per_token_keys(&[&[1.0_f32, 0.0, 0.0, 0.0][..]]);
    let first = r.query(&q, 1, Some(OWNER));
    let refined = WhiteningStrategy.refine(&mut r, &q, first, 1, Some(OWNER));
    assert!(!refined.is_empty());
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p tdb-retrieval corpus_covariance whitened`
Expected: FAIL.

- [ ] **Step 3: Add `faer` dependency**

In `crates/tdb-retrieval/Cargo.toml` under `[dependencies]`:
```toml
faer = "0.24"
```

- [ ] **Step 4: Implement covariance tracking in per_token.rs**

Add `corpus_sq_sum: Vec<f32>` field, `whitening_cache: Option<Vec<f32>>`, `whitening_cache_count: usize` fields. Add `corpus_covariance()` and `whitening_matrix()` methods. Update `accumulate_corpus()` to track outer product sums. Use `faer::Mat` for eigendecomposition in `compute_whitening_matrix()`.

(See previous plan Task 1 steps 3-4 and Task 2 step 4 for exact code — the implementation is the same, only the ordering changes: tests are already written.)

- [ ] **Step 5: Implement `WhiteningStrategy`**

In `crates/tdb-retrieval/src/refinement.rs`:

```rust
pub struct WhiteningStrategy;

impl RefinementStrategy for WhiteningStrategy {
    fn name(&self) -> &str { "whitened" }
    fn refine(&self, retriever: &mut PerTokenRetriever, query_key: &[f32],
              first_stage: Vec<RetrievalResult>, k: usize, _o: Option<OwnerId>,
    ) -> Vec<RetrievalResult> {
        whitened_rescore(retriever, query_key, first_stage, k)
    }
}
```

The `whitened_rescore` function uses `faer` for the matrix-vector multiply (via `retriever.whitening_matrix()`), falls back to `mean_centered_rescore` if whitening unavailable.

Add `"whitened"` to `strategy_from_name`:
```rust
"whitened" => Box::new(WhiteningStrategy),
```

- [ ] **Step 6: Update PyO3 — add "whitened" alias**

Already handled by `strategy_from_name` — just verify the Python binding calls it.

- [ ] **Step 7: Run all tests**

Run: `cargo test --workspace --exclude tdb-python`
Expected: All pass.

Run: `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop && pytest tests/python/test_vague_refinement.py -v`
Expected: All pass.

- [ ] **Step 8: Commit**

```bash
git add crates/tdb-retrieval/ crates/tdb-python/src/lib.rs
git commit -m "✨ feat(retrieval): ZCA whitening strategy — faer eigendecomposition + corpus covariance"
```

---

### Task 3: Token Importance Reweighting (Decorator Pattern)

**Files:**
- Create: `crates/tdb-retrieval/src/token_weighter.rs`
- Modify: `crates/tdb-retrieval/src/lib.rs`
- Modify: `crates/tdb-retrieval/src/refinement.rs` (integrate into rescore fns)
- Modify: `crates/tdb-engine/src/engine.rs` (field + setter)
- Modify: `crates/tdb-python/src/lib.rs` (binding)

- [ ] **Step 1: Write ATDD tests FIRST**

Create `crates/tdb-retrieval/src/token_weighter.rs` with tests only:

```rust
//! Per-token importance weighting — Decorator pattern.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_near_mean_is_low() {
        let mean = [1.0_f32, 0.0, 0.0];
        let token = [0.9_f32, 0.1, 0.0];
        assert!(token_weight(&token, &mean) < 0.3);
    }

    #[test]
    fn weight_orthogonal_is_high() {
        let mean = [1.0_f32, 0.0, 0.0];
        let token = [0.0_f32, 1.0, 0.0];
        assert!(token_weight(&token, &mean) > 0.8);
    }

    #[test]
    fn disabled_returns_one() {
        assert!((weighted_or_unit(false, &[0.5, 0.5], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn weight_clamped_non_negative() {
        let mean = [1.0_f32, 0.0];
        let token = [1.0_f32, 0.0];
        assert!(token_weight(&token, &mean) >= 0.0);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p tdb-retrieval token_weight`
Expected: FAIL.

- [ ] **Step 3: Implement token_weighter.rs**

```rust
//! Per-token importance weighting — Decorator pattern.
//!
//! Weight = `1 - cosine(token, corpus_mean)`. Distinctive tokens
//! (orthogonal to mean) get high weight; common tokens get low weight.

pub fn token_weight(token: &[f32], corpus_mean: &[f32]) -> f32 {
    let dot: f32 = token.iter().zip(corpus_mean).map(|(a, b)| a * b).sum();
    let norm_t = token.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_m = corpus_mean.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_t < 1e-9 || norm_m < 1e-9 {
        return 1.0;
    }
    (1.0 - dot / (norm_t * norm_m)).clamp(0.0, 1.0)
}

pub fn weighted_or_unit(enabled: bool, token: &[f32], corpus_mean: &[f32]) -> f32 {
    if enabled { token_weight(token, corpus_mean) } else { 1.0 }
}
```

Export in `crates/tdb-retrieval/src/lib.rs`:
```rust
pub mod token_weighter;
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p tdb-retrieval token_weight`
Expected: 4 tests PASS.

- [ ] **Step 5: Integrate into refinement + engine**

Add `reweight: bool` parameter to `RefinementStrategy::refine()` and all implementations. Update `apply()` and all existing call sites. Add `token_reweighting: bool` field to Engine with setter. Add PyO3 binding. Multiply each dot product by `weighted_or_unit(reweight, &query_token, &mean)` in the rescore functions.

- [ ] **Step 6: Run full test suite**

Run: `cargo test --workspace --exclude tdb-python`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add crates/tdb-retrieval/ crates/tdb-engine/src/engine.rs crates/tdb-python/src/lib.rs
git commit -m "✨ feat(retrieval): token importance reweighting — corpus-mean distance (Decorator pattern)"
```

---

### Task 4: Multi-Layer Query Fusion (Python, Composite Pattern)

**Files:**
- Create: `python/tardigrade_hooks/multi_layer_query.py`
- Create: `tests/python/test_multi_layer_query.py`
- Modify: `python/tardigrade_hooks/client.py`

- [ ] **Step 1: Write ATDD tests FIRST**

Create `tests/python/test_multi_layer_query.py`:

```python
"""ATDD tests for multi-layer RRF fusion — Composite pattern."""

import pytest
from tardigrade_hooks.multi_layer_query import rrf_fuse


class TestRRFFusion:
    def test_single_list_preserves_order(self):
        ranked = [{"pack_id": 1, "score": 0.9}, {"pack_id": 2, "score": 0.5}]
        fused = rrf_fuse([ranked], k=60)
        assert [r["pack_id"] for r in fused] == [1, 2]

    def test_shared_packs_rank_highest(self):
        a = [{"pack_id": 1, "score": 0.9}, {"pack_id": 2, "score": 0.5}]
        b = [{"pack_id": 2, "score": 0.8}, {"pack_id": 3, "score": 0.4}]
        fused = rrf_fuse([a, b], k=60)
        assert fused[0]["pack_id"] == 2

    def test_empty_input(self):
        assert rrf_fuse([], k=60) == []
        assert rrf_fuse([[]], k=60) == []

    def test_disjoint_lists(self):
        a = [{"pack_id": 1, "score": 0.9}]
        b = [{"pack_id": 2, "score": 0.9}]
        c = [{"pack_id": 3, "score": 0.9}]
        fused = rrf_fuse([a, b, c], k=60)
        assert len(fused) == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=python pytest tests/python/test_multi_layer_query.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement multi_layer_query.py**

Create `python/tardigrade_hooks/multi_layer_query.py` with `rrf_fuse()` function and `MultiLayerQuery` class. (See previous plan Task 4 step 3 for exact code.)

- [ ] **Step 4: Update client.py with `multi_layer` parameter**

- [ ] **Step 5: Run tests**

Run: `pip install -e . && PYTHONPATH=python pytest tests/python/test_multi_layer_query.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add python/tardigrade_hooks/multi_layer_query.py python/tardigrade_hooks/client.py tests/python/test_multi_layer_query.py
git commit -m "✨ feat(hooks): multi-layer query fusion via RRF (Composite pattern)"
```

---

### Task 5: Stacking Experiment + CLAUDE.md

- [ ] **Step 1: Write experiment** testing all configurations on 10-fact Sonia corpus
- [ ] **Step 2: Run experiment, record R@5**
- [ ] **Step 3: Update CLAUDE.md**
- [ ] **Step 4: Commit and push**

Success criteria: Specific ≥ 100%, Moderate ≥ 80%, Vague > 60%.
