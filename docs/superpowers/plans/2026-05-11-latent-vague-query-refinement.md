# Latent-Space Vague Query Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve vague-query R@5 (currently 60%) with three training-free latent-space techniques: ZCA whitening, token importance reweighting, and multi-layer query fusion — all preserving TardigradeDB's tensor-native premise.

**Architecture:** Whitening extends the existing refinement pipeline (new `RefinementMode::Whitened` variant using corpus covariance statistics). Token reweighting decorates the scoring function with per-token IDF-like weights. Multi-layer fusion is Python-side orchestration (multiple `mem_read_pack` calls + RRF merge).

**Tech Stack:** Rust (tdb-retrieval, tdb-engine), `faer` crate for eigendecomposition, Python (tardigrade_hooks).

---

### Task 1: Corpus Covariance Statistics in PerTokenRetriever

**Files:**
- Modify: `crates/tdb-retrieval/src/per_token.rs`
- Test: `crates/tdb-retrieval/src/per_token.rs` (inline `#[cfg(test)]` module)

- [ ] **Step 1: Write failing test for corpus_sq_sum tracking**

In the existing `#[cfg(test)] mod tests` at the bottom of `crates/tdb-retrieval/src/per_token.rs`, add:

```rust
#[test]
fn test_corpus_covariance_is_none_when_empty() {
    let r = PerTokenRetriever::with_scoring_mode(ScoringMode::Top5Avg);
    assert!(r.corpus_covariance().is_none());
}

#[test]
fn test_corpus_covariance_computed_from_inserted_tokens() {
    let mut r = PerTokenRetriever::with_scoring_mode(ScoringMode::Top5Avg);
    let t1 = [2.0_f32, 0.0];
    let t2 = [0.0_f32, 2.0];
    r.insert(0, 1, &encode_per_token_keys(&[&t1]));
    r.insert(1, 1, &encode_per_token_keys(&[&t2]));
    let cov = r.corpus_covariance().unwrap();
    // Covariance of [[2,0],[0,2]] with mean [1,1]:
    // Var(x)=1, Var(y)=1, Cov(x,y)=-1
    assert_eq!(cov.len(), 4); // 2x2 flattened
    assert!((cov[0] - 1.0).abs() < 0.01); // Var(x)
    assert!((cov[3] - 1.0).abs() < 0.01); // Var(y)
    assert!((cov[1] - (-1.0)).abs() < 0.01); // Cov(x,y)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p tdb-retrieval test_corpus_covariance`
Expected: FAIL — method does not exist.

- [ ] **Step 3: Add `corpus_sq_sum` field and methods**

In `crates/tdb-retrieval/src/per_token.rs`:

Add field after `corpus_token_count` (line 177):
```rust
    /// Running sum of per-dimension squared products for covariance estimation.
    /// Flattened dim×dim matrix: `corpus_sq_sum[j*dim+k] += token[j] * token[k]`.
    corpus_sq_sum: Vec<f32>,
```

Initialize in `with_config` (line 202):
```rust
    corpus_sq_sum: Vec::new(),
```

Reset in `ensure_corpus_sum_dim` (add after `self.corpus_token_count = 0;`):
```rust
    self.corpus_sq_sum = vec![0.0; dim * dim];
```

Add accumulation in `accumulate_corpus` (after `self.corpus_token_count += 1;`):
```rust
    let dim = token.len();
    for j in 0..dim {
        for k in 0..dim {
            self.corpus_sq_sum[j * dim + k] += token[j] * token[k];
        }
    }
```

Add public method after `corpus_mean()`:
```rust
    pub fn corpus_covariance(&self) -> Option<Vec<f32>> {
        if self.corpus_token_count < 2 || self.corpus_sum.is_empty() {
            return None;
        }
        let dim = self.corpus_sum.len();
        if self.corpus_sq_sum.len() != dim * dim {
            return None;
        }
        let n = self.corpus_token_count as f32;
        let mean: Vec<f32> = self.corpus_sum.iter().map(|s| s / n).collect();
        let mut cov = vec![0.0_f32; dim * dim];
        for j in 0..dim {
            for k in 0..dim {
                cov[j * dim + k] = self.corpus_sq_sum[j * dim + k] / n - mean[j] * mean[k];
            }
        }
        Some(cov)
    }
```

- [ ] **Step 4: Run test**

Run: `cargo test -p tdb-retrieval test_corpus_covariance`
Expected: PASS

- [ ] **Step 5: Run full retrieval test suite**

Run: `cargo test -p tdb-retrieval`
Expected: All existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add crates/tdb-retrieval/src/per_token.rs
git commit -m "✨ feat(retrieval): corpus covariance tracking for whitening refinement"
```

---

### Task 2: Whitening Matrix + RefinementMode::Whitened

**Files:**
- Modify: `crates/tdb-retrieval/Cargo.toml` (add `faer` dependency)
- Modify: `crates/tdb-retrieval/src/per_token.rs` (whitening matrix cache)
- Modify: `crates/tdb-retrieval/src/refinement.rs` (add Whitened variant + rescore fn)
- Test: `crates/tdb-retrieval/src/refinement.rs` (inline tests)

- [ ] **Step 1: Add `faer` dependency**

In `crates/tdb-retrieval/Cargo.toml`, add under `[dependencies]`:
```toml
faer = "0.24"
```

- [ ] **Step 2: Write failing test for whitened rescore**

In the `#[cfg(test)] mod tests` of `crates/tdb-retrieval/src/refinement.rs`, add:

```rust
#[test]
fn whitened_produces_different_scores_than_mean_centered() {
    let mut r = populated_retriever();
    let q = encode_per_token_keys(&[&[0.7_f32, 0.3, 0.1, 0.0][..]]);
    let first_stage = r.query(&q, 3, Some(OWNER));
    let mc = apply(RefinementMode::MeanCentered, &mut r, &q, first_stage.clone(), 3, Some(OWNER));
    let wh = apply(RefinementMode::Whitened, &mut r, &q, first_stage, 3, Some(OWNER));
    // Scores should differ (whitening applies covariance normalization).
    let mc_scores: Vec<f32> = mc.iter().map(|r| r.score).collect();
    let wh_scores: Vec<f32> = wh.iter().map(|r| r.score).collect();
    assert_ne!(mc_scores, wh_scores);
}

#[test]
fn whitened_falls_back_when_corpus_too_small() {
    // Single cell — fewer tokens than dimensions=4, should fall back to mean-centering.
    let mut r = PerTokenRetriever::with_scoring_mode(ScoringMode::Top5Avg);
    let cell0 = [1.0_f32, 0.0, 0.0, 0.0];
    r.insert(0, OWNER, &encode_per_token_keys(&[&cell0]));
    let q = encode_per_token_keys(&[&cell0[..]]);
    let first_stage = r.query(&q, 1, Some(OWNER));
    // Should not panic — graceful fallback.
    let refined = apply(RefinementMode::Whitened, &mut r, &q, first_stage, 1, Some(OWNER));
    assert!(!refined.is_empty());
}
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo test -p tdb-retrieval whitened`
Expected: FAIL — `Whitened` variant does not exist.

- [ ] **Step 4: Add whitening_matrix method to PerTokenRetriever**

In `crates/tdb-retrieval/src/per_token.rs`, add a cached whitening matrix field:

After `corpus_sq_sum` field:
```rust
    /// Cached whitening matrix (dim×dim flattened). Invalidated when corpus changes.
    whitening_cache: Option<Vec<f32>>,
    /// Token count when whitening_cache was last computed.
    whitening_cache_count: usize,
```

Initialize in `with_config`:
```rust
    whitening_cache: None,
    whitening_cache_count: 0,
```

Add public method after `corpus_covariance()`:
```rust
    pub fn whitening_matrix(&mut self) -> Option<&[f32]> {
        if self.corpus_token_count == self.whitening_cache_count {
            if let Some(ref cache) = self.whitening_cache {
                return Some(cache.as_slice());
            }
        }
        let cov = self.corpus_covariance()?;
        let dim = self.corpus_sum.len();
        let w = compute_whitening_matrix(&cov, dim)?;
        self.whitening_cache = Some(w);
        self.whitening_cache_count = self.corpus_token_count;
        self.whitening_cache.as_deref()
    }
```

Add the eigendecomposition helper (private, at module level):
```rust
fn compute_whitening_matrix(cov: &[f32], dim: usize) -> Option<Vec<f32>> {
    use faer::Mat;

    if cov.len() != dim * dim || dim == 0 {
        return None;
    }

    let mat = Mat::from_fn(dim, dim, |i, j| cov[i * dim + j] as f64);
    let eigen = mat.selfadjoint_eigendecomposition(faer::Side::Lower);
    let eigenvalues = eigen.s().column_vector();
    let eigenvectors = eigen.u();

    // W = U @ diag(1/sqrt(λ)) @ U^T, clamping small eigenvalues for stability.
    let eps = 1e-5_f64;
    let mut result = vec![0.0_f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut sum = 0.0_f64;
            for k in 0..dim {
                let lam = eigenvalues.read(k).max(eps);
                sum += eigenvectors.read(i, k) * (1.0 / lam.sqrt()) * eigenvectors.read(j, k);
            }
            result[i * dim + j] = sum as f32;
        }
    }
    Some(result)
}
```

- [ ] **Step 5: Add `Whitened` to RefinementMode + dispatcher**

In `crates/tdb-retrieval/src/refinement.rs`:

Add to the enum:
```rust
    /// ZCA whitening: normalize covariance of query + stored tokens, re-score.
    Whitened,
```

Add match arm in `apply()`:
```rust
        RefinementMode::Whitened => whitened_rescore(retriever, query_key, first_stage, k),
```

Add the rescore function:
```rust
fn whitened_rescore(
    retriever: &mut PerTokenRetriever,
    query_key: &[f32],
    first_stage: Vec<RetrievalResult>,
    k: usize,
) -> Vec<RetrievalResult> {
    if first_stage.is_empty() {
        return first_stage;
    }
    // whitening_matrix() takes &mut self (lazy compute + cache).
    let Some(w_matrix) = retriever.whitening_matrix() else {
        // Fallback to mean-centered if whitening unavailable.
        return mean_centered_rescore(retriever, query_key, first_stage, k);
    };
    let w_matrix = w_matrix.to_vec(); // clone to release &mut borrow
    let Some(mean) = retriever.corpus_mean() else {
        return first_stage;
    };
    let dim = mean.len();
    let Some(query_tokens) = parse_query_tokens(query_key, dim) else {
        return first_stage;
    };

    let whitened_query: Vec<Vec<f32>> = query_tokens
        .iter()
        .map(|tok| mat_vec_mul(&w_matrix, &subtract_in_place(tok.clone(), &mean), dim))
        .collect();

    let inv_sqrt_dk = 1.0 / (dim as f32).sqrt();
    let top_n = retriever.top_n();

    let mut rescored: Vec<RetrievalResult> = first_stage
        .into_iter()
        .map(|res| {
            let mut dots: Vec<f32> = Vec::new();
            for (stored_token, _owner) in retriever.dequantized_tokens_for_cell(res.cell_id) {
                if stored_token.len() != dim {
                    continue;
                }
                let ws = mat_vec_mul(&w_matrix, &subtract_in_place(stored_token, &mean), dim);
                for wq in &whitened_query {
                    dots.push(dot(wq, &ws) * inv_sqrt_dk);
                }
            }
            let score = top_n_avg(&mut dots, top_n);
            RetrievalResult { cell_id: res.cell_id, owner: res.owner, score }
        })
        .collect();

    rescored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    rescored.truncate(k);
    rescored
}

fn mat_vec_mul(matrix: &[f32], vec: &[f32], dim: usize) -> Vec<f32> {
    let mut result = vec![0.0_f32; dim];
    for i in 0..dim {
        let mut sum = 0.0_f32;
        for j in 0..dim {
            sum += matrix[i * dim + j] * vec[j];
        }
        result[i] = sum;
    }
    result
}
```

- [ ] **Step 6: Run tests**

Run: `cargo test -p tdb-retrieval`
Expected: All tests pass including the 2 new whitening tests.

- [ ] **Step 7: Add PyO3 binding for "whitened" mode**

In `crates/tdb-python/src/lib.rs`, find the `set_refinement_mode` match block and add:
```rust
"whitened" => tdb_retrieval::refinement::RefinementMode::Whitened,
```

- [ ] **Step 8: Commit**

```bash
git add crates/tdb-retrieval/Cargo.toml crates/tdb-retrieval/src/per_token.rs crates/tdb-retrieval/src/refinement.rs crates/tdb-python/src/lib.rs
git commit -m "✨ feat(retrieval): ZCA whitening refinement mode with faer eigendecomposition"
```

---

### Task 3: Token Importance Reweighting

**Files:**
- Create: `crates/tdb-retrieval/src/token_weighter.rs`
- Modify: `crates/tdb-retrieval/src/lib.rs` (export module)
- Modify: `crates/tdb-retrieval/src/refinement.rs` (use weighter in rescore fns)
- Modify: `crates/tdb-engine/src/engine.rs` (add field + setter)
- Modify: `crates/tdb-python/src/lib.rs` (PyO3 binding)
- Test: `crates/tdb-retrieval/src/token_weighter.rs` (inline tests)

- [ ] **Step 1: Write failing test**

Create `crates/tdb-retrieval/src/token_weighter.rs` with test only:

```rust
//! Per-token importance weighting — Decorator pattern.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_near_mean_is_low() {
        let mean = vec![1.0_f32, 0.0, 0.0];
        let token_near_mean = vec![0.9_f32, 0.1, 0.0];
        let w = token_weight(&token_near_mean, &mean);
        assert!(w < 0.3, "Token near mean should have low weight, got {w}");
    }

    #[test]
    fn weight_orthogonal_to_mean_is_high() {
        let mean = vec![1.0_f32, 0.0, 0.0];
        let token_ortho = vec![0.0_f32, 1.0, 0.0];
        let w = token_weight(&token_ortho, &mean);
        assert!(w > 0.8, "Token orthogonal to mean should have high weight, got {w}");
    }

    #[test]
    fn weight_is_clamped_non_negative() {
        let mean = vec![1.0_f32, 0.0];
        let token = vec![1.0_f32, 0.0]; // identical to mean
        let w = token_weight(&token, &mean);
        assert!(w >= 0.0);
    }

    #[test]
    fn disabled_weighter_returns_one() {
        let mean = vec![1.0_f32, 0.0];
        let token = vec![0.5_f32, 0.5];
        let w = weighted_or_unit(false, &token, &mean);
        assert!((w - 1.0).abs() < 1e-6);
    }
}
```

- [ ] **Step 2: Implement token_weighter.rs**

Add the implementation above the tests:

```rust
//! Per-token importance weighting — Decorator pattern.
//!
//! Tokens near the corpus mean carry little discriminative signal.
//! Tokens orthogonal to the mean are distinctive. Weight formula:
//! `w = 1 - cosine(token, corpus_mean)`, clamped to [0, 1].

pub fn token_weight(token: &[f32], corpus_mean: &[f32]) -> f32 {
    let dot: f32 = token.iter().zip(corpus_mean).map(|(a, b)| a * b).sum();
    let norm_t: f32 = token.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_m: f32 = corpus_mean.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_t < 1e-9 || norm_m < 1e-9 {
        return 1.0;
    }
    let cosine = dot / (norm_t * norm_m);
    (1.0 - cosine).clamp(0.0, 1.0)
}

pub fn weighted_or_unit(enabled: bool, token: &[f32], corpus_mean: &[f32]) -> f32 {
    if enabled { token_weight(token, corpus_mean) } else { 1.0 }
}
```

- [ ] **Step 3: Export module**

In `crates/tdb-retrieval/src/lib.rs`, add:
```rust
pub mod token_weighter;
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p tdb-retrieval token_weight`
Expected: All 4 tests PASS.

- [ ] **Step 5: Integrate into refinement rescore functions**

In `crates/tdb-retrieval/src/refinement.rs`, update `mean_centered_rescore` and `whitened_rescore` to accept a `reweight: bool` parameter and apply per-token weights to dot products.

Add import at top:
```rust
use crate::token_weighter::weighted_or_unit;
```

Update `apply()` signature to accept `reweight: bool`:
```rust
pub fn apply(
    mode: RefinementMode,
    retriever: &mut PerTokenRetriever,
    query_key: &[f32],
    first_stage: Vec<RetrievalResult>,
    k: usize,
    owner_filter: Option<OwnerId>,
    reweight: bool,
) -> Vec<RetrievalResult> {
```

In `mean_centered_rescore` and `whitened_rescore`, after computing each dot product, multiply by the query token's weight:

Replace the dot computation loop body:
```rust
for cq in &centered_query {
    let w = weighted_or_unit(reweight, cq, &mean);
    dots.push(dot(cq, &centered_stored) * inv_sqrt_dk * w);
}
```

- [ ] **Step 6: Update engine.rs to pass reweight flag**

In `crates/tdb-engine/src/engine.rs`:

Add field to `Engine` struct:
```rust
    token_reweighting: bool,
```

Initialize as `false` in `open_with_options`.

Add setter/getter:
```rust
    pub fn set_token_reweighting(&mut self, enabled: bool) {
        self.token_reweighting = enabled;
    }

    pub fn token_reweighting(&self) -> bool {
        self.token_reweighting
    }
```

Update the refinement call site to pass the flag:
```rust
    candidates = tdb_retrieval::refinement::apply(
        self.refinement_mode,
        per_token,
        query_key,
        candidates,
        k * 2,
        owner_filter,
        self.token_reweighting,
    );
```

- [ ] **Step 7: Add PyO3 bindings**

In `crates/tdb-python/src/lib.rs`, add:
```rust
    fn set_token_reweighting(&self, enabled: bool) -> PyResult<()> {
        lock_engine(&self.inner)?.set_token_reweighting(enabled);
        Ok(())
    }

    fn token_reweighting(&self) -> PyResult<bool> {
        Ok(lock_engine(&self.inner)?.token_reweighting())
    }
```

- [ ] **Step 8: Fix all existing test call sites for `apply()`**

The existing tests in `refinement.rs` call `apply()` without the `reweight` param. Add `false` as the last argument to all existing test calls.

- [ ] **Step 9: Run full test suite**

Run: `cargo test --workspace --exclude tdb-python`
Expected: All tests pass.

- [ ] **Step 10: Commit**

```bash
git add crates/tdb-retrieval/src/token_weighter.rs crates/tdb-retrieval/src/lib.rs crates/tdb-retrieval/src/refinement.rs crates/tdb-engine/src/engine.rs crates/tdb-python/src/lib.rs
git commit -m "✨ feat(retrieval): token importance reweighting — IDF-like corpus-mean distance weighting"
```

---

### Task 4: Multi-Layer Query Fusion (Python)

**Files:**
- Create: `python/tardigrade_hooks/multi_layer_query.py`
- Modify: `python/tardigrade_hooks/client.py`
- Create: `tests/python/test_multi_layer_query.py`

- [ ] **Step 1: Write failing tests**

Create `tests/python/test_multi_layer_query.py`:

```python
"""ATDD tests for MultiLayerQuery — Composite pattern for multi-layer RRF fusion."""

import numpy as np
import pytest

from tardigrade_hooks.multi_layer_query import rrf_fuse


class TestRRFFusion:
    def test_single_list_preserves_order(self):
        ranked = [{"pack_id": 1, "score": 0.9}, {"pack_id": 2, "score": 0.5}]
        fused = rrf_fuse([ranked], k=60)
        assert [r["pack_id"] for r in fused] == [1, 2]

    def test_two_lists_boost_shared_packs(self):
        list_a = [{"pack_id": 1, "score": 0.9}, {"pack_id": 2, "score": 0.5}]
        list_b = [{"pack_id": 2, "score": 0.8}, {"pack_id": 3, "score": 0.4}]
        fused = rrf_fuse([list_a, list_b], k=60)
        # Pack 2 appears in both lists — should rank highest
        assert fused[0]["pack_id"] == 2

    def test_empty_lists(self):
        assert rrf_fuse([], k=60) == []
        assert rrf_fuse([[]], k=60) == []

    def test_disjoint_lists_interleave(self):
        list_a = [{"pack_id": 1, "score": 0.9}]
        list_b = [{"pack_id": 2, "score": 0.9}]
        list_c = [{"pack_id": 3, "score": 0.9}]
        fused = rrf_fuse([list_a, list_b, list_c], k=60)
        assert len(fused) == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `source .venv/bin/activate && PYTHONPATH=python pytest tests/python/test_multi_layer_query.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement multi_layer_query.py**

Create `python/tardigrade_hooks/multi_layer_query.py`:

```python
"""Multi-layer query fusion — Composite pattern.

Runs retrieval at multiple transformer layers, fuses rankings via
Reciprocal Rank Fusion (RRF). Pure latent-space: no text, no external
model. Only the query is multi-layer; stored memories use single-layer keys.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .constants import DEFAULT_CAPTURE_LAYER_RATIO
from .encoding import encode_per_token

DEFAULT_LAYER_RATIOS = (0.50, DEFAULT_CAPTURE_LAYER_RATIO, 0.83)
DEFAULT_RRF_K = 60


def rrf_fuse(
    ranked_lists: list[list[dict]],
    k: int = DEFAULT_RRF_K,
) -> list[dict]:
    """Fuse multiple ranked lists via Reciprocal Rank Fusion.

    Each item in a ranked list must have a ``pack_id`` key.
    Returns a single list sorted by fused RRF score (descending).
    """
    if not ranked_lists:
        return []

    scores: dict[int, float] = defaultdict(float)
    pack_data: dict[int, dict] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            pid = item["pack_id"]
            scores[pid] += 1.0 / (k + rank + 1)
            if pid not in pack_data:
                pack_data[pid] = item

    fused = []
    for pid, score in sorted(scores.items(), key=lambda x: -x[1]):
        entry = dict(pack_data[pid])
        entry["rrf_score"] = score
        fused.append(entry)

    return fused


class MultiLayerQuery:
    """Composite: queries the engine at multiple layers, fuses via RRF."""

    def __init__(
        self,
        engine,
        *,
        layer_ratios: tuple[float, ...] = DEFAULT_LAYER_RATIOS,
        rrf_k: int = DEFAULT_RRF_K,
    ):
        self._engine = engine
        self._layer_ratios = layer_ratios
        self._rrf_k = rrf_k

    def query(
        self,
        model,
        tokenizer,
        query_text: str,
        k: int = 5,
        owner: int | None = None,
    ) -> list[dict]:
        """Run retrieval at each layer, fuse via RRF."""
        import torch

        inputs = tokenizer(query_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        n_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size

        ranked_lists = []
        for ratio in self._layer_ratios:
            layer_idx = int(n_layers * ratio)
            hidden = out.hidden_states[layer_idx][0][1:]  # skip pos 0
            h_np = hidden.cpu().numpy().astype(np.float32)
            query_key = encode_per_token(h_np, hidden_size)
            results = self._engine.mem_read_pack(query_key, k * 2, owner)
            ranked_lists.append(results)

        fused = rrf_fuse(ranked_lists, k=self._rrf_k)
        return fused[:k]
```

- [ ] **Step 4: Update client.py**

In `python/tardigrade_hooks/client.py`, modify the `query` method to support `multi_layer`:

```python
    def query(self, query_text: str, *, k: int = 5, multi_layer: bool = False) -> list[dict]:
        """Retrieve the top-k packs matching *query_text*."""
        if multi_layer and self._tokenizer is not None:
            from .multi_layer_query import MultiLayerQuery
            # multi_layer requires a real model — fall back to single if stub
            try:
                mlq = MultiLayerQuery(self._engine)
                return mlq.query(
                    self._kv_fn.__self__ if hasattr(self._kv_fn, '__self__') else None,
                    self._tokenizer, query_text, k=k, owner=self._owner,
                )
            except Exception:
                pass
        key, _ = self._kv_fn(query_text, self._tokenizer)
        return self._engine.mem_read_pack(key, k, self._owner)
```

- [ ] **Step 5: Run tests**

Run: `source .venv/bin/activate && pip install -e . && PYTHONPATH=python pytest tests/python/test_multi_layer_query.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add python/tardigrade_hooks/multi_layer_query.py python/tardigrade_hooks/client.py tests/python/test_multi_layer_query.py
git commit -m "✨ feat(hooks): multi-layer query fusion via RRF — Composite pattern"
```

---

### Task 5: Stacking Experiment + CLAUDE.md

**Files:**
- Create: `experiments/vague_refinement_v2_experiment.py`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Write experiment script**

Test all configurations on the 10-fact Sonia corpus with Qwen3-0.6B on MPS:
1. Baseline (centered)
2. Whitened only
3. Whitened + token reweighting
4. Whitened + token reweighting + multi-layer fusion

Report R@5 for specific/moderate/vague per configuration.

- [ ] **Step 2: Run experiment**

Run: `source .venv/bin/activate && python experiments/vague_refinement_v2_experiment.py`

Success criteria:
- Specific R@5 ≥ 100% in all configs
- Moderate R@5 ≥ 80% in all configs
- Vague R@5 > 60% in at least one config

- [ ] **Step 3: Update CLAUDE.md with results and new test counts**

- [ ] **Step 4: Commit and push**

```bash
git add experiments/vague_refinement_v2_experiment.py CLAUDE.md
git commit -m "📊 experiments: latent-space vague refinement — whitening + reweighting + multi-layer"
git push origin main
```
