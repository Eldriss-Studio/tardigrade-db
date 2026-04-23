# Sonia Parallel Subagent Validation

**Date:** April 23, 2026  
**Status:** Complete  
**Execution mode:** Two parallel Codex subagents, each using a different model

## Hypothesis

Sonia memory experiments should execute reproducibly when delegated to independent subagents, and both runs should return coherent recall and SNR metrics without operational failures.

## Setup

Two subagents were spawned in parallel:

| Subagent | Agent model | Script |
|---|---|---|
| A | `gpt-5.4-mini` | `experiments/sonia_real_kv_cache.py` |
| B | `gpt-5.2` | `experiments/sonia_production_sim.py` |

Both runs used the same environment pattern:

```bash
cd /Users/storylight/Dev/tardigrade-db
PYTHONPATH=python PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 .venv/bin/python <script>
```

## Models Used and How

### 1) Agent models (orchestration layer)

These models only executed commands and reported outputs:

| Purpose | Model |
|---|---|
| Subagent A runner | `gpt-5.4-mini` |
| Subagent B runner | `gpt-5.2` |

### 2) Experiment model (memory-retrieval layer)

Both Sonia scripts load and evaluate the same HF model:

| Script | Model | Core representation used |
|---|---|---|
| `sonia_real_kv_cache.py` | `Qwen/Qwen3-0.6B` | **Real KV keys** from `past_key_values` |
| `sonia_production_sim.py` | `Qwen/Qwen3-0.6B` | Hidden states from one semantic layer (`query_layer = int(0.67 * n_layers)`) |

How vectors are built:
- `sonia_real_kv_cache.py`
  - Mean-pool mode: mean over sequence of projected K vectors at query layer.
  - Per-token mode: stores each token's projected K vector at query layer.
- `sonia_production_sim.py`
  - Mean-pool mode: mean over sequence hidden states at the chosen semantic layer.
  - Per-token mode: stores each token hidden-state vector at that layer.

In both scripts, retrieval query vectors are mean-pooled (same representation family as the stored vectors for that script), and `engine.mem_read(query_vec, k, owner=1)` is used for ranking.

## Results

### Run A — `sonia_real_kv_cache.py`

- Model under test: `Qwen/Qwen3-0.6B`
- KV shape reported by script: `28L, ql=18`
- Mean-pool (real KV): `10/16` (`62.5%`), SNR `-1.1`
- Per-token (real KV): `12/16` (`75.0%`), SNR `-3.1`
- Recall delta (per-token vs mean-pool): `+12.5%`
- Unique top-1 memories: mean-pool `5`, per-token `7`
- Process outcome: success (exit `0`)

Script-provided prior hidden-state baseline (for comparison):
- Hidden mean-pool: `31.2% | SNR -267.7`
- Hidden per-token: `31.2% | SNR +11096.9`

### Run B — `sonia_production_sim.py`

- Mean-pool: `5/16` (`31.2%`), SNR `-267.7`
- Per-token: `5/16` (`31.2%`), SNR `+11096.9`
- Recall delta: `+0.0%`
- Process outcome: success (exit `0`)

### Local rerun validation

Both scripts were re-run directly (outside subagents) before commit and reproduced the same headline metrics:
- `sonia_real_kv_cache.py`: `62.5%` (mean-pool) vs `75.0%` (per-token)
- `sonia_production_sim.py`: `31.2%` vs `31.2%`

## Interpretation

1. Parallel delegated execution is operationally stable for both Sonia scripts.
2. `sonia_real_kv_cache.py` shows higher recall for per-token real-KV storage than mean-pool in this run (`75.0%` vs `62.5%`).
3. `sonia_production_sim.py` showed no recall lift in this run, despite a very large SNR separation on per-token mode.
4. Across both runs, no blocking runtime errors occurred, so this path is suitable for repeated delegated validation.

## Reproduction

Run the same two scripts in parallel terminals:

```bash
cd /Users/storylight/Dev/tardigrade-db
PYTHONPATH=python PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 .venv/bin/python experiments/sonia_real_kv_cache.py
```

```bash
cd /Users/storylight/Dev/tardigrade-db
PYTHONPATH=python PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 .venv/bin/python experiments/sonia_production_sim.py
```
