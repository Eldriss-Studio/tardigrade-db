# Agent RLS Benchmark — CUDA Runbook

Run the LLM agent reformulation experiment on a CUDA GPU (RTX 3070 Ti or similar).

## What This Tests

DeepSeek-chat generates vocabulary-bridged query reformulations (e.g., "athletic achievements" → "ultramarathon", "marathon training"). The experiment measures whether this breaks the 68.2% LoCoMo ceiling.

Two modes available — run both for comparison:

| Mode | What it tests | Expected time |
|---|---|---|
| `in_memory` (lexical) | Reformulation + word-overlap matching | ~70 min (API-bound) |
| `native` (KV engine) | Reformulation + latent-space retrieval | ~2-3 hrs (GPU-bound) |

## Prerequisites

```bash
cd ~/Dev/tardigrade-db
git pull origin main
source .venv/bin/activate
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop
pip install torch transformers  # CUDA version
```

Verify CUDA:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## DeepSeek API Key

```bash
export DEEPSEEK_API_KEY=$(grep DEEPSEEK_API_KEY .env.bench | cut -d= -f2)
```

## Test 1: In-Memory Mode (lexical, same as Mac baseline)

This replicates the 68.2% baseline environment but with agent reformulation.

```bash
TDB_RLS_MODE=agent TDB_BENCH_FORCE_FALLBACK=1 \
LOCOMO_DATA_PATH=benchmarks/datasets/phase1_oracle/locomo_phase1.jsonl \
LOCOMO_DATA_REV=phase1_oracle \
LONGMEMEVAL_DATA_PATH=benchmarks/datasets/phase1_oracle/longmemeval_phase1.jsonl \
LONGMEMEVAL_DATA_REV=phase1_oracle \
PYTHONPATH=python python -m tdb_bench run --mode full --system tardigrade \
  --config python/tdb_bench/config/default.json \
  --output target/bench-agent-rls-inmemory.json
```

**Baseline comparison:** LoCoMo 68.2%, LongMemEval 88.8% (without reformulation).

## Test 2: Native Mode (latent-space KV retrieval)

This uses the real Rust engine with Qwen3-0.6B hidden states. NOT possible on MPS (51+ hours).

```bash
TDB_RLS_MODE=agent TDB_BENCH_DEVICE=cuda \
TDB_REFINEMENT_MODE=centered \
LOCOMO_DATA_PATH=benchmarks/datasets/phase1_oracle/locomo_phase1.jsonl \
LOCOMO_DATA_REV=phase1_oracle \
LONGMEMEVAL_DATA_PATH=benchmarks/datasets/phase1_oracle/longmemeval_phase1.jsonl \
LONGMEMEVAL_DATA_REV=phase1_oracle \
PYTHONPATH=python python -m tdb_bench run --mode full --system tardigrade \
  --config python/tdb_bench/config/default.json \
  --output target/bench-agent-rls-native.json
```

**Note:** Native mode uses `confidence_threshold=inf` for agent mode (always reformulates). This establishes the ceiling. Confidence gating calibration is a follow-up.

## Test 3: Native Baseline (no reformulation)

Run without agent RLS to get the native-mode baseline on this hardware:

```bash
TDB_RLS_MODE=none TDB_BENCH_DEVICE=cuda \
TDB_REFINEMENT_MODE=centered \
LOCOMO_DATA_PATH=benchmarks/datasets/phase1_oracle/locomo_phase1.jsonl \
LOCOMO_DATA_REV=phase1_oracle \
LONGMEMEVAL_DATA_PATH=benchmarks/datasets/phase1_oracle/longmemeval_phase1.jsonl \
LONGMEMEVAL_DATA_REV=phase1_oracle \
PYTHONPATH=python python -m tdb_bench run --mode full --system tardigrade \
  --config python/tdb_bench/config/default.json \
  --output target/bench-native-baseline.json
```

## Quick Validation (run first!)

Before the full run, validate DeepSeek is actually being called:

```bash
TDB_RLS_MODE=agent TDB_BENCH_FORCE_FALLBACK=1 \
PYTHONPATH=python python -m tdb_bench run --mode smoke --system tardigrade \
  --config python/tdb_bench/config/default.json \
  --output target/bench-smoke-validate.json
```

Expected: 6 items, latencies ~1.5-2s each (DeepSeek API round-trip), score=1.0.
If latencies are <100ms, DeepSeek is NOT being called — check API key.

## Reading Results

```bash
python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
s = d['aggregates']['systems']['tardigrade']
print(f'Score: {s[\"avg_score\"]:.4f} ({s[\"ok\"]} ok, {s[\"failed\"]} failed)')
meta = d['manifest']['adapter_meta']['tardigrade']
print(f'Mode: {meta[\"mode\"]}, RLS: {meta[\"rls_mode\"]}')
locomo = [i for i in d['items'] if i['dataset'] == 'locomo']
lme = [i for i in d['items'] if i['dataset'] == 'longmemeval']
if locomo: print(f'LoCoMo: {sum(i[\"score\"] for i in locomo)/len(locomo):.4f} ({len(locomo)} items)')
if lme: print(f'LongMemEval: {sum(i[\"score\"] for i in lme)/len(lme):.4f} ({len(lme)} items)')
" target/bench-agent-rls-native.json
```

## Known Results (Mac, in_memory mode)

| Config | LoCoMo | LongMemEval |
|---|---|---|
| Baseline (no RLS) | 68.2% | 88.8% |
| Agent RLS, naive fusion | 52.9% (-15.3pp) | 77.8% (-11.0pp) |

Naive always-reformulate + max-score fusion degrades both. The follow-up experiment will implement margin-based acceptance (only use reformulated result if it scores ≥2x the original).

## Cost Estimate

- DeepSeek-chat: ~$0.07/1M input tokens
- ~2,042 items × ~150 tokens/call ≈ 306K tokens → ~$0.02
- LLM judge (also DeepSeek): ~2,042 calls → ~$0.04
- **Total: ~$0.06 per full run**
