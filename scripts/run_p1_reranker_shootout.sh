#!/usr/bin/env bash
# P1 reranker shootout — LLM-gated retrieve-then-read variant.
#
# Two full-corpus runs (1533 LoCoMo + 500 LongMemEval) against both
#   - cross-encoder/ms-marco-MiniLM-L-6-v2
#   - Qwen/Qwen3-Reranker-0.6B
# at TDB_RETRIEVER_TOP_K=25, TDB_REFINEMENT_MODE=whitened.
#
# Adapter: tardigrade-llm-gated (Phase 1B.5)
#   retrieval (KV+rerank) → DeepSeek answer generation → DeepSeek judge
#
# Cost: ~$0.57 per LoCoMo run + ~$0.18 per LongMemEval run + judge
#       ≈ $1.50 total for both reranker variants.
# Wall time: ~95 min per reranker (LoCoMo phase dominates).
#
# Decision rule: higher LoCoMo LLM-Judge aggregate wins.
# Tiebreak on LongMemEval LLM-Judge aggregate.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env.bench ]; then
  echo "missing .env.bench (DEEPSEEK_API_KEY for answerer + judge)" >&2
  exit 1
fi
if [ ! -d .venv ]; then
  echo "missing .venv" >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
set -a; source .env.bench; set +a

export TDB_RLS_MODE=none
export TDB_BENCH_DEVICE=cuda
export TDB_REFINEMENT_MODE=whitened
export TDB_RETRIEVER_TOP_K=25
export TDB_LLM_GATE_PROVIDER=deepseek
# Response cache survives restarts so a re-run of identical
# (model, prompt_template_version, prompt_hash) hits free.
export TDB_LLM_GATE_CACHE_DIR=target/llm-gate-cache
mkdir -p "${TDB_LLM_GATE_CACHE_DIR}"

export LOCOMO_DATA_PATH=benchmarks/datasets/phase1_oracle/locomo_phase1.jsonl
export LOCOMO_DATA_REV=phase1_oracle
export LONGMEMEVAL_DATA_PATH=benchmarks/datasets/phase1_oracle/longmemeval_phase1.jsonl
export LONGMEMEVAL_DATA_REV=phase1_oracle

mkdir -p target

run_one() {
  local label="$1"
  local model="$2"
  local out="target/bench-${label}-llmgated-k25-full.json"
  echo "=== [$(date -Iseconds)] P1-LLM-gated run: ${label} (${model}) ==="
  TDB_BENCH_RERANK_MODEL="${model}" \
    PYTHONPATH=python python -u -m tdb_bench run --mode full \
      --system tardigrade-llm-gated \
      --config python/tdb_bench/config/default.json \
      --output "${out}"
  echo "=== [$(date -Iseconds)] done: ${out} ==="
}

run_one minilm     "cross-encoder/ms-marco-MiniLM-L-6-v2"
run_one qwen3rr    "Qwen/Qwen3-Reranker-0.6B"

echo "=== [$(date -Iseconds)] P1 LLM-gated shootout complete ==="
