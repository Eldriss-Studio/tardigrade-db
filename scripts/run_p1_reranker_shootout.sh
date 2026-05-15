#!/usr/bin/env bash
# P1 reranker shootout: full-corpus LoCoMo + LongMemEval against both
# cross-encoder/ms-marco-MiniLM-L-6-v2 and Qwen/Qwen3-Reranker-0.6B
# at TDB_RETRIEVER_TOP_K=25, TDB_REFINEMENT_MODE=whitened.
#
# Decision: higher LoCoMo aggregate wins; tiebreak on LongMemEval.
# Documented in docs/experiments/2026-05-14-bench-audit.md Phase 1B.5.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env.bench ]; then
  echo "missing .env.bench (DEEPSEEK_API_KEY for judge step)" >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "missing .venv (run: python3 -m venv .venv && source .venv/bin/activate && pip install ...)" >&2
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
export LOCOMO_DATA_PATH=benchmarks/datasets/phase1_oracle/locomo_phase1.jsonl
export LOCOMO_DATA_REV=phase1_oracle
export LONGMEMEVAL_DATA_PATH=benchmarks/datasets/phase1_oracle/longmemeval_phase1.jsonl
export LONGMEMEVAL_DATA_REV=phase1_oracle

mkdir -p target

run_one() {
  local label="$1"
  local model="$2"
  local out="target/bench-${label}-k25-full.json"
  echo "=== [$(date -Iseconds)] P1 run: ${label} (${model}) ==="
  TDB_BENCH_RERANK_MODEL="${model}" \
    PYTHONPATH=python python -u -m tdb_bench run --mode full \
      --system tardigrade \
      --config python/tdb_bench/config/default.json \
      --output "${out}"
  echo "=== [$(date -Iseconds)] done: ${out} ==="
}

run_one minilm     "cross-encoder/ms-marco-MiniLM-L-6-v2"
run_one qwen3rr    "Qwen/Qwen3-Reranker-0.6B"

echo "=== [$(date -Iseconds)] P1 shootout complete ==="
