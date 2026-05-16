#!/usr/bin/env bash
# Fast 50-item LoCoMo + 50-item LongMemEval smoke against the real
# LLM-gated adapter on the GPU. Run this BEFORE kicking the multi-hour
# full-corpus shootout — it verifies:
#
#   1) the native KV path runs under the new adapter,
#   2) DeepSeek answer generation works at non-trivial scale,
#   3) cost-per-item is roughly $0.0005 (so $1.50 full-run estimate is right),
#   4) LLM-Judge scoring runs through cleanly.
#
# Wall time: ~3-5 min. Cost: ~$0.05.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env.bench ]; then
  echo "missing .env.bench" >&2
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
export TDB_BENCH_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
export TDB_LLM_GATE_PROVIDER=deepseek
export TDB_LLM_GATE_CACHE_DIR=target/llm-gate-cache
mkdir -p "${TDB_LLM_GATE_CACHE_DIR}"

# Use the 50-item subset configs (already in benchmarks/datasets/).
# Override max_items by writing a derived config.
cat > target/bench-llmgated-smoke50-config.json <<'JSON'
{
  "version": 1,
  "profiles": {
    "full": {
      "seed": 42,
      "timeout_seconds": 30,
      "datasets": [
        {
          "name": "locomo",
          "revision": "phase1_oracle_dated",
          "path": "benchmarks/datasets/phase1_oracle_dated/locomo_phase1.jsonl",
          "max_items": 50
        },
        {
          "name": "longmemeval",
          "revision": "phase1_oracle_dated",
          "path": "benchmarks/datasets/phase1_oracle_dated/longmemeval_phase1.jsonl",
          "max_items": 50
        }
      ],
      "systems": ["tardigrade-llm-gated"],
      "evaluator": {
        "mode": "llm_gated",
        "answerer_model": "deepseek-chat",
        "judge_model": "deepseek-chat"
      },
      "top_k": 5,
      "prompts": {
        "answer": "Answer concisely using retrieved evidence.",
        "judge": "Score factual correctness and evidence relevance only."
      }
    }
  }
}
JSON

PYTHONPATH=python python -u -m tdb_bench run --mode full \
  --system tardigrade-llm-gated \
  --config target/bench-llmgated-smoke50-config.json \
  --output target/bench-llmgated-smoke50.json

# Pretty summary
python3 - <<'PY'
import json
d = json.load(open("target/bench-llmgated-smoke50.json"))
print("aggregates:", json.dumps(d["aggregates"], indent=2, sort_keys=True))
print("status:", d["status_summary"])
PY
