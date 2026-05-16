#!/usr/bin/env bash
# Pre-flight smoke for the challenger headline run (slice B5).
#
# Runs the *challenger* profile against a 30-item LoCoMo + 30-item
# LongMemEval subset under the same stack the full run uses. Verifies
# the wiring end-to-end before committing to the ~4 hr / ~$3-5 full
# headline run.
#
# Wall time: ~5-8 min. Cost: ~$0.10.

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

export TDB_BENCH_MODEL="${TDB_BENCH_MODEL:-Qwen/Qwen3-1.7B}"
export TDB_BENCH_DEVICE="${TDB_BENCH_DEVICE:-cuda}"
export TDB_REFINEMENT_MODE="${TDB_REFINEMENT_MODE:-whitened}"
export TDB_RETRIEVER_TOP_K="${TDB_RETRIEVER_TOP_K:-25}"
export TDB_LLM_GATE_PROVIDER="${TDB_LLM_GATE_PROVIDER:-deepseek}"
export TDB_LLM_GATE_CACHE_DIR="${TDB_LLM_GATE_CACHE_DIR:-target/llm-gate-cache}"
mkdir -p "${TDB_LLM_GATE_CACHE_DIR}" target

# Derive a 30-item subset profile from the challenger.
cat > target/bench-challenger-smoke30-config.json <<'JSON'
{
  "version": 1,
  "profiles": {
    "challenger": {
      "_comment": "Smoke variant of the challenger profile — 30 items per dataset for end-to-end wiring verification.",
      "seed": 42,
      "timeout_seconds": 30,
      "datasets": [
        {
          "name": "locomo",
          "revision": "phase1_oracle_full",
          "path": "benchmarks/datasets/phase1_oracle_full/locomo_phase1.jsonl",
          "max_items": 30
        },
        {
          "name": "longmemeval",
          "revision": "phase1_oracle_full",
          "path": "benchmarks/datasets/phase1_oracle_full/longmemeval_phase1.jsonl",
          "max_items": 30
        }
      ],
      "systems": ["tardigrade-llm-gated"],
      "evaluator": {
        "mode": "justify_then_judge",
        "answerer_model": "deepseek-chat",
        "judge_model": "deepseek-chat"
      },
      "top_k": 5,
      "prompts": {
        "answer": "Answer concisely using retrieved evidence; use evidence dates to resolve relative temporal references.",
        "judge": "Score factual correctness; the reasoning trace is an aid, not the score target."
      }
    }
  }
}
JSON

OUT="${1:-target/bench-challenger-smoke30.json}"

PYTHONPATH=python python -u -m tdb_bench run --mode challenger \
  --config target/bench-challenger-smoke30-config.json \
  --output "${OUT}" \
  --workers "${TDB_BENCH_WORKERS:-8}"

python3 - <<PY
import json
d = json.load(open("${OUT}"))
print("=== smoke30 summary ===")
print("aggregates:", json.dumps(d.get("aggregates", {}), indent=2, sort_keys=True))
print("status:", d.get("status_summary", {}))
locomo = [it for it in d["items"] if it.get("dataset") == "locomo"]
lme    = [it for it in d["items"] if it.get("dataset") == "longmemeval"]
def avg(items):
    return sum(it.get("score", 0.0) for it in items) / max(1, len(items))
def idk_rate(items):
    return sum(
        1 for it in items
        if it.get("answer", "").strip().lower().startswith("i don")
    ) / max(1, len(items))
print(f"LoCoMo:      {len(locomo)} items, avg={avg(locomo):.4f}, IDK={idk_rate(locomo)*100:.1f}%")
print(f"LongMemEval: {len(lme)} items, avg={avg(lme):.4f}, IDK={idk_rate(lme)*100:.1f}%")
PY
