#!/usr/bin/env bash
# Track B headline measurement — slice B5.
#
# Executes the full LoCoMo (1542 items) + LongMemEval (500 items) bench
# under the `challenger` profile:
#   * capture model:   Qwen/Qwen3-1.7B (env-overridable)
#   * dataset:         phase1_oracle_full (full-conversation context)
#   * adapter:         tardigrade-llm-gated (retrieve → answer)
#   * evaluator:       justify_then_judge (DeepSeek for both stages)
#
# Cost: ~$3-5 (two DeepSeek calls per item × 2042 items × ~512 + 60
# response tokens). Wall time: ~3-4 hours on RTX 3070 Ti.
# Writes the headline measurement to target/bench-challenger-headline.json.
#
# Output is *measured*, not asserted. We document whatever the bench
# returns honestly — that's the slice B5 contract.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env.bench ]; then
  echo "missing .env.bench (DEEPSEEK_API_KEY for justify + judge stages)" >&2
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

# Challenger profile expects these env values for the dataset paths
# but supplies sane defaults in default.json itself. We export the
# capture model + LLM-gating budget + reranker pick + cache dir here
# so the run is reproducible from the script alone.
export TDB_BENCH_MODEL="${TDB_BENCH_MODEL:-Qwen/Qwen3-1.7B}"
export TDB_BENCH_DEVICE="${TDB_BENCH_DEVICE:-cuda}"
export TDB_REFINEMENT_MODE="${TDB_REFINEMENT_MODE:-whitened}"
export TDB_RETRIEVER_TOP_K="${TDB_RETRIEVER_TOP_K:-25}"
export TDB_LLM_GATE_PROVIDER="${TDB_LLM_GATE_PROVIDER:-deepseek}"
export TDB_LLM_GATE_CACHE_DIR="${TDB_LLM_GATE_CACHE_DIR:-target/llm-gate-cache}"
mkdir -p "${TDB_LLM_GATE_CACHE_DIR}"

OUT="${1:-target/bench-challenger-headline.json}"
mkdir -p "$(dirname "${OUT}")"

echo "=== [$(date -Iseconds)] CHALLENGER HEADLINE RUN ==="
echo "  model:       ${TDB_BENCH_MODEL}"
echo "  device:      ${TDB_BENCH_DEVICE}"
echo "  refinement:  ${TDB_REFINEMENT_MODE}"
echo "  retr top_k:  ${TDB_RETRIEVER_TOP_K}"
echo "  gate prov:   ${TDB_LLM_GATE_PROVIDER}"
echo "  cache dir:   ${TDB_LLM_GATE_CACHE_DIR}"
echo "  output:      ${OUT}"
echo ""

PYTHONPATH=python python -u -m tdb_bench run --mode challenger \
  --config python/tdb_bench/config/default.json \
  --output "${OUT}"

echo ""
echo "=== [$(date -Iseconds)] CHALLENGER RUN COMPLETE ==="
echo ""

# Print honest summary
python3 - <<PY
import json
d = json.load(open("${OUT}"))
print("aggregates:")
print(json.dumps(d.get("aggregates", {}), indent=2, sort_keys=True))
print()
print("status:", d.get("status_summary", {}))
print()

# Per-dataset split if available.
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
