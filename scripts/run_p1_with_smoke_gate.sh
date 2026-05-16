#!/usr/bin/env bash
# Two-stage gate:
#   1) 50-item LLM-gated smoke (~$0.05, ~5 min). Sanity-check the path.
#   2) Only if smoke passes: full 1533+500 shootout (~$1.50, ~3 hr).
#
# "Green" gate:
#   - smoke exits 0
#   - status_summary has zero failed/skipped items
#   - LoCoMo score > 0.20 (50-item floor; below this implies a wiring
#     bug — the broken adapter already scores ~0.30 on the same data)

set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/smoke_llm_gated_50.sh

python3 - <<'PY' || exit 1
import json
d = json.load(open("target/bench-llmgated-smoke50.json"))
status = d["status_summary"]
agg = d["aggregates"]

failed = status.get("failed", 0)
skipped = status.get("skipped", 0)
ok = status.get("ok", 0)
print(f"smoke status: ok={ok} failed={failed} skipped={skipped}")
print(f"smoke aggregates: {json.dumps(agg, indent=2, sort_keys=True)}")

if failed > 2 or skipped > 2:
    print(f"GATE FAIL: too many non-ok items (failed={failed}, skipped={skipped})")
    raise SystemExit(2)

# Find the LoCoMo per-system score under either flat or nested layout.
locomo_score = None
for k, v in agg.items():
    if "locomo" in k.lower() and "score" in str(k).lower() and isinstance(v, (int, float)):
        locomo_score = float(v)
        break
if locomo_score is None and "by_dataset" in agg:
    locomo_score = agg["by_dataset"].get("locomo", {}).get("score")
if locomo_score is None and "by_system_dataset" in agg:
    for k, v in agg["by_system_dataset"].items():
        if "locomo" in k.lower():
            locomo_score = v.get("score") if isinstance(v, dict) else None
            break

print(f"smoke locomo score: {locomo_score}")
if locomo_score is None:
    print("GATE FAIL: couldn't locate LoCoMo aggregate score")
    raise SystemExit(3)
if locomo_score < 0.20:
    print(f"GATE FAIL: LoCoMo score {locomo_score:.3f} < 0.20 floor — wiring suspect")
    raise SystemExit(4)
print(f"GATE PASS: smoke clean, LoCoMo at {locomo_score:.3f} on 50 items")
PY

echo ""
echo "=== [$(date -Iseconds)] smoke gate passed — kicking full shootout ==="
bash scripts/run_p1_reranker_shootout.sh
