# Two Experiments: Larger Capture Model + Agent-Driven RLS

## Experiment A: Larger Capture Model

### Problem
Qwen3-0.6B's hidden states don't bridge vocabulary gaps ("athletic" ≠ "ultramarathon" in 0.6B hidden space). Larger models encode richer semantic relationships in their hidden states.

### Design
Swap the capture model from Qwen3-0.6B to Qwen2.5-3B (float16, cached locally) or Qwen3-1.7B. The adapter already supports `TDB_BENCH_MODEL` env var. No code changes — just env var and benchmark run.

### ATDD Acceptance Criteria
1. `TDB_BENCH_MODEL=Qwen/Qwen2.5-3B` loads and ingests without crash
2. Score ratios differentiate (NOT all 1.000) with the larger model
3. LoCoMo score differs from 68.2% (any direction proves the model matters)

### Implementation
Zero code changes. Single benchmark run:
```bash
TDB_BENCH_MODEL=Qwen/Qwen2.5-3B TDB_BENCH_DEVICE=mps \
TDB_REFINEMENT_MODE=centered \
LOCOMO_DATA_PATH=benchmarks/datasets/phase1_oracle/locomo_phase1.jsonl \
LOCOMO_DATA_REV=phase1_oracle \
LONGMEMEVAL_DATA_PATH=benchmarks/datasets/phase1_oracle/longmemeval_phase1.jsonl \
LONGMEMEVAL_DATA_REV=phase1_oracle \
PYTHONPATH=python python -m tdb_bench run --mode full --system tardigrade \
  --config python/tdb_bench/config/default.json \
  --output target/bench-3b-capture.json
```

**Risk:** Qwen2.5-3B in float16 on MPS may be too slow for 2042 items × chunked ingestion (128 chunks each). Fallback: use float32 with `max_length=64` (fewer tokens per chunk) or test on a 200-item subset first.

**Risk:** `_load_model_cached()` hardcodes `attn_implementation="eager"`. Qwen2.5 may need different config. Check if it loads cleanly before full run.

---

## Experiment B: Agent-Driven RLS via MCP Protocol

### Problem
The 0.6B capture model can't reason about vocabulary bridges. A capable agent (Claude, GPT-4o) using TardigradeDB via MCP tools CAN reason: "I'm looking for athletic activities" → "let me search for running, marathon, sports" → multiple `mem_query` calls → fused answer.

### Design

#### Pattern: Protocol (not code)

RLS for MCP is a **prompt protocol**, not a code feature. The agent follows a sequence of MCP tool calls:

```
1. Agent calls mem_query("What athletic achievements does Sonia have?")
2. Agent inspects results — low relevance
3. Agent reasons: "athletic → running, marathon, endurance sports"
4. Agent calls mem_query("Sonia running marathon endurance")
5. Agent merges both result sets, picks best answer
```

This is what DCI-Agent does (grep/file reads), adapted to TardigradeDB's MCP tools. The agent IS the reformulation engine.

#### What TardigradeDB provides
- Existing MCP tools: `mem_query`, `mem_store`, `mem_status` etc.
- A documented **RLS protocol prompt** that tells the agent how to do multi-step retrieval
- A benchmark harness that simulates an agent following the protocol

#### ATDD Acceptance Criteria
1. RLS protocol prompt documented as a system message
2. Benchmark harness calls an LLM API (DeepSeek/Claude) with the protocol prompt + query
3. LLM produces reformulated queries
4. Harness executes them against the engine
5. LoCoMo score differs from 68.2%

#### Implementation

**New file:** `python/tdb_bench/adapters/agent_rls.py` — a new benchmark adapter that:
1. Takes a query from the benchmark runner
2. Sends it to an LLM API (DeepSeek) with the RLS protocol prompt
3. Parses the LLM's reformulation suggestions
4. Runs each suggestion through the engine
5. Fuses results, returns answer

**RLS Protocol Prompt:**
```
You are querying a memory system. If your first search doesn't find
a good answer, rephrase the query using different vocabulary and
search again. Return the final answer.

Original question: {question}

Search strategy:
1. Search with the original question
2. If results seem irrelevant, think about what words the stored
   memory might actually use
3. Search again with those words
4. Return the best answer from all searches
```

**Env var:** `TDB_RLS_AGENT_MODEL=deepseek-chat` (or `claude-sonnet-4-6`)

### SOLID
- **SRP:** Protocol prompt handles reasoning. Adapter handles execution. Engine handles retrieval.
- **OCP:** Different LLM backends via env var. Different protocol prompts via config.
- **Strategy:** `AgentRLSAdapter` is a new `BenchmarkAdapter` implementation.

### Files
- Create: `python/tdb_bench/adapters/agent_rls.py`
- Modify: `python/tdb_bench/registry.py` (register new adapter)
- Create: `tests/python/test_agent_rls.py`
- Create: `python/tdb_bench/prompts/rls_protocol.txt`

---

## Execution Order

1. **Experiment A first** (zero code, just a benchmark run). If the 3B capture model improves LoCoMo, we know the model size is the bottleneck.
2. **Experiment B second** (new adapter + protocol). This is the bigger bet — proving that agent reasoning bridges what latent-space geometry cannot.
3. Document all results. Update CLAUDE.md.
