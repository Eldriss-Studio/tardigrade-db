> **⚠️ RETRACTED — 2026-05-14 (motivation section only).** The "Why This Exists" section's empirical justification is invalidated by the 2026-05-14 bench audit:
>
> - **"LongMemEval: 88.8% — strong, beats the field"** and **"LoCoMo: 67.2% — below vanilla GPT-4o baseline (74%)"** were measured against the **lexical fallback adapter** on a corrupted LoCoMo dataset, not the native KV engine. They are not measurements of TardigradeDB. The "vocabulary-mismatched ceiling on LoCoMo" framing has no empirical support from those runs.
> - The DeepMind LIMIT paper (ICLR 2026) is a valid theoretical result, but the claim "we proved this is a mathematical property of vector-space retrieval, not a model quality issue" via TardigradeDB's numbers is retracted — we did not prove that with our data.
> - **More importantly:** on the clean LoCoMo dataset, all RLS strategies measured to date (keyword, agent/DeepSeek reformulation) **underperform** the no-RLS baseline (-5.3pp to -12.7pp at 50 items). The premise that RLS bridges a vocabulary gap is not supported; in current form RLS is net negative on clean data.
>
> **What survives in this doc:**
> - The **"Vague R@5: 60% on our 10-fact corpus, same 3 queries miss in every configuration"** claim — measured on the synthetic Sonia 10-fact corpus, not LoCoMo. Unaffected by the bench bug (but note: a 10-fact corpus is not a basis for a fundamental-ceiling claim).
> - The **RLS protocol design** (steps 1-5: RETRIEVE → EVALUATE → REFORMULATE → RE-RETRIEVE → FUSE), the **engine primitives** section, **protocol layers**, and the **research-references survey** — these are literature review and design, independent of our broken bench numbers.
> - All external research citations (DeepMind LIMIT, DCI, CRAG, Self-RAG, NeMo, Context-1, MMLF, Agentic RAG Survey) — the papers themselves are unaffected.
>
> **The honest current state:** RLS as currently implemented (naive fusion, threshold tuned against the broken baseline) hurts retrieval. The redesign hypotheses in the audit doc — fusion picks wrong items, reformulations dilute the latent signal, threshold mis-calibrated — are the open questions.
>
> Forensic record: [`../experiments/2026-05-14-bench-audit.md`](../experiments/2026-05-14-bench-audit.md).

# Reflective Latent Search (RLS): Research Foundation

## What Is RLS

Reflective Latent Search is TardigradeDB's answer to the vector-space retrieval
ceiling. Instead of making the similarity metric smarter (we proved that
whitening, reweighting, and multi-layer fusion all produce 0% improvement),
RLS lets the **agent reason about what to retrieve** and try multiple
approaches — while keeping all retrieval in latent space.

The name: **Reflective** (the agent evaluates its own results and decides
whether to try again), **Latent** (retrieval stays on KV tensors, no text
search), **Search** (iterative, not single-shot).

## Why This Exists

TardigradeDB's latent-space retrieval hits a theoretical ceiling on
vocabulary-mismatched queries:

- **LongMemEval: 88.8%** — strong, beats the field. These queries have
  vocabulary overlap with stored memories.
- **LoCoMo: 67.2%** — below vanilla GPT-4o baseline (74%). These queries
  are conversational, vague, vocabulary-mismatched.
- **Vague R@5: 60%** on our 10-fact corpus. Same 3 queries miss in every
  configuration we tested.

DeepMind's LIMIT paper (ICLR 2026) proved this is a **mathematical property
of vector-space retrieval**, not a model quality issue. The sign-rank of the
relevance matrix bounds what any fixed-dimension vector space can retrieve.

The DCI paper (May 2026) showed the solution: don't make retrieval smarter —
**increase the resolution of the interface** between the agent and the corpus.
DCI uses grep/file reads. RLS uses **the agent's own forward pass** on
reformulated queries — staying latent-native.

## How RLS Works

```
Agent query: "What does Sonia know about languages?"
    ↓
STEP 1: RETRIEVE
    engine.mem_read_pack(query_hidden_states, k=5)
    → results with scores
    ↓
STEP 2: EVALUATE
    Is top-1 score above confidence threshold?
    Does the returned text (pack_text) actually answer the query?
    ├─ YES → return results (fast path, ~88% of queries)
    └─ NO → continue to STEP 3
    ↓
STEP 3: REFORMULATE
    Agent reasons about why retrieval failed:
    "languages → translation, linguistic, foreign language work"
    Agent generates reformulated query text.
    Forward pass on reformulated text → new hidden states.
    ↓
STEP 4: RE-RETRIEVE
    engine.mem_read_pack(reformulated_hidden_states, k=5)
    → new results
    ↓
STEP 5: FUSE
    RRF merge of original + reformulated results.
    Return top-k from fused list.
    ↓
(Optional: repeat STEPS 3-5 up to max_attempts times)
```

## What Makes RLS Different from Existing Patterns

| Pattern | How it retrieves | What bridges vocabulary gap |
|---------|-----------------|---------------------------|
| **Corrective RAG** | Text embedding re-retrieval | LLM reformulates text query |
| **Self-RAG** | Reflection tokens in generation | Model critiques own output |
| **DCI-Agent** | grep/file reads on raw corpus | Agent uses terminal tools |
| **RLS** | **Latent-space dot products on KV tensors** | **Agent's forward pass on reformulated query generates new hidden states** |

RLS is the only pattern where the retrieval primitive stays tensor-native.
Each reformulation generates genuinely different hidden states (the model
processes different tokens), which means the retrieval key explores a
different region of latent space. The agent's reasoning IS the query
expansion — it happens implicitly through the forward pass, not through
explicit query rewriting.

## Engine Primitives Needed

RLS requires the engine to provide three things that it mostly already has:

### 1. Confidence Signal (mostly exists)

`mem_read_pack` already returns scores. The missing piece: a calibrated
confidence indicator that tells the agent "this result is likely wrong."

**Options (increasing complexity):**
- **Score ratio:** `top1_score / top2_score`. High ratio = confident
  (one result dominates). Low ratio = ambiguous (many results score
  similarly). Training-free, no calibration needed.
- **Score vs corpus statistics:** compare top-1 score against the
  corpus's historical score distribution. Scores below the 25th
  percentile trigger reformulation.
- **Binary flag:** `is_confident: bool` in the result, computed from
  the score ratio. Simplest for the agent to consume.

### 2. Multi-Query Fusion (partially exists)

`MultiLayerQuery.rrf_fuse()` exists in Python. The engine-level
equivalent would be:

```rust
pub fn mem_read_pack_fused(
    &mut self,
    query_keys: &[Vec<f32>],  // multiple query keys
    k: usize,
    owner_filter: Option<OwnerId>,
) -> Result<Vec<PackReadResult>>
```

This runs retrieval for each key, deduplicates by pack_id (keeping
best score per pack), and returns the fused top-k. Eliminates N
round-trips between Python and Rust.

### 3. Reformulation Hints (new)

When confidence is low, what should the agent reformulate toward?
The engine can provide hints:

- **Top-scoring pack's text** — "the best match I found was about X,
  try asking about X more specifically"
- **Corpus topics** — summary of what's stored (via governance tier
  distribution or text clustering)

These are convenience APIs, not core primitives.

## Protocol Layers

### Layer 1: Engine Primitives (Rust)

- `mem_read_pack` with confidence signal (score ratio or percentile)
- `mem_read_pack_fused` for multi-key retrieval + RRF
- Existing: refinement modes, pack text, view keys

### Layer 2: Reference Implementation (Python — `TardigradeClient`)

- `client.query()` gains `reflective=True` parameter
- Implements the RETRIEVE → EVALUATE → REFORMULATE → RE-RETRIEVE loop
- Uses the model for reformulation (forward pass on rephrased text)
- Configurable: `max_attempts`, `confidence_threshold`, `reformulation_strategy`

### Layer 3: MCP Protocol (for agent-native use)

- Document the RLS protocol as a sequence of MCP tool calls
- Agent calls `mem_query` → inspects results → calls `mem_query` again
  with different phrasing → merges results
- No new MCP tools needed — the agent drives the loop using existing tools
- Protocol documented as a prompt/instruction pattern

## Research References

| Source | Key insight |
|--------|-------------|
| [DeepMind LIMIT (ICLR 2026)](https://arxiv.org/abs/2508.21038) | Vector retrieval has a mathematical ceiling — RLS bypasses it via agent reasoning |
| [DCI-Agent (May 2026)](https://arxiv.org/abs/2605.05242) | Interface resolution matters more than similarity metric — RLS increases resolution via reformulation |
| [Agentic RAG Survey](https://arxiv.org/abs/2501.09136) | Iterative retrieval with confidence-based reformulation outperforms single-shot |
| [Corrective RAG](https://www.codebrains.co.in/blog/2025/ai/corrective-rag-self-healing-retrieval-layer-your-rag-system-desperately-needs) | RETRIEVE → EVALUATE → REFORMULATE loop pattern |
| [Self-RAG](https://arxiv.org/abs/2310.11511) | Reflection tokens for self-critique — 5.8% hallucination rate |
| [NVIDIA NeMo Agentic Retrieval](https://huggingface.co/blog/nvidia/nemo-retriever-agentic-retrieval) | Agentic loop adapts strategy dynamically; distillable to smaller agents |
| [Chroma Context-1](https://www.trychroma.com/research/context-1) | Self-editing search agent — reformulates queries based on results |
| [MMLF (NAACL 2025)](https://aclanthology.org/2025.findings-naacl.367.pdf) | Multi-query multi-passage late fusion at engine level |

## What RLS Preserves

- **Tensor-native premise:** All retrieval is latent-space dot products
  on the model's own hidden states. No text search, no BM25, no external
  embedding model.
- **Engine as kernel:** The engine stores and retrieves KV tensors. It
  doesn't reason, reformulate, or generate text. The agent does that.
- **Zero-training:** RLS is a protocol, not a trained model. Any LLM
  agent can follow it using existing MCP tools.
- **Backward compatible:** `reflective=False` (default) gives exactly
  the current single-shot behavior.

## What RLS Adds

- **Agent-driven vocabulary bridging:** The agent's reasoning ("languages
  → translation work") generates new hidden states that bridge the
  vocabulary gap — something no geometric transform can do.
- **Bounded iteration:** Max 3 attempts, configurable. The fast path
  handles 88% of queries without any reformulation.
- **Composable:** Stacks with existing features (mean-centering, cross-
  encoder reranker, multi-view keys).
