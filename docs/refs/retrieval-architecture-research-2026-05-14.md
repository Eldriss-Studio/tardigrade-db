# Retrieval Architecture Research — 2026-05-14

**Companion to:** [`docs/experiments/2026-05-14-bench-audit.md`](../experiments/2026-05-14-bench-audit.md)

Literature review commissioned after the 2026-05-14 audit measured
honest TardigradeDB numbers on the clean LoCoMo + LongMemEval
datasets (29.62% / 3.14% deterministic). The audit confirmed the
engine is not broken; this memo records what the published evidence
says about the architectural choices that determine how high those
numbers can go.

Every claim below is sourced. Where evidence is thin or absent,
that's stated explicitly.

---

## TL;DR

1. **The 3.14% LongMemEval result is not a "single-signal vs hybrid"
   retrieval problem.** Pure BM25 alone — the simplest published
   retriever — reaches **68% R@5** on the same benchmark per the
   LongMemEval paper. A 22× gap to the weakest baseline is not
   recoverable by retriever architecture; the dominant lever
   somewhere else.
2. **The LongMemEval paper itself names indexing strategy as the
   dominant lever**, not retriever choice. Round-level chunking,
   fact-augmented keys, and time-aware query expansion each move
   recall more than swapping BM25 for a 7B trained dense retriever.
3. **Raw decoder-LM hidden states as a retrieval signal has no
   published support at sub-1B scale.** Every adjacent paper that
   gets useful retrieval out of decoder hidden states adds a
   projection head, contrastive training, or both. PromptReps
   explicitly reports raw dense hidden-state retrieval underperforms
   sparse.
4. **Production LLM-agent memory systems are mostly hybrid
   (Mem0, Mem0g, Zep) or pure-BM25 with LLM-agentic search
   (ByteRover).** Zero use raw LM hidden states.
5. **Counter-evidence exists.** ByteRover hits LoCoMo 92.2% with
   BM25-only retrieval and heavy LLM compute at score time. RRF
   fusion is empirically suboptimal — Bruch et al. (TOIS 2023) show
   tuned convex combination beats it. Cross-encoder reranking gives
   bigger uplifts (15–25pp Hit@1) than fusion typically does
   (~5–15pp).

---

## Q1 — What do production LLM-agent memory systems actually use?

### Letta (formerly MemGPT)

Tiered memory (core / archival / recall). Archival memory is a vector
store using a configurable embedding model — the framework defaults
to OpenAI's `text-embedding-3-small`. Retrieval is **single-signal
dense** via semantic search; `conversation_search` exposes keyword
search over recall storage as a **separate tool**, not fused with
archival search. No built-in reranker.

- Sources: [docs.letta.com/concepts/memgpt](https://docs.letta.com/concepts/memgpt/),
  [github.com/letta-ai/letta](https://github.com/letta-ai/letta)

### Mem0 / Mem0g

Explicitly **multi-signal**. The Mem0 paper
([arXiv:2504.19413](https://arxiv.org/abs/2504.19413)) and public
docs describe three parallel scoring passes — (a) semantic
similarity (dense embedding, default `text-embedding-3-small`),
(b) BM25 keyword matching, (c) entity matching — followed by
fusion. Reranker stage is pluggable (Cohere, ZeroEntropy, HF
cross-encoders, Sentence-Transformers, LLM-as-reranker). Mem0g
adds graph traversal via entity linking.

- Sources: [arXiv:2504.19413](https://arxiv.org/abs/2504.19413),
  [docs.mem0.ai/core-concepts/memory-evaluation](https://docs.mem0.ai/core-concepts/memory-evaluation)

### ByteRover 2.0

**No vector DB, no embedding model.** Stores knowledge as
human-readable Markdown files in a hierarchical "Context Tree" and
uses a five-tier retrieval cascade:

- Tier 0: exact cache (hash match, ~0 ms)
- Tier 1: fuzzy cache (Jaccard, ~50 ms)
- Tier 2: BM25 only (MiniSearch, ~100 ms)
- Tier 3: constrained single LLM call (<5 s)
- Tier 4: full agentic loop (8-15 s)

Reports **92.2% on LoCoMo** with this BM25 + LLM-agentic stack.
Caveat: Tiers 3-4 burn substantial LLM compute, so the comparison
isn't apples-to-apples with a pure retrieval engine.

- Sources: [byterover.dev/blog/introducing-byterover-cli-2.0](https://www.byterover.dev/blog/introducing-byterover-cli-2.0-reinvent-memory-for-autonomous-agents),
  [arXiv:2604.01599](https://arxiv.org/abs/2604.01599)

### Zep (Graphiti)

Explicitly **hybrid + graph**. The paper
([arXiv:2501.13956](https://arxiv.org/abs/2501.13956)) describes
three-step retrieval: (1) candidate identification via dense
embedding (1024-d, cosine) + BM25 full-text search + direct graph
traversal, (2) candidate ranking, (3) context construction.
Reports +18.5% accuracy and 90% latency reduction vs MemGPT on the
Deep Memory Retrieval (DMR) benchmark. Reranking is via graph
reranker, not a cross-encoder.

- Source: [arXiv:2501.13956](https://arxiv.org/abs/2501.13956)

### LMCache

**Not a comparable system.** KV-cache infrastructure with prefix
matching and a CacheBlend extension for non-prefix reuse. Retrieval
is exact token-sequence hash matching plus a non-prefix variant
where ~15% of KV is recomputed for correctness. No semantic
retrieval — the lookup key is the literal token sequence.

- Sources: [lmcache.ai/tech_report.pdf](https://lmcache.ai/tech_report.pdf),
  [arXiv:2510.09665](https://arxiv.org/html/2510.09665v2)

### Summary

| System | Signal | Reranker | Notes |
|---|---|---|---|
| Letta | Dense (single) | None | Sparse as separate tool |
| Mem0 | Dense + BM25 + entity (fused) | Pluggable cross-encoder/LLM | Most pipeline-like |
| Mem0g | Dense + graph traversal | Same | Entity-anchor graph walks |
| ByteRover 2.0 | BM25 + cache + LLM-agentic | None | No embeddings |
| Zep | Dense + BM25 + graph | None (graph rerank) | Temporal graph |
| LMCache | Exact token-prefix hash | n/a | Not semantic memory |

Three hybrid, one BM25+agentic, one single-signal dense, one
non-comparable. **Zero use raw LM hidden states.** The closest
analog to TardigradeDB's "no trained retriever" position is
ByteRover — but ByteRover replaces the embedding with BM25, not
raw hidden states.

---

## Q2 — Hybrid vs single-signal: empirical evidence

### BEIR (Thakur et al., NeurIPS 2021)

The canonical zero-shot retrieval benchmark
([arXiv:2104.08663](https://arxiv.org/abs/2104.08663)). Key findings
relevant here:

- **In-domain (MS MARCO):** dense beats BM25 by 7-18 nDCG points.
- **Out-of-domain (BEIR's 18 zero-shot datasets):** BM25 is
  competitive or better than most dense models trained on MS MARCO.
  Quote: "BM25 is a strong generalizable baseline... many dense
  single-vector embedding models trained on MS MARCO labels are
  outperformed by BM25 in an out-of-domain setting."
- **Best zero-shot performance:** late-interaction (ColBERT) or
  BM25 + cross-encoder reranking.

Hybrid numbers across reproduced experiments:

- BEIR aggregate: hybrid lifts nDCG@10 from **43.42 (BM25)** to
  **52.59** (~21% relative improvement).
- Natural Questions top-1: BM25 22.1%, DPR 48.7%, **hybrid 53.4%**
  (+4.7pp over dense alone).
- Financial QA: Hybrid + cross-encoder R@5 = **0.816** vs
  Hybrid-RRF alone 0.695 vs BM25 0.644 vs dense 0.587. The
  cross-encoder dominates the lift.

### RRF specifically

- **Cormack et al. (SIGIR 2009)** — original RRF paper, showed RRF
  outperforms Condorcet and individual rankers.
- **Bruch et al., "An Analysis of Fusion Functions for Hybrid
  Retrieval"** (TOIS 2023,
  [arXiv:2210.11934](https://arxiv.org/abs/2210.11934)): **Convex
  combination (CC) of lexical and semantic scores outperforms RRF**
  in both in-domain and out-of-domain. CC requires only a small
  tuning set; RRF disregards score distribution and that's an
  information loss. *So RRF is the safe zero-tuning baseline, but a
  tuned linear combination is strictly better when labeled
  validation data exists.*

### LongMemEval specifically

The paper (Wu et al., ICLR 2025,
[arXiv:2410.10813](https://arxiv.org/abs/2410.10813)) reports these
R@5 baselines on LongMemEval-M:

| Retriever | R@5 |
|---|---|
| flat-bm25 | **0.683** |
| flat-contriever | 0.723 |
| flat-stella (1.5B) | 0.732 |
| flat-gte (Qwen2-7B-instruct) | similar |
| Contriever + fact-expansion | **0.762** (paper's best) |

**The paper's key claim:** indexing strategy matters more than
retriever choice. Specifically:

- Fact-augmented key expansion: **+4% retrieval, +5% downstream
  accuracy**
- Time-aware query expansion: **+11.4% recall (round-level), +6.7%
  (session-level)**
- Session decomposition into rounds: largest single intervention

**Implication for TardigradeDB:** Going from BM25 (0.683) to a 7B
trained dense retriever buys ~5-10 points. Going from "raw session
indexing" to "fact-augmented rounds + time-aware queries" buys
~10-15 points. Our current 3.14% is **two orders of magnitude below
the worst baseline** (BM25 at 68%). That gap is not caused by
retriever architecture choice. It's caused by something more
fundamental — almost certainly indexing granularity given the
paper's own findings.

### LoCoMo specifically

The original paper
([arXiv:2402.17753](https://arxiv.org/abs/2402.17753), Maharana
et al., ACL 2024) did not deeply ablate retrieval architectures.
Recent claimed numbers:

- **ByteRover 92.2%** (BM25 + LLM-agentic, no embeddings)
- **MemPalace 96.6%** (architecture details not fully public; uses
  simpler retrieval + heavier LLM reasoning per their
  [benchmarks page](https://www.mempalace.tech/benchmarks))
- **Mem0** (hybrid) reports strong numbers

---

## Q3 — Raw LM hidden states as a retrieval signal

This is the foundational architectural bet of TardigradeDB. The
published evidence is **mixed-to-negative for raw hidden states
without retrieval-specific training**:

### LLM2Vec (McGill, ICLR 2024 spotlight)

[arXiv:2404.05961](https://arxiv.org/abs/2404.05961). Decoder LLMs
can be turned into strong retrievers, but only after: (1) enabling
bidirectional attention, (2) masked next-token training, (3)
unsupervised contrastive learning. MTEB score: **56.8 (unsupervised
SOTA)**. Without those three steps, decoder hidden states underperform
encoder baselines.

### RepLLaMA / Llama2Vec

[arXiv:2312.15503](https://arxiv.org/abs/2312.15503). LLaMA-2
fine-tuned for retrieval reaches **BEIR average 56.4 (zero-shot
SOTA)**. Fine-tuned, not raw.

### PromptReps (most directly relevant)

[arXiv:2404.18424](https://arxiv.org/abs/2404.18424). Prompting LLMs
to produce dense + sparse representations. Critical quote:

> "Dense embeddings from hidden states alone perform poorly for
> document retrieval tasks with some LLMs, but sparse representations
> are much more robust, and the best retrieval effectiveness is
> achieved with hybrid retrieval systems."

This is the most direct published evidence against TardigradeDB's
foundational bet — that raw decoder hidden states (Top5Avg over
position-skipped tokens) is a sufficient retrieval signal.

### "One Model Is Enough: Native Retrieval Embeddings from LLM Agent Hidden States"

[arXiv:2603.08429](https://arxiv.org/abs/2603.08429). Equips an agent
with native retrieval by adding **a lightweight projection head** on
top of hidden states. The direction is right; the implementation
requires a learned projection, not raw hidden states.

### Causal2Vec

[arXiv:2507.23386](https://arxiv.org/abs/2507.23386). Modifies
decoder-only LLMs for embedding via retrieval-specific training.

### Honest summary

I could not find any published paper that benchmarks **raw,
untrained, decoder-LM hidden states of a sub-1B model with
per-token Top5Avg aggregation** against trained retrievers on
long-document or conversational tasks. The closest evidence
(PromptReps, LLM2Vec ablations) is **negative for the raw variant**.
This does not prove TardigradeDB's approach can't work — but **no
published evidence supports it competing with trained retrievers**,
and the closest adjacent papers explicitly say raw decoder hidden
states underperform.

---

## Q4 — The LongMemEval failure mode

Our hypothesis going in: top-K dominated by within-session chunks
that share surface vocabulary, not the chunk containing the actual
answer.

**This is a documented and named phenomenon** with multiple labels
across subfields:

- **Vocabulary mismatch / lexical overlap dominance** —
  Furnas et al., CACM 1987,
  [doi.org/10.1145/32206.32212](https://dl.acm.org/doi/abs/10.1145/32206.32212)
  ("The vocabulary problem in human-system communication"). Classic
  name; surface-form similarity overrides semantic relevance.
- **Topic drift** in query expansion literature.
- **Session-coarseness** in conversational retrieval. **The
  LongMemEval paper itself (Finding 1)** says: "Round is the best
  granularity for storing and utilizing interactive history" — i.e.,
  session-level chunking dilutes relevance signal precisely because
  off-topic surface tokens in the same session bleed into the chunk
  representation. *This is exactly the failure mode we observed.*
- **Conversational distractor problem** — SelRoute (Wang 2026,
  [arXiv:2604.02431](https://arxiv.org/abs/2604.02431)) and broader
  CIS literature.

### Published fixes that aren't "use a trained dense retriever"

From the LongMemEval paper and surrounding literature:

1. **Finer chunking granularity** (round-level, not session-level)
   — largest published intervention.
2. **Fact-augmented key expansion** — extract atomic facts at
   indexing time and add them as additional retrieval keys.
   **+4% retrieval per the paper.** TardigradeDB already has the
   `add_view_keys` primitive for this.
3. **Time-aware query expansion** when temporal cues are relevant
   — +6.7% to +11.4% recall.
4. **Query rewriting / decomposition** — RankRAG (NeurIPS 2024,
   [proceedings link](https://proceedings.neurips.cc/paper_files/paper/2024/file/db93ccb6cf392f352570dd5af0a223d3-Paper-Conference.pdf))
   — an LLM rewrites the user's vague conversational query into a
   retrieval-shaped form before scoring.
5. **BM25** is largely immune to this specific failure for *content*
   queries (no session-level pooling, no semantic-drift toward
   session-mean). Part of why hybrid helps here.
6. **Multi-view / multi-key indexing** — which TardigradeDB already
   has via `add_view_keys`.

The LongMemEval paper is unusually clear: **the dominant lever is
indexing, not retriever choice.**

---

## Q5 — Counter-evidence to "hybrid is the universal answer"

We looked specifically for this. Findings:

1. **ByteRover 2.0 LoCoMo 92.2% with BM25-only retrieval.** Strongest
   counter-evidence in the conversational-memory domain. Caveat: spends
   substantial LLM compute at Tier 3/4.
2. **MemPalace 96.6% LongMemEval.** Architecture less public, but
   reportedly doesn't lean on hybrid embedding fusion.
3. **Bruch et al. (TOIS 2023)** — RRF is suboptimal; tuned convex
   combination wins. So "hybrid via RRF" is not the strongest variant.
4. **In-domain MS MARCO:** a single well-trained dense retriever (ColBERTv2 or strong cross-encoder reranker) **beats hybrid** by
   7-18 nDCG points (per BEIR). Hybrid's advantage is specifically
   out-of-domain / zero-shot.
5. **PromptReps**: hidden-state dense alone is weak, but **the
   dominant component of their hybrid is sparse**, not dense. For
   some LLM-derived retrievers, the sparse side carries most of the
   load — suggesting that if we *do* go hybrid, BM25 may matter more
   than the latent signal.
6. **LongMemEval's own finding:** switching retrievers (BM25 → 7B
   GTE) moves R@5 by ~5-8 points; switching indexing strategy moves
   it by ~15 points. **Indexing >> retriever architecture.**

### Cross-encoder reranking as alternative

The reported 62.67% → 83.00% Hit@1 lift from cross-encoder reranking
(industry benchmarks) is **larger than the typical hybrid uplift
over single-signal dense** (~5-15 points). If reranker quality is
high and you can afford it, you may not need fusion — though hybrid
still helps recall in the candidate pool feeding the reranker.

---

## Synthesis for TardigradeDB

### What's true

1. The 3.14% LongMemEval result is not a "single vs hybrid" problem.
   It's a 22× gap to the weakest baseline — that's an indexing /
   chunking / key-construction problem before it's an architecture
   problem.
2. The foundational bet ("raw LM hidden states are a sufficient
   retrieval signal") has no published support at sub-1B scale.
   Every adjacent paper adds projection or contrastive training.
3. The LongMemEval paper's own ablation is unambiguous: indexing
   > retriever choice. And TardigradeDB already has the indexing
   primitives (`TextChunker`, `add_view_keys`, multi-view
   consolidation).
4. Cross-encoder reranking — already shipped — gives larger lifts
   than fusion typically does, but only if the right item is in the
   top-K it sees.

### What's actionable

The next debug should be:

1. **Fix indexing first** — round-level chunking, `add_view_keys`
   with fact extraction, time-aware query expansion. Largest
   published lever. Tests TardigradeDB's existing primitives.
2. **Then, if gap remains, add a second retrieval signal.** BM25
   over the `TextStore` content. Tuned convex combination, not RRF.
   Tests the "hidden states benefit from a complementary signal"
   hypothesis with the empirically-strongest complement.
3. **Then, if gap still remains, learn a projection head over
   hidden states.** LLM2Vec / RepLLaMA pattern. Tests whether the
   foundational bet survives with retrieval-specific training.
4. **Last resort, swap to a trained dense retriever** and reposition
   TardigradeDB's KV-native side as an injection mechanism rather
   than the retrieval signal.

The staged debug plan is in
[`docs/superpowers/plans/2026-05-14-retrieval-debug-plan.md`](../superpowers/plans/2026-05-14-retrieval-debug-plan.md).

### Honest framing of the bet

The literature does not currently support
"raw LM hidden states + per-token Top5Avg replaces trained
retrieval." But the architectural surface area of TardigradeDB
(persistent KV-cache as memory, latent-space injection, AKL
governance, multi-view consolidation, semantic edges) is novel and
intact. The retrieval signal is one component of the kernel, not
the whole product. Reframing TardigradeDB as "a KV-native memory
kernel that uses model hidden states as ONE signal in a hybrid
retrieval stack" is defensible.

---

## Sources

- Letta: [docs.letta.com/concepts/memgpt](https://docs.letta.com/concepts/memgpt/),
  [github.com/letta-ai/letta](https://github.com/letta-ai/letta)
- Mem0: [arXiv:2504.19413](https://arxiv.org/abs/2504.19413),
  [docs.mem0.ai](https://docs.mem0.ai/core-concepts/memory-evaluation)
- Zep / Graphiti: [arXiv:2501.13956](https://arxiv.org/abs/2501.13956)
- ByteRover 2.0:
  [byterover.dev/blog/introducing-byterover-cli-2.0](https://www.byterover.dev/blog/introducing-byterover-cli-2.0-reinvent-memory-for-autonomous-agents),
  [arXiv:2604.01599](https://arxiv.org/abs/2604.01599),
  [byterover.dev/blog/benchmark-ai-agent-memory](https://www.byterover.dev/blog/benchmark-ai-agent-memory)
- LMCache: [lmcache.ai/tech_report.pdf](https://lmcache.ai/tech_report.pdf),
  [arXiv:2510.09665](https://arxiv.org/html/2510.09665v2),
  [blog.lmcache.ai/2024-10-09-cacheblend](https://blog.lmcache.ai/2024-10-09-cacheblend/)
- LongMemEval (ICLR 2025): [arXiv:2410.10813](https://arxiv.org/abs/2410.10813),
  [github.com/xiaowu0162/longmemeval](https://github.com/xiaowu0162/longmemeval),
  [xiaowu0162.github.io/long-mem-eval](https://xiaowu0162.github.io/long-mem-eval/)
- LoCoMo (ACL 2024): [arXiv:2402.17753](https://arxiv.org/abs/2402.17753)
- BEIR (NeurIPS 2021): [arXiv:2104.08663](https://arxiv.org/abs/2104.08663),
  [NeurIPS proceedings](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/65b9eea6e1cc6bb9f0cd2a47751a186f-Paper-round2.pdf)
- Cormack RRF (SIGIR 2009):
  [cormack.uwaterloo.ca/cormacksigir09-rrf.pdf](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)
- Bruch fusion analysis (TOIS 2023): [arXiv:2210.11934](https://arxiv.org/abs/2210.11934),
  [ACM TOIS](https://dl.acm.org/doi/10.1145/3596512)
- LLM2Vec (ICLR 2024): [arXiv:2404.05961](https://arxiv.org/abs/2404.05961)
- RepLLaMA / Llama2Vec: [arXiv:2312.15503](https://arxiv.org/abs/2312.15503)
- PromptReps: [arXiv:2404.18424](https://arxiv.org/abs/2404.18424)
- Native Retrieval Embeddings from Agent Hidden States:
  [arXiv:2603.08429](https://arxiv.org/abs/2603.08429)
- Causal2Vec: [arXiv:2507.23386](https://arxiv.org/abs/2507.23386)
- Furnas, vocabulary problem (CACM 1987):
  [ACM DL](https://dl.acm.org/doi/abs/10.1145/32206.32212)
- Survey of Conversational Search: [arXiv:2410.15576](https://arxiv.org/html/2410.15576v1)
- RankRAG (NeurIPS 2024):
  [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/db93ccb6cf392f352570dd5af0a223d3-Paper-Conference.pdf)
- MemPalace benchmarks: [mempalace.tech/benchmarks](https://www.mempalace.tech/benchmarks)
- Cognaptus analysis of ByteRover architecture:
  [cognaptus.com/blog/2026-04-05-memory-rewritten-why-byterover-kills-the-pipeline-and-maybe-saves-agents](https://cognaptus.com/blog/2026-04-05-memory-rewritten-why-byterover-kills-the-pipeline-and-maybe-saves-agents/)
