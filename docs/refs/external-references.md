# TardigradeDB: External References Audit

> **⚠️ RETRACTED — 2026-05-14 (narrow scope).** One citation *context* in this file is invalidated by the 2026-05-14 bench audit. **The DeepMind LIMIT paper itself ([arXiv:2508.21038](https://arxiv.org/abs/2508.21038), ICLR 2026) is unaffected** — it is a valid mathematical result and remains a legitimate reference. What is retracted is **TardigradeDB's empirical application of it**:
>
> - **Section A3d entry on LIMIT** ("Explains why TardigradeDB's vague-query ceiling (60% R@5) cannot be broken by geometric transforms alone…") — the "60% R@5 ceiling" claim on LoCoMo data was measured against a corrupted dataset and the lexical fallback adapter, not the native engine. The mathematical statement of LIMIT is correct; our claim that our empirical numbers *demonstrate* the LIMIT ceiling has no support from these runs.
> - **Section H decision-justification row** ("Latent-space retrieval has a theoretical ceiling for vocabulary-mismatched queries" — citing LIMIT + "experimentally confirmed: whitening, token reweighting, multi-layer fusion all 0% improvement on the 3 persistently-missed vague queries") — the LIMIT half is fine as theory; the "experimentally confirmed" half is retracted to the extent it was measured on the LoCoMo bench. The 10-fact synthetic Sonia corpus results survive, but they do not by themselves confirm a fundamental ceiling.
>
> **All other ~40 references in this file (Knowledge Packs, MemArt, DiskANN/Vamana, ByteRover 2.0, PagedAttention, CacheBlend, RoFormer, ColBERT-PRF, BERT-flow, Rocchio, Hindsight, RAG-Fusion, CRAG, Self-RAG, DCI, MMLF, Mem0, Letta, Zep, etc.) are unaffected** — they are external work, cited correctly, and TardigradeDB's design decisions that referenced them are mostly grounded in things other than the broken LoCoMo numbers.
>
> Forensic record: [`../experiments/2026-05-14-bench-audit.md`](../experiments/2026-05-14-bench-audit.md).

# TardigradeDB: External References Audit

Complete catalog of every paper, blog post, algorithm, system, and benchmark that
influenced TardigradeDB's architecture or implementation. Sources: codebase, docs,
plans, `.claude/plans/`, Codex/Resumancer sessions (2026-01 through 2026-05), and
experiment docs.

---

## A. Papers & Academic Research

### A1. Core Architecture — Load-Bearing Citations

These directly justify fundamental design decisions. If any were retracted or
contradicted, the corresponding subsystem would require redesign.

| Paper | arXiv / DOI | What it justifies | Where cited |
|---|---|---|---|
| **Knowledge Packs** | arXiv:2604.03270 | KV injection is mathematically equivalent to text-in-prompt for causal transformers; sequential multi-memory approach; naive concatenation failure mode | `docs/experiments/multi-memory-injection.md`, `docs/experiments/kv-cache-validation.md`, `README.md` |
| **MemArt** (KV Cache as Persistent/Shared Memory) | arXiv:2409.17264 | Latent-space retrieval without text round-trip; brute-force SIMD matmul beats ANN at <10K blocks per agent; compressed key handling | `crates/tdb-retrieval/src/lib.rs`, `crates/tdb-retrieval/src/attention.rs`, `docs/experiments/kv-cache-validation.md`, `README.md`, `CLAUDE.md` |
| **DiskANN / Vamana** (Subramanya et al., 2019) | NEURIPS 2019 | Cold-path ANN graph index; single-layer small-world graph; angular diversity pruning (robust pruning); chose over HNSW because HNSW fails for Q/K distribution shift | `crates/tdb-index/src/vamana/mod.rs`, `crates/tdb-index/src/vamana/prune.rs`, `docs/learning-roadmap.md`, `CLAUDE.md`, `README.md` |
| **ByteRover 2.0** (Agent-Native Memory) | arXiv:2604.01599 | Source of the Adaptive Knowledge Lifecycle (AKL): importance scoring formula, maturity tier thresholds, hysteresis, recency decay τ=30d. Applied to tensor blocks instead of text files | `crates/tdb-governance/`, `docs/refs/AI Agentic Memory System Efficiency.md`, `docs/learning-roadmap.md`, `CLAUDE.md` |
| **PagedAttention / vLLM** (Kwon et al., 2023) | SOSP 2023 | Paged KV cache management; continuous batching; defines the production connector API target (KV Connector v1) | `docs/learning-roadmap.md`, `docs/experiments/README.md`, `README.md` |
| **CacheBlend** | EuroSys 2025 Best Paper | RoPE position corruption in concatenated KV caches; cross-prompt KV composition reuse strategy; informed `RoPECorrectedConcatComposer` implementation; tested and found position corruption is NOT the primary multi-memory failure mode | `docs/experiments/multi-memory-injection.md` |

### A2. Retrieval & Encoding

| Paper | arXiv / DOI | What it justifies | Where cited |
|---|---|---|---|
| **RoFormer** (Su et al., 2021) | arXiv:2104.09864 | Rotary Position Embeddings (RoPE); rotation applied to Q and K; critical for cross-context KV reuse and position remapping via `RoPEPositionEncoder.remap_keys()` | `docs/learning-roadmap.md`, `docs/experiments/kv-cache-validation.md` |
| **"Attention Is All You Need"** (Vaswani et al., 2017) | arXiv:1706.03762 | Transformer fundamentals; Q/K/V projections; scaled dot-product attention formula `score(q,k) = (q·k)/√d_k` | `docs/learning-roadmap.md`, `crates/tdb-retrieval/src/lib.rs` |
| **FIER** | arXiv ID not confirmed in codebase docs | Confirms Q*K (not K*K) as the correct retrieval scoring: all use Q for queries; K*K produces symmetric attention that can't capture directional relationships | `docs/experiments/kv-cache-validation.md` |
| **ShadowKV** | arXiv ID not confirmed in codebase docs | Same confirmation of Q*K over K*K; cited alongside FIER and "From QKV to K/KV" as literature consensus | `docs/experiments/kv-cache-validation.md` |
| **"From QKV to K/KV"** | arXiv ID not confirmed in codebase docs | Explains why K*K retrieval fails: symmetric attention cannot represent directional query-to-memory relationships; the correct design uses Q for queries and K for stored memories | `docs/experiments/kv-cache-validation.md` |
| **ColBERT** (Khattab & Zaharia, 2020) | arXiv:2004.12832 | Late interaction scoring via cosine sum-of-max; **tested and rejected** — `cosine_sum_max` scored 53.3% R@5 vs Top5Avg's 100% on 100-memory corpus | `docs/experiments/kv-cache-validation.md` |
| **GPTQ** (Frantar et al., 2022) | arXiv:2210.17323 | Post-training quantization intuition for weight quantization; same principles apply to KV cache Q4 quantization in `tdb-storage` | `docs/learning-roadmap.md` |
| **AWQ** (Lin et al., 2023) | arXiv:2306.00978 | Activation-aware weight quantization; cited alongside GPTQ for KV quantization background | `docs/learning-roadmap.md` |
| **Orthogonal Procrustes** (classical linear algebra; Schönemann 1966) | — | Cross-model space alignment; used for same-family retrieval via linear projection; empirically shown to plateau at ~47% R@5 for cross-family (Qwen→GPT-2), motivating the MLP adapter instead | `docs/experiments/README.md` |
| **Orca** (Yu et al., 2022) | OSDI 2022 | Iteration-level scheduling for LLM inference; concurrent prefill/decode scheduling relevant to multi-agent session management | `docs/learning-roadmap.md` |
| **Pancake** | arXiv:2602.21477 | Multi-tier memory for multi-agent serving; ANN indices + cache tier + GPU/CPU placement optimization | `docs/competitors/competitors-search-2.md` |
| **Rocchio Algorithm** (Rocchio, 1971) | Salton, *The SMART Retrieval System*, Prentice-Hall 1971 | Classical pseudo-relevance feedback (PRF): `q' = α·q_orig + β·avg(relevant) − γ·avg(non_rel)`. Foundation for the latent-space PRF refinement designed for vague-query retrieval. The query gets pulled toward the centroid of plausible top-k results, closing the vocabulary mismatch that produces the 48% R@5 cliff on vague queries | `crates/tdb-retrieval/src/refinement.rs` (planned), `docs/experiments/vague_queries/` (planned) |
| **Improving Query Representations for Dense Retrieval with PRF** (Yu et al., CIKM 2021) | arXiv:2108.13454 | Vector-based PRF for dense retrievers: Rocchio adapted to embedding space using pre-generated passage embeddings; both Average and Rocchio fusion variants. Justifies pure latent-space PRF over text-based query expansion for TardigradeDB's tensor-native architecture | `crates/tdb-retrieval/src/refinement.rs` (planned) |
| **ColBERT-PRF** (Wang et al., TWeb 2022) | arXiv:2106.11251 | Late-interaction PRF with ColBERT-style per-token MaxSim scoring. Closest existing analogue to TardigradeDB's per-token Top5Avg + PRF design. Confirms PRF works with late-interaction scoring (not just bi-encoders) | `docs/experiments/vague_queries/` (planned) |
| **PRF with Deep LMs and Dense Retrievers: Successes and Pitfalls** (Li et al., TOIS 2023) | doi:10.1145/3570724 | Survey of dense PRF; documents the drift failure mode (when first-stage top-1 is wrong, PRF amplifies the error) and motivates conservative α/β values (α≥0.7) and small k_prime (≤5). Informs the LatentPrfRefinement default hyperparameters | `crates/tdb-retrieval/src/refinement.rs` (planned) |
| **Reciprocal Rank Fusion** (Cormack, Clarke, Büttcher, SIGIR 2009) | doi:10.1145/1571941.1572114 | Fusion formula `score(d) = Σ 1/(k+rank_i(d))`, k=60 default. Combines first-stage and PRF-stage rankings without score normalization. Designated as the safer alternative to pure replacement when PRF drifts; potential Stage-3 fusion if hybrid sparse retrieval is added later | `crates/tdb-retrieval/src/refinement.rs` (planned) |
| **Decoding a Neural Retriever's Latent Space for Query Suggestion** (Adolphs et al., EMNLP 2022) | arXiv:2210.12084 | Demonstrates that meaningful semantics are decodable from neural retrieval latent spaces and that moving in the right direction in latent space retrieves the relevant passage. Supports the premise that PRF in K-space carries usable signal | `docs/experiments/vague_queries/` (planned) |

### A3. Adapters & Weights-as-Memory

| Paper | arXiv / DOI | What it justifies | Where cited |
|---|---|---|---|
| **LoRA** (Hu et al., 2021) | arXiv:2106.09685 | Low-rank weight adaptation `W' = W + αBA`; frozen base + per-agent trainable rank-r matrices; foundation for SynapticBank design | `docs/learning-roadmap.md`, `crates/tdb-core/src/` |
| **LRAgent** | arXiv:2602.01053 | Shared base KV cache + per-agent LoRA adapter splitting; 3× memory savings in multi-agent serving; inspired synaptic bank + multi-agent isolation design | `docs/competitors/competitors-search-2.md`, `docs/learning-roadmap.md`, `CLAUDE.md` |
| **MemoryLLM (Apple)** | arXiv:2602.00398 | FFN parameters as token-indexed neural memory; "weights as memory" framing; not productized — confirms SynapticBank is novel | `docs/competitors/competitors-search-2.md`, `CLAUDE.md` |
| **PRIME** (Dual-Memory Personalization) | arXiv:2507.04607 | Episodic + semantic dual-memory architecture; slow weights (semantic) vs fast episodic memory; neuroscience-grounded split inspires KV bank (episodic) + synaptic bank (semantic) | `docs/competitors/competitors-search-2.md`, `docs/learning-roadmap.md` |
| **FwPKM** (Fast-weight Product Key Memory) | arXiv:2601.00671 | Dynamic fast-weight episodic memory updated during inference; alternative to fixed storage | `docs/competitors/competitors-search-2.md` |
| **Generalized Key-Value Memory** | arXiv:2203.06223 | Decoupled memory dimension from support vectors; external KV memory in hardware | `docs/competitors/competitors-search-2.md` |

### A3b. Index-Time Augmentation & Multi-View Retrieval

| Paper | arXiv / DOI | What it justifies | Where cited |
|---|---|---|---|
| **HyPE** (Hypothetical Prompt Embeddings; Vake et al., 2025) | [SSRN:5139335](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5139335) | Index-time question generation per chunk — embed questions as retrieval keys, return parent chunk for synthesis. +42pp precision, +45pp recall on certain datasets. Key insight: LLM-generated questions (not rule-based) are required for discriminative views; views are retrieval keys only, never returned as results | `docs/refs/file-ingest-as-kv-memory.md`, `experiments/multiview_diagnosis.py` |
| **Doc2Query--: When Less is More** (Gospodinov & MacAvaney, ECIR 2023) | [arXiv:2301.03266](https://arxiv.org/abs/2301.03266) | Relevance filtering of generated query expansions before indexing. Uncontrolled generation produces hallucinated/redundant queries that harm retrieval — **filtering improves effectiveness by 16% while cutting index size by 33%**. Directly explains TardigradeDB's multi-view failure: rule-based views are low-diversity expansions that dilute the index | `experiments/multiview_diagnosis.py` |
| **Doc2Query++** (Topic-Coverage Dual-Index Fusion, October 2025) | [arXiv:2510.09557](https://arxiv.org/abs/2510.09557) | Dual-Index Fusion — text and query expansions in separate indexes, fused via RRF at retrieval. Solves the noise-from-concatenation problem that degrades dense retrieval when expansions are appended directly. Proposed fix architecture for TardigradeDB's view dilution | `experiments/multiview_diagnosis.py` |
| **When More Reformulations Hurt: Avoiding Drift** (May 2026) | [arXiv:2605.00560](https://arxiv.org/html/2605.00560) | PRF-induced query drift when initial retrieved set contains off-topic documents. Adding expansion terms shifts query away from original intent. Explains why TardigradeDB's question-framing views ("What did Sonia translated do?") act as self-generated off-topic PRF, degrading moderate R@5 from 80%→20% | `experiments/multiview_diagnosis.py` |
| **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval; Sarthi et al., ICLR 2024) | [arXiv:2401.18059](https://arxiv.org/abs/2401.18059) | Hierarchical summarization tree — abstract nodes find neighborhoods, leaf nodes provide answers. Prevents abstract summaries from competing with specific leaves. +20% absolute accuracy on QuALITY. Relevant pattern: hierarchy prevents the view-canonical competition problem | `experiments/multiview_diagnosis.py` |
| **Deliberation in Latent Space via Differentiable Cache Augmentation** (Liu et al., Google DeepMind, ICML 2025) | [arXiv:2412.17747](https://arxiv.org/abs/2412.17747) | Offline coprocessor augments KV cache with trained latent embeddings — extends cache, doesn't replace it. +10% GSM8K. Closest to what TardigradeDB could do natively: augment KV cache with learned "view embeddings" rather than storing separate view packs | `experiments/multiview_diagnosis.py` |
| **Multi-Vector Retriever / Parent Document Retriever** (LangChain pattern) | [blog.langchain.com](https://blog.langchain.com/semi-structured-multi-modal-rag/) | Decouples retrieval representation from answer source. Summaries/questions/paraphrases as retrieval keys all resolve to the same parent document — parent is returned, never the summary itself. The architectural pattern TardigradeDB's consolidator should follow | `experiments/multiview_diagnosis.py` |
| **ExpandR** (Yao et al., EMNLP 2025) | [arXiv:2502.17057](https://arxiv.org/abs/2502.17057) | Jointly optimizes LLM + retriever for query expansion via DPO. LLM generates expansions, retriever is trained to use them. +5% retrieval improvement. Relevant: shows vocabulary mismatch can be attacked by training the retriever, not just expanding queries | vague-query research (2026-05-11) |
| **SoftQE** (Amazon, 2024) | [arXiv:2402.12663](https://arxiv.org/abs/2402.12663) | Distills LLM query expansions into a retriever encoder — no LLM at inference. Trains query encoder to produce embeddings similar to expanded-query embeddings. Relevant: shows vocabulary bridging can be baked into the retriever itself, eliminating inference-time cost | vague-query research (2026-05-11) |
| **Query Expansion Survey** (2025) | [arXiv:2509.07794](https://arxiv.org/abs/2509.07794) | Comprehensive survey organizing QE along four dimensions: injection point, grounding/interaction, learning/alignment, and KG integration. Covers encoder-only through instruction-tuned variants | vague-query research (2026-05-11) |
| **vstash** (April 2026) | [arXiv:2604.15484](https://arxiv.org/abs/2604.15484) | Local-first hybrid retrieval (BM25 + dense vector + adaptive RRF) in a single SQLite file. 33M-param pipeline matches ColBERTv2 on 3/5 BEIR datasets. Self-supervised embedding refinement via vector/FTS disagreement. Production-grade with schema versioning + integrity checks | vague-query research (2026-05-11) |

### A3c. Latent-Space Query Transformation (Pure Latent, No Text)

| Paper | arXiv / DOI | What it justifies | Where cited |
|---|---|---|---|
| **LaSER** (March 2026) | [arXiv:2603.01425](https://arxiv.org/abs/2603.01425) | Self-distillation framework: internalizes explicit CoT reasoning into latent space of dense retrievers. Multi-grained alignment (output + trajectory). At inference, latent reasoning only — no text generation. Directly applicable: train a small adapter to map vague query hidden states toward correct memory regions in K-space | vague-query research (2026-05-11) |
| **DEBATER** (February 2025, SIGIR Asia Pacific 2025) | [arXiv:2502.12974](https://arxiv.org/abs/2502.12974) | Chain-of-Deliberation: iteratively refines document embeddings through multiple reasoning steps. Self-distillation fuses most informative steps. Shows deliberation in latent space improves retrieval without text round-trip. Applicable at write-time consolidation | vague-query research (2026-05-11) |
| **KV Packet** (April 2026) | [arXiv:2604.13226](https://arxiv.org/abs/2604.13226) | Trainable soft-token adapters wrap cached KV blocks to bridge context discontinuity. Zero recomputation, near-zero FLOPs. Directly relevant: adapters could make stored memories' hidden states more universally retrievable regardless of query phrasing. Fits SynapticBank | vague-query research (2026-05-11) |
| **AdaQR / Dense Retriever as Reasoner** (October 2025) | [arXiv:2510.21727](https://arxiv.org/abs/2510.21727) | Adaptive Query Reasoning: router directs queries to fast dense reasoning or deep LLM reasoning. Shows query transformation can happen in embedding space without LLM calls for most queries | vague-query research (2026-05-11) |

### A3d. Fundamental Limitations of Vector-Space Retrieval

| Paper | arXiv / DOI | What it justifies | Where cited |
|---|---|---|---|
| **On the Theoretical Limitations of Embedding-Based Retrieval** (Google DeepMind, ICLR 2026) | [arXiv:2508.21038](https://arxiv.org/abs/2508.21038) | Proves mathematically that single-vector retrieval has a dimensional ceiling: the number of distinct top-k outcomes is upper-bounded by embedding dimension via sign-rank. Even SOTA models fail <20% recall@100 on the LIMIT stress dataset. Explains why TardigradeDB's vague-query ceiling (60% R@5) cannot be broken by geometric transforms alone — the remaining misses require world-knowledge reasoning, not similarity | vague-query research (2026-05-11) |
| **Beyond Semantic Similarity: Rethinking Retrieval for Agentic Search via Direct Corpus Interaction** (May 2026) | [arXiv:2605.05242](https://arxiv.org/abs/2605.05242) | Direct Corpus Interaction (DCI): agent searches raw corpus with tools (grep, file reads, scripts) — no embeddings, no vector index. +11% agentic search, +30.7% multi-hop QA, +21.5% IR ranking over embedding baselines. Key insight: retrieval quality depends on interface resolution, not similarity metric. More aligned with TardigradeDB's thesis than BM25 — the agent reasons about what to retrieve | vague-query research (2026-05-11) |
| **Agentic RAG Survey** (January 2025) | [arXiv:2501.09136](https://arxiv.org/abs/2501.09136) | Comprehensive survey of agentic RAG patterns: reflection, planning, tool use, multi-agent collaboration. Key insight: iterative retrieval with query reformulation outperforms single-shot. Self-RAG achieves 5.8% hallucination rate (lowest of 12 variants). Corrective RAG evaluates retrieval quality and triggers reformulation on failure | agentic retrieval research (2026-05-11) |
| **NVIDIA NeMo Retriever Agentic Pipeline** (2025) | [HuggingFace blog](https://huggingface.co/blog/nvidia/nemo-retriever-agentic-retrieval) | #1 ViDoRe v3, #2 BRIGHT. Key: agentic loop naturally adapts strategy per dataset without heuristics. Research direction: distill agentic reasoning into smaller specialized agents for latency/cost reduction | agentic retrieval research (2026-05-11) |
| **Chroma Context-1: Self-Editing Search Agent** (2026) | [trychroma.com](https://www.trychroma.com/research/context-1) | Agent that reformulates its own search queries based on retrieval results. Demonstrates the self-editing search pattern TardigradeDB's RLS protocol would implement | agentic retrieval research (2026-05-11) |
| **MMLF: Multi-Query Multi-Passage Late Fusion** (NAACL 2025) | [ACL Anthology](https://aclanthology.org/2025.findings-naacl.367.pdf) | Engine-level fusion of multiple query representations into a single retrieval pass. Relevant primitive for RLS: batch multiple reformulated queries and fuse at the engine level | agentic retrieval research (2026-05-11) |

### A3e. Adaptive Retrieval Gating & Confidence-Based Reformulation

| Paper | arXiv / DOI | What it justifies | Where cited |
|---|---|---|---|
| **CRAG** (Corrective RAG; Yan et al., 2024) | [arXiv:2401.15884](https://arxiv.org/abs/2401.15884) | Three-way confidence gate (Correct/Ambiguous/Incorrect) using a T5-large retrieval evaluator. Two-threshold system with dataset-specific calibration. Lower thresholds as permissive as -0.99 on [-1,1] scale. Key insight: a healthy corrective system should trigger on 20-40% of queries — if 0%, thresholds are miscalibrated. Directly informed the diagnosis that TardigradeDB's 1.10 score-ratio confidence threshold was gating out nearly all queries | LLM agent reformulation experiment (2026-05-12) |
| **Self-RAG** (Asai et al., 2023) | [arXiv:2310.11511](https://arxiv.org/abs/2310.11511) | Learned reflection tokens (Retrieve/ISREL/ISSUP/ISUSE) trained into the LM vocabulary. Does NOT re-query on irrelevant results — filters and generates without retrieval or uses only relevant subset. Retrieval threshold default 0.2 controls frequency. More "retrieve or don't" than "reformulate and retry" | LLM agent reformulation experiment (2026-05-12) |
| **TARG** (Training-Free Adaptive Retrieval Gating, 2025) | [arXiv:2511.09803](https://arxiv.org/abs/2511.09803) | Margin gate using top-1 vs top-2 logit gap as uncertainty signal. Optimal retrieval budgets of 1-33% depending on model strength. At 0.1% retrieval rate, exceeded Always-RAG accuracy (83.8 vs 80.8 EM). Directly analogous to TardigradeDB's score-ratio threshold — confirms the gate must be calibrated to a retrieval budget, not set arbitrarily | LLM agent reformulation experiment (2026-05-12) |
| **DMQR-RAG** (Diverse Multi-Query Rewriting, 2024) | [arXiv:2411.13154](https://arxiv.org/abs/2411.13154) | Tested 4 rewriting strategies; applying all simultaneously is NOT optimal. Adaptive selection reduces rewrites by ~40% while improving performance (+2.17% on FreshQA). Key insight: for experiments, establish the always-reformulate ceiling first, then add adaptive selection | LLM agent reformulation experiment (2026-05-12) |
| **RAG-Fusion Industry Deployment** (2025) | [arXiv:2603.02153](https://arxiv.org/abs/2603.02153) | Production deployment: fusion consistently increased upstream recall BUT gains were statistically insignificant after reranking (p_adj >= 0.125). Hit@10 degraded by -6.08pp. Fusion benefits "largely confined to recall-scarce query regimes." Sub-queries beyond 5 cause informational redundancy. Directly relevant: TardigradeDB's vague-query failures ARE the recall-scarce regime where fusion helps | LLM agent reformulation experiment (2026-05-12) |
| **FLARE** (Active Retrieval Augmented Generation; Jiang et al., 2023) | [arXiv:2305.06983](https://arxiv.org/abs/2305.06983) | Triggers retrieval on low-confidence generated tokens. Best results when retrieval triggers on 40-80% of sentences. Confirms that selective but generous retrieval outperforms both always-retrieve and never-retrieve | LLM agent reformulation experiment (2026-05-12) |
| **Weighted RRF** (Confidence-Aware Fusion; CCNC 2026) | [uregina.ca](https://uregina.ca/~nss373/papers/Rag-CCNC2026.pdf) | Standard RRF treats all queries as equally important, diluting precision when low-quality reformulations are included. WRRF incorporates confidence scores: score range 0.6712 at rank 1 with gradual decline vs standard RRF's near-flat 0.0159-0.0667. Implication: when fusing original + reformulated results, weight the original higher proportional to confidence | LLM agent reformulation experiment (2026-05-12) |

### A3f. Quantization for Retrieval — Outlier Crush & Per-Channel Scaling

Investigated 2026-05-13 after the lazy-`PerTokenRetriever` cutover dropped LoCoMo recall from 68% to 3.3%. Probe showed Q4_0 group-of-32 quantization rounding 31 of 32 dims to zero when one outlier dim dominated the group abs_max — a known failure mode of naive low-bit quantization on LLM hidden states. These sources establish that **Q4 is the wrong storage format if retrieval scores are computed from the round-tripped data**; production retrieval engines all use INT8 or higher as their scoring floor, and the standard countermeasure for activation outliers is per-channel scaling (SmoothQuant pattern applied to storage).

| Paper / Source | arXiv / Link | What it justifies | Where cited |
|---|---|---|---|
| **LLM.int8()** (Dettmers et al., 2022) | [arXiv:2208.07339](https://arxiv.org/abs/2208.07339) | Discovered systematic **activation-outlier channels** in LLMs ≥6.7B (now also in modern smaller models like Qwen3): a small set of hidden dimensions whose magnitudes are 10-100× the rest, active in up to 75% of sequence positions. Names the failure mode that crushed our Q4 round-trip — outliers dominate the group scale and force every non-outlier value below the half-step rounding threshold | `docs/refs/AI-db-discussion.md` (planned: quantization-for-retrieval section); next-iteration `crates/tdb-retrieval/` redesign |
| **Tim Dettmers — Emergent Features** | [timdettmers.com](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/) | Companion blog clarifying that outlier channels are emergent, not noise; mixed-precision (FP16 outliers + INT8 rest) is the canonical fix in inference; same principle transfers to retrieval storage | next-iteration `tdb-retrieval/` redesign |
| **Quantizable Transformers** (Bondarenko et al., NeurIPS 2023) | [NeurIPS 2023 paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/edbcb7583fd8921dad78adecfe06a99b-Paper-Conference.pdf) | Traces emergent outliers to softmax "no-op" attention heads. Implication for TardigradeDB: outliers are a property of the captured representation, not a quantization artifact — must be handled at the storage layer, not assumed away | next-iteration `tdb-retrieval/` redesign |
| **SmoothQuant** (Xiao et al., 2022) | [arXiv:2211.10438](https://arxiv.org/abs/2211.10438) | Migrate outlier difficulty from activations to weights via per-channel scale `σ[d]`. **Transferable to storage**: pre-divide hidden states by `σ[d]` at store time, multiply back at score time. This is the standard recipe and the recommended fix for our regression | next-iteration `tdb-retrieval/` redesign — per-channel scaling + INT8 group quantization replaces Q4 as the scoring format |
| **AWQ** (Lin et al., MLSys 2024) | [MLSys 2024 paper](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf) | Per-channel scaling on weights driven by activation statistics. Storage-transferable as a calibration-time per-channel pre-scaling step. Confirms the SmoothQuant-for-storage approach is not novel | next-iteration `tdb-retrieval/` redesign |
| **OmniQuant / Outlier Suppression+** (Wei et al., 2023) | [arXiv:2304.09145](https://arxiv.org/abs/2304.09145) | Channel-wise shift + scale; combined transformation handles asymmetric outlier distributions. Optional refinement on top of plain SmoothQuant if pure scaling proves insufficient | next-iteration `tdb-retrieval/` redesign |
| **ColBERTv2** (Santhanam et al., 2021) | [arXiv:2112.01488](https://arxiv.org/abs/2112.01488) | Closest published shape to TardigradeDB's per-token retrieval: per-token vectors at scale, on disk. Explicitly **rejected naive quantization** in favor of **centroid + 1-or-2-bit residual** (PQ-like). At 1-bit residual: MRR@10 = 35.5, Recall@50 = 81.6 on MS MARCO with 6-10× compression vs FP16 ColBERT. The centroid absorbs the outlier structure; residuals only encode small deviations — that is why 1-bit per dim *works* there but Q4_0 *fails* on raw hidden states | next-iteration `tdb-retrieval/` — deferred to ≥5M cells |
| **PLAID** (Santhanam et al., 2022) | [arXiv:2205.09707](https://arxiv.org/abs/2205.09707) | Centroid-pruning on top of ColBERTv2; two-tier scoring (cheap compressed candidate gen → high-precision rerank). Same pattern as Faiss IVF-PQ + rescore. Architectural reference for TardigradeDB's eventual disk-tier path | next-iteration `tdb-retrieval/` — long-term reference |
| **ScaNN / Anisotropic VQ** (Guo et al., ICML 2020) | [PMLR](http://proceedings.mlr.press/v119/guo20h/guo20h.pdf) | SOTA on MIPS via anisotropic vector quantization (different precision along different directions). Production-validated; informs the principle that retrieval-aware quantization beats blind uniform quantization | next-iteration `tdb-retrieval/` redesign |
| **Faiss Index Selection Guidelines** | [Faiss wiki](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index) | Production guidance: SQ8 (INT8 scalar) is the default low-bit floor; PQ/OPQ for further compression always paired with a rescoring stage. Confirms the floor for in-storage retrieval scoring is INT8, not Q4 | next-iteration `tdb-retrieval/` redesign |
| **Pinecone — Product Quantization explainer** | [pinecone.io/learn](https://www.pinecone.io/learn/series/faiss/product-quantization/) | Plain-language overview of PQ economics: up to 97% compression at usable recall; always rescored. Useful in-codebase reference for explaining the two-tier pattern | next-iteration docs |
| **Qdrant — Quantization docs** | [qdrant.tech](https://qdrant.tech/documentation/manage-data/quantization/) | Production system uses SQ8 (INT8) as default; Binary Quantization (BQ) is opt-in with rescoring. Reports SQ8 retains <1% recall loss, BQ ~5% with rescore. Validates INT8 as the safe retrieval floor | next-iteration `tdb-retrieval/` redesign |
| **Qdrant — Binary Quantization** | [qdrant.tech/articles](https://qdrant.tech/articles/binary-quantization/) | Companion to the docs above; explains the always-rescore pattern that makes sub-INT8 quantization viable | reference |
| **Hugging Face — Embedding Quantization** | [HF blog](https://huggingface.co/blog/embedding-quantization) | Empirical recall retention benchmarks: FP16 99-100%, INT8 97-99%, binary 85-95% (with rescore). The empirical floor for retrieval scoring is INT8 with per-channel scaling; below that, structured compression (PQ/centroids) is required | next-iteration `tdb-retrieval/` redesign |
| **Vespa — Scaling ColPali to billions** | [Vespa blog](https://blog.vespa.ai/scaling-colpali-to-billions/) | Production deployment of disk-tier per-token retrieval (ColPali = ColBERT-style on visual embeddings). Confirms the "offload neural representation to disk, only touch in second-phase ranking" pattern at scale. Architectural reference for the path TardigradeDB needs after the immediate fix | next-iteration architecture |
| **llama.cpp Q4_0 format discussion** | [GitHub discussion](https://github.com/ggml-org/llama.cpp/discussions/1121) | Definitive reference on Q4_0 internals (group of 32 floats, one f32 scale, 4-bit signed [-7, 7]). Confirms our diagnosis of the mechanism, and confirms the format was designed for *weight* quantization in inference, not retrieval over activations | `crates/tdb-storage/src/quantization.rs` docstrings (planned update) |
| **Brenndoerfer — INT4 group-wise quantization tutorial** | [mbrenndoerfer.com](https://mbrenndoerfer.com/writing/int4-quantization-group-wise-nf4-format-llms) | Explicit explanation of the *"outliers dominate the scale; non-outlier values lose precision"* failure mode that crushed our recall. Plain-language reference for the bug we hit | `docs/perf/ingestion-roadmap.md` (planned update) |

**Architectural conclusion from this body of work.** The right storage format for per-token retrieval is **INT8 group quantization paired with a per-channel scaling sidecar** (SmoothQuant-for-storage pattern). Q4_0 is appropriate for transport/archival but not for scoring. Centroid + residual (ColBERTv2 PLAID) is the long-term direction once corpus exceeds ~5M cells. These references gate the post-OOM-fix work: the next retrieval-storage redesign defers to this section's recommendations.

### A3g. LLM-Agent Memory Retrieval Architecture (2026-05-14 audit)

Commissioned after the 2026-05-14 bench audit (`docs/experiments/2026-05-14-bench-audit.md`) measured TardigradeDB at 29.62% LoCoMo / 3.14% LongMemEval (deterministic, clean data). Research synthesis lives in `docs/refs/retrieval-architecture-research-2026-05-14.md`; this table captures the gating citations. Together these sources show: (a) production LLM-agent memory systems are mostly hybrid (dense + BM25 + reranker) or BM25-only with LLM-agentic search — zero use raw decoder-LM hidden states; (b) raw hidden-state retrieval at sub-1B model scale has no published support; (c) for LongMemEval, indexing strategy dominates retriever architecture as a recall lever.

| Paper / Source | arXiv / Link | What it justifies | Where cited |
|---|---|---|---|
| **LongMemEval** (Wu et al., ICLR 2025) | [arXiv:2410.10813](https://arxiv.org/abs/2410.10813) | Reports baseline R@5 = 0.683 (flat-bm25) / 0.732 (flat-stella 1.5B) / 0.762 (Contriever + fact-expansion, paper's best). Paper's headline finding: **indexing strategy dominates retriever choice**. Switching from BM25 to a 7B dense retriever moves R@5 by ~5-8 points; round-level chunking + fact-augmented keys + time-aware query expansion together move it by ~15 points. TardigradeDB at 3.14% on this benchmark is two orders of magnitude below the worst baseline — gating evidence that the bottleneck is indexing, not retriever architecture | retrieval architecture research (2026-05-14); planned staged debug |
| **Zep (Graphiti) paper** | [arXiv:2501.13956](https://arxiv.org/abs/2501.13956) | Production hybrid retrieval pipeline: dense embedding (1024-d cosine) + BM25 full-text + direct graph traversal, no LLM during retrieval, then graph reranker. +18.5% accuracy and 90% latency reduction vs MemGPT on DMR. Reference production architecture | retrieval architecture research (2026-05-14) |
| **PromptReps** | [arXiv:2404.18424](https://arxiv.org/abs/2404.18424) | Most direct counter-evidence to TardigradeDB's foundational bet: explicitly reports **dense embeddings from raw LM hidden states underperform sparse representations on document retrieval**; best results from hybrid (dense + sparse) systems. Forces honest framing of the "raw hidden states as retrieval signal" thesis | retrieval architecture research (2026-05-14) |
| **LLM2Vec** (McGill, ICLR 2024 spotlight) | [arXiv:2404.05961](https://arxiv.org/abs/2404.05961) | Decoder LLMs become strong retrievers only after (1) bidirectional attention, (2) masked next-token training, (3) unsupervised contrastive learning — MTEB 56.8 unsupervised SOTA. Establishes the contemporary recipe for getting retrieval signal from decoder hidden states; the recipe is *trained*, not raw | retrieval architecture research (2026-05-14); reference for future projection-head experiments |
| **RepLLaMA / Llama2Vec** | [arXiv:2312.15503](https://arxiv.org/abs/2312.15503) | LLaMA-2 fine-tuned for retrieval reaches BEIR average 56.4 (zero-shot SOTA). Confirms that fine-tuned decoder LMs are competitive retrievers — but only after retrieval-specific training | retrieval architecture research (2026-05-14) |
| **Native Retrieval Embeddings from LLM Agent Hidden States** | [arXiv:2603.08429](https://arxiv.org/abs/2603.08429) | Adds a lightweight projection head on top of agent hidden states for native retrieval. Supports the *direction* of TardigradeDB's bet (hidden states are usable) but explicitly requires a trained projection — not raw signal | retrieval architecture research (2026-05-14); planned Phase 3 of debug plan |
| **Causal2Vec** | [arXiv:2507.23386](https://arxiv.org/abs/2507.23386) | Decoder-only LLM embedding via retrieval-specific training. Same pattern as LLM2Vec / RepLLaMA | retrieval architecture research (2026-05-14) |
| **BEIR** (Thakur et al., NeurIPS 2021) | [arXiv:2104.08663](https://arxiv.org/abs/2104.08663) | Canonical zero-shot retrieval benchmark. Findings: in-domain dense beats BM25 by 7-18 nDCG; out-of-domain BM25 is competitive or better than most dense models trained on MS MARCO; best zero-shot is late-interaction (ColBERT) or BM25 + cross-encoder reranking. Hybrid lifts BEIR aggregate nDCG@10 from 43.42 (BM25) to 52.59 (~21% relative). Establishes "hybrid as zero-shot safe default" | retrieval architecture research (2026-05-14) |
| **Cormack et al. — Reciprocal Rank Fusion** (SIGIR 2009) | [cormack.uwaterloo.ca/cormacksigir09-rrf.pdf](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) | Original RRF paper. Outperforms Condorcet fusion and individual rankers; the zero-tuning safe default | retrieval architecture research (2026-05-14) |
| **Bruch et al. — Analysis of Fusion Functions** (TOIS 2023) | [arXiv:2210.11934](https://arxiv.org/abs/2210.11934) | **Tuned convex combination (CC) of lexical and semantic scores OUTPERFORMS RRF** in both in-domain and out-of-domain. CC needs only a small tuning set; RRF discards score distribution information. Implication for TardigradeDB: if/when we add a second retrieval signal, prefer tuned CC over RRF | retrieval architecture research (2026-05-14); planned Phase 2 of debug plan |
| **Furnas et al. — The Vocabulary Problem** (CACM 1987) | [dl.acm.org/doi/10.1145/32206.32212](https://dl.acm.org/doi/abs/10.1145/32206.32212) | Names the classic failure mode: lexical-overlap dominance overrides semantic relevance. Identical underlying phenomenon to TardigradeDB's LongMemEval failure (top-K dominated by within-session chunks sharing surface vocabulary, not the chunk with the actual answer) | retrieval architecture research (2026-05-14) |
| **RankRAG** (NeurIPS 2024) | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/db93ccb6cf392f352570dd5af0a223d3-Paper-Conference.pdf) | LLM rewrites the user's vague conversational query into a retrieval-shaped form before scoring. Architectural reference for "query-side fix" — analogous to the time-aware query expansion LongMemEval recommends. Already partially implemented in TardigradeDB as RLS strategies (now quarantined; could be revived with retrieval-shape rewriting rather than synonym expansion) | retrieval architecture research (2026-05-14) |
| **Survey of Conversational Search** | [arXiv:2410.15576](https://arxiv.org/html/2410.15576v1) | Broad survey of conversational IR; documents the "conversational distractor problem" (within-session noise drowning out the relevant turn) that LongMemEval's session-vs-round chunking ablation specifically addresses | retrieval architecture research (2026-05-14) |
| **MemPalace benchmarks** | [mempalace.tech/benchmarks](https://www.mempalace.tech/benchmarks) | Reports 96.6% LongMemEval. Architecture less public, but reportedly relies on simpler retrieval + heavier LLM reasoning rather than hybrid embedding fusion. Counter-evidence to "hybrid is the universal answer" | retrieval architecture research (2026-05-14) |
| **Cognaptus — ByteRover architecture analysis** | [cognaptus.com](https://cognaptus.com/blog/2026-04-05-memory-rewritten-why-byterover-kills-the-pipeline-and-maybe-saves-agents/) | Third-party teardown of ByteRover's "kill the pipeline" architecture (BM25 + LLM-agentic, no embeddings). Useful as a counter-narrative to dense-embedding orthodoxy | retrieval architecture research (2026-05-14) |

**Architectural conclusion from this body of work.** TardigradeDB's foundational bet — raw decoder-LM hidden states as a sufficient retrieval signal at sub-1B model scale — has no published support and at least one direct counter-result (PromptReps). The honest 2026-05-14 numbers (LoCoMo 29.62%, LongMemEval 3.14%) are consistent with that absence of evidence. *However*, the immediate gap to published baselines (e.g. BM25's 68% R@5 on LongMemEval) is so large that the bottleneck is almost certainly indexing strategy, not retriever architecture. The staged debug plan at `docs/superpowers/plans/2026-05-14-retrieval-debug-plan.md` derives directly from this section's findings.

### A4. Agent Memory Systems

| Paper | arXiv / DOI | What it justifies | Where cited |
|---|---|---|---|
| **MemoryOS** (BAI-LAB) | arXiv:2506.06326 | Hierarchical memory (short/mid/long-term); FIFO paging, dynamic information movement; 49.1% F1 improvement on LoCoMo | `docs/refs/AI Agentic Memory System Efficiency.md`, `docs/refs/AI-db-discussion.md` |
| **A-Mem** | NeurIPS 2025 virtual/2025/poster/119020 | Zettelkasten-inspired agentic memory; dynamic node organization; shows SOTA still doesn't beat simple RAG on MemoryBench consistently | `docs/refs/AI Agentic Memory System Efficiency.md`, `docs/refs/AI-db-discussion.md`, `docs/experiments/multi-memory-injection.md` |
| **Mem0 2025** | arXiv:2504.19413 | Production memory system: 26% better than OpenAI memory, 91% lower p95 latency, 90% token savings on LoCoMo; multi-scope model (user/agent/session/app); actor-aware memory for hallucination contagion prevention | `docs/refs/AI Agentic Memory System Efficiency.md` |
| **"From Prompt-Response to Goal-Directed Systems"** | arXiv:2602.10479 | Agentic AI architecture framework; context for evolution toward stateful memory-equipped systems | `docs/refs/AI Agentic Memory System Efficiency.md` |
| **Cost and accuracy of long-term graph memory in distributed LLM-based multi-agent systems** | arXiv:2601.07978 | Trade-offs in distributed graph memory; cost/accuracy analysis for graph-based approaches | `docs/refs/AI Agentic Memory System Efficiency.md` |
| **Kelle** | arXiv:2510.16040 | KV caching on eDRAM for edge devices; hardware-aware KV persistence; eviction and hardware layout optimization | `docs/competitors/competitors-search-2.md` |
| **SimpleMem** (Liu et al., 2026) | [arXiv:2601.02553](https://arxiv.org/html/2601.02553v1) | Three-stage pipeline: entropy-aware filtering → recursive memory consolidation (merges related units with dedup) → adaptive query-aware retrieval. Multi-view indexing (dense + BM25 + metadata). +26.4% F1, 30× token reduction. Omni-SimpleMem (April 2026) achieves SOTA on LoCoMo (F1=0.613) and Mem-Gallery (F1=0.810). Key insight: consolidation *reduces* redundancy rather than adding competing entries | `experiments/multiview_diagnosis.py` |
| **ENGRAM** (2025) | [arXiv:2511.12960](https://arxiv.org/abs/2511.12960) | Typed memory partitioning (episodic/semantic/procedural) with per-type top-k retrieval then merge with dedup. Beats full-context baseline by +15 points on LongMemEval using ~1% of tokens. Key insight: typed partitioning prevents cross-type competition — directly relevant to preventing view-canonical competition in TardigradeDB | `experiments/multiview_diagnosis.py` |
| **LiCoMemory** (2025) | [arXiv:2511.01448](https://arxiv.org/abs/2511.01448) | CogniGraph: lightweight hierarchical graph using entities and relations as semantic indexing layers, with temporal + hierarchy-aware search and unified reranking. +26.6pp on multi-session, +20.7pp on temporal reasoning subsets of LongMemEval. Architecture: session → entity-relation → chunk levels | `experiments/multiview_diagnosis.py` |
| **MemOS** (MemTensor, 2025) | [arXiv:2505.22101](https://arxiv.org/abs/2505.22101) | Memory Operating System: elevates memory to first-class operational resource. Three-layer architecture (API / scheduling+management / storage+infrastructure). Three memory types: parametric, activation, plaintext. Governance: scheduling, layering, permission control, exception handling. Confirms architectural direction of treating memory as a managed OS resource | `experiments/multiview_diagnosis.py` |
| **MemoryOS** (BAI-LAB, EMNLP 2025 Oral) | [GitHub](https://github.com/BAI-LAB/MemoryOS) | Three-level storage: short/mid/long-term. FIFO paging, dynamic information movement between levels | `experiments/multiview_diagnosis.py` |
| **Hindsight** (Vectorize.io + Virginia Tech, 2026) | [arXiv:2512.12818](https://arxiv.org/abs/2512.12818) | TEMPR: four parallel retrievals (semantic vector + BM25 keyword + entity graph traversal + temporal filtering) fused. 89.6% LoCoMo (OSS-120B), 91.4% LongMemEval (Gemini-3 Pro). Four memory networks: world facts, agent experiences, entity summaries, evolving beliefs. Key insight: no single retrieval modality covers all query types | vague-query research (2026-05-11) |

### A5. Additional arXiv Citations (refs documents)

The following appear by ID in `docs/refs/AI Agentic Memory System Efficiency.md` and
`docs/refs/AI-db-discussion.md`. Not all have been read fully; included for
completeness.

- arXiv:2510.17281v4 — LoCoMo benchmark paper
- arXiv:2603.07670v1 — Agent memory survey
- arXiv:2512.13564 — Memory evaluation frameworks
- arXiv:2602.16313v1 — Memory evaluation metrics
- arXiv:2312.11970v1 — Multi-session memory architecture
- arXiv:2408.04948 — Retrieval or composition related
- arXiv:2506.07398 — H-MEM: agentic memory design (cited in `docs/refs/AI-db-discussion.md` design evolution)
- arXiv:2507.03608v1 — Graph-based memory systems
- arXiv:2507.22925 — G-Memory: memory system architecture (cited in `docs/refs/AI-db-discussion.md`)
- arXiv:2511.16131v1 — Advanced memory architectures
- arXiv:2402.01763v3 — Transformer memory architecture papers
- arXiv:2604.01707v1 — Memory in the LLM Era: Modular Architectures and Strategies [benchmark + analysis]
- arXiv:2605.03675 — MemTier: Tiered Memory Architecture and Retrieval Bottleneck Analysis for Long-Running Agents
- arXiv:2603.04814v1 — Beyond the Context Window: Cost-Performance Analysis of Fact-Based Memory vs Long-Context LLMs

### A6. Neuroscience Foundation

| Source | What it justifies | Where cited |
|---|---|---|
| **"Cognition without Consciousness"** (Nature, doi:10.1057/s41599-024-03611-3) | Episodic vs semantic memory split grounded in neuroscience; theoretical basis for KV bank (episodic, fast, specific) vs synaptic bank (semantic, slow, generalizing) | `docs/refs/AI-db-discussion.md` |

---

## B. Systems, Frameworks & Libraries

### B1. Production LLM Serving

| System | Version tested | Architectural decision it informed |
|---|---|---|
| **vLLM** | 0.19.1 | KV Connector v1 API integration; prefix-cache architecture; `build_connector_meta` DTO pattern; fingerprint lifecycle verified against `kv_transfer_utils.py:56`, `example_connector.py:332`, `scheduler.py:1823`; **critical discovery**: `base.py:451-480` confirms v1 API is prefix-cache-only — no mechanism for cross-prompt KV injection |
| **SGLang** | — | RadixAttention investigated; ruled out — prefix-cache only, no cross-prompt KV injection possible. Closed Path 3 deployment option |
| **llama.cpp** | — | Q4_0 quantization scheme (`crates/tdb-storage/src/quantization.rs`); GGUF model file format for `GGUFModelResolver` |
| **HuggingFace Transformers** | — | `past_key_values` API for KV cache capture and injection; `generate()` auto-handles position ID offsetting for RoPE; zero-copy PyO3 NumPy interop |

### B2. Agent Memory Competitors

| System | LoCoMo score | What TardigradeDB takes / rejects |
|---|---|---|
| **ByteRover 2.0** | 92.2% | AKL algorithm (taken, adapted to tensors instead of text files); 5-tier progressive retrieval (fuzzy text → keyword → LLM-driven); Context Tree (Domain→Topic→Subtopic→Entry) as file-based knowledge graph; **does not use vector search** — structured text retrieval only; even with Gemini Flash hits 90.9% — architecture, not model, is the driver |
| **MemOS 2.0 (Stardust)** | ~75.8 | Commercial "Memory OS" with +43.7% accuracy over OpenAI Memory, ~35% token savings; graph-structured memory store; explicit OpenClaw plugin. pypi.org/project/MemoryOS/ — confirmed competitor, tensor-native approach is TardigradeDB's differentiation |
| **Zep (Graphiti)** | 75.14% | Temporal knowledge graph concept (noted); ~104% higher CPU than Mem0 — cost concern |
| **Letta (MemGPT)** | 74.0% | Tiered memory concept (Core/Recall/Archival); architectural lock-in via MemFS rejected |
| **Mem0** | 66.9% / 68.4% (graph) | Multi-scope identity model (user/agent/session/app); 21 framework integrations; production baseline |
| **LangMem** (LangChain) | — | Native LangChain orchestration integration; **rejected for real-time use** — independent evaluations record 59.82s p95 search latency; restricted to offline/batch-mode operations only (`docs/refs/AI Agentic Memory System Efficiency.md`) |
| **AgentCore** (AWS) | — | AWS long-term memory deep dive; commercial managed memory layer for agents; confirms enterprise demand but cloud lock-in (`docs/refs/AI Agentic Memory System Efficiency.md`, footnote 4) |
| **OpenClaw** | 100K+ GitHub stars | File-based memory validated at scale; rejected Redis/Pinecone in favor of local optimization |
| **Genesys-Memory** | — | Causal graph with pruning (text-based); Trace graph in TardigradeDB is the tensor-native equivalent |
| **SpaceTimeDB** | — | Database-as-execution-runtime pattern; WASM reducers inside kernel; specific lessons from local code exploration (commit `d5c1738c1`, 2026-04-21): durability boundary trait, commitlog actor pattern (bounded async queue + batched writes), confirmed-reads contract, incremental subscription engine |
| **LMCache** | — | Persistent cross-request KV reuse; infrastructure-level, not cognitive — confirms TardigradeDB's differentiated position |
| **RelayCaching** | — | Cross-agent decode-to-prefill KV reuse (~4.7× TTFT reduction); different problem domain |
| **HiAgent** | — | Hierarchical working-memory manager; subgoal-level summarization (text-based) |
| **Cognee** | — | AI memory evaluation tools; comparison point |
| **Hindsight / Memobase / Memvid / Recallio** | — | Named in competitive comparisons; limited architectural influence |

### B3. Infrastructure & Storage

| System | Architectural role |
|---|---|
| **TiKV** | Concurrency patterns (crossbeam-epoch); Rust + CUDA integration model |
| **Neon (Postgres proxy)** | Rust + CUDA integration pattern reference |
| **PyO3** | Rust-Python bindings; zero-copy NumPy float32 arrays (`PyReadonlyArray1`) |
| **Crossbeam** | Lock-free concurrency via epoch-based reclamation |
| **SafeTensors** | Import/export format; rejected as primary storage (100MB header cap, no append, O(n) parse) |
| **Criterion** | Rust benchmarking framework for all Criterion bench suites |
| **PyTorch** | Model inference in experiments; RTX 3070 Ti, CUDA 12.8 |
| **llama-cpp-python** | Python bindings for llama.cpp; used by `LlamaCppHook` to extract final-layer embeddings from GGUF models. Retrieval-only path (llama.cpp does not expose KV cache externally); informed the dual-store split between HF (full KV) and GGUF (embedding-only) paths | `.claude/plans/llama-cpp-hook-plan.md` |
| **Ollama** | GGUF blob manifest resolution; `GGUFModelResolver` resolves `"llama3.2:3b"` → `/path/to/blob` via Ollama manifest files; used alongside LM Studio as the two primary GGUF distribution formats | `.claude/plans/llama-cpp-hook-plan.md` |
| **LM Studio** | Alternative GGUF model path format; supported in `GGUFModelResolver` alongside Ollama manifest paths | `.claude/plans/llama-cpp-hook-plan.md` |
| **BGE-Reranker** (BAAI, 2024) | Cross-encoder reranker (`BAAI/bge-reranker-v2-m3`, ~278M params, multilingual). Architecture: query+document concatenated as `[CLS] query [SEP] doc [SEP]`, single scalar relevance score. Fast enough for CPU on small batches, single-GPU for larger. Available as a drop-in via `CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3")`. arXiv:2402.03216 | `python/tardigrade_hooks/reranker.py` (configurable) |
| **MiniLM** (Wang et al., NeurIPS 2020) | 6-layer / 22M-parameter distilled transformer architecture. The default cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is built on this — small enough to run interactively on CPU, sub-50ms on GPU for <100 query/doc pairs. arXiv:2002.10957. Chosen over BGE-Reranker as the default because the latency budget for an in-the-loop reranker is tight | `python/tardigrade_hooks/reranker.py` (default model) |
| **Sentence-BERT / sentence-transformers** (Reimers & Gurevych, EMNLP 2019) | Library + training methodology that produced both the bi-encoders and cross-encoders we consume. The `sentence_transformers.CrossEncoder` API is what `CrossEncoderReranker` wraps. arXiv:1908.10084 | `python/tardigrade_hooks/reranker.py` |
| **MS MARCO Passage Ranking** (Bajaj et al., 2018) | Training corpus for `cross-encoder/ms-marco-MiniLM-L-6-v2`. ~1M Bing search queries with crowd-judged relevance labels over 8.8M passages. Defines "passage relevance" the way the reranker learned to score it — short, factual, English. Caveat: does NOT match agent-memory-style queries 1:1 (memos are first-person diary text, not web passages); empirical validation on TardigradeDB's own corpus is the source of truth. arXiv:1611.09268 | `python/tardigrade_hooks/reranker.py` (training-data caveat) |
| **Rerankers and Two-Stage Retrieval** (Pinecone, 2023) | Reference implementation pattern: bi-encoder first stage retrieves top-K, cross-encoder reranks. Validates the architectural split we use — bi-encoder (latent K vectors) wide-net, cross-encoder narrow-pass over the candidates that have memo text | `python/tardigrade_hooks/reranker.py` |
| **Hybrid Search: BM25 Still Wins** (TianPan.co, April 2026) | [tianpan.co](https://tianpan.co/blog/2026-04-12-hybrid-search-production-bm25-dense-embeddings) | Production analysis showing BM25 + dense hybrid lifts recall@10 from 65-78% (single modality) to 91%. RRF fusion takes 6ms. BM25 catches exact entities; dense catches semantic similarity. Complementary signals | vague-query research (2026-05-11) |
| **Hybrid Search Guide** (Supermemory, April 2026) | [supermemory.ai](https://blog.supermemory.ai/hybrid-search-guide/) | Practical guide to hybrid search for agent memory: adaptive RRF fusion, IDF-based per-query weight adjustment, temporal decay inspired by Ebbinghaus forgetting curve | vague-query research (2026-05-11) |
| **intfloat/e5-small-v2** | Traditional embedding RAG comparison baseline throughout all scale experiments; `passage: ...` / `query: ...` encoding convention; achieved 100% recall@1-5 on 100-memory corpus and 5/10 multi-hop retrieval (vs TardigradeDB 0/10) — key calibration point for retrieval quality assessment | `docs/experiments/kv-cache-validation.md`, `docs/experiments/multi-memory-injection.md` |

### B4. Vector & Graph Databases (Compared, Not Used)

| System | Why not used |
|---|---|
| **Pinecone** | Rejected by OpenClaw and TardigradeDB; embedding-based, no KV-native memory |
| **Qdrant** | Standard vector store; supported by Mem0; not tensor-native |
| **Neo4j** | Used by Zep (Graphiti); ~104% higher CPU cost than Mem0; replaced by custom Trace graph |
| **Chroma** | Standard vector store; supported by Mem0 |
| **Redis** | Distributed cache; rejected by OpenClaw in favor of local optimization |

### B5. Educational/Reference Projects

| Source | What it taught |
|---|---|
| **Jay Alammar "The Illustrated Transformer"** | Transformer mechanics foundation |
| **Andrej Karpathy "Let's build GPT from scratch" (YouTube)** | GPT implementation details |
| **Apache Kafka** | Event streaming patterns for high-velocity agent state |
| **Apache Cassandra** | Distributed state management at scale |
| **PostgreSQL** | Reference architecture: Postgres + vector DB = what TardigradeDB replaces |
| **Model Context Protocol (MCP)** | Open standard for LLM tool communication; TardigradeDB exposes 7 MCP tools |
| **Obsidian (PKM tool)** | Backlink-density concept: notes with more backlinks are more important/interconnected. Cited in `docs/experiments/multi-memory-injection.md` as direct inspiration for Trace-Boosted Retrieval alongside PageRank |
| **AGENTS.db** | Layered context system for AI agents; cited in `docs/refs/AI-db-discussion.md` as part of the design evolution toward KV-native memory; different architectural approach (context layering) vs TardigradeDB's tensor-native memory |

---

## C. Named Algorithms & Data Structures

### C1. Implemented in TardigradeDB

| Algorithm / Structure | Crate / File | Source |
|---|---|---|
| **Vamana graph** (single-layer small-world, angular diversity pruning) | `crates/tdb-index/src/vamana/` | DiskANN paper (Subramanya et al., 2019) |
| **Medoid seeding** (entry point = centroid-nearest node, updated incrementally) | `crates/tdb-index/src/vamana/mod.rs:229` | DiskANN paper; standard graph index entry point strategy for small-world graphs |
| **Q4_0 quantization** (4-bit INT, scale+zero-point per block) | `crates/tdb-storage/src/quantization.rs` | llama.cpp (`ggerganov/llama.cpp`) |
| **INT8 symmetric quantization** (SLB hot path) | `crates/tdb-retrieval/src/slb.rs` | Standard; NEON SDOT intrinsics |
| **NEON SIMD dot product** (vmull_s8, vpadalq_s16, vaddvq_s32) | `crates/tdb-retrieval/src/simd_distance.rs` | ARM NEON intrinsics |
| **AVX2 INT8 dot product** (_mm256_maddubs_epi16 / _mm256_madd_epi16) | `crates/tdb-retrieval/src/simd_distance.rs` | x86_64 AVX2 intrinsics; 16x throughput over scalar path on x86_64; complementary to NEON path |
| **Top5Avg scoring** (mean of top-5 per-token dot products) | `crates/tdb-retrieval/` | Custom — TardigradeDB-native, no external citation |
| **Semantic Lookaside Buffer (SLB)** (INT8 hot cache, CPU TLB analogy) | `crates/tdb-retrieval/src/slb.rs` | TLB concept from CPU architecture |
| **LRU eviction** (OrderedDict with domain-invariant capacity = max_num_seqs) | `python/tardigrade_vllm/connector.py` | Classic; capacity bound from vLLM `scheduler_config.max_num_seqs` |
| **Write-Ahead Log (WAL)** | `crates/tdb-index/` | Standard database engineering |
| **Mark-Sweep GC** (segment compaction, live ratio threshold 50%) | `crates/tdb-storage/` | Classic CS; applied to quantized segment files |
| **Adaptive Knowledge Lifecycle (AKL)** (ι scoring, tier hysteresis, decay r=exp(-Δt/τ)) | `crates/tdb-governance/` | ByteRover 2.0 (arXiv:2604.01599), adapted to tensors |
| **Epoch-based reclamation** | via crossbeam | Crossbeam crate (lock-free GC for concurrent data structures) |
| **RoPE position remapping** | `python/tardigrade_hooks/` | RoFormer (Su et al., 2021) |
| **GQA (Grouped Query Attention) head expansion** | `crates/tdb-retrieval/src/` | Qwen3-0.6B model spec; K head expansion vs Q averaging |
| **Attention sink skip** (position-0 token causes recall cliff) | `crates/tdb-retrieval/src/` | Discovered empirically; 96.7% → 3.3% recall without skip |
| **Trace-Boosted Retrieval** (link-count score boost: `final_score = retrieval_score * (1 + link_count * 0.3)`) | `python/tardigrade_hooks/` | PageRank-inspired (linked pages rank higher) + Obsidian (backlink-dense notes are more important); hard plateau at boost_factor=0.3 — any non-zero value fixes the same 3 queries |
| **PageANN** (page-node Vamana, disk-aligned cold storage) | `README.md` (future roadmap) | PageANN research; designed as disk-aware extension of Vamana for billion-scale cold storage; not yet implemented |
| **Mean-centered rescoring** (subtract corpus K mean from query + stored vectors before dot product) | `crates/tdb-retrieval/src/refinement.rs::mean_centered_rescore` | Whitening-the-embedding-space approach; cheap version of BERT-flow (Li et al., EMNLP 2020 arXiv:2011.05864) — does mean subtraction only, skipping the normalizing-flow pass. Empirically validated: +31pp on moderate-tier R@5 (28% → 59%) with zero regression on specific (`docs/experiments/vague_queries/results.md`) |
| **Latent PRF / Rocchio in K-space** (`q' = α·q + β·centroid(top_K')` then re-retrieve) | `crates/tdb-retrieval/src/refinement.rs::latent_prf` | Rocchio (1971); generalized to dense retrievers by Yu et al. CIKM 2021 (arXiv:2108.13454); per-token validation by ColBERT-PRF (arXiv:2106.11251). Tested 8 hyperparameter configurations, none usable for our corpus geometry — see C2 below |
| **Cross-encoder reranker** (Stage-2 over text-bearing candidates) | `python/tardigrade_hooks/reranker.py::CrossEncoderReranker` | MiniLM (arXiv:2002.10957) trained on MS MARCO (arXiv:1611.09268), wrapped via sentence-transformers `CrossEncoder` (arXiv:1908.10084). Two-stage retrieval pattern (Pinecone 2023). Default model `cross-encoder/ms-marco-MiniLM-L-6-v2`; user can swap to BGE-Reranker via constructor param |

### C2. Evaluated but Rejected

| Algorithm | Reason for rejection |
|---|---|
| **HNSW** (Hierarchical Navigable Small World) | Fails for attention retrieval due to Q/K distribution shift in latent space |
| **IVF** (Inverted File Index) | Not chosen; Vamana preferred for disk-locality and small-world navigation |
| **BM25** (lexical search) | Noted as hybrid retrieval option; not implemented |
| **Mean-pooled hidden states for injection** | Catastrophically broken: hidden states live in d_model space, KV cache in head_dim space — category error. Recall: 26× to 829× improvement when switching to full per-token KV |
| **ColBERT-style cosine sum-of-max** (`cosine_sum_max` scorer) | Tested on 100-memory corpus: 53.3% R@5 — **worse** than per_head_max (63.3%) and far below Top5Avg (100%). Late interaction over stored hidden states does not improve on simpler max-sim aggregation for this retrieval task (`docs/experiments/kv-cache-validation.md`) |
| **Zettelkasten auto-linking on store** (A-MEM pattern, NeurIPS 2025) | Phase 33 experiment: latent hidden-state similarity measures topic similarity, not event identity. Auto-linking at threshold=200-250 produced 99 within-domain noise links and only 8/20 correct cross-reference pairs; final accuracy 30% — *worse* than explicit `store_linked()` at 70%. Architecture decision: the agent provides intelligence about what to link; the engine stores the decision. (`docs/experiments/multi-memory-injection.md`) |
| **K\*K retrieval** (query with K, store K) | K vectors share a massive common component across all sequences (~4000 cross-sentence dot product at non-sink positions vs ~200 content-specific signal difference). Confirmed bad by FIER, ShadowKV, "From QKV to K/KV" — symmetric attention cannot represent directional query-to-memory relationships |
| **HyDE** (Hypothetical Document Embeddings; Gao et al., ACL 2023, arXiv:2212.10496) | Generate a hypothetical answer with an LLM, embed it, retrieve. Effective for vocabulary-mismatched/vague queries in text-based RAG, but **rejected for TardigradeDB's retrieval path**: requires an extra LLM forward pass per query (500–2000ms per the dev community, e.g., dev.to/aarjay_singh "Why I stopped putting LLMs in my agent memory retrieval path"). The agent IS the LLM — having it generate a hypothetical inside its own retrieval loop is architecturally backwards and breaks the agent-step latency budget. Latent-space PRF (Rocchio in K-space) achieves the same vocabulary-bridging effect with no LLM call |
| **Cross-encoder reranking on stored text** (BGE-Reranker, MiniLM as a primary stage) | Operates on text. TardigradeDB's primary stored unit is KV tensors, not text. Memo text in `text_store` is optional and often absent. Making vague-query handling depend on memo text would split the retrieval contract. Kept as a **future optional Stage-3** when memos are present (see B1 BGE-Reranker entry); rejected as the primary fix |
| **ZCA whitening, token reweighting, multi-layer fusion** (latent-space geometric transforms) | All three tested 2026-05-11 on 10-fact Sonia corpus: 0% improvement on any vague query in any configuration or combination. Same 3 queries miss in every config. The vocabulary-mismatch failures require world-knowledge reasoning that no geometric transform can provide — confirmed by DeepMind's LIMIT theoretical proof (ICLR 2026). Architecture preserved (RefinementStrategy trait, TokenWeighter, MultiLayerQuery) but does not improve recall |
| **RLS embedding expansion** (nearest-neighbor vocabulary bridge from model's embedding table) | Tested 2026-05-11 on full LoCoMo (1,542 items): 67.2%→67.2% (0% improvement). Embedding neighbors capture lexical similarity ("athletic"→"athletics"), not conceptual ("athletic"→"ultramarathon"). The vocabulary gap in LoCoMo requires reasoning-level knowledge that shallow embedding proximity cannot provide. Architecture sound (EmbeddingExpansionStrategy), synonym source insufficient |
| **Rule-based multi-view consolidation** (views as competing packs in same index) | Tested 2026-05-11: 3 rule-based framings (summary/question/paraphrase) stored as separate packs. Moderate R@5 dropped from 80%→20% — views dilute top-k. Question views near-identical across facts (cos~0.75), crowd out canonicals. Doc2Query-- predicted this: uncontrolled/low-diversity expansions harm retrieval. Fix requires either (a) dual-index fusion (Doc2Query++), (b) parent-document pattern (HyPE), or (c) LLM-powered views + quality filter |
| **Text-based BM25 + RRF hybrid** (as primary stage) | Same text dependency as cross-encoder reranking. Adds value only when memos exist and are descriptive. RRF (Cormack 2009) is still adopted as a fusion mechanism for the latent-space PRF stage in case PRF drifts (see A2) — we use the rank-fusion math without the BM25 sparse signal |
| **Aggressive confidence gating for RLS** (score-ratio threshold 1.10) | Tested 2026-05-12: RLS confidence gate (`top1_score / top2_score >= 1.10`) blocked nearly all queries from reaching the reformulation strategies, rendering keyword/embedding/generative RLS ineffective regardless of strategy quality. Literature confirms (CRAG, TARG, FLARE): healthy reformulation systems trigger on 20-40% of queries; 0% means miscalibrated thresholds. For the LLM agent reformulation experiment, bypassed the gate entirely (`confidence_threshold=0.0`) to establish the ceiling per DMQR-RAG's recommendation. Gate will be re-calibrated to a proper retrieval budget if the experiment shows positive results |

### C3. Architectural Patterns Named in Plans & Docs

| Pattern | Applied to |
|---|---|
| **Adapter** | `RetrievalKeyView`/`RetrievalKeyAdapter` (Rust retrieval key plan); `LlamaCppHook` translating llama-cpp-python API; `HuggingFaceHook` |
| **Data Transfer Object (DTO)** | `_TardigradeConnectorMetadata` crossing scheduler→worker IPC boundary in vLLM connector |
| **Strategy** | `RetrievalKeyStrategy` ABC (embedding table, last token, raw K); `Quantizer` trait (Q4/Q8/FP16); `RefinementStrategy` trait (NoOp, MeanCentered, Whitening, LatentPrf) |
| **Decorator** | `TokenWeighter` (per-token importance weighting wraps scoring function) |
| **Composite** | `MultiLayerQuery` (fuses N engine queries via RRF) |
| **Template Method** | `TardigradeHook` ABC; retrieval key `compute_for_save` unified interface |
| **Factory Method with Fallback Chain** | `_get_embed_weights`: safetensors → hf_hub_download → AutoModel.from_pretrained |
| **Bounded Cache (LRU with domain invariant)** | `_pack_id_by_fingerprint` map bounded by `max_num_seqs` |
| **Chain of Responsibility** | Three-stage retrieval pipeline: SLB → PerTokenRetriever → BruteForceRetriever |
| **Memento** | Engine crash recovery; state rebuild from durable sources |
| **Active Object** | `MaintenanceWorker` background governance sweep |
| **Value Object** | `RetrievalKeyView` parsed key representation |
| **Specification** | ATDD acceptance tests encoding which stage receives raw tokens vs pooled vectors |
| **Template Method (Benchmarks)** | Consistent benchmark flow across 100/1K/10K corpora |
| **Object Mother / Fixture Builder** | Test fixture factories for broad-match, spike, malformed-header, Vamana-threshold |
| **Schmitt Trigger (Hysteresis)** | Tier promotion/demotion thresholds to prevent oscillation (from control theory) |
| **Monitor Object** | Python Engine wrapped in `Arc<Mutex<>>` with GIL release via `py.detach()` |
| **Observer** | Hook lifecycle (on_generate, on_prefill) |
| **PageRank** (Brin & Page, 1998) | Cited as direct inspiration for Trace-Boosted Retrieval: memories with more trace links rank higher, analogous to pages with more inbound links. `docs/experiments/multi-memory-injection.md` (Phase 32) |

---

## D. Benchmarks & Evaluation Datasets

### D1. External Benchmarks (Used for Competitive Comparison)

| Benchmark | Origin | What was measured | TardigradeDB status |
|---|---|---|---|
| **LoCoMo** (Long-term Conversational Memory) | ACL 2024 | Ultra-long conversations (300-600 turns, 9K-26K tokens); BLEU/F1/LLM Score | Internal 25-item sample run (2026-04-22): Tardigrade 1.0000 vs Letta 0.2002 |
| **LongMemEval** | — | Long-term memory retention across thousands of documents | Internal 25-item sample run (2026-04-22): Tardigrade 1.0000 avg score |
| **MemoryBench** | — | Long-horizon continual learning; shows SOTA doesn't consistently beat simple RAG | Used to calibrate synthetic fact test methodology |
| **MemoryArena** | — | Memory evaluation framework | Mentioned in refs |
| **MemoryAgentBench** | — | Agent memory benchmark | Mentioned in refs |
| **PrefEval-10** | — | Preference evaluation (Mem0 reports gains) | Mentioned in refs |
| **PersonaMem** | — | Persona memory evaluation | Mentioned in refs |
| **2026 AI Index (Stanford HAI)** | Stanford HAI | Benchmark integrity analysis; "jagged intelligence"; up to 64% ground truth error rate in LoCoMo | Informs caution about benchmarking claims |

### D2. Internal Experiments (TardigradeDB-Native)

| Experiment | Key result | File |
|---|---|---|
| **100-memory corpus** | 100% R@5 with Top5Avg + hidden states; 97ms latency | `docs/experiments/kv-cache-validation.md` |
| **5000-memory scale** | 100% R@1-R@5; 3.2s GPU latency (model inference dominates from 2K→5K) | `docs/experiments/README.md` |
| **Vague query retrieval** | Specific: 100% R@5; moderate/vague: ~46% R@5 (100 queries/tier); cliff = vocabulary overlap not vagueness | `docs/experiments/README.md` |
| **Cross-model retrieval** | Same-family (Qwen 0.6B→1.7B): 90% R@5 via linear projection; cross-family (Qwen→GPT-2): 76.7% via MLP adapter (~400K params); Orthogonal Procrustes ceiling ~47% cross-family | `docs/experiments/README.md` |
| **Synthetic gibberish facts** | 9/10 recall on Qwen3-0.6B (100% ratio vs text RAG); novel knowledge transfer proven | `docs/experiments/kv-cache-validation.md` |
| **Multi-memory injection** | Sequential (Knowledge Packs paper) works; naive concatenation fails; RoPE position corruption not the primary cause | `docs/experiments/multi-memory-injection.md` |
| **SGLang investigation** | Confirmed prefix-cache only; no cross-prompt KV injection path | `docs/experiments/sglang-investigation.md` |
| **Tardigrade vs Letta benchmark** (internal, 2026-04-22) | 50-item sample (25 LoCoMo + 25 LongMemEval), deterministic lexical evaluator: Tardigrade `1.0000` vs Letta `0.2002`; latency 7.44ms vs 81.10ms. Smoke fixture also validated (3 repeats, quality tie). Full matrix invalid (Letta ingest failures 2042/2042). | `docs/bench/v1-results.md` |
| **Scorer comparison** | ColBERT cosine_sum_max: 53.3%; per_head_max Q*K: 63.3%; hidden states + Top5Avg: 100%. K*K causes gravity well (cross-sentence dot product ~4000, content signal ~200) | `docs/experiments/kv-cache-validation.md` |
| **RAG baseline comparison** | intfloat/e5-small-v2 achieves 100% recall@1-5 on 100 memories (30 positive queries); multi-hop: RAG 5/10, TardigradeDB 0/10 (latent hidden states miss second-hop entity names) | `docs/experiments/kv-cache-validation.md`, `docs/experiments/multi-memory-injection.md` |
| **Trace-Boosted Retrieval sweep** | Hard plateau: any boost_factor > 0 fixes the same 3/9 failures; optimal = 0.3; 5 failures immune to boosting (background memories scored higher) | `docs/experiments/multi-memory-injection.md` |
| **Multi-view consolidation diagnosis** | Rule-based views (summary/question/paraphrase) **degrade** moderate R@5 from 80%→20% by diluting the result set. Root cause: question-framing views ("What did Sonia translated do?") are near-identical across facts (cos~0.75 between question views of different facts), crowding top-k with views from wrong facts. Paraphrase views cos=0.99 to canonical (no retrieval value). File ingest 100% R@3 (works perfectly). Three fix architectures identified: dual-index (Doc2Query++), parent-document (HyPE), LLM-powered views + quality filter | `experiments/multiview_diagnosis.py`, `experiments/file_ingest_and_multiview_experiment.py` |

---

## E. Blog Posts & Articles

### E1. Architecture-Influencing Articles

| Article | Source | Architectural impact |
|---|---|---|
| **"LLM-Native Databases: What They Are and When to Use Them"** | alexsmale.com | Conceptual framing for LLM-native database design |
| **"Why Vector Databases Aren't Enough for Real AI Memory and What to Do Instead"** | blog.recallio.ai | Vector DB limitations; motivation for tensor-native approach |
| **"ByteRover: Agent-Native Memory"** | byterover.dev/blog | AKL algorithm deep dive; 92.2% LoCoMo methodology |
| **"Architecture Deep Dive: ByteRover CLI 2.0"** | byterover.dev/blog | Open-source memory system architecture details |
| **"Milvus: We Extracted OpenClaw's Memory System (memsearch)"** | milvus.io/blog | Markdown-based memory at scale; validated file-based approach |
| **"Inside OpenClaw: How Its Memory Architecture Powers Self-Hosted AI Agents"** | ubos.tech | OpenClaw tripartite architecture (channel, brain, body) |
| **"How Agentic AI Can Strain Modern Memory Hierarchies"** | The Register, 2026-01-28 | Hardware and infrastructure realities of persistent agent memory |
| **"Persistent Architectural Memory cut our Token Costs by ~55%"** | Reddit r/PromptEngineering | Empirical token savings from architectural memory |
| **"Nature: Cognition without Consciousness"** | Nature (doi:s41599-024-03611-3) | Neuroscience grounding for episodic/semantic memory split |

### E2. Competitive Analysis Articles

| Article | Source |
|---|---|
| **"Best AI Agent Memory Systems in 2026"** | vectorize.io |
| **"Mem0 vs Letta (MemGPT): AI Agent Memory Compared"** | vectorize.io |
| **"Mem0 vs Zep (Graphiti): AI Agent Memory Compared"** | vectorize.io |
| **"AI Agent Memory Systems in 2026" (Yogesh Yadav)** | blog.devgenius.io |
| **"Memory in Agents: What, Why, and How"** | mem0.ai/blog |
| **"Agentic RAG vs Traditional RAG"** | mem0.ai/blog |
| **"Build Persistent Memory for Agentic AI" (Mem0 + ElastiCache + Neptune)** | aws.amazon.com/blogs/database |
| **"Building Smarter AI Agents: AgentCore Long-Term Memory Deep Dive"** | aws.amazon.com/blogs/machine-learning |
| **"Letta v1 Agent Release"** | letta.com/blog |
| **"Benchmarking AI Agent Memory"** | byterover.dev/blog |
| **"AI Memory Tools Evaluation"** | cognee.ai/blog |
| **"The Problem with Comparing AI Memory System Benchmarks"** | Reddit r/MachineLearning |
| **"AI Agent Memory: Building Stateful AI Systems"** | redis.io/blog |
| **"The Agent Stack: Memory, Tools, Schedulers, and Observability That Scale"** | LinkedIn (Alex Smale) |
| **"Designing Memory Architectures for Production-Grade GenAI Systems"** | Medium (Avijit Swain) |
| **"A Practical Guide to Memory for Autonomous LLM Agents"** | Towards Data Science |
| **"Reverse Engineering Latest ChatGPT Memory Feature"** | agentman.ai/blog |
| **"Agentic AI 2025: Coordination, Memory, Path to Maturity"** | LinkedIn (Ersin Yumer) |
| **"Memory in AI: What Separates Agents from Chatbots in 2025"** | LinkedIn (Deepak Kamboj) |
| **"FreeCodeCamp: How to Build and Secure a Personal AI Agent with OpenClaw"** | freecodecamp.org |
| **"OpenClaw: Agentic AI in the Wild - Architecture, Adoption, and Emerging Security Risks"** | acronis.com |
| **"Hybrid vs RAG vs Vector"** | glean.com/blog |
| **"Why Hybrid RAG"** | memgraph.com/blog |
| **"How AI Agents Remember Things: Vector Stores in LLM Memory"** | freecodecamp.org |
| **"Long-Term Memory for LLMs Using Vector Store"** | dev.to (Einar Cesar) |
| **LangChain Blog: "Continual Learning for AI Agents"** | blog.langchain.com |

### E3. Educational & General

| Article | Source |
|---|---|
| **Jay Alammar "The Illustrated Transformer"** | jalammar.github.io |
| **Andrej Karpathy "Let's build GPT from scratch"** | YouTube |
| **OpenAI "Memory and New Controls for ChatGPT"** | openai.com/index |
| **OpenAI "Introducing ChatGPT Agent"** | openai.com/index |
| **OpenAI Developers "Session Memory Cookbook"** | developers.openai.com/cookbook |
| **Microsoft AutoGen Ecosystem Docs** | microsoft.github.io/autogen |
| **Hacker News: "Using the Agent Framework to Expand LLM Capabilities"** | news.ycombinator.com (id=44827101) |
| **Reddit r/AI_Agents "What is your full AI Agent stack in 2026?"** | reddit.com |
| **Reddit r/ClaudeAI: "Built with Claude Project Showcase Megathread"** | reddit.com |

---

## F. GitHub Repositories Cited

| Repository | Cited for |
|---|---|
| `github.com/ggerganov/llama.cpp` | Q4_0 quantization scheme; GGUF format spec |
| `github.com/vllm-project/vllm` | KV Connector v1 API; `ExampleConnector` reference implementation; `base.py:451-480` prefix-cache-only contract |
| `github.com/mem0ai/mem0` | Production memory orchestration reference |
| `github.com/letta-ai/letta` | Tiered memory architecture; MemFS design |
| `github.com/HiAgent2024/HiAgent` | Hierarchical working-memory for agents |
| `github.com/IAAR-Shanghai/Awesome-AI-Memory` | Curated AI memory resources list |
| `github.com/punkpeye/awesome-mcp-servers` | MCP server collection |
| `byterover.dev/blog` (assumed repo) | AKL algorithm reference |
| `github.com/MemTensor/MemOS` | MemOS 2.0 source; OpenClaw plugin and memory OS architecture |
| `github.com/aiming-lab/SimpleMem` | SimpleMem: parallel multi-view agent memory retrieval; Omni-SimpleMem multimodal extension |
| `github.com/Shichun-Liu/Agent-Memory-Paper-List` | Curated paper list for "Memory in the Age of AI Agents" survey (arXiv:2512.13564) |
| `github.com/DEEP-PolyU/Awesome-GraphMemory` | Survey of graph-based agent memory: taxonomy, techniques, applications |
| `github.com/BAI-LAB/MemoryOS` | MemoryOS (EMNLP 2025 Oral): three-level storage for personalized AI agents |
| `github.com/parthsarthi03/raptor` | RAPTOR: recursive abstractive processing for tree-organized retrieval |
| `github.com/NirDiamant/RAG_Techniques` | RAG techniques collection including HyPE implementation notebook |
| `github.com/google-deepmind/limit` | LIMIT dataset: stress test for embedding-based retrieval (ICLR 2026) |
| `github.com/DCI-Agent/DCI-Agent-Lite` | Direct Corpus Interaction agent: retrieval via grep/file reads instead of embeddings |

---

## G. Key Architectural Decisions Traceable to External Sources

This section maps each major TardigradeDB design decision to its external validation.

| Decision | External source | Confidence |
|---|---|---|
| Brute-force SIMD matmul beats ANN at <10K blocks | MemArt (arXiv:2409.17264) | High — cited directly |
| KV injection = text-in-prompt for causal transformers | Knowledge Packs (arXiv:2604.03270) | High — cited directly |
| Vamana over HNSW for latent-space retrieval | DiskANN paper + internal observation re: Q/K distribution shift | Medium — principle from paper, distribution shift reasoning is internal |
| AKL importance scoring, tier hysteresis, τ=30d decay | ByteRover 2.0 (arXiv:2604.01599) | High — cited directly, adapted |
| Q4_0 storage format over safetensors | llama.cpp + safetensors limitations documented in CLAUDE.md | High |
| Sequential multi-memory injection (not concatenation) | Knowledge Packs (arXiv:2604.03270) | High |
| Per-token hidden states (not mean-pooled) for retrieval key | Empirical discovery (96.7%→100% recall); MemArt supports | Medium — empirical + literature |
| Attention sink skip (position 0) | Empirical discovery (96.7%→3.3% recall without skip) | High — empirical, not from paper |
| **Q\*K over K\*K for retrieval** | Empirically discovered (K*K gravity well, ~4000 common component); confirmed by FIER, ShadowKV, "From QKV to K/KV" — all use Q for queries because K*K produces symmetric attention that can't capture directional relationships | High — empirical + literature consensus |
| Top5Avg scoring over other aggregations | Empirical discovery: Hidden states + top5_pair_avg = 100%; no external paper validates the specific heuristic | Low — **no external citation** — internal heuristic |
| Episodic (KV bank) + Semantic (synaptic bank) split | PRIME (arXiv:2507.04607) + Nature neuroscience paper | Medium |
| GQA head expansion for Qwen3 | Qwen3 model spec + empirical observation | High |
| vLLM connector fingerprint identity via block_indices[0] | vLLM source verification: `kv_transfer_utils.py:56`, `example_connector.py:332`, `scheduler.py:1823` | High — verified against framework source |
| Bounded LRU at max_num_seqs | vLLM `scheduler_config.max_num_seqs` invariant + domain reasoning | High |
| **vLLM KV Connector v1 is prefix-cache only** (no cross-prompt injection possible) | vLLM source `base.py:451-480`: contract documentation + `scheduler.py` tracing + synthetic-fact A/B test (cold and primed byte-identical) | High — verified against source + experiment |
| MLP adapter over Orthogonal Procrustes for cross-family retrieval | Procrustes empirically plateaus at ~47% R@5; MLP achieves 76.7% on same Qwen→GPT-2 corpus | High — empirical ablation |
| Agent provides link intelligence; engine stores the decision | Phase 33 experiment: auto-linking via hidden-state similarity failed (30% accuracy vs 70% with explicit links); confirmed by field (Mem0, Cognee, Hindsight all require LLM extraction or trained probes for entity linking) | High — empirical + competitive validation |
| **Multi-view consolidation: views as retrieval keys, not competing results** | Doc2Query-- (arXiv:2301.03266) documents saturation/dilution failure; Doc2Query++ (arXiv:2510.09557) proposes dual-index fusion; HyPE (SSRN:5139335) shows LLM-generated questions as retrieval-only keys with parent resolution; ENGRAM (arXiv:2511.12960) shows typed partitioning prevents cross-type competition; multi-view diagnosis experiment (2026-05-11) confirmed rule-based views degrade moderate R@5 from 80%→20% via index dilution | High — empirical + strong literature consensus |
| **Latent-space retrieval has a theoretical ceiling for vocabulary-mismatched queries** | DeepMind LIMIT paper (ICLR 2026, arXiv:2508.21038) proves vector-space retrieval is bounded by sign-rank of the relevance matrix. Experimentally confirmed: whitening, token reweighting, multi-layer fusion all 0% improvement on the 3 persistently-missed vague queries. The remaining failures require world-knowledge reasoning ("ultramarathons are athletic events"), not geometric transformation. DCI (arXiv:2605.05242) shows agentic corpus interaction bypasses this ceiling. | High — theoretical proof + empirical confirmation on TardigradeDB |
| **Latent-space PRF (Rocchio in K-space) over HyDE/cross-encoder for vague-query refinement** | Three-way constraint analysis: (1) HyDE adds 500–2000ms LLM call per query, breaks agent-loop budget; (2) cross-encoder/BM25 require memo text which is optional in TardigradeDB; (3) Rocchio (1971) generalized to dense vectors by Yu et al. CIKM 2021 (arXiv:2108.13454) and validated for late-interaction by ColBERT-PRF (arXiv:2106.11251) — pure latent-space, no LLM, no text dependency, operates on K vectors already stored. Empirical validation pending the refinement implementation | Medium — strong literature foundation; empirical confirmation pending |

---

## H. Known Gaps — Decisions Without External Validation

These design choices currently have no external citation. They may be correct but
would need justification if TardigradeDB is published academically or critically
reviewed:

1. **Top5Avg scoring heuristic** — Taking the mean of the top-5 per-token dot
   products. No paper validates this aggregation function vs alternatives (mean-all,
   max, sum, BM25-style term frequency weighting). Ablation study needed.

2. **τ=30d half-life for recency decay** — Adopted from ByteRover 2.0's AKL but the
   original paper does not explain how τ was chosen. No literature validates this
   specific value for tensor-based memories.

3. **Tier thresholds (ι: 65/85 for promotion, 35/60 for demotion)** — Same source
   issue. ByteRover 2.0 uses similar thresholds for text memories; no validation that
   these transfer to KV tensors.

4. **50% live-ratio threshold for segment compaction** — Chosen as the GC trigger.
   No external citation; may be overly conservative or aggressive depending on workload.

5. **match_threshold=150 for connector retrieval** — Acknowledged in the connector
   hardening plan as needing recalibration after retrieval key unification. Currently
   tuned for the old (broken) mean-pooled key strategy.

6. **Cross-model retrieval MLP adapter (~400K params)** — Validated empirically at
   76.7% R@5 (Qwen→GPT-2). No literature on optimal adapter size for KV-space
   projection between model families.

7. **arXiv IDs for FIER, ShadowKV, and "From QKV to K/KV"** — These three papers
   are cited in `docs/experiments/kv-cache-validation.md` as confirming the Q*K > K*K
   decision, but their full citations and arXiv IDs do not appear in any codebase doc.
   If TardigradeDB is published or peer-reviewed, these need to be tracked down and
   formally cited.

8. **PageANN paper citation** — CLAUDE.md and README.md both reference
   "PageANN-inspired" page-node Vamana as the cold-path design target, but the
   specific PageANN paper/authors are not cited anywhere in the codebase.

---

*Generated: 2026-05-01. Updated: 2026-05-12. Sources: codebase + docs + `.claude/plans/` +
Resumancer session journal (40 entries, 2026-04 tardigrade-db branch) + Codex/Claude sessions
2026-01 through 2026-05. May 2026 updates: multi-view consolidation diagnosis (20 refs);
vague-query research (ExpandR, SoftQE, vstash, Hindsight, ByteRover architecture); latent-space
transforms (LaSER, DEBATER, KV Packet, AdaQR); fundamental limitations (DeepMind LIMIT ICLR 2026,
DCI-Agent); adaptive retrieval gating (CRAG, Self-RAG, TARG, DMQR-RAG, RAG-Fusion, FLARE, WRRF).
Total: 47+ new references across sessions.*
