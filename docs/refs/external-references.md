# TardigradeDB: External References Audit

Complete catalog of every paper, blog post, algorithm, system, and benchmark that
influenced TardigradeDB's architecture or implementation. Sources: codebase, docs,
plans, `.remember/` buffer, `.claude/plans/`, and Codex sessions (2026-01 through
2026-04).

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
| **Orca** (Yu et al., 2022) | OSDI 2022 | Iteration-level scheduling for LLM inference; concurrent prefill/decode scheduling relevant to multi-agent session management | `docs/learning-roadmap.md` |
| **Pancake** | arXiv:2602.21477 | Multi-tier memory for multi-agent serving; ANN indices + cache tier + GPU/CPU placement optimization | `docs/competitors/competitors-search-2.md` |

### A3. Adapters & Weights-as-Memory

| Paper | arXiv / DOI | What it justifies | Where cited |
|---|---|---|---|
| **LoRA** (Hu et al., 2021) | arXiv:2106.09685 | Low-rank weight adaptation `W' = W + αBA`; frozen base + per-agent trainable rank-r matrices; foundation for SynapticBank design | `docs/learning-roadmap.md`, `crates/tdb-core/src/` |
| **LRAgent** | arXiv:2602.01053 | Shared base KV cache + per-agent LoRA adapter splitting; 3× memory savings in multi-agent serving; inspired synaptic bank + multi-agent isolation design | `docs/competitors/competitors-search-2.md`, `docs/learning-roadmap.md`, `CLAUDE.md` |
| **MemoryLLM (Apple)** | arXiv:2602.00398 | FFN parameters as token-indexed neural memory; "weights as memory" framing; not productized — confirms SynapticBank is novel | `docs/competitors/competitors-search-2.md`, `CLAUDE.md` |
| **PRIME** (Dual-Memory Personalization) | arXiv:2507.04607 | Episodic + semantic dual-memory architecture; slow weights (semantic) vs fast episodic memory; neuroscience-grounded split inspires KV bank (episodic) + synaptic bank (semantic) | `docs/competitors/competitors-search-2.md`, `docs/learning-roadmap.md` |
| **FwPKM** (Fast-weight Product Key Memory) | arXiv:2601.00671 | Dynamic fast-weight episodic memory updated during inference; alternative to fixed storage | `docs/competitors/competitors-search-2.md` |
| **Generalized Key-Value Memory** | arXiv:2203.06223 | Decoupled memory dimension from support vectors; external KV memory in hardware | `docs/competitors/competitors-search-2.md` |

### A4. Agent Memory Systems

| Paper | arXiv / DOI | What it justifies | Where cited |
|---|---|---|---|
| **MemoryOS** (BAI-LAB) | arXiv:2506.06326 | Hierarchical memory (short/mid/long-term); FIFO paging, dynamic information movement; 49.1% F1 improvement on LoCoMo | `docs/refs/AI Agentic Memory System Efficiency.md`, `docs/refs/AI-db-discussion.md` |
| **A-Mem** | NeurIPS 2025 virtual/2025/poster/119020 | Zettelkasten-inspired agentic memory; dynamic node organization; shows SOTA still doesn't beat simple RAG on MemoryBench consistently | `docs/refs/AI Agentic Memory System Efficiency.md`, `docs/refs/AI-db-discussion.md` |
| **Mem0 2025** | arXiv:2504.19413 | Production memory system: 26% better than OpenAI memory, 91% lower p95 latency, 90% token savings on LoCoMo; multi-scope model (user/agent/session/app); actor-aware memory for hallucination contagion prevention | `docs/refs/AI Agentic Memory System Efficiency.md` |
| **"From Prompt-Response to Goal-Directed Systems"** | arXiv:2602.10479 | Agentic AI architecture framework; context for evolution toward stateful memory-equipped systems | `docs/refs/AI Agentic Memory System Efficiency.md` |
| **Cost and accuracy of long-term graph memory in distributed LLM-based multi-agent systems** | arXiv:2601.07978 | Trade-offs in distributed graph memory; cost/accuracy analysis for graph-based approaches | `docs/refs/AI Agentic Memory System Efficiency.md` |
| **Kelle** | arXiv:2510.16040 | KV caching on eDRAM for edge devices; hardware-aware KV persistence; eviction and hardware layout optimization | `docs/competitors/competitors-search-2.md` |

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
- arXiv:2506.07398 — Agentic memory design
- arXiv:2507.03608v1 — Graph-based memory systems
- arXiv:2507.22925 — Memory system architecture
- arXiv:2511.16131v1 — Advanced memory architectures
- arXiv:2402.01763v3 — Transformer memory architecture papers

### A6. Neuroscience Foundation

| Source | What it justifies | Where cited |
|---|---|---|
| **"Cognition without Consciousness"** (Nature, doi:10.1057/s41599-024-03611-3) | Episodic vs semantic memory split grounded in neuroscience; theoretical basis for KV bank (episodic, fast, specific) vs synaptic bank (semantic, slow, generalizing) | `docs/refs/AI-db-discussion.md` |

---

## B. Systems, Frameworks & Libraries

### B1. Production LLM Serving

| System | Version tested | Architectural decision it informed |
|---|---|---|
| **vLLM** | 0.19.1 | KV Connector v1 API integration; prefix-cache architecture; `build_connector_meta` DTO pattern; fingerprint lifecycle verified against `kv_transfer_utils.py:56`, `example_connector.py:332`, `scheduler.py:1823` |
| **SGLang** | — | RadixAttention investigated; ruled out — prefix-cache only, no cross-prompt KV injection possible. Closed Path 3 deployment option |
| **llama.cpp** | — | Q4_0 quantization scheme (`crates/tdb-storage/src/quantization.rs`); GGUF model file format for `GGUFModelResolver` |
| **HuggingFace Transformers** | — | `past_key_values` API for KV cache capture and injection; `generate()` auto-handles position ID offsetting for RoPE; zero-copy PyO3 NumPy interop |

### B2. Agent Memory Competitors

| System | LoCoMo score | What TardigradeDB takes / rejects |
|---|---|---|
| **ByteRover 2.0** | 92.2% | AKL algorithm (taken, adapted to tensors instead of text files); 5-tier progressive retrieval |
| **Zep (Graphiti)** | 75.14% | Temporal knowledge graph concept (noted); ~104% higher CPU than Mem0 — cost concern |
| **Letta (MemGPT)** | 74.0% | Tiered memory concept (Core/Recall/Archival); architectural lock-in via MemFS rejected |
| **Mem0** | 66.9% / 68.4% (graph) | Multi-scope identity model (user/agent/session/app); 21 framework integrations; production baseline |
| **OpenClaw** | 100K+ GitHub stars | File-based memory validated at scale; rejected Redis/Pinecone in favor of local optimization |
| **Genesys-Memory** | — | Causal graph with pruning (text-based); Trace graph in TardigradeDB is the tensor-native equivalent |
| **SpaceTimeDB** | — | Database-as-execution-runtime pattern; WASM reducers inside kernel; informed embedded logic design |
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

---

## C. Named Algorithms & Data Structures

### C1. Implemented in TardigradeDB

| Algorithm / Structure | Crate / File | Source |
|---|---|---|
| **Vamana graph** (single-layer small-world, angular diversity pruning) | `crates/tdb-index/src/vamana/` | DiskANN paper (Subramanya et al., 2019) |
| **Q4_0 quantization** (4-bit INT, scale+zero-point per block) | `crates/tdb-storage/src/quantization.rs` | llama.cpp (`ggerganov/llama.cpp`) |
| **INT8 symmetric quantization** (SLB hot path) | `crates/tdb-retrieval/src/slb.rs` | Standard; NEON SDOT intrinsics |
| **NEON SIMD dot product** (vmull_s8, vpadalq_s16, vaddvq_s32) | `crates/tdb-retrieval/src/simd_distance.rs` | ARM NEON intrinsics |
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

### C2. Evaluated but Rejected

| Algorithm | Reason for rejection |
|---|---|
| **HNSW** (Hierarchical Navigable Small World) | Fails for attention retrieval due to Q/K distribution shift in latent space |
| **IVF** (Inverted File Index) | Not chosen; Vamana preferred for disk-locality and small-world navigation |
| **BM25** (lexical search) | Noted as hybrid retrieval option; not implemented |
| **Mean-pooled hidden states for injection** | Catastrophically broken: hidden states live in d_model space, KV cache in head_dim space — category error. Recall: 26× to 829× improvement when switching to full per-token KV |

### C3. Architectural Patterns Named in Plans & Docs

| Pattern | Applied to |
|---|---|
| **Adapter** | `RetrievalKeyView`/`RetrievalKeyAdapter` (Rust retrieval key plan); `LlamaCppHook` translating llama-cpp-python API; `HuggingFaceHook` |
| **Data Transfer Object (DTO)** | `_TardigradeConnectorMetadata` crossing scheduler→worker IPC boundary in vLLM connector |
| **Strategy** | `RetrievalKeyStrategy` ABC (embedding table, last token, raw K); `Quantizer` trait (Q4/Q8/FP16) |
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

---

## D. Benchmarks & Evaluation Datasets

### D1. External Benchmarks (Used for Competitive Comparison)

| Benchmark | Origin | What was measured | TardigradeDB status |
|---|---|---|---|
| **LoCoMo** (Long-term Conversational Memory) | ACL 2024 | Ultra-long conversations (300-600 turns, 9K-26K tokens); BLEU/F1/LLM Score | Not yet evaluated; used as competitor comparison point |
| **LongMemEval** | — | Long-term memory retention across thousands of documents | Not yet evaluated |
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
| **Cross-model retrieval** | Same-family (Qwen 0.6B→1.7B): 90% R@5 via linear projection; cross-family (Qwen→GPT-2): 76.7% via MLP adapter (~400K params) | `docs/experiments/README.md` |
| **Synthetic gibberish facts** | 9/10 recall on Qwen3-0.6B (100% ratio vs text RAG); novel knowledge transfer proven | `docs/experiments/kv-cache-validation.md` |
| **Multi-memory injection** | Sequential (Knowledge Packs paper) works; naive concatenation fails; RoPE position corruption not the primary cause | `docs/experiments/multi-memory-injection.md` |
| **SGLang investigation** | Confirmed prefix-cache only; no cross-prompt KV injection path | `docs/experiments/sglang-investigation.md` |

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
| `github.com/vllm-project/vllm` | KV Connector v1 API; `ExampleConnector` reference implementation |
| `github.com/mem0ai/mem0` | Production memory orchestration reference |
| `github.com/letta-ai/letta` | Tiered memory architecture; MemFS design |
| `github.com/HiAgent2024/HiAgent` | Hierarchical working-memory for agents |
| `github.com/IAAR-Shanghai/Awesome-AI-Memory` | Curated AI memory resources list |
| `github.com/punkpeye/awesome-mcp-servers` | MCP server collection |
| `byterover.dev/blog` (assumed repo) | AKL algorithm reference |

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
| Top5Avg scoring over Q\*K per-token | Empirical discovery; no external paper | Low — **no external citation** — internal heuristic |
| Episodic (KV bank) + Semantic (synaptic bank) split | PRIME (arXiv:2507.04607) + Nature neuroscience paper | Medium |
| GQA head expansion for Qwen3 | Qwen3 model spec + empirical observation | High |
| vLLM connector fingerprint identity via block_indices[0] | vLLM source verification: `kv_transfer_utils.py:56`, `example_connector.py:332`, `scheduler.py:1823` | High — verified against framework source |
| Bounded LRU at max_num_seqs | vLLM `scheduler_config.max_num_seqs` invariant + domain reasoning | High |

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

---

*Generated: 2026-05-01. Sources: codebase + docs + `.claude/plans/` +
`.remember/` buffer + Codex sessions 2026-01 through 2026-04.*
