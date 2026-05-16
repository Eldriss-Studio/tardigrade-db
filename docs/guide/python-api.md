# Python API Reference

## TardigradeClient

The recommended high-level entry point. Facade pattern that wraps Engine, FileIngestor, MemoryConsolidator, and the query path behind a unified API.

### Constructor

```python
TardigradeClient(
    db_path,
    *,
    tokenizer=None,
    owner=1,
    kv_capture_fn=None,
    vamana_threshold=9999,
)
```

- `db_path` — `str | Path`. Directory for persistent storage; engine is created internally.
- `tokenizer` — tokenizer with `.encode()` / `.decode()`. Required for real KV capture.
- `owner` — owner ID for multi-agent isolation (default: 1).
- `kv_capture_fn` — `(chunk_text, tokenizer) -> (key, layer_payloads)`. If `None`, uses a random-key stub (testing only).
- `vamana_threshold` — engine's Vamana ANN activation threshold.

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `store(fact_text, *, salience=80.0)` | `int` (pack_id) | Store a single fact as a KV pack |
| `query(query_text, *, k=5)` | `list[dict]` | Retrieve top-k packs |
| `ingest_text(text, *, document_id=None, chunk_size=512)` | `IngestResult` | Chunk and ingest a text document |
| `ingest_file(path, *, document_id=None, chunk_size=512)` | `IngestResult` | Read a file and ingest it |
| `consolidate(pack_id)` | `int` | Attach multi-view retrieval keys to one pack; returns views attached |
| `consolidate_all()` | `dict[int, int]` | Consolidate all eligible packs; returns `{pack_id: views_attached}` |
| `list_packs()` | `list[dict]` | All packs for this owner |
| `pack_count()` | `int` | Number of packs for this owner |
| `engine` | `Engine` | Direct access to the underlying `tardigrade_db.Engine` |

---

## KnowledgePackStore

Full end-to-end injection via HuggingFace models. Use when you need KV cache injection directly into `model.generate()`.

### Constructor

```python
KnowledgePackStore(engine, model, tokenizer, owner=1, query_layer=None)
```

- `engine` — pre-built `tardigrade_db.Engine` instance
- `model` — HuggingFace causal LM
- `tokenizer` — matching tokenizer with chat template
- `owner` — owner ID (default: 1)
- `query_layer` — hidden layer index for retrieval keys (default: 67% of model depth)

### Storage Methods

#### `store(fact_text, salience=80.0)`

Store a fact as a KV cache pack. Returns the assigned `pack_id`.

```python
pack_id = kps.store("User prefers morning meetings before 10am")
```

#### `store_and_link(fact_text, related_pack_id, salience=80.0)`

Store a fact and link it to an existing memory.

```python
existing = kps.store("Went to bookstore in Pilsen")
kps.store_and_link("The bookstore is called Casa Azul", existing)
```

#### `store_linked(facts, salience=80.0)`

Store multiple related facts and link them all to each other.

```python
kps.store_linked([
    "Lucia's instructor is Tomoko",
    "Tomoko drives a Honda Civic",
])
```

#### `forget(pack_id)`

Delete a memory permanently.

### Retrieval + Injection Methods

#### `generate(query_text, **gen_kwargs)`

Retrieve the best matching memory, inject its KV cache, generate a response.

Returns `(generated_text, prompt_tokens, had_memory)`.

#### `generate_with_trace(query_text, k=1, composer=None, boost_factor=0.3, **gen_kwargs)`

Retrieve with trace-boosted scoring, follow trace links, compose multiple packs, inject, generate.

#### `retrieve_and_inject(query_text)`

Lower-level: retrieve and build DynamicCache without generating.

Returns `(cache, query_ids, attention_mask)` or `(None, query_ids, None)`.

#### `generate_multi(query_text, k=3, composer=None, **gen_kwargs)`

Retrieve k packs and compose them. No trace link following.

---

## Reflective Latent Search (RLS)

Agentic retrieval loop that reformulates queries when the initial retrieval is not confident.

### ReflectiveLatentSearch

```python
ReflectiveLatentSearch(
    engine,
    model,
    tokenizer,
    query_layer,
    hidden_size,
    owner=1,
    k=5,
    strategies=None,         # default: [KeywordExpansionStrategy()]
    confidence_threshold=1.10,
    max_attempts=2,
)
```

The loop: RETRIEVE → EVALUATE → REFORMULATE → RE-RETRIEVE → FUSE.

Confidence is measured as `score[0] / score[1]`. If the ratio is below `confidence_threshold`, RLS iterates through the configured strategies and fuses results with RRF.

#### `query(question, top_k=None) → list[MemoryCellHandle]`

Run the full RLS loop. Returns list of `MemoryCellHandle`.

```python
from tardigrade_hooks.rls import ReflectiveLatentSearch, KeywordExpansionStrategy

rls = ReflectiveLatentSearch(
    engine, model, tokenizer,
    query_layer=16, hidden_size=1024,
    strategies=[KeywordExpansionStrategy()],
)
handles = rls.query("What outdoor activities does this person enjoy?")
```

### Reformulation Strategies

All strategies implement `ReformulationStrategy.reformulate(query_text) -> list[str]`.

| Class | Requires | Description |
|-------|----------|-------------|
| `KeywordExpansionStrategy()` | nothing | Extract content words, expand via synonym table |
| `MultiPhrasingStrategy()` | nothing | Template-based variants (keyword-only + WH-question form) |
| `EmbeddingExpansionStrategy(tokenizer, embed_weights, top_k=10)` | embedding table | Nearest-neighbor lookup in the model's own vocabulary |
| `GenerativeReformulationStrategy(model, tokenizer, max_new_tokens=40)` | local LLM | Small model rephrases the query (e.g. Qwen2.5-3B) |
| `LLMAgentReformulationStrategy(api_key, model=None)` | external API | Calls DeepSeek (or any OpenAI-compatible API) for vocabulary-bridged reformulations |

Cost order (lowest → highest): keyword < multiphrasing < embedding < generative < agent.

### RRF Fusion

```python
from tardigrade_hooks.rls import rrf_fuse_handles

fused = rrf_fuse_handles(handle_lists, k=60)
```

Fuses multiple `MemoryCellHandle` lists via Reciprocal Rank Fusion, deduplicating by `cell_id`.

### Constants (`tardigrade_hooks.constants`)

| Constant | Value | Meaning |
|----------|-------|---------|
| `RLS_DEFAULT_CONFIDENCE_THRESHOLD` | `1.10` | Min score ratio before RLS re-retrieves |
| `RLS_DEFAULT_MAX_ATTEMPTS` | `2` | Max reformulation iterations |
| `RLS_MODE_NONE` | `"none"` | No reformulation |
| `RLS_MODE_KEYWORD` | `"keyword"` | KeywordExpansionStrategy |
| `RLS_MODE_MULTIPHRASING` | `"multiphrasing"` | MultiPhrasingStrategy |
| `RLS_MODE_EMBEDDING` | `"embedding"` | EmbeddingExpansionStrategy |
| `RLS_MODE_GENERATIVE` | `"generative"` | GenerativeReformulationStrategy |
| `RLS_MODE_AGENT` | `"agent"` | LLMAgentReformulationStrategy |

---

## CrossEncoderReranker

Stage-2 re-ranking over text-bearing candidates using a cross-encoder model.

```python
from tardigrade_hooks.reranker import CrossEncoderReranker

reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",  # 22M params
)

reranked = reranker.rerank(
    query_text="What does Zara do for work?",
    candidates=handles,
    get_text=lambda h: engine.pack_text(h.cell_id),
)
```

Requires candidates to have associated text (stored via `mem_write_pack(..., text=...)` or `set_pack_text()`). ~30% latency overhead vs retrieval alone (~86ms vs ~67ms p95 on Qwen3-0.6B).

---

## File Ingestion

### TextChunker

Token-bounded chunker with configurable overlap.

```python
from tardigrade_hooks.chunker import TextChunker

chunker = TextChunker(
    tokenizer,
    max_tokens=512,    # DEFAULT_CHUNK_TOKENS
    overlap_tokens=64, # CHUNK_OVERLAP_TOKENS
    min_tokens=32,     # MIN_CHUNK_TOKENS
)

chunks = chunker.chunk(text)  # list of Chunk(text, token_count, start_char, end_char)
```

### FileIngestor

Ingests a text document as sequential KV memory packs. Consecutive chunks are linked via `Supports` edges.

```python
from tardigrade_hooks.file_ingestor import FileIngestor

ingestor = FileIngestor(
    engine,
    tokenizer=tokenizer,
    owner=1,
    chunker=chunker,       # optional; uses TextChunker(512) by default
    salience=70.0,         # DEFAULT_FILE_INGEST_SALIENCE
    kv_capture_fn=fn,      # (chunk_text, tokenizer) -> (key, layer_payloads)
)

result = ingestor.ingest(text, document_id="readme")
# IngestResult(pack_ids=[1, 2, 3], chunk_count=3, edge_count=2, document_id="readme")

result = ingestor.ingest_file("/path/to/doc.txt")
```

### IngestResult

```python
@dataclass
class IngestResult:
    pack_ids: list[int]
    chunk_count: int
    edge_count: int
    document_id: str | None
```

---

## Multi-view Consolidation v2

The parent-document pattern: the canonical pack stores the KV tensor; views are additional retrieval surfaces on the same fact, stored as linked packs.

### ViewGenerator

```python
from tardigrade_hooks.view_generator import ViewGenerator

gen = ViewGenerator(
    # All keyword-only (constructor uses `*` after `self`)
    model=None,                    # required for mode="llm"
    tokenizer=None,                # required for mode="llm"
    framings=("summary", "question", "paraphrase"),  # DEFAULT_VIEW_FRAMINGS
    mode="rule",                   # "rule" (no model) or "llm" (HyPE-style LLM questions)
)

views = gen.generate("Tomoko Nishida teaches swimming at the Pilsen aquatic center")
# Returns a list of view strings — one per framing — generated by the
# rule-based strategies (summary / question / paraphrase). Output shape and
# wording depend on the input text and the active framing set.
```

Available framing names: `"summary"`, `"question"`, `"paraphrase"`, `"llm_question"`.

### MemoryConsolidator

Tier-gated, idempotent multi-view attachment. Only consolidates packs at or above the configured minimum tier.

```python
from tardigrade_hooks.consolidator import MemoryConsolidator

consolidator = MemoryConsolidator(
    engine,
    owner=1,
    view_generator=gen,
    min_tier=1,    # CONSOLIDATION_MIN_TIER: Validated tier
)

n = consolidator.consolidate(pack_id)          # int: views attached to this pack
all_n = consolidator.consolidate_all(owner=1)  # dict[int, int]
```

### ConsolidationSweepThread

Background Active Object daemon that runs consolidation sweeps automatically.

```python
from tardigrade_hooks.consolidation_sweep import ConsolidationSweepThread

sweep = ConsolidationSweepThread(consolidator, interval_seconds=60)
sweep.start()
# ...
sweep.stop()
print(f"Total views attached: {sweep.views_attached}")
```

### Engine methods (multi-view)

| Method | Description |
|--------|-------------|
| `engine.add_view_keys(pack_id, keys)` | Attach additional retrieval keys to an existing canonical pack |
| `engine.view_count(pack_id)` | Number of views currently attached to a pack |

---

## Engine (Rust)

The low-level Rust engine exposed via PyO3.

```python
import tardigrade_db

engine = tardigrade_db.Engine("/path/to/storage")
```

### Core Methods

| Method | Description |
|--------|-------------|
| `mem_write(owner, layer, key, value, salience, parent_cell_id)` | Write a single cell |
| `mem_read(query_key, k, owner)` | Read top-k cells |
| `mem_write_pack(owner, retrieval_key, layer_payloads, salience, text=None)` | Write a multi-layer KV pack with optional fact text |
| `mem_read_pack(query_key, k, owner)` | Read top-k packs |
| `mem_read_pack_with_trace_boost(query_key, k, owner, boost_factor)` | Read with trace-boosted scoring |
| `mem_read_tokens(tokens, k, owner)` | Direct token-level retrieval. `tokens`: `np.ndarray` of shape `(n_tokens, d_model)` `float32`. `k`: top-k results. `owner`: optional owner filter. Returns the same `ReadResult` as `mem_read_pack`. Skips the Python encode/parse round-trip used by `mem_read_pack`. |

### Pack Management

| Method | Description |
|--------|-------------|
| `load_pack_by_id(pack_id)` | Load a pack directly by ID |
| `add_pack_link(pack_id_1, pack_id_2)` | Create durable trace link between packs |
| `add_pack_edge(pack_id_1, pack_id_2, edge_type)` | Create a typed edge (use constants: `EDGE_SUPPORTS`, `EDGE_CONTRADICTS`, etc.) |
| `pack_supports(pack_id)` / `pack_contradicts(pack_id)` | Query semantic edges |
| `pack_links(pack_id)` | All packs linked to a given pack |
| `pack_count()` | Total number of packs stored |
| `pack_importance(pack_id)` | Current importance score |
| `pack_text(pack_id)` | Get stored fact text (None if not stored) |
| `set_pack_text(pack_id, text)` | Set or update fact text |
| `delete_pack(pack_id)` | Permanently delete a pack |
| `add_view_keys(pack_id, keys)` | Attach additional retrieval keys (multi-view v2) |
| `view_count(pack_id)` | Views attached to a pack |

### Governance

| Method | Description |
|--------|-------------|
| `cell_importance(cell_id)` | Current importance score |
| `cell_tier(cell_id)` | Current tier (Draft/Validated/Core) |
| `advance_days(days)` | Simulate time passage for decay |
| `evict_draft_packs(owner)` | Remove all Draft-tier packs for an owner |

### Other

| Method | Description |
|--------|-------------|
| `cell_count()` | Total cells in engine |
| `trace_ancestors(cell_id)` | Get causal parent chain |
| `has_vamana()` | Whether ANN index is active |
| `status()` | Engine health + metrics dict |
| `compact()` | Trigger segment compaction (mark-sweep GC) |
| `refresh()` | Reload WAL + rebuild derived state |
| `set_refinement_mode(mode, **kwargs)` | Configure query-side refinement. `mode` is `"none"` (raw retrieval), `"centered"` (subtract corpus mean from query/keys before scoring), or `"prf"` (Rocchio-style pseudo-relevance feedback in K-space). See [`docs/experiments/vague_queries/results.md`](../experiments/vague_queries/results.md). |

### Edge type constants (`tardigrade_hooks.constants`)

```python
from tardigrade_hooks.constants import (
    EDGE_CAUSED_BY,   # 0
    EDGE_FOLLOWS,     # 1
    EDGE_CONTRADICTS, # 2
    EDGE_SUPPORTS,    # 3
)
```
