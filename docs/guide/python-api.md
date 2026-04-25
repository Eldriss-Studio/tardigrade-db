# Python API Reference

## KnowledgePackStore

The main interface for storing, retrieving, and injecting memories.

### Constructor

```python
KnowledgePackStore(engine, model, tokenizer, owner=1, query_layer=None)
```

- `engine` — `tardigrade_db.Engine` instance
- `model` — HuggingFace causal LM (e.g., from `AutoModelForCausalLM.from_pretrained`)
- `tokenizer` — matching tokenizer with chat template
- `owner` — owner ID for multi-agent memory isolation (default: 1)
- `query_layer` — which hidden layer to use for retrieval keys (default: 67% of model depth)

### Storage Methods

#### `store(fact_text, salience=80.0, auto_link=True, auto_link_threshold=None)`

Store a fact as a KV cache pack. Wraps the fact in a chat template before computing KV.

Returns the assigned `pack_id` (integer).

```python
pack_id = kps.store("User prefers morning meetings before 10am")
```

#### `store_and_link(fact_text, related_pack_id, salience=80.0)`

Store a fact and link it to an existing memory. Use when learning a new detail about something already remembered.

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

### Single-Memory Retrieval + Injection

#### `generate(query_text, **gen_kwargs)`

Retrieve the best matching memory, inject its KV cache, generate a response.

Returns `(generated_text, prompt_tokens, had_memory)`.

```python
text, tokens, had_memory = kps.generate(
    "When should we schedule the review?",
    max_new_tokens=100, do_sample=False,
)
```

#### `retrieve_and_inject(query_text)`

Lower-level: retrieve and build DynamicCache without generating.

Returns `(cache, query_ids, attention_mask)` or `(None, query_ids, None)`.

### Multi-Memory Retrieval (Trace-Linked)

#### `generate_with_trace(query_text, k=1, composer=None, boost_factor=0.3, **gen_kwargs)`

Retrieve with trace-boosted scoring, follow trace links, compose multiple packs, inject, generate.

```python
text, tokens, had = kps.generate_with_trace(
    "What car does Lucia's instructor drive?",
    k=1, max_new_tokens=100, do_sample=False,
)
```

#### `retrieve_with_trace(query_text, k=1, composer=None, boost_factor=0.3)`

Lower-level: retrieve with trace hop, build composed DynamicCache.

### Multi-Memory Retrieval (Without Trace)

#### `generate_multi(query_text, k=3, composer=None, **gen_kwargs)`

Retrieve k packs and compose them. No trace link following.

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
| `mem_write_pack(owner, retrieval_key, layer_payloads, salience)` | Write a multi-layer KV pack |
| `mem_read_pack(query_key, k, owner)` | Read top-k packs |
| `mem_read_pack_with_trace_boost(query_key, k, owner, boost_factor)` | Read with trace-boosted scoring |

### Pack Management

| Method | Description |
|--------|-------------|
| `load_pack_by_id(pack_id)` | Load a pack directly by ID |
| `add_pack_link(pack_id_1, pack_id_2)` | Create durable trace link between packs |
| `pack_links(pack_id)` | Get all packs linked to a given pack |
| `pack_count()` | Total number of packs stored |
| `pack_importance(pack_id)` | Current importance score of a pack |

### Governance

| Method | Description |
|--------|-------------|
| `cell_importance(cell_id)` | Current importance score |
| `cell_tier(cell_id)` | Current tier (Draft/Validated/Core) |
| `advance_days(days)` | Simulate time passage for decay |

### Other

| Method | Description |
|--------|-------------|
| `cell_count()` | Total cells in engine |
| `trace_ancestors(cell_id)` | Get causal parent chain |
| `has_vamana()` | Whether ANN index is active |
