# Quickstart: TardigradeDB in 5 Minutes

## What is TardigradeDB?

A persistent memory engine for AI agents. Store memories as KV cache tensors, retrieve them using the model's own latent space, inject them directly into generation at zero prompt token cost.

## Setup

```bash
git clone https://github.com/Eldriss-Studio/tardigrade-db.git
cd tardigrade-db
./scripts/setup.sh
```

This creates a virtual environment, installs dependencies, and downloads the default model (Qwen3-0.6B, ~600MB).

## Option A: Use with Claude Code (MCP)

The MCP server delivers memories as text in tool responses. This works with any LLM client but consumes normal prompt tokens. For zero-token KV injection, see Option B.

Add to your Claude Code MCP settings (the setup script prints the exact config):

```json
{
  "mcpServers": {
    "tardigrade": {
      "command": "/path/to/tardigrade-db/.venv/bin/python",
      "args": ["-m", "tardigrade_mcp"],
      "env": {
        "PYTHONPATH": "/path/to/tardigrade-db/python",
        "TARDIGRADE_DB_PATH": "./tardigrade-memory",
        "TARDIGRADE_MODEL": "Qwen/Qwen3-0.6B"
      }
    }
  }
}
```

Then in any conversation, the agent has memory tools:

- `tardigrade_store` — remember a fact
- `tardigrade_store_and_link` — attach a detail to an existing memory
- `tardigrade_recall` — find relevant memories
- `tardigrade_recall_with_trace` — follow links for multi-hop queries
- `tardigrade_list_links` — see connected memories
- `tardigrade_list_all` — list all stored memories
- `tardigrade_forget` — delete a memory permanently

## Option B: Use with Python (TardigradeClient)

The Python `TardigradeClient` facade is the recommended entry point. It combines store, query, file ingest, and consolidation behind one object. Pass a `db_path` and it creates and manages the engine internally.

```python
from tardigrade_hooks.client import TardigradeClient

# Create client — engine is created automatically at db_path
client = TardigradeClient("./my-agent-memory", owner=1)

# Store a single fact
pack_id = client.store("User prefers morning meetings before 10am")

# Query — returns list of pack result dicts
results = client.query("When should we schedule the review?", k=3)

# Ingest a file with automatic token-bounded chunking
result = client.ingest_file("context.txt")  # IngestResult with pack_ids, chunk_count
print(f"Ingested {result.chunk_count} chunks as {len(result.pack_ids)} packs")

# Consolidate views for a pack (multi-view v2)
n = client.consolidate(pack_id)        # int: views attached
all_views = client.consolidate_all()   # dict[int, int]: {pack_id: views_attached}
```

> ⚠️ **Production note:** The snippet above omits `kv_capture_fn`, so `TardigradeClient` uses its built-in random-vector stub (`_random_kv_stub`) for retrieval keys. That's fine for smoke-testing the API surface, but retrieval will not be semantically meaningful. For real use, pass a `kv_capture_fn` tied to a loaded model — see [`docs/guide/python-api.md`](python-api.md) for the HF KV-hook bridge.

### What happens under the hood during `query()`

1. **Retrieve:** `kv_capture_fn` encodes the query into a retrieval key via a model forward pass
2. **Evaluate:** Top5Avg per-token latent scoring checks confidence (score ratio between rank 1 and rank 2)
3. **Reformulate:** If confidence is below threshold, RLS runs keyword expansion + multi-phrasing variants
4. **Re-retrieve:** Each variant is scored independently
5. **Fuse:** Results from all variants are merged via Reciprocal Rank Fusion (RRF)

## Linking Related Memories

For multi-hop queries, link facts at store time so retrieval can follow connections:

```python
from tardigrade_db import Engine
from tardigrade_hooks.kp_injector import KnowledgePackStore

engine = Engine("./my-agent-memory")
kps = KnowledgePackStore(engine, model, tokenizer, owner=1)

# Store initial fact
existing = kps.store("Went to a bookstore in Pilsen")

# Later, learn the name — link it
kps.store_and_link("The bookstore is called Casa Azul", existing)

# Query follows the link automatically
text, _, _ = kps.generate_with_trace("What is the bookstore called?")
```

## What Happens Under the Hood

1. **Store:** Text wrapped in chat template, model forward pass, KV cache extracted, Q4 quantized, persisted to disk
2. **Retrieve:** Query hidden states scored against stored memories via per-token Top5Avg
3. **Inject:** KV cache reconstructed as DynamicCache, injected into model.generate() at zero prompt tokens
4. **Trace links:** Related memories connected via durable graph edges, retrieval follows connections

## Next Steps

- [MCP Setup Guide](mcp-setup.md) — detailed configuration for different clients
- [Python API Reference](python-api.md) — full API including TardigradeClient, RLS, file ingestion, and multi-view
- [Concepts](concepts.md) — KV injection, RLS, trace links, multi-view consolidation, governance explained
