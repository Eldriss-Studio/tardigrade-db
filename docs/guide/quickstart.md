# Quickstart: TardigradeDB in 5 Minutes

You've installed TardigradeDB (or are about to) and you want it running in something useful within the next few minutes. There are two paths, and which one you pick depends on what you're hooking it into.

If you're driving an LLM through Claude Code, Cursor, or any client that speaks the Model Context Protocol, take **Option A (MCP)**. The integration is essentially zero glue — you paste a config block, restart the client, and the agent gets memory tools. The tradeoff is that recalled memories arrive as text in tool responses, which costs prompt tokens on every turn that uses them.

If you're calling the model yourself — running HuggingFace `transformers`, building a server, embedding inference in your own app — take **Option B (Python API)**. The integration is more code, but recalled memories get injected into the model's KV cache directly, with zero prompt-token cost on the way in. This is the path the project's design is optimised for.

This guide covers both.

## Setup

```bash
git clone https://github.com/Eldriss-Studio/tardigrade-db.git
cd tardigrade-db
./scripts/setup.sh
```

The setup script creates a virtual environment, installs dependencies, and downloads the default model (Qwen3-0.6B, ~600 MB). When it finishes it prints the paths you'll paste into the MCP config in Option A — copy them somewhere.

## Option A — Use it with Claude Code or Cursor (MCP)

Memories come back as text. That makes the integration universal (any MCP-speaking client works) but means every recalled memory costs prompt tokens on the turn that uses it. The token cost is real but the integration cost is near zero, which is the right trade for an IDE-level agent.

Paste this into either `~/.claude/claude_desktop_config.json` (applies to all your Claude Code projects) or a `.mcp.json` at the root of a specific project (applies only there):

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

Restart the client. The agent now has seven memory tools available — `tardigrade_store` to remember a fact, `tardigrade_store_and_link` to attach a detail to an existing memory, `tardigrade_recall` to find relevant memories, `tardigrade_recall_with_trace` to follow links for multi-hop queries, plus `list_links`, `list_all`, and `forget`. The agent decides when to call them; you don't have to script anything.

For per-client details (paths, environment variables, the Cursor UI), see [`mcp-setup.md`](mcp-setup.md).

## Option B — Use it from Python (TardigradeClient)

The Python API gives you zero-token KV injection, but it has one precondition you need to know about before you start.

**You need a model.** `TardigradeClient` computes retrieval keys by running a forward pass through whatever model you give it — passing a `kv_capture_fn` tied to that model is what makes retrieval semantically meaningful. If you skip that argument, the client falls back to a random-vector stub that's fine for smoke-testing the API shape but produces near-random retrieval. The snippet below uses the stub so you can run it with a fresh `pip install` and no model loaded; switch to a real `kv_capture_fn` once you've confirmed the API works (see [`python-api.md`](python-api.md) for the HuggingFace KV-hook bridge that gives you one).

With that caveat:

```python
from tardigrade_hooks.client import TardigradeClient

# Engine is created automatically at db_path. owner scopes memories per
# agent or tenant — see consumers.md for how this is the isolation primitive.
client = TardigradeClient("./my-agent-memory", owner=1)

# Store a single fact
pack_id = client.store("User prefers morning meetings before 10am")

# Query — returns a list of pack result dicts
results = client.query("When should we schedule the review?", k=3)

# Ingest a file with automatic token-bounded chunking
result = client.ingest_file("context.txt")
print(f"Ingested {result.chunk_count} chunks as {len(result.pack_ids)} packs")

# Consolidate views for a pack (multi-view v2)
n = client.consolidate(pack_id)        # int: views attached
all_views = client.consolidate_all()   # dict[int, int]: {pack_id: views_attached}
```

`TardigradeClient` is a facade — one object that bundles `store`, `query`, `ingest_file`, and `consolidate` so you don't have to wire up `Engine`, `FileIngestor`, and `MemoryConsolidator` separately. If you need finer control (custom write buffers, direct engine access, multi-layer queries), drop down to those primitives — they're all documented in [`python-api.md`](python-api.md).

## Linking related memories

For multi-hop queries — *"what's the bookstore's name?"* when the bookstore was mentioned in one memory and named in another — link facts at store time so retrieval can follow connections:

```python
from tardigrade_db import Engine
from tardigrade_hooks.kp_injector import KnowledgePackStore

engine = Engine("./my-agent-memory")
kps = KnowledgePackStore(engine, model, tokenizer, owner=1)

# Store the original fact
existing = kps.store("Went to a bookstore in Pilsen")

# Later, learn the name — link the new fact to the old one
kps.store_and_link("The bookstore is called Casa Azul", existing)

# A traced query follows the link automatically
text, _, _ = kps.generate_with_trace("What is the bookstore called?")
```

The link tells the engine that the second fact extends the first. A subsequent query that matches either memory can follow the trace edge to recover both.

## What's actually happening under the hood

When you call `client.store(text)`, the text is wrapped in the model's chat template, a forward pass extracts the KV cache from `past_key_values`, the cache is Q4-quantized (4-bit integer compression that loses ~0.1 % of signal fidelity in practice) and written to disk as a pack.

When you call `client.query(text)`, the engine computes a retrieval key from the query and scores it against every stored pack using *per-token Top5Avg* — for each pack, it takes the dot products between the query's hidden states and the pack's hidden states, keeps the top five matches across all token positions, and averages them. This avoids the gravity-well failure mode where mean-pooled scoring collapses to one dominant memory.

If you're using [Reflective Latent Search](concepts.md#reflective-latent-search) (`engine.set_refinement_mode("centered+prf")` or similar), there's an extra step: the engine compares the score ratio between rank 1 and rank 2, and if confidence is low it reformulates the query with keyword-expansion or multi-phrasing variants, re-retrieves for each, and fuses the rankings via Reciprocal Rank Fusion (RRF — a rank-based merge that weights items by `1 / (k + rank)`). RLS is documented but not currently the recommended path on clean benchmark data; see [`concepts.md`](concepts.md#reflective-latent-search) for the honest status.

## Next steps

- [`mcp-setup.md`](mcp-setup.md) — full MCP configuration including the seven tools, environment variables, model swapping.
- [`python-api.md`](python-api.md) — complete Python API: `TardigradeClient`, `KnowledgePackStore`, RLS, file ingestion, multi-view consolidation.
- [`concepts.md`](concepts.md) — KV injection, trace links, multi-view consolidation, governance, and the honest status of each.
- [`consumers.md`](consumers.md) — three reference patterns for wiring TardigradeDB into agents, multi-tenant apps, and document-QA systems.
