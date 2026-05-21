# MCP Setup Guide

You're running Claude Code, Cursor, or another LLM client that speaks the Model Context Protocol, and you want to give the agent persistent memory it can recall across sessions. TardigradeDB ships an MCP server that wires that up in seven tool calls. This guide walks the setup for Claude Code and Cursor, and explains the one important tradeoff before you start.

## The tradeoff to know before you wire this up

The MCP server delivers retrieved memories as **text** in the tool's response, because the Model Context Protocol speaks JSON-over-stdio and can't carry KV tensors directly. That makes the MCP path universally compatible (any LLM that can call MCP tools can use it) but means every recalled memory costs prompt tokens in the next turn — exactly the cost embedding RAG has, and exactly the cost the TardigradeDB design is otherwise trying to avoid.

So:

- **Use the MCP server** when you want zero-glue persistent memory inside Claude Code, Cursor, or any MCP-speaking client. The token cost is real but the integration cost is near zero.
- **Use the [Python API](python-api.md) directly** when you control the model and care about prompt-token cost — that path injects retrieved memories as pre-computed KV cache, skipping the re-tokenization entirely.

The MCP path is the right tradeoff for IDE-level agents. The Python path is the right tradeoff for production model serving.

## Prerequisites

Run `./scripts/setup.sh` from the tardigrade-db repository root. It installs the Python dependencies into a `.venv` and downloads the embedding model (Qwen3-0.6B by default). When the script finishes it prints the three paths you'll paste into your MCP config in the next step — the Python interpreter inside the venv, the `python/` source directory (which becomes `PYTHONPATH`), and a default storage directory. Copy those somewhere; the wiring section below uses them.

## Claude Code

Claude Code looks for MCP server configuration in two places, and either works for this. `~/.claude/claude_desktop_config.json` is your user-level file and applies to every project you open in Claude Code. A `.mcp.json` at the root of a specific project applies only to that project — useful if you want one TardigradeDB store per project rather than a shared one. Add the following block inside the `mcpServers` object of whichever file you picked:

```json
{
  "mcpServers": {
    "tardigrade": {
      "command": "/path/to/tardigrade-db/.venv/bin/python",
      "args": ["-m", "tardigrade_mcp"],
      "env": {
        "PYTHONPATH": "/path/to/tardigrade-db/python",
        "TARDIGRADE_DB_PATH": "/path/to/memory-storage",
        "TARDIGRADE_MODEL": "Qwen/Qwen3-0.6B"
      }
    }
  }
}
```

`PYTHONPATH` is set because `tardigrade_mcp` lives in the repo's `python/` directory rather than being pip-installed — pointing Python at the source tree is how the MCP runner finds the module. `TARDIGRADE_DB_PATH` is the directory where persisted memory cells live; pick a stable location outside `/tmp`. `TARDIGRADE_MODEL` is the HuggingFace model used to compute retrieval keys; the default Qwen3-0.6B works well and runs on CPU.

## Cursor

Cursor exposes MCP wiring through Settings → MCP Servers. The fields map directly to the JSON above:

- **Name:** `tardigrade`
- **Command:** `/path/to/tardigrade-db/.venv/bin/python`
- **Args:** `-m tardigrade_mcp`
- **Environment:** add `PYTHONPATH`, `TARDIGRADE_DB_PATH`, and `TARDIGRADE_MODEL` with the same values as the Claude Code example.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TARDIGRADE_DB_PATH` | `./tardigrade-memory` | Directory for persistent storage |
| `TARDIGRADE_MODEL` | `Qwen/Qwen3-0.6B` | HuggingFace model for KV computation |
| `TARDIGRADE_OWNER` | `1` | Owner ID (for multi-agent setups) |

## Available Tools

### `tardigrade_store`

Store a fact as a persistent memory.

**Parameters:**
- `text` (string): The fact to remember

**Returns:** `{"pack_id": int, "status": "stored"}`

### `tardigrade_store_and_link`

Store a fact linked to an existing memory. Use when learning a new detail about something already remembered.

**Parameters:**
- `text` (string): The new detail
- `related_pack_id` (integer): Pack ID of the existing memory

**Returns:** `{"pack_id": int, "linked_to": int, "status": "stored_and_linked"}`

### `tardigrade_recall`

Retrieve the most relevant memories for a query using latent-space scoring.

**Parameters:**
- `query` (string): What to search for
- `k` (integer, default 1): Number of results

**Returns:** List of `{"pack_id": int, "text": string, "score": float}`

### `tardigrade_recall_with_trace`

Retrieve memories following trace links for multi-hop queries. Finds the best match, then follows connections to discover related facts.

**Parameters:**
- `query` (string): What to search for
- `k` (integer, default 1): Number of initial results (trace expands this)

**Returns:** List of `{"pack_id": int, "text": string, "score": float, "linked_packs": [int]}`

### `tardigrade_list_links`

Show what memories are connected to a given memory.

**Parameters:**
- `pack_id` (integer): The memory to inspect

**Returns:** List of `{"pack_id": int, "text": string}`

### `tardigrade_list_all`

List all stored memories with their pack IDs and link counts.

**Parameters:** None

**Returns:** List of `{"pack_id": int, "text": string, "links": int}`

### `tardigrade_forget`

Delete a stored memory permanently. Irreversible.

**Parameters:**
- `pack_id` (integer): The memory to delete

**Returns:** `{"pack_id": int, "status": "deleted"}`

## Performance Notes

- **First tool call** loads the model (~2-3 seconds for Qwen3-0.6B on Apple Silicon)
- **Subsequent calls** are fast (~100ms for store, ~50ms for recall)
- **Storage:** ~730 KB per memory (Q4 quantized KV cache)
- **Memory:** ~1.2 GB RAM for Qwen3-0.6B model

## Switching Models

```bash
# Use a larger model for better quality
TARDIGRADE_MODEL="Qwen/Qwen2.5-3B" python -m tardigrade_mcp

# Or set in your MCP config:
"env": { "TARDIGRADE_MODEL": "Qwen/Qwen2.5-3B" }
```

Larger models produce richer KV representations but use more RAM and storage.

## Python API Features Not Available in MCP

The 7 MCP tools expose base-level KV store + trace operations. The following features are available via the Python API ([`TardigradeClient`](python-api.md#tardigradeclient)) but are not currently exposed through MCP:

- **Reflective Latent Search (RLS)** — confidence-gated query reformulation with 5 strategy options
- **File ingestion** — `ingest_file()` / `ingest_text()` with automatic token-bounded chunking
- **Multi-view consolidation v2** — `consolidate()` / `consolidate_all()` for multi-framing retrieval surfaces
- **CrossEncoderReranker** — Stage-2 re-ranking over text-bearing candidates

MCP RLS routing is planned for a future phase.
