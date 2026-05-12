# MCP Setup Guide

TardigradeDB provides an MCP (Model Context Protocol) server that gives any LLM agent persistent memory through 7 tool calls.

**Note:** The MCP server delivers memories as text in tool responses for universal LLM compatibility. This uses normal prompt tokens. For zero-token KV injection, use the [Python API](python-api.md) directly.

## Prerequisites

1. Run `./scripts/setup.sh` to install dependencies and download the model
2. Note the paths printed by the setup script

## Claude Code

Add to `~/.claude/claude_desktop_config.json` or your project's `.mcp.json`:

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

## Cursor

Add to Cursor's MCP settings (Settings > MCP Servers):

- **Name:** tardigrade
- **Command:** `/path/to/tardigrade-db/.venv/bin/python`
- **Args:** `-m tardigrade_mcp`
- **Environment:** set PYTHONPATH, TARDIGRADE_DB_PATH, TARDIGRADE_MODEL

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
