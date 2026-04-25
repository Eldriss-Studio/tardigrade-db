#!/usr/bin/env python3
"""TardigradeDB MCP Server — persistent KV cache memory for LLM agents.

Facade pattern: 5 MCP tools wrap KnowledgePackStore methods.
Adapter pattern: text-based MCP protocol adapted to tensor-based storage.

Configuration via environment variables:
    TARDIGRADE_DB_PATH  — engine storage directory (default: ./tardigrade-memory)
    TARDIGRADE_MODEL    — HuggingFace model name (default: Qwen/Qwen3-0.6B)
    TARDIGRADE_OWNER    — owner ID for memories (default: 1)

Usage:
    python -m tardigrade_mcp.server
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from mcp.server.fastmcp import FastMCP
from transformers import AutoModelForCausalLM, AutoTokenizer

import tardigrade_db
from tardigrade_hooks.encoding import encode_per_token
from tardigrade_hooks.kp_injector import KnowledgePackStore

# -- Server setup -------------------------------------------------------------

mcp = FastMCP("TardigradeDB")

_kps = None
_engine = None


def _get_kps():
    """Lazy-initialize KnowledgePackStore on first use."""
    global _kps, _engine

    if _kps is not None:
        return _kps

    db_path = os.environ.get("TARDIGRADE_DB_PATH", "./tardigrade-memory")
    model_name = os.environ.get("TARDIGRADE_MODEL", "Qwen/Qwen3-0.6B")
    owner = int(os.environ.get("TARDIGRADE_OWNER", "1"))

    print(f"[TardigradeDB] Loading model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    print(f"[TardigradeDB] Opening engine: {db_path}", flush=True)
    _engine = tardigrade_db.Engine(db_path)
    _kps = KnowledgePackStore(_engine, model, tokenizer, owner=owner)

    print(
        f"[TardigradeDB] Ready. {_engine.pack_count()} memories loaded.",
        flush=True,
    )
    return _kps


def _query_key(kps, query):
    """Compute per-token retrieval key for a query string."""
    query_input = kps.tokenizer.encode(query, return_tensors="pt")
    with torch.no_grad():
        out = kps.model(query_input, output_hidden_states=True)
    hidden = out.hidden_states[kps.query_layer][0]
    h_tokens = hidden[1:].numpy().astype(np.float32)
    return encode_per_token(h_tokens, kps.hidden_size)


# -- MCP Tools ----------------------------------------------------------------


@mcp.tool()
def tardigrade_store(text: str) -> dict:
    """Store a fact the user wants you to remember across conversations.

    Call this when the user shares personal preferences, important dates,
    project details, names, or any information they expect you to recall
    later. The fact is stored permanently and survives restarts.

    Returns a pack_id you can use to link related facts later with
    tardigrade_store_and_link.
    """
    kps = _get_kps()
    pack_id = kps.store(text)
    return {"pack_id": pack_id, "status": "stored"}


@mcp.tool()
def tardigrade_store_and_link(text: str, related_pack_id: int) -> dict:
    """Store a new detail and link it to a memory you already stored.

    Use when learning additional information about something you previously
    remembered. For example, learning a person's name after already storing
    where they work. Pass the pack_id of the original memory.

    This creates a connection so both facts are found together in future
    queries via tardigrade_recall_with_trace.
    """
    kps = _get_kps()
    pack_id = kps.store_and_link(text, related_pack_id)
    return {
        "pack_id": pack_id,
        "linked_to": related_pack_id,
        "status": "stored_and_linked",
    }


@mcp.tool()
def tardigrade_recall(query: str, k: int = 1) -> list:
    """Search stored memories for facts relevant to the current question.

    Call this BEFORE answering questions that might depend on previously
    stored information. Returns the most relevant stored facts with
    confidence scores. If no relevant memories exist, returns an empty
    list — answer from your own knowledge instead.
    """
    kps = _get_kps()
    key = _query_key(kps, query)
    packs = kps.engine.mem_read_pack(key, k, kps.owner)

    return [
        {
            "pack_id": p["pack_id"],
            "text": kps._text_registry.get(p["pack_id"], ""),
            "score": round(float(p["score"]), 2),
        }
        for p in packs
    ]


@mcp.tool()
def tardigrade_recall_with_trace(query: str, k: int = 1) -> list:
    """Search memories and follow connections between linked facts.

    Use this instead of tardigrade_recall when the answer might require
    combining information from multiple related memories. For example,
    "What car does the user's instructor drive?" requires knowing who
    the instructor is AND what car they drive. Returns all connected facts.
    """
    kps = _get_kps()
    key = _query_key(kps, query)

    packs = kps.engine.mem_read_pack_with_trace_boost(key, k, kps.owner, 0.3)

    retrieved_ids = {p["pack_id"] for p in packs}
    for p in list(packs):
        for linked_id in kps.engine.pack_links(p["pack_id"]):
            if linked_id not in retrieved_ids:
                try:
                    packs.append(kps.engine.load_pack_by_id(linked_id))
                    retrieved_ids.add(linked_id)
                except Exception:
                    pass

    return [
        {
            "pack_id": p["pack_id"],
            "text": kps._text_registry.get(p["pack_id"], ""),
            "score": round(float(p.get("score", 0.0)), 2),
            "linked_packs": kps.engine.pack_links(p["pack_id"]),
        }
        for p in packs
    ]


@mcp.tool()
def tardigrade_list_links(pack_id: int) -> list:
    """Show all memories connected to a specific memory.

    Use to explore what you know about a topic or to find the pack_id
    of a related fact before storing a new detail with
    tardigrade_store_and_link.
    """
    kps = _get_kps()
    return [
        {
            "pack_id": lid,
            "text": kps._text_registry.get(lid, ""),
        }
        for lid in kps.engine.pack_links(pack_id)
    ]


@mcp.tool()
def tardigrade_list_all() -> list:
    """List all stored memories with their pack IDs and link counts.

    Use to see everything you remember. Helpful when exploring what you
    know or when looking for a pack_id to link a new detail to.
    """
    kps = _get_kps()
    results = []
    for pack_id, text in sorted(kps._text_registry.items()):
        links = kps.engine.pack_links(pack_id)
        results.append({
            "pack_id": pack_id,
            "text": text,
            "links": len(links),
        })
    return results


# -- Entry point --------------------------------------------------------------


def main():
    mcp.run()


if __name__ == "__main__":
    main()
