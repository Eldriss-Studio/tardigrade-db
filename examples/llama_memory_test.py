#!/usr/bin/env python3
"""TardigradeDB memory test with local Llama models via llama-cpp-python.

Uses actual model hidden states from Ollama's GGUF files — no API needed,
no separate download. Extracts per-token embeddings from the model's final
layer and stores them as KV cache tensors in TardigradeDB.

Usage:
    python examples/llama_memory_test.py store "memory one" "memory two" ...
    python examples/llama_memory_test.py query "related question"
    python examples/llama_memory_test.py info
    python examples/llama_memory_test.py clear
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from llama_cpp import Llama

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import tardigrade_db

# --- Configuration ---

# Ollama blob path for llama3.2:3b (2.0 GB, Q4_K_M)
MODEL_PATH = (
    "/Users/storylight/.ollama/models/blobs/"
    "sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff"
)
MODEL_NAME = "llama3.2:3b"

DB_DIR = Path(tempfile.gettempdir()) / "tardigrade_llama_test"
MEMORY_MAP = DB_DIR / "memory_map.json"

_llm = None


def get_model():
    """Load the model once (cached)."""
    global _llm
    if _llm is None:
        print(f"  Loading {MODEL_NAME}...", end=" ", flush=True)
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=512,
            n_gpu_layers=-1,
            verbose=False,
            embedding=True,
        )
        print(f"OK (d_model={_llm.n_embd()})")
    return _llm


def text_to_hidden_state(text):
    """Extract hidden state embedding from the model's final layer.

    Returns the mean-pooled embedding across all tokens, producing
    a single d_model-dimensional vector that represents the full
    semantic content of the input text.
    """
    llm = get_model()
    output = llm.create_embedding(text)
    # output['data'][0]['embedding'] is (n_tokens, d_model)
    embeddings = np.array(output["data"][0]["embedding"], dtype=np.float32)

    if embeddings.ndim == 1:
        return embeddings

    # Mean-pool across tokens
    return embeddings.mean(axis=0)


def cmd_store(sentences):
    DB_DIR.mkdir(parents=True, exist_ok=True)
    engine = tardigrade_db.Engine(str(DB_DIR))

    memory_map = {}
    if MEMORY_MAP.exists():
        memory_map = json.loads(MEMORY_MAP.read_text())

    for i, sentence in enumerate(sentences):
        hidden = text_to_hidden_state(sentence)
        salience = min(80.0 + i * 2, 100.0)

        cell_id = engine.mem_write(
            owner=1,
            layer=0,  # single-layer since we use final hidden state
            key=hidden,
            value=hidden,
            salience=salience,
            parent_cell_id=None,
        )
        memory_map[str(cell_id)] = sentence
        print(f"  Stored cell {cell_id}: \"{sentence[:70]}\"")

    MEMORY_MAP.write_text(json.dumps(memory_map, indent=2))
    print(f"\n  Total cells in DB: {engine.cell_count()}")
    print(f"  Vector dimensions: {hidden.shape[0]}")


def cmd_query(query_text, k=5):
    engine = tardigrade_db.Engine(str(DB_DIR))

    memory_map = {}
    if MEMORY_MAP.exists():
        memory_map = json.loads(MEMORY_MAP.read_text())

    query_vec = text_to_hidden_state(query_text)

    results = engine.mem_read(query_vec, k, 1)

    print(f"  Query: \"{query_text}\"")
    print(f"  Results ({len(results)} retrieved):\n")

    for i, r in enumerate(results):
        sentence = memory_map.get(str(r.cell_id), "<unknown>")
        tier_name = ["Draft", "Validated", "Core"][r.tier]
        print(f"    #{i+1} | score={r.score:>12.2f} | cell={r.cell_id:>3} | {tier_name} | imp={r.importance:.0f}")
        print(f"         \"{sentence[:80]}\"")


def cmd_info():
    if not DB_DIR.exists():
        print("  No database found. Run 'store' first.")
        return
    engine = tardigrade_db.Engine(str(DB_DIR))
    memory_map = {}
    if MEMORY_MAP.exists():
        memory_map = json.loads(MEMORY_MAP.read_text())

    print(f"  DB path: {DB_DIR}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Total cells: {engine.cell_count()}")
    print(f"  Mapped memories: {len(memory_map)}\n")

    for cid, text in memory_map.items():
        imp = engine.cell_importance(int(cid))
        tier = ["Draft", "Validated", "Core"][engine.cell_tier(int(cid))]
        print(f"    Cell {cid:>3} [{tier:>9} imp={imp:>5.1f}]: \"{text[:75]}\"")


def cmd_clear():
    import shutil
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
        print("  Database cleared.")
    else:
        print("  No database to clear.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: store <sentences...> | query <text> | info | clear")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "store":
        cmd_store(sys.argv[2:])
    elif cmd == "query":
        cmd_query(sys.argv[2] if len(sys.argv) > 2 else "")
    elif cmd == "info":
        cmd_info()
    elif cmd == "clear":
        cmd_clear()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
