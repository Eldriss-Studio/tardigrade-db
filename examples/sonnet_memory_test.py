#!/usr/bin/env python3
"""TardigradeDB memory cycle test.

Accepts memories and queries via CLI args, stores/retrieves from TardigradeDB.
Used by Claude subagents to test the full memory cycle.

Usage:
    # Store memories
    python examples/sonnet_memory_test.py store "fact one" "fact two" "fact three"

    # Query memories
    python examples/sonnet_memory_test.py query "related question"

    # Full info
    python examples/sonnet_memory_test.py info
"""

import hashlib
import sys
import tempfile
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import tardigrade_db

DB_DIR = Path(tempfile.gettempdir()) / "tardigrade_sonnet_test"
D_MODEL = 768


def text_to_vector(text: str) -> np.ndarray:
    """Deterministic text → vector via seeded RNG.

    Words that share tokens produce overlapping vector components,
    so semantically related phrases have higher dot-product similarity.
    """
    vec = np.zeros(D_MODEL, dtype=np.float32)
    words = text.lower().split()
    for word in words:
        seed = int(hashlib.sha256(word.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.RandomState(seed)
        vec += rng.randn(D_MODEL).astype(np.float32)
    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def cmd_store(memories: list[str]):
    DB_DIR.mkdir(parents=True, exist_ok=True)
    engine = tardigrade_db.Engine(str(DB_DIR))

    results = []
    for i, memory in enumerate(memories):
        key = text_to_vector(memory)
        value = text_to_vector(memory + " value")
        salience = min(80.0 + i * 5, 100.0)
        cell_id = engine.mem_write(
            owner=1,
            layer=0,
            key=key,
            value=value,
            salience=salience,
            parent_cell_id=None,
        )
        results.append({
            "cell_id": cell_id,
            "memory": memory,
            "salience": salience,
        })
        print(f"  Stored cell {cell_id}: {memory[:60]}...")

    print(f"\nTotal cells in DB: {engine.cell_count()}")
    # Save memory text mapping for retrieval display
    mapping_file = DB_DIR / "memory_map.json"
    existing = {}
    if mapping_file.exists():
        existing = json.loads(mapping_file.read_text())
    for r in results:
        existing[str(r["cell_id"])] = r["memory"]
    mapping_file.write_text(json.dumps(existing, indent=2))

    return results


def cmd_query(query_text: str, k: int = 5):
    engine = tardigrade_db.Engine(str(DB_DIR))
    mapping_file = DB_DIR / "memory_map.json"
    memory_map = {}
    if mapping_file.exists():
        memory_map = json.loads(mapping_file.read_text())

    query_vec = text_to_vector(query_text)
    results = engine.mem_read(query_vec, k, 1)

    print(f"Query: {query_text}")
    print(f"Results ({len(results)} retrieved):\n")
    retrieved = []
    for r in results:
        text = memory_map.get(str(r.cell_id), "<unknown>")
        tier_name = ["Draft", "Validated", "Core"][r.tier]
        print(f"  Cell {r.cell_id} | score={r.score:.4f} | tier={tier_name} | importance={r.importance:.1f}")
        print(f"    → {text}")
        retrieved.append({"cell_id": r.cell_id, "score": r.score, "text": text})

    return retrieved


def cmd_info():
    if not DB_DIR.exists():
        print("No database found. Run 'store' first.")
        return
    engine = tardigrade_db.Engine(str(DB_DIR))
    print(f"DB path: {DB_DIR}")
    print(f"Total cells: {engine.cell_count()}")

    mapping_file = DB_DIR / "memory_map.json"
    if mapping_file.exists():
        memory_map = json.loads(mapping_file.read_text())
        print(f"Mapped memories: {len(memory_map)}")
        for cid, text in memory_map.items():
            imp = engine.cell_importance(int(cid))
            tier = ["Draft", "Validated", "Core"][engine.cell_tier(int(cid))]
            print(f"  Cell {cid}: imp={imp:.1f} tier={tier} → {text[:80]}")


def cmd_clear():
    import shutil
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
        print("Database cleared.")
    else:
        print("No database to clear.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: store <memories...> | query <text> | info | clear")
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
