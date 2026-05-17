"""TardigradeDB starter — Python.

Stores a fact, queries it, prints the result.

Run::

    pip install tardigrade-db
    python main.py
"""

from __future__ import annotations

from pathlib import Path

from tardigrade_hooks import TardigradeClient


def main() -> int:
    engine_dir = Path(__file__).parent / "engine_dir"
    client = TardigradeClient(db_path=engine_dir, owner=1)

    pack_id = client.store("Alice moved to Berlin in 2021.")
    print(f"stored pack #{pack_id}")

    results = client.query("Where did Alice move?", k=3)
    for i, hit in enumerate(results, 1):
        print(f"  [{i}] score={hit.get('score', 0):.3f}  {hit.get('text', '')[:80]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
