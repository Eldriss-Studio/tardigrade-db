#!/usr/bin/env python3
"""TardigradeDB KV cache memory test — real transformer tensors.

Two-phase test using GPT-2's actual hidden states:
  1. STORE: Process sentences through GPT-2, capture KV cache tensors per layer
  2. QUERY: Process query sentences, retrieve matching memories via latent-space attention

Usage:
    # Store memories from sentences
    python examples/kv_memory_test.py store "sentence one" "sentence two" ...

    # Query with a new sentence
    python examples/kv_memory_test.py query "related question"

    # Show all stored memories
    python examples/kv_memory_test.py info

    # Clear database
    python examples/kv_memory_test.py clear
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import tardigrade_db
from tardigrade_hooks.hf_hook import HuggingFaceHook

DB_DIR = Path(tempfile.gettempdir()) / "tardigrade_kv_test"
MEMORY_MAP = DB_DIR / "memory_map.json"

# Target layer for single-layer retrieval (layer 8 captures high-level semantics in GPT-2).
TARGET_LAYER = 8


def load_model():
    """Load GPT-2 and tokenizer."""
    print("  Loading GPT-2...", end=" ", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True)
    print(f"OK ({model.config.n_layer} layers, d={model.config.n_embd})")
    return model, tokenizer


def get_hidden_states(model, tokenizer, text):
    """Run text through GPT-2 and return hidden states per layer."""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return [outputs.hidden_states[i + 1].numpy() for i in range(model.config.n_layer)]


def cmd_store(sentences):
    DB_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model()
    engine = tardigrade_db.Engine(str(DB_DIR))
    hook = HuggingFaceHook(engine, owner=1, k=5, norm_threshold=0.0)

    memory_map = {}
    if MEMORY_MAP.exists():
        memory_map = json.loads(MEMORY_MAP.read_text())

    for sentence in sentences:
        hidden = get_hidden_states(model, tokenizer, sentence)

        cells_written = 0
        first_cell_id = None
        for layer_idx, h in enumerate(hidden):
            decision = hook.on_generate(layer=layer_idx, hidden_states=h)
            if decision.should_write and decision.key is not None:
                cell_id = engine.mem_write(
                    1, layer_idx, decision.key, decision.value,
                    decision.salience, None
                )
                if first_cell_id is None:
                    first_cell_id = cell_id
                cells_written += 1

        if first_cell_id is not None:
            for cid in range(first_cell_id, first_cell_id + cells_written):
                memory_map[str(cid)] = sentence

        print(f"  Stored: \"{sentence[:70]}\" -> {cells_written} cells (layers 0-{cells_written - 1})")

    MEMORY_MAP.write_text(json.dumps(memory_map, indent=2))
    print(f"\n  Total cells in DB: {engine.cell_count()}")


def cmd_query(query_text, k=5):
    model, tokenizer = load_model()
    engine = tardigrade_db.Engine(str(DB_DIR))

    memory_map = {}
    if MEMORY_MAP.exists():
        memory_map = json.loads(MEMORY_MAP.read_text())

    hidden = get_hidden_states(model, tokenizer, query_text)

    print(f"  Query: \"{query_text}\"\n")

    # Query at the target layer for clearest semantic signal.
    h = hidden[TARGET_LAYER]
    if h.ndim == 3:
        h = h[0]
    mean_query = h.mean(axis=0).astype(np.float32)
    results = engine.mem_read(mean_query, k, 1)

    seen_sentences = set()
    print(f"  Results from layer {TARGET_LAYER} (top-{k}):\n")
    rank = 1
    for r in results:
        sentence = memory_map.get(str(r.cell_id), "<unknown>")
        if sentence in seen_sentences:
            continue
        seen_sentences.add(sentence)
        tier_name = ["Draft", "Validated", "Core"][r.tier]
        print(f"    #{rank} | score={r.score:>10.2f} | cell={r.cell_id:>3} | layer={r.layer} | {tier_name}")
        print(f"         \"{sentence[:80]}\"")
        rank += 1

    # Also show all-layer breakdown for the top result
    print(f"\n  Per-layer scores for best match:")
    for layer_idx, h_layer in enumerate(hidden):
        if h_layer.ndim == 3:
            h_layer = h_layer[0]
        q = h_layer.mean(axis=0).astype(np.float32)
        layer_results = engine.mem_read(q, 1, 1)
        if layer_results:
            best = layer_results[0]
            sentence = memory_map.get(str(best.cell_id), "?")
            print(f"    Layer {layer_idx:>2}: score={best.score:>10.2f} -> \"{sentence[:50]}\"")


def cmd_info():
    if not DB_DIR.exists():
        print("  No database found. Run 'store' first.")
        return
    engine = tardigrade_db.Engine(str(DB_DIR))
    memory_map = {}
    if MEMORY_MAP.exists():
        memory_map = json.loads(MEMORY_MAP.read_text())

    print(f"  DB path: {DB_DIR}")
    print(f"  Total cells: {engine.cell_count()}")

    # Group by sentence
    sentences = {}
    for cid, text in memory_map.items():
        if text not in sentences:
            sentences[text] = []
        sentences[text].append(int(cid))

    print(f"  Unique memories: {len(sentences)}\n")
    for text, cells in sentences.items():
        imp = engine.cell_importance(cells[0])
        tier = ["Draft", "Validated", "Core"][engine.cell_tier(cells[0])]
        print(f"    [{tier:>9} imp={imp:>5.1f}] cells {cells[0]}-{cells[-1]}: \"{text[:75]}\"")


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
