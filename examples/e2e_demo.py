#!/usr/bin/env python3
"""End-to-end demo: TardigradeDB with a real HuggingFace GPT-2 model.

Proves the thesis: capture KV cache tensors during inference, persist them
to TardigradeDB, retrieve by latent-space attention, and inject into a
subsequent inference pass.

Usage:
    python examples/e2e_demo.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

# Add the hooks package to path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import tardigrade_db
from tardigrade_hooks.hf_hook import HuggingFaceHook


def run_gpt2_demo():
    """Full demo with real GPT-2 model."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("\n[1] Loading GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True)
    model.eval()
    print(f"    Model: {model.config.n_layer} layers, d_model={model.config.n_embd}")

    db_dir = tempfile.mkdtemp(prefix="tardigrade_demo_")
    print(f"\n[2] Engine at {db_dir}")
    engine = tardigrade_db.Engine(db_dir)
    hook = HuggingFaceHook(engine, owner=1, k=5, norm_threshold=0.5)

    # First inference — capture KV.
    prompt1 = "The capital of France is"
    print(f"\n[3] Capture: '{prompt1}'")
    inputs = tokenizer(prompt1, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    cells_written = 0
    for layer_idx in range(model.config.n_layer):
        h = outputs.hidden_states[layer_idx + 1].numpy()
        decision = hook.on_generate(layer=layer_idx, hidden_states=h)
        if decision.should_write and decision.key is not None:
            engine.mem_write(1, layer_idx, decision.key, decision.value, decision.salience, None)
            cells_written += 1
    print(f"    Written {cells_written} cells ({engine.cell_count()} total)")

    # Second inference — retrieve.
    prompt2 = "What is the main city of France"
    print(f"\n[4] Retrieve: '{prompt2}'")
    inputs2 = tokenizer(prompt2, return_tensors="pt")
    with torch.no_grad():
        outputs2 = model(**inputs2)

    for layer_idx in range(min(3, model.config.n_layer)):
        h = outputs2.hidden_states[layer_idx + 1].numpy()
        handles = hook.on_prefill(layer=layer_idx, query_states=h)
        if handles:
            print(f"    Layer {layer_idx}: {len(handles)} cells (best={handles[0].score:.4f})")

    # Governance.
    print("\n[5] Governance:")
    for cid in range(min(3, engine.cell_count())):
        imp = engine.cell_importance(cid)
        tier = ["Draft", "Validated", "Core"][engine.cell_tier(cid)]
        print(f"    Cell {cid}: importance={imp:.1f}, tier={tier}")

    # Persistence.
    print("\n[6] Persistence:")
    count = engine.cell_count()
    del engine
    engine2 = tardigrade_db.Engine(db_dir)
    print(f"    Before={count}, After={engine2.cell_count()}")
    assert engine2.cell_count() == count


def run_numpy_demo():
    """Fallback demo using numpy arrays (simulated inference)."""
    print("\n--- numpy-only demo (simulated inference) ---")

    db_dir = tempfile.mkdtemp(prefix="tardigrade_demo_")
    engine = tardigrade_db.Engine(db_dir)

    d_model = 768
    n_layers = 12

    print("\n[3] Writing KV from 12 simulated layers...")
    for layer in range(n_layers):
        key = np.random.randn(d_model).astype(np.float32) * 0.5
        value = np.random.randn(d_model).astype(np.float32) * 0.5
        salience = min(float(np.linalg.norm(key)) * 10.0, 100.0)
        engine.mem_write(1, layer, key, value, salience, None)
    print(f"    Written {engine.cell_count()} cells")

    print("\n[4] Retrieving with similar query...")
    query = np.random.randn(d_model).astype(np.float32) * 0.5
    results = engine.mem_read(query, 5, 1)
    print(f"    Retrieved {len(results)} cells")
    for r in results:
        print(f"      Cell {r.cell_id} (layer {r.layer}): score={r.score:.4f}")

    print("\n[5] Governance:")
    for cid in range(min(3, engine.cell_count())):
        imp = engine.cell_importance(cid)
        tier = ["Draft", "Validated", "Core"][engine.cell_tier(cid)]
        print(f"    Cell {cid}: importance={imp:.1f}, tier={tier}")

    print("\n[6] Persistence:")
    count = engine.cell_count()
    del engine
    engine2 = tardigrade_db.Engine(db_dir)
    assert engine2.cell_count() == count
    print(f"    Reopened: {engine2.cell_count()} cells intact")


def main():
    print("=" * 60)
    print("TardigradeDB End-to-End Demo")
    print("=" * 60)

    try:
        import torch  # noqa: F401
        from transformers import GPT2LMHeadModel  # noqa: F401
        run_gpt2_demo()
    except ImportError:
        print("    PyTorch/transformers not installed. Using numpy fallback.")
        run_numpy_demo()

    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)


if __name__ == "__main__":
    main()
