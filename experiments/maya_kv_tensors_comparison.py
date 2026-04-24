#!/usr/bin/env python3
"""Head-to-head: GPT-2 vs Qwen3 — per-layer KV tensors on Maya's memories.

Both models use the SAME vectorization strategy that TardigradeDB was
designed for: per-layer hidden states captured via output_hidden_states=True,
mean-pooled across the sequence dimension.

This is NOT embeddings or pooled representations. These are the model's
actual internal activations at each transformer layer — the K/V tensors
that TardigradeDB natively stores and retrieves.

Usage:
    source .venv/bin/activate
    python experiments/maya_kv_tensors_comparison.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import tardigrade_db

# ── Maya's memories ──────────────────────────────────────────────────────────

MEMORIES = [
    "Arrived at 6:15am to pre-round on post-op patients. The overnight tech left a sticky note on cage 14: 'Biscuit vomited twice, still eating.' Small relief.",
    "Emergency at 7:30am — a golden retriever named Captain swallowed a corn cob three days ago. Radiographs showed complete obstruction at the jejunum. Dr. Nakamura said I'd assist on the enterotomy.",
    "Four hours in surgery. My hands were shaking when Dr. Nakamura let me close the intestinal incision. Two-layer closure, 4-0 PDS, interrupted appositional. She said 'adequate' which from her is basically a standing ovation.",
    "Captain's owner, a retired firefighter named Glen, was sitting in the lobby the entire time. When I told him Captain made it through surgery, he grabbed my hand with both of his and couldn't speak for ten seconds.",
    "Ate cold leftover pasta standing up in the pharmacy because there was no time to sit down. Dropped marinara sauce on my scrub top. Third time this month.",
    "Showed the two new externs how to read abdominal radiographs. One of them asked if the spleen was a tumor. I remembered asking the exact same question my first week.",
    "Got into it with Dr. Patel about whether to discharge the diabetic cat on Lantus or ProZinc. He pulled rank. I'm still convinced ProZinc was the right call for this cat's insulin curve.",
    "The hospital's orange tabby, Chairman Meow, was sleeping in the centrifuge room again. Someone put a tiny surgical cap on him. Nobody knows who.",
    "Missed a call from the board certification office. Spent twenty minutes catastrophizing that my application had a problem before calling back. It was just confirming my exam date.",
    "Stayed late to help Raj place a jugular catheter on a fractious feral cat. Took three attempts and a towel burrito. Raj's hands were steadier than mine by the third try.",
    "Sat with Biscuit in cage 14 for ten minutes after everyone left. She put her head on my lap. Post-op day three and her appetite is back. That's the whole reason I do this.",
    "Drove home at 9pm listening to a podcast about compassion fatigue in veterinary medicine. Cried a little at the part about learning to hold other people's grief without absorbing it.",
]

SPECIFIC_QUERIES = [
    ("The dog that swallowed something and needed surgery", [1, 2]),
    ("The owner who was waiting in the lobby during surgery", [3]),
    ("What did I eat for lunch and where", [4]),
    ("Teaching the new students about reading x-rays", [5]),
    ("Argument with another doctor about medication for a cat", [6]),
    ("The hospital cat wearing something funny", [7]),
    ("Phone call that made me anxious about my certification", [8]),
    ("Helping someone with a difficult procedure on a feral cat", [9]),
    ("Sitting quietly with a recovering patient after hours", [10]),
    ("Driving home and listening to something that made me cry", [11]),
    ("How was Biscuit doing this morning", [0]),
    ("Did I do any suturing or wound closure", [2]),
]

NEGATIVE_QUERIES = [
    "Did I attend a faculty meeting about budget cuts",
    "Was there a horse brought in for colic surgery",
    "Did I go to the gym after work",
    "Meeting with my thesis advisor about the research paper",
]


def run_model(model_name, friendly_name, query_layer_ratio=0.67):
    """Run the two-agent experiment with per-layer KV tensors.

    Args:
        model_name: HuggingFace model ID.
        friendly_name: Display name.
        query_layer_ratio: Which layer to query from (0.67 = ~2/3 depth, semantic).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n  Loading {friendly_name}...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,

        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    query_layer = int(n_layers * query_layer_ratio)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"OK ({n_layers} layers, d_model={d_model}, query_layer={query_layer})")

    db_dir = Path(tempfile.mkdtemp(prefix=f"tardigrade_maya_kv_"))
    engine = tardigrade_db.Engine(str(db_dir))

    # ── Store: per-layer hidden states ───────────────────────────────────

    for mem_idx, memory in enumerate(MEMORIES):
        inputs = tokenizer(memory, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)

        for layer_idx in range(n_layers):
            hidden = outputs.hidden_states[layer_idx + 1].numpy()[0]  # (seq, d_model)
            mean_hidden = hidden.mean(axis=0).astype(np.float32)      # (d_model,)
            salience = min(75.0 + mem_idx * 2, 100.0)
            engine.mem_write(1, layer_idx, mean_hidden, mean_hidden, salience, None)

    total_cells = engine.cell_count()
    print(f"  Stored: {total_cells} cells ({len(MEMORIES)} memories x {n_layers} layers)")

    # ── Retrieve: query from semantic layer ──────────────────────────────

    results_map = {}
    genuine_scores = []
    negative_scores = []

    for query_text, expected_mems in SPECIFIC_QUERIES:
        inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)

        query_hidden = outputs.hidden_states[query_layer + 1].numpy()[0]
        query_vec = query_hidden.mean(axis=0).astype(np.float32)

        results = engine.mem_read(query_vec, 5, 1)
        # Map cell ID to memory index: cell_id // n_layers
        retrieved_mems = [r.cell_id // n_layers for r in results]
        hit = any(m in expected_mems for m in retrieved_mems)
        genuine_scores.extend(r.score for r in results)

        rank = "—"
        if hit:
            for j, m in enumerate(retrieved_mems):
                if m in expected_mems:
                    rank = f"#{j+1}"
                    break

        top_mem_idx = retrieved_mems[0] if retrieved_mems else -1
        results_map[query_text] = {
            "hit": hit,
            "rank": rank,
            "top_score": results[0].score if results else 0,
            "top_mem": top_mem_idx,
            "retrieved_mems": retrieved_mems,
        }

    for query_text in NEGATIVE_QUERIES:
        inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
        query_hidden = outputs.hidden_states[query_layer + 1].numpy()[0]
        query_vec = query_hidden.mean(axis=0).astype(np.float32)
        results = engine.mem_read(query_vec, 3, 1)
        if results:
            negative_scores.append(results[0].score)

    shutil.rmtree(db_dir)

    hits = sum(1 for r in results_map.values() if r["hit"])
    recall = hits / len(SPECIFIC_QUERIES) * 100
    avg_genuine = np.mean(genuine_scores) if genuine_scores else 0
    avg_negative = np.mean(negative_scores) if negative_scores else 0
    snr = avg_genuine - avg_negative

    return {
        "name": friendly_name,
        "model": model_name,
        "params": f"{param_count / 1e6:.0f}M",
        "layers": n_layers,
        "d_model": d_model,
        "query_layer": query_layer,
        "recall": recall,
        "hits": hits,
        "total": len(SPECIFIC_QUERIES),
        "avg_genuine": avg_genuine,
        "avg_negative": avg_negative,
        "snr": snr,
        "details": results_map,
    }


def main():
    print("=" * 70)
    print("Per-Layer KV Tensor Comparison — Maya's Memories")
    print("Same vectorization (output_hidden_states), different models")
    print("=" * 70)

    models = [
        ("openai-community/gpt2", "GPT-2"),
        ("Qwen/Qwen3-0.6B", "Qwen3-0.6B"),
    ]

    all_results = []
    for model_id, name in models:
        print(f"\n{'━' * 70}")
        print(f"  {name} ({model_id})")
        print(f"{'━' * 70}")
        r = run_model(model_id, name)
        all_results.append(r)

        # Print per-query results inline.
        print()
        for query_text, expected in SPECIFIC_QUERIES:
            d = r["details"][query_text]
            status = f"✓ {d['rank']}" if d["hit"] else "✗   "
            top_mem = MEMORIES[d["top_mem"]][:50] if d["top_mem"] >= 0 else "—"
            short_q = query_text[:45]
            print(f"  {status:>6}  {short_q:<45} → mem {d['top_mem']:>2}")

    # ── Side-by-side ─────────────────────────────────────────────────────

    print(f"\n{'━' * 70}")
    print("  COMPARISON")
    print(f"{'━' * 70}\n")

    headers = [r["name"] for r in all_results]
    col_w = 16

    def row(label, vals):
        print(f"  {label:<25}", end="")
        for v in vals:
            print(f" {v:>{col_w}}", end="")
        print()

    row("", headers)
    print(f"  {'─' * (25 + (col_w + 1) * len(all_results))}")
    row("Parameters", [r["params"] for r in all_results])
    row("Layers", [str(r["layers"]) for r in all_results])
    row("d_model", [str(r["d_model"]) for r in all_results])
    row("Query layer", [str(r["query_layer"]) for r in all_results])
    row("Cells stored", [str(r["layers"] * len(MEMORIES)) for r in all_results])
    row("Recall", [f"{r['recall']:.1f}%" for r in all_results])
    row("Avg genuine score", [f"{r['avg_genuine']:.1f}" for r in all_results])
    row("Avg negative score", [f"{r['avg_negative']:.1f}" for r in all_results])
    row("Signal-to-noise", [f"{r['snr']:+.1f}" for r in all_results])

    # Per-query comparison.
    print(f"\n  ── Per-Query ──\n")
    print(f"  {'Query':<45}", end="")
    for h in headers:
        print(f"  {h:>{col_w - 2}}", end="")
    print()
    print(f"  {'─' * (45 + col_w * len(all_results))}")

    for query_text, _ in SPECIFIC_QUERIES:
        short_q = query_text[:43]
        print(f"  {short_q:<45}", end="")
        for r in all_results:
            d = r["details"][query_text]
            mark = f"✓{d['rank']}" if d["hit"] else "✗"
            print(f"  {mark:>{col_w - 2}}", end="")
        print()

    # Misses.
    for r in all_results:
        misses = [q for q, _ in SPECIFIC_QUERIES if not r["details"][q]["hit"]]
        if misses:
            print(f"\n  {r['name']} misses ({len(misses)}):")
            for q in misses:
                expected = [e for qt, e in SPECIFIC_QUERIES if qt == q][0]
                print(f"    ✗ \"{q}\"")
                print(f"      Expected: memory {expected} — \"{MEMORIES[expected[0]][:55]}\"")

    print(f"\n{'=' * 70}")
    for r in all_results:
        print(f"  {r['name']}: {r['hits']}/{r['total']} ({r['recall']:.1f}%) | SNR: {r['snr']:+.1f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
