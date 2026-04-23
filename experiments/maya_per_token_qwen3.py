#!/usr/bin/env python3
"""Per-token KV storage with Qwen3-0.6B on Maya's memories.

This is how TardigradeDB is actually designed to work:
  - Each token's hidden state is stored as a separate cell
  - Retrieval matches individual token activations, not averaged blobs
  - The model's fine-grained understanding of each word is preserved

Previous experiments used mean-pooling (average all tokens into one vector),
which destroyed the distinguishing details. This test preserves them.

Compared against mean-pooled Qwen3 from the previous experiment (41.7% recall).

Usage:
    source .venv/bin/activate
    python experiments/maya_per_token_qwen3.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import tardigrade_db

# ── Maya's memories ──────────────────────────────────────────────────────────

MEMORIES = [
    "Arrived at 6:15am to pre-round on post-op patients. The overnight tech left a sticky note on cage 14: 'Biscuit vomited twice, still eating.' Small relief.",
    "Emergency at 7:30am, a golden retriever named Captain swallowed a corn cob three days ago. Radiographs showed complete obstruction at the jejunum. Dr. Nakamura said I would assist on the enterotomy.",
    "Four hours in surgery. My hands were shaking when Dr. Nakamura let me close the intestinal incision. Two-layer closure, 4-0 PDS, interrupted appositional. She said adequate which from her is basically a standing ovation.",
    "Captain's owner, a retired firefighter named Glen, was sitting in the lobby the entire time. When I told him Captain made it through surgery, he grabbed my hand with both of his and couldn't speak for ten seconds.",
    "Ate cold leftover pasta standing up in the pharmacy because there was no time to sit down. Dropped marinara sauce on my scrub top. Third time this month.",
    "Showed the two new externs how to read abdominal radiographs. One of them asked if the spleen was a tumor. I remembered asking the exact same question my first week.",
    "Got into it with Dr. Patel about whether to discharge the diabetic cat on Lantus or ProZinc. He pulled rank. I am still convinced ProZinc was the right call for this cat's insulin curve.",
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

MODEL_NAME = "Qwen/Qwen3-0.6B"


def run_mean_pool(model, tokenizer, n_layers, query_layer):
    """Baseline: mean-pooled per-layer (one cell per memory per layer)."""
    print(f"\n  -- Mean-Pooled (baseline) --")

    db_dir = Path(tempfile.mkdtemp(prefix="tardigrade_maya_meanpool_"))
    engine = tardigrade_db.Engine(str(db_dir))

    for mem_idx, memory in enumerate(MEMORIES):
        inputs = tokenizer(memory, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
        for layer_idx in range(n_layers):
            hidden = outputs.hidden_states[layer_idx + 1].numpy()[0]
            mean_vec = hidden.mean(axis=0).astype(np.float32)
            salience = min(75.0 + mem_idx * 2, 100.0)
            engine.mem_write(1, layer_idx, mean_vec, mean_vec, salience, None)

    print(f"  Stored: {engine.cell_count()} cells ({len(MEMORIES)} mem x {n_layers} layers)")

    results = retrieve_and_score(engine, model, tokenizer, n_layers, query_layer, "mean-pool")
    shutil.rmtree(db_dir)
    return results


def run_per_token(model, tokenizer, n_layers, query_layer):
    """Per-token: each token's hidden state stored as a separate cell."""
    print(f"\n  -- Per-Token (TardigradeDB design) --")

    db_dir = Path(tempfile.mkdtemp(prefix="tardigrade_maya_pertoken_"))
    engine = tardigrade_db.Engine(str(db_dir))

    cell_to_mem = {}

    for mem_idx, memory in enumerate(MEMORIES):
        inputs = tokenizer(memory, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)

        seq_len = inputs["input_ids"].shape[1]

        for layer_idx in range(n_layers):
            hidden = outputs.hidden_states[layer_idx + 1].numpy()[0]  # (seq, d_model)

            for tok_idx in range(seq_len):
                tok_vec = hidden[tok_idx].astype(np.float32)
                salience = min(75.0 + mem_idx * 2, 100.0)
                cell_id = engine.mem_write(1, layer_idx, tok_vec, tok_vec, salience, None)
                cell_to_mem[cell_id] = mem_idx

    print(f"  Stored: {engine.cell_count()} cells ({len(MEMORIES)} mem x {n_layers} layers x tokens)")

    results = retrieve_and_score(
        engine, model, tokenizer, n_layers, query_layer, "per-token",
        cell_to_mem=cell_to_mem,
    )
    shutil.rmtree(db_dir)
    return results


def retrieve_and_score(engine, model, tokenizer, n_layers, query_layer, label, cell_to_mem=None):
    """Run queries and compute recall + SNR."""
    results_map = {}
    genuine_scores = []
    negative_scores = []

    for query_text, expected_mems in SPECIFIC_QUERIES:
        inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)

        query_hidden = outputs.hidden_states[query_layer + 1].numpy()[0]
        query_vec = query_hidden.mean(axis=0).astype(np.float32)

        results = engine.mem_read(query_vec, 10, 1)

        if cell_to_mem is not None:
            retrieved_mems = [cell_to_mem.get(r.cell_id, -1) for r in results]
        else:
            retrieved_mems = [r.cell_id // n_layers for r in results]

        # Deduplicate: first occurrence of each unique memory.
        seen_mems = set()
        unique_mems = []
        for m in retrieved_mems:
            if m not in seen_mems:
                seen_mems.add(m)
                unique_mems.append(m)

        hit = any(m in expected_mems for m in unique_mems[:5])
        genuine_scores.extend(r.score for r in results[:5])

        rank = "---"
        if hit:
            for j, m in enumerate(unique_mems[:5]):
                if m in expected_mems:
                    rank = f"#{j+1}"
                    break

        top_mem_idx = unique_mems[0] if unique_mems else -1
        results_map[query_text] = {
            "hit": hit,
            "rank": rank,
            "top_score": results[0].score if results else 0,
            "top_mem": top_mem_idx,
            "unique_mems": unique_mems[:5],
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

    hits = sum(1 for r in results_map.values() if r["hit"])
    recall = hits / len(SPECIFIC_QUERIES) * 100
    avg_genuine = np.mean(genuine_scores) if genuine_scores else 0
    avg_negative = np.mean(negative_scores) if negative_scores else 0
    snr = avg_genuine - avg_negative

    return {
        "label": label,
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
    print("Per-Token vs Mean-Pool -- Qwen3-0.6B on Maya's Memories")
    print("Same model, same memories, different storage granularity")
    print("=" * 70)

    print(f"\n  Loading {MODEL_NAME}...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        output_hidden_states=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    query_layer = int(n_layers * 0.67)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"OK ({param_count/1e6:.0f}M, {n_layers} layers, d_model={d_model}, query_layer={query_layer})")

    mean_pool = run_mean_pool(model, tokenizer, n_layers, query_layer)
    per_token = run_per_token(model, tokenizer, n_layers, query_layer)

    # ── Per-query comparison ─────────────────────────────────────────────

    print(f"\n{'=' * 70}")
    print("  PER-QUERY COMPARISON")
    print(f"{'=' * 70}\n")

    print(f"  {'Query':<45} {'Mean-Pool':>10} {'Per-Token':>10}")
    print(f"  {'-' * 67}")

    for query_text, expected in SPECIFIC_QUERIES:
        mp = mean_pool["details"][query_text]
        pt = per_token["details"][query_text]
        mp_mark = f"Y{mp['rank']}" if mp["hit"] else "N"
        pt_mark = f"Y{pt['rank']}" if pt["hit"] else "N"
        short_q = query_text[:43]
        print(f"  {short_q:<45} {mp_mark:>10} {pt_mark:>10}")

    # ── Summary ──────────────────────────────────────────────────────────

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}\n")

    print(f"  {'Metric':<25} {'Mean-Pool':>15} {'Per-Token':>15}")
    print(f"  {'-' * 57}")
    mp_recall = f"{mean_pool['recall']:.1f}%"
    pt_recall = f"{per_token['recall']:.1f}%"
    mp_gen = f"{mean_pool['avg_genuine']:.1f}"
    pt_gen = f"{per_token['avg_genuine']:.1f}"
    mp_neg = f"{mean_pool['avg_negative']:.1f}"
    pt_neg = f"{per_token['avg_negative']:.1f}"
    mp_snr = f"{mean_pool['snr']:+.1f}"
    pt_snr = f"{per_token['snr']:+.1f}"
    print(f"  {'Recall':<25} {mp_recall:>15} {pt_recall:>15}")
    print(f"  {'Avg genuine score':<25} {mp_gen:>15} {pt_gen:>15}")
    print(f"  {'Avg negative score':<25} {mp_neg:>15} {pt_neg:>15}")
    print(f"  {'Signal-to-noise':<25} {mp_snr:>15} {pt_snr:>15}")

    # Top memory distribution.
    print(f"\n  -- Top Memory Distribution --")
    mp_tops = [mean_pool["details"][q]["top_mem"] for q, _ in SPECIFIC_QUERIES]
    pt_tops = [per_token["details"][q]["top_mem"] for q, _ in SPECIFIC_QUERIES]
    mp_unique = len(set(mp_tops))
    pt_unique = len(set(pt_tops))
    print(f"  Mean-pool: {mp_unique} unique memories in top-1 across 12 queries")
    print(f"  Per-token: {pt_unique} unique memories in top-1 across 12 queries")

    # Misses.
    for r in [mean_pool, per_token]:
        misses = [q for q, _ in SPECIFIC_QUERIES if not r["details"][q]["hit"]]
        if misses:
            print(f"\n  {r['label']} misses ({len(misses)}):")
            for q in misses:
                expected = [e for qt, e in SPECIFIC_QUERIES if qt == q][0]
                d = r["details"][q]
                print(f"    X \"{q[:55]}\"")
                print(f"      Got mems: {d['unique_mems'][:5]} | Expected: {expected}")

    print(f"\n{'=' * 70}")
    print(f"  Mean-pool: {mean_pool['hits']}/{mean_pool['total']} ({mean_pool['recall']:.1f}%) | SNR: {mean_pool['snr']:+.1f}")
    print(f"  Per-token: {per_token['hits']}/{per_token['total']} ({per_token['recall']:.1f}%) | SNR: {per_token['snr']:+.1f}")
    delta = per_token["recall"] - mean_pool["recall"]
    print(f"  Delta:     {delta:+.1f}% recall")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
