#!/usr/bin/env python3
"""Two-Agent Memory Cycle — Fresh Character, Llama 3.2:3b.

A completely new character with original memories. Agent 1 (Experiencer)
stores 12 vivid memories from a day. Agent 2 (Rememberer) retrieves blind.

Character: Maya, a second-year veterinary resident at a university
animal hospital. Day 47 of her surgical rotation.

Usage:
    source .venv/bin/activate
    python experiments/llama3_fresh_agent_validation.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import tardigrade_db

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH = (
    "/Users/storylight/.ollama/models/blobs/"
    "sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff"
)
MODEL_NAME = "llama3.2:3b"

# ── Maya's Day (12 memories) ─────────────────────────────────────────────────

MEMORIES = [
    # 0 — Morning routine
    "Arrived at 6:15am to pre-round on post-op patients. The overnight tech left a sticky note on cage 14: 'Biscuit vomited twice, still eating.' Small relief.",
    # 1 — Difficult case
    "Emergency at 7:30am — a golden retriever named Captain swallowed a corn cob three days ago. Radiographs showed complete obstruction at the jejunum. Dr. Nakamura said I'd assist on the enterotomy.",
    # 2 — Surgery
    "Four hours in surgery. My hands were shaking when Dr. Nakamura let me close the intestinal incision. Two-layer closure, 4-0 PDS, interrupted appositional. She said 'adequate' which from her is basically a standing ovation.",
    # 3 — Emotional moment
    "Captain's owner, a retired firefighter named Glen, was sitting in the lobby the entire time. When I told him Captain made it through surgery, he grabbed my hand with both of his and couldn't speak for ten seconds.",
    # 4 — Lunch
    "Ate cold leftover pasta standing up in the pharmacy because there was no time to sit down. Dropped marinara sauce on my scrub top. Third time this month.",
    # 5 — Teaching moment
    "Showed the two new externs how to read abdominal radiographs. One of them asked if the spleen was a tumor. I remembered asking the exact same question my first week.",
    # 6 — Conflict
    "Got into it with Dr. Patel about whether to discharge the diabetic cat on Lantus or ProZinc. He pulled rank. I'm still convinced ProZinc was the right call for this cat's insulin curve.",
    # 7 — Small observation
    "The hospital's orange tabby, Chairman Meow, was sleeping in the centrifuge room again. Someone put a tiny surgical cap on him. Nobody knows who.",
    # 8 — Anxiety
    "Missed a call from the board certification office. Spent twenty minutes catastrophizing that my application had a problem before calling back. It was just confirming my exam date.",
    # 9 — Helping a colleague
    "Stayed late to help Raj place a jugular catheter on a fractious feral cat. Took three attempts and a towel burrito. Raj's hands were steadier than mine by the third try.",
    # 10 — Quiet moment
    "Sat with Biscuit in cage 14 for ten minutes after everyone left. She put her head on my lap. Post-op day three and her appetite is back. That's the whole reason I do this.",
    # 11 — Drive home
    "Drove home at 9pm listening to a podcast about compassion fatigue in veterinary medicine. Cried a little at the part about learning to hold other people's grief without absorbing it.",
]

# ── Queries (blind — as if Agent 2 is recalling the day) ─────────────────────

BROAD_QUERIES = [
    ("What happened at the hospital today", list(range(12))),
    ("Were there any emergencies", [1, 2]),
    ("Did anything make me emotional", [3, 10, 11]),
    ("What surgeries did I do or assist with", [1, 2]),
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


def load_model():
    from llama_cpp import Llama

    print(f"  Loading {MODEL_NAME}...", end=" ", flush=True)
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=512,
        n_gpu_layers=-1,
        verbose=False,
        embedding=True,
    )
    d_model = llm.n_embd()
    print(f"OK (d_model={d_model})")
    return llm, d_model


def text_to_vector(llm, text):
    output = llm.create_embedding(text)
    embeddings = np.array(output["data"][0]["embedding"], dtype=np.float32)
    if embeddings.ndim == 1:
        return embeddings
    return embeddings.mean(axis=0)


def run_experiment():
    print("=" * 70)
    print("Two-Agent Memory Cycle — Fresh Character")
    print("Character: Maya, veterinary surgical resident, day 47")
    print("Model: Llama 3.2:3b (3072-dim pooled embeddings)")
    print("=" * 70)

    llm, d_model = load_model()

    db_dir = Path(tempfile.mkdtemp(prefix="tardigrade_maya_"))
    engine = tardigrade_db.Engine(str(db_dir))

    # ── Phase 1: Store ───────────────────────────────────────────────────

    print(f"\n{'─' * 70}")
    print("Phase 1: Storing Memories (Agent 1 — Experiencer)")
    print(f"{'─' * 70}\n")

    for i, memory in enumerate(MEMORIES):
        vec = text_to_vector(llm, memory)
        salience = min(75.0 + i * 2, 100.0)
        cell_id = engine.mem_write(1, 0, vec, vec, salience, None)
        print(f"  [{cell_id:>2}] \"{memory[:75]}\"")

    print(f"\n  Total cells: {engine.cell_count()}, dim: {d_model}")

    # ── Phase 2: Retrieve ────────────────────────────────────────────────

    print(f"\n{'─' * 70}")
    print("Phase 2: Blind Retrieval (Agent 2 — Rememberer)")
    print(f"{'─' * 70}")

    genuine_scores = []
    negative_scores = []

    # Broad.
    print("\n  ── Broad Queries ──\n")
    for query_text, expected in BROAD_QUERIES:
        query_vec = text_to_vector(llm, query_text)
        results = engine.mem_read(query_vec, 5, 1)
        retrieved = [r.cell_id for r in results]
        hit = any(r.cell_id in expected for r in results)
        genuine_scores.extend(r.score for r in results)

        status = "✓" if hit else "✗"
        top = results[0] if results else None
        top_mem = MEMORIES[top.cell_id][:60] if top else "—"
        print(f"  {status} \"{query_text}\"")
        print(f"    → score={top.score:>10.2f} | cell={top.cell_id} | \"{top_mem}\"")

    # Specific.
    print("\n  ── Specific Queries ──\n")
    hits = 0
    misses = []

    for query_text, expected in SPECIFIC_QUERIES:
        query_vec = text_to_vector(llm, query_text)
        results = engine.mem_read(query_vec, 5, 1)
        retrieved = [r.cell_id for r in results]
        hit = any(r.cell_id in expected for r in results)
        genuine_scores.extend(r.score for r in results)

        rank = "—"
        if hit:
            hits += 1
            for j, r in enumerate(results):
                if r.cell_id in expected:
                    rank = f"#{j+1}"
                    break

        status = "✓" if hit else "✗"
        top = results[0] if results else None
        top_mem = MEMORIES[top.cell_id][:60] if top else "—"
        print(f"  {status} \"{query_text}\"")
        print(f"    → score={top.score:>10.2f} | rank={rank} | \"{top_mem}\"")

        if not hit:
            misses.append((query_text, expected, retrieved))

    # Negative.
    print("\n  ── Negative Queries ──\n")
    for query_text in NEGATIVE_QUERIES:
        query_vec = text_to_vector(llm, query_text)
        results = engine.mem_read(query_vec, 3, 1)
        top = results[0] if results else None
        if top:
            negative_scores.append(top.score)
        top_mem = MEMORIES[top.cell_id][:60] if top else "—"
        print(f"  ○ \"{query_text}\"")
        print(f"    → score={top.score:>10.2f} | \"{top_mem}\"")

    # ── Summary ──────────────────────────────────────────────────────────

    print(f"\n{'─' * 70}")
    print("Results")
    print(f"{'─' * 70}\n")

    total = len(SPECIFIC_QUERIES)
    recall = hits / total * 100
    avg_genuine = np.mean(genuine_scores) if genuine_scores else 0
    avg_negative = np.mean(negative_scores) if negative_scores else 0
    snr_gap = avg_genuine - avg_negative

    print(f"  Character:          Maya, vet surgical resident")
    print(f"  Model:              {MODEL_NAME} ({d_model}-dim)")
    print(f"  Memories stored:    {len(MEMORIES)}")
    print(f"  Specific recall:    {hits}/{total} ({recall:.1f}%)")
    print(f"  Avg genuine score:  {avg_genuine:.2f}")
    print(f"  Avg negative score: {avg_negative:.2f}")
    print(f"  Signal-to-noise:    {snr_gap:+.2f}")

    if misses:
        print(f"\n  ── Missed ({len(misses)}) ──")
        for query, expected, retrieved in misses:
            print(f"    Query:    \"{query}\"")
            print(f"    Expected: cell {expected} — \"{MEMORIES[expected[0]][:65]}\"")
            print(f"    Got:      cells {retrieved}")

    # Comparison.
    print(f"\n  ── Cross-Experiment Comparison ──")
    print(f"  {'Experiment':<35} {'Recall':>8} {'SNR':>8}")
    print(f"  {'─' * 53}")
    print(f"  {'Kael / word-hash (768d)':<35} {'91.7%':>8} {'~0.01':>8}")
    print(f"  {'Kael / GPT-2 KV (768d, 117M)':<35} {'80.0%':>8} {'~1600':>8}")
    print(f"  {'Kael / Llama 3.2 emb (3072d, 3B)':<35} {'100.0%':>8} {'~0':>8}")
    llama_label = f"Maya / Llama 3.2 emb ({d_model}d, 3B)"
    print(f"  {llama_label:<35} {f'{recall:.1f}%':>8} {f'{snr_gap:+.0f}':>8}")

    # Governance.
    print(f"\n  ── Governance ──")
    for cid in range(min(5, engine.cell_count())):
        imp = engine.cell_importance(cid)
        tier = ["Draft", "Validated", "Core"][engine.cell_tier(cid)]
        print(f"    Cell {cid}: importance={imp:.1f}, tier={tier}")

    shutil.rmtree(db_dir)

    print(f"\n{'=' * 70}")
    verdict = "PASS" if recall >= 90 else "PARTIAL" if recall >= 70 else "FAIL"
    print(f"VERDICT: {verdict} — {hits}/{total} ({recall:.1f}% recall)")
    print(f"{'=' * 70}")

    return recall, snr_gap


if __name__ == "__main__":
    run_experiment()
