#!/usr/bin/env python3
"""Two-Agent Memory Cycle — Llama 3.2:3b Validation.

Replicates the GPT-2 two-agent experiment with Llama 3.2:3b (3B params,
3072-dim hidden states) to validate that richer latent representations
improve recall over GPT-2's 80%.

Same character (Kael, day 3 at NovaBridge), same memories, same queries.
Direct apples-to-apples comparison.

Usage:
    source .venv/bin/activate
    python experiments/llama3_two_agent_validation.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import tardigrade_db

# ── Configuration ────────────────────────────────────────────────────────────

# Ollama blob path for llama3.2:3b (2.0 GB, Q4_K_M)
MODEL_PATH = (
    "/Users/storylight/.ollama/models/blobs/"
    "sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff"
)
MODEL_NAME = "llama3.2:3b"

# ── Memories (same as GPT-2 test, from the Kael character) ───────────────────

MEMORIES = [
    "Walked in at 8:47am, the coffee machine line was three people deep. Quiet dread settling in, third day and still no routine.",
    "Marcus flagged 502 errors in the auth service during standup. Priya's PR is blocked on my review, I should look at it after lunch.",
    "Spent two hours on a KeyError in user_session.py line 74. Turned out I was on the wrong branch the whole time. Wanted to disappear.",
    "Walked into the kitchen mid-tense conversation between two people I don't know. Backed out immediately, face hot.",
    "Ate a sad arugula salad alone at the window table. Watched the parking lot and spiraled into imposter thoughts.",
    "Explained React 18 strict mode double-effect behavior to Priya at her desk. She seemed genuinely grateful.",
    "Overheard Dev and Tomas at the standing desks deep in a debate about rewriting the billing service from scratch.",
    "Jordan sent me a cryptic Slack at 3:30pm: 'Hey — do you have a sec?' Heart rate spiked. Turned out to be about the printer.",
    "Noticed Sol has a framed tardigrade microscope print on his desk. Made me smile. Didn't say anything.",
    "Left at 6:15pm. Alone on the BART platform, heavy imposter weight on my shoulders, wondering if I'll ever feel like I belong.",
]

# ── Queries ──────────────────────────────────────────────────────────────────

# Each query maps to expected memory indices (ground truth).
BROAD_QUERIES = [
    ("What happened at work today", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ("Did anything embarrassing happen", [2, 3]),
    ("What technical problems did I deal with", [1, 2]),
    ("Who did I talk to or interact with", [1, 5, 7]),
]

SPECIFIC_QUERIES = [
    ("Dev and Tomas standing desks conversation", [6]),
    ("Marcus auth service standup morning", [1]),
    ("Jordan Slack message manager afternoon", [7]),
    ("Sol engineer framed photo desk quiet", [8]),
    ("What did I eat for lunch", [4]),
    ("KeyError bug in Python code", [2]),
    ("How did I feel leaving the office at the end of the day", [9]),
    ("Walking into kitchen awkward conversation", [3]),
    ("Helping Priya with React code", [5]),
    ("Coffee machine morning arrival", [0]),
]

NEGATIVE_QUERIES = [
    "Meeting with CEO about funding round",
    "Fire alarm at the office building",
    "Team outing at the bowling alley",
]


def load_model():
    """Load Llama 3.2:3b via llama-cpp-python."""
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
    """Extract embedding from model's final layer, mean-pooled across tokens."""
    output = llm.create_embedding(text)
    embeddings = np.array(output["data"][0]["embedding"], dtype=np.float32)
    if embeddings.ndim == 1:
        return embeddings
    return embeddings.mean(axis=0)


def run_experiment():
    """Run the full two-agent memory cycle experiment."""
    print("=" * 70)
    print("Two-Agent Memory Cycle — Llama 3.2:3b Validation")
    print("=" * 70)

    llm, d_model = load_model()

    db_dir = Path(tempfile.mkdtemp(prefix="tardigrade_llama3_exp_"))
    print(f"  DB: {db_dir}")
    print(f"  Memories: {len(MEMORIES)}")
    print(f"  Broad queries: {len(BROAD_QUERIES)}")
    print(f"  Specific queries: {len(SPECIFIC_QUERIES)}")
    print(f"  Negative queries: {len(NEGATIVE_QUERIES)}")

    engine = tardigrade_db.Engine(str(db_dir))

    # ── Phase 1: Agent 1 stores memories ─────────────────────────────────

    print(f"\n{'─' * 70}")
    print("Phase 1: Storing Memories (Agent 1 — Experiencer)")
    print(f"{'─' * 70}\n")

    memory_vectors = []
    for i, memory in enumerate(MEMORIES):
        vec = text_to_vector(llm, memory)
        memory_vectors.append(vec)
        salience = min(80.0 + i * 2, 100.0)
        cell_id = engine.mem_write(1, 0, vec, vec, salience, None)
        print(f"  [{cell_id:>2}] \"{memory[:75]}...\"")

    print(f"\n  Total cells: {engine.cell_count()}, dim: {d_model}")

    # ── Phase 2: Agent 2 retrieves (blind) ───────────────────────────────

    print(f"\n{'─' * 70}")
    print("Phase 2: Blind Retrieval (Agent 2 — Rememberer)")
    print(f"{'─' * 70}")

    # Track all results for summary.
    all_results = {}
    genuine_scores = []
    negative_scores = []

    # Broad queries.
    print("\n  ── Broad Queries ──\n")
    for query_text, expected_indices in BROAD_QUERIES:
        query_vec = text_to_vector(llm, query_text)
        results = engine.mem_read(query_vec, 5, 1)

        retrieved_ids = [r.cell_id for r in results]
        top_score = results[0].score if results else 0
        hit = any(r.cell_id in expected_indices for r in results)

        genuine_scores.extend([r.score for r in results])
        all_results[query_text] = {
            "type": "broad",
            "expected": expected_indices,
            "retrieved": retrieved_ids,
            "top_score": top_score,
            "hit": hit,
        }

        status = "✓" if hit else "✗"
        top_memory = MEMORIES[results[0].cell_id] if results else "N/A"
        print(f"  {status} \"{query_text}\"")
        print(f"    → score={top_score:>10.2f} | cell={retrieved_ids[0] if retrieved_ids else '?'} | \"{top_memory[:60]}\"")

    # Specific queries.
    print("\n  ── Specific Queries ──\n")
    specific_hits = 0
    specific_total = len(SPECIFIC_QUERIES)

    for query_text, expected_indices in SPECIFIC_QUERIES:
        query_vec = text_to_vector(llm, query_text)
        results = engine.mem_read(query_vec, 5, 1)

        retrieved_ids = [r.cell_id for r in results]
        top_score = results[0].score if results else 0
        hit = any(r.cell_id in expected_indices for r in results)
        if hit:
            specific_hits += 1

        genuine_scores.extend([r.score for r in results])
        all_results[query_text] = {
            "type": "specific",
            "expected": expected_indices,
            "retrieved": retrieved_ids,
            "top_score": top_score,
            "hit": hit,
        }

        status = "✓" if hit else "✗"
        top_memory = MEMORIES[results[0].cell_id] if results else "N/A"
        rank = "—"
        for j, r in enumerate(results):
            if r.cell_id in expected_indices:
                rank = f"#{j+1}"
                break
        print(f"  {status} \"{query_text}\"")
        print(f"    → score={top_score:>10.2f} | rank={rank} | \"{top_memory[:60]}\"")

    # Negative queries.
    print("\n  ── Negative Queries (should score lower) ──\n")
    for query_text in NEGATIVE_QUERIES:
        query_vec = text_to_vector(llm, query_text)
        results = engine.mem_read(query_vec, 3, 1)

        top_score = results[0].score if results else 0
        negative_scores.append(top_score)

        top_memory = MEMORIES[results[0].cell_id] if results else "N/A"
        print(f"  ○ \"{query_text}\"")
        print(f"    → score={top_score:>10.2f} | \"{top_memory[:60]}\"")

    # ── Summary ──────────────────────────────────────────────────────────

    print(f"\n{'─' * 70}")
    print("Summary")
    print(f"{'─' * 70}\n")

    recall = specific_hits / specific_total * 100
    avg_genuine = np.mean(genuine_scores) if genuine_scores else 0
    avg_negative = np.mean(negative_scores) if negative_scores else 0
    snr_gap = avg_genuine - avg_negative

    print(f"  Model:              {MODEL_NAME} ({d_model}-dim)")
    print(f"  Memories stored:    {len(MEMORIES)}")
    print(f"  Specific recall:    {specific_hits}/{specific_total} ({recall:.1f}%)")
    print(f"  Avg genuine score:  {avg_genuine:.2f}")
    print(f"  Avg negative score: {avg_negative:.2f}")
    print(f"  Signal-to-noise:    {snr_gap:.2f}")
    print()

    # Comparison with previous results.
    print("  ── Comparison ──")
    print(f"  {'Method':<25} {'Recall':>8} {'SNR Gap':>10}")
    print(f"  {'─' * 45}")
    print(f"  {'Word-hash (768d)':<25} {'91.7%':>8} {'~0.01':>10}  {'word overlap'}")
    print(f"  {'GPT-2 KV (768d, 117M)':<25} {'80.0%':>8} {'~1,600':>10}  {'per-layer KV tensors'}")
    llama_label = f"Llama 3.2 ({d_model}d, 3B)"
    print(f"  {llama_label:<25} {f'{recall:.1f}%':>8} {f'{snr_gap:.0f}':>10}  {'pooled embeddings (*)'}")
    print()
    print("  (*) SNR is low because pooled embeddings cluster densely. Per-layer KV")
    print("      tensors (like the GPT-2 test) would give wider score separation.")

    # ── Missed memories analysis ─────────────────────────────────────────

    missed = []
    for query_text, expected_indices in SPECIFIC_QUERIES:
        r = all_results[query_text]
        if not r["hit"]:
            missed.append((query_text, expected_indices, r["retrieved"]))

    if missed:
        print(f"\n  ── Missed Memories ({len(missed)}) ──")
        for query, expected, retrieved in missed:
            print(f"\n    Query: \"{query}\"")
            print(f"    Expected: memory {expected} — \"{MEMORIES[expected[0]][:70]}\"")
            print(f"    Got: cells {retrieved}")

    # Governance check.
    print(f"\n  ── Governance ──")
    for cid in range(min(5, engine.cell_count())):
        imp = engine.cell_importance(cid)
        tier = ["Draft", "Validated", "Core"][engine.cell_tier(cid)]
        print(f"    Cell {cid}: importance={imp:.1f}, tier={tier}")

    # Cleanup.
    shutil.rmtree(db_dir)

    print(f"\n{'=' * 70}")
    verdict = "PASS" if recall >= 90 else "PARTIAL" if recall >= 70 else "FAIL"
    print(f"VERDICT: {verdict} ({recall:.1f}% recall)")
    print(f"{'=' * 70}")

    return recall, snr_gap, all_results


if __name__ == "__main__":
    run_experiment()
