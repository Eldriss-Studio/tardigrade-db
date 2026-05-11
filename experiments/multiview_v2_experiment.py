"""Experiment: Multi-View v2 — parent-document pattern with LLM-generated views.

Tests the B+C combined approach:
- B: Views are retrieval cells on the canonical pack (add_view_keys)
- C: LLM generates diverse questions, cosine filter rejects near-duplicates

Compares against baseline (centered refinement, no views) on the same
10-fact Sonia corpus used in the v1 diagnosis.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import tardigrade_db
from tardigrade_hooks.constants import DEFAULT_CAPTURE_LAYER_RATIO
from tardigrade_hooks.encoding import encode_per_token
from tardigrade_hooks.view_generator import ViewGenerator, filter_diverse

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OWNER = 1

print(f"Loading {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float32, attn_implementation="eager",
).to(DEVICE)
model.requires_grad_(False)

n_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
query_layer = int(n_layers * DEFAULT_CAPTURE_LAYER_RATIO)
print(f"Model: {n_layers} layers, hidden={hidden_size}, query_layer={query_layer}")


def capture_hidden(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[query_layer][0][1:].cpu().numpy().astype(np.float32)


def capture_key(text: str) -> np.ndarray:
    h = capture_hidden(text)
    return encode_per_token(h, hidden_size)


FACTS = [
    "Sonia translated a pharmaceutical patent from German to English for a Berlin-based biotech startup in March 2024.",
    "Sonia's grandmother taught her to play the balalaika every Sunday afternoon in their apartment in Kazan.",
    "Sonia completed a 42-kilometer ultramarathon through the Scottish Highlands in under five hours.",
    "Sonia designed a waste-heat recovery system for a cement kiln in Rajasthan that cut emissions by 18 percent.",
    "Sonia published a peer-reviewed paper on mycorrhizal networks in boreal peatlands in the Journal of Ecology.",
    "Sonia built a mass spectrometer calibration jig from surplus lab equipment for her university's chemistry department.",
    "Sonia organized a neighborhood composting cooperative in Kreuzberg that now processes two tonnes of organic waste per month.",
    "Sonia wrote a Python library for simulating fluid dynamics in microfluidic chip designs.",
    "Sonia negotiated a three-year supply contract for rare-earth magnets with a mining cooperative in Baotou, Inner Mongolia.",
    "Sonia restored a 1967 Jawa 350 motorcycle she found rusted in a barn outside Brno.",
]

SPECIFIC_QUERIES = [
    ("Who translated a pharmaceutical patent from German?", 0),
    ("Who ran an ultramarathon in Scotland?", 2),
    ("Who published a paper on mycorrhizal networks?", 4),
    ("Who restored a 1967 Jawa motorcycle?", 9),
    ("Who built a mass spectrometer jig?", 5),
]

MODERATE_QUERIES = [
    ("What did Sonia do related to translation work?", 0),
    ("Tell me about Sonia's athletic achievements", 2),
    ("What scientific research has Sonia done?", 4),
    ("What engineering projects has Sonia worked on?", 3),
    ("What environmental work has Sonia been involved in?", 6),
]

VAGUE_QUERIES = [
    ("What does Sonia know about languages?", 0),
    ("Has Sonia done anything outdoorsy?", 2),
    ("What's Sonia's connection to nature or ecology?", 4),
    ("Tell me something mechanical about Sonia", 9),
    ("What's Sonia's background with chemicals or materials?", 5),
]


def run_queries(engine, queries, pack_ids, label, k=5):
    hits = 0
    for query_text, fact_idx in queries:
        query_key = capture_key(query_text)
        results = engine.mem_read_pack(query_key, k, OWNER)
        found_packs = {r["pack_id"] for r in results}
        expected = pack_ids[fact_idx]
        hit = expected in found_packs
        if hit:
            hits += 1
        else:
            print(f"    MISS: '{query_text[:55]}' expected={expected}, got={found_packs}")
    r = hits / len(queries) if queries else 0
    print(f"  {label}: R@{k} = {hits}/{len(queries)} = {r:.0%}")
    return r


# ===========================================================================
# Experiment 1: BASELINE
# ===========================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 1: Baseline — 10 facts, centered, no views")
print("=" * 70)

tmpdir1 = tempfile.mkdtemp()
engine1 = tardigrade_db.Engine(tmpdir1, vamana_threshold=9999)

pack_ids1 = []
for fact in FACTS:
    key = capture_key(fact)
    pid = engine1.mem_write_pack(
        OWNER, key, [(0, np.zeros(8, dtype=np.float32))], 80.0, text=fact,
    )
    pack_ids1.append(pid)

engine1.set_refinement_mode("centered")
print(f"Stored {len(pack_ids1)} facts")

r_sp_base = run_queries(engine1, SPECIFIC_QUERIES, pack_ids1, "Specific")
r_mo_base = run_queries(engine1, MODERATE_QUERIES, pack_ids1, "Moderate")
r_va_base = run_queries(engine1, VAGUE_QUERIES, pack_ids1, "Vague")


# ===========================================================================
# Experiment 2: MULTI-VIEW v2
# ===========================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 2: Multi-view v2 — LLM questions + add_view_keys")
print("=" * 70)

tmpdir2 = tempfile.mkdtemp()
engine2 = tardigrade_db.Engine(tmpdir2, vamana_threshold=9999)

pack_ids2 = []
for fact in FACTS:
    key = capture_key(fact)
    pid = engine2.mem_write_pack(
        OWNER, key, [(0, np.zeros(8, dtype=np.float32))], 80.0, text=fact,
    )
    pack_ids2.append(pid)

view_gen = ViewGenerator(model=model, tokenizer=tokenizer, mode="llm")
total_views = 0
t0 = time.time()

for i, pid in enumerate(pack_ids2):
    fact_text = FACTS[i]

    view_texts = view_gen.generate(fact_text)
    print(f"  Fact {i}: generated {len(view_texts)} questions")
    for vt in view_texts:
        print(f"    -> {vt[:70]}")

    view_hiddens = [capture_hidden(vt) for vt in view_texts]
    diverse_hiddens = filter_diverse(view_hiddens, threshold=0.92, max_kept=3)
    print(f"    Diversity filter: {len(view_hiddens)} -> {len(diverse_hiddens)} kept")

    view_keys = [encode_per_token(h, hidden_size) for h in diverse_hiddens]
    if view_keys:
        count = engine2.add_view_keys(pid, view_keys)
        total_views += count

t_consolidate = time.time() - t0
print(f"\nAttached {total_views} view keys in {t_consolidate:.1f}s")
engine2.set_refinement_mode("centered")

r_sp_mv = run_queries(engine2, SPECIFIC_QUERIES, pack_ids2, "Specific")
r_mo_mv = run_queries(engine2, MODERATE_QUERIES, pack_ids2, "Moderate")
r_va_mv = run_queries(engine2, VAGUE_QUERIES, pack_ids2, "Vague")


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'':30s} {'Specific':>10s} {'Moderate':>10s} {'Vague':>10s}")
print(f"{'Baseline (centered)':30s} {r_sp_base:>9.0%} {r_mo_base:>9.0%} {r_va_base:>9.0%}")
print(f"{'+ Multi-view v2 (LLM+dedup)':30s} {r_sp_mv:>9.0%} {r_mo_mv:>9.0%} {r_va_mv:>9.0%}")
d_mo = r_mo_mv - r_mo_base
d_va = r_va_mv - r_va_base
print(f"{'Delta':30s} {'--':>10s} {d_mo:>+9.0%} {d_va:>+9.0%}")
print(f"\nView keys attached: {total_views} ({total_views / len(FACTS):.1f} per fact)")
print(f"Model: {MODEL_NAME}, Device: {DEVICE}, Layer: {query_layer}/{n_layers}")

print("\nSuccess criteria:")
print(f"  Specific >= 100%: {'PASS' if r_sp_mv >= 1.0 else 'FAIL'} ({r_sp_mv:.0%})")
print(f"  Moderate >= 80%:  {'PASS' if r_mo_mv >= 0.8 else 'FAIL'} ({r_mo_mv:.0%})")
print(f"  Vague > 60%:      {'PASS' if r_va_mv > 0.6 else 'FAIL'} ({r_va_mv:.0%})")
