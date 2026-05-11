"""Experiment: Reflective Latent Search on 10-fact Sonia corpus.

Tests keyword expansion, multi-phrasing, and both combined.
Compares against baseline (centered, single-shot).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import tardigrade_db
from tardigrade_hooks.constants import DEFAULT_CAPTURE_LAYER_RATIO
from tardigrade_hooks.encoding import encode_per_token
from tardigrade_hooks.rls import (
    KeywordExpansionStrategy,
    MultiPhrasingStrategy,
    ReflectiveLatentSearch,
)

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


def capture_key(text):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    h = out.hidden_states[query_layer][0][1:].cpu().numpy().astype(np.float32)
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

SPECIFIC = [
    ("Who translated a pharmaceutical patent from German?", 0),
    ("Who ran an ultramarathon in Scotland?", 2),
    ("Who published a paper on mycorrhizal networks?", 4),
    ("Who restored a 1967 Jawa motorcycle?", 9),
    ("Who built a mass spectrometer jig?", 5),
]

MODERATE = [
    ("What did Sonia do related to translation work?", 0),
    ("Tell me about Sonia's athletic achievements", 2),
    ("What scientific research has Sonia done?", 4),
    ("What engineering projects has Sonia worked on?", 3),
    ("What environmental work has Sonia been involved in?", 6),
]

VAGUE = [
    ("What does Sonia know about languages?", 0),
    ("Has Sonia done anything outdoorsy?", 2),
    ("What's Sonia's connection to nature or ecology?", 4),
    ("Tell me something mechanical about Sonia", 9),
    ("What's Sonia's background with chemicals or materials?", 5),
]


def run_baseline(engine, queries, pack_ids, label, k=5):
    hits = 0
    for text, idx in queries:
        qk = capture_key(text)
        results = engine.mem_read_pack(qk, k, OWNER)
        if pack_ids[idx] in {r["pack_id"] for r in results}:
            hits += 1
        else:
            print(f"    MISS: '{text[:50]}' expected={pack_ids[idx]}")
    r = hits / len(queries) if queries else 0
    print(f"  {label}: R@{k} = {hits}/{len(queries)} = {r:.0%}")
    return r


def run_rls(rls, engine, queries, pack_ids, cell_to_pack, label, k=5):
    hits = 0
    for text, idx in queries:
        handles = rls.query(text, top_k=k)
        found_packs = set()
        for h in handles:
            pid = cell_to_pack.get(int(h.cell_id))
            if pid is not None:
                found_packs.add(pid)
        if pack_ids[idx] in found_packs:
            hits += 1
        else:
            print(f"    MISS: '{text[:50]}' expected={pack_ids[idx]}, got packs={found_packs}")
    r = hits / len(queries) if queries else 0
    print(f"  {label}: R@{k} = {hits}/{len(queries)} = {r:.0%}")
    return r


configs = [
    ("Baseline (centered)", None),
    ("RLS keyword", [KeywordExpansionStrategy()]),
    ("RLS multiphrasing", [MultiPhrasingStrategy()]),
    ("RLS both", [KeywordExpansionStrategy(), MultiPhrasingStrategy()]),
]

results = {}

for name, strategies in configs:
    print(f"\n{'=' * 60}")
    print(f"CONFIG: {name}")
    print(f"{'=' * 60}")

    tmpdir = tempfile.mkdtemp()
    engine = tardigrade_db.Engine(tmpdir, vamana_threshold=9999)
    engine.set_refinement_mode("centered")

    pack_ids = []
    cell_to_pack = {}
    for fact in FACTS:
        key = capture_key(fact)
        pid = engine.mem_write_pack(
            OWNER, key, [(0, np.zeros(8, dtype=np.float32))], 80.0, text=fact,
        )
        pack_ids.append(pid)

    # Build cell_to_pack mapping from pack directory
    for pid in pack_ids:
        packs = engine.list_packs(OWNER)
        for p in packs:
            cell_to_pack[p["pack_id"]] = p["pack_id"]

    if strategies is None:
        sp = run_baseline(engine, SPECIFIC, pack_ids, "Specific")
        mo = run_baseline(engine, MODERATE, pack_ids, "Moderate")
        va = run_baseline(engine, VAGUE, pack_ids, "Vague")
    else:
        # Show what reformulation produces for the vague queries
        for s in strategies:
            for text, _ in VAGUE[:2]:
                reformulated = s.reformulate(text)
                print(f"  {s.__class__.__name__}: '{text[:40]}' -> {reformulated}")

        rls = ReflectiveLatentSearch(
            engine=engine, model=model, tokenizer=tokenizer,
            query_layer=query_layer, hidden_size=hidden_size,
            owner=OWNER, k=5, strategies=strategies,
            confidence_threshold=1.5, max_attempts=3,
        )

        # For RLS we need cell_id -> pack_id mapping
        # mem_read_tokens returns cell-level results, not pack-level
        # Use mem_read_pack for consistency
        sp = run_baseline(engine, SPECIFIC, pack_ids, "Specific")  # specific doesn't need RLS

        # For moderate/vague, use RLS via direct engine query with reformulated text
        mo_hits = 0
        for text, idx in MODERATE:
            # Try original first
            qk = capture_key(text)
            orig_results = engine.mem_read_pack(qk, 5, OWNER)
            found = {r["pack_id"] for r in orig_results}

            if pack_ids[idx] not in found:
                # Reformulate and try again
                for s in strategies:
                    for variant in s.reformulate(text):
                        vk = capture_key(variant)
                        var_results = engine.mem_read_pack(vk, 5, OWNER)
                        found.update(r["pack_id"] for r in var_results)

            if pack_ids[idx] in found:
                mo_hits += 1
            else:
                print(f"    MISS: '{text[:50]}' expected={pack_ids[idx]}")
        mo = mo_hits / len(MODERATE)
        print(f"  Moderate: R@5 = {mo_hits}/{len(MODERATE)} = {mo:.0%}")

        va_hits = 0
        for text, idx in VAGUE:
            qk = capture_key(text)
            orig_results = engine.mem_read_pack(qk, 5, OWNER)
            found = {r["pack_id"] for r in orig_results}

            if pack_ids[idx] not in found:
                for s in strategies:
                    for variant in s.reformulate(text):
                        vk = capture_key(variant)
                        var_results = engine.mem_read_pack(vk, 5, OWNER)
                        found.update(r["pack_id"] for r in var_results)

            if pack_ids[idx] in found:
                va_hits += 1
            else:
                print(f"    MISS: '{text[:50]}' expected={pack_ids[idx]}")
        va = va_hits / len(VAGUE)
        print(f"  Vague: R@5 = {va_hits}/{len(VAGUE)} = {va:.0%}")

    results[name] = (sp, mo, va)


print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"{'Config':30s} {'Specific':>10s} {'Moderate':>10s} {'Vague':>10s}")
for name, (sp, mo, va) in results.items():
    print(f"{name:30s} {sp:>9.0%} {mo:>9.0%} {va:>9.0%}")

baseline = results["Baseline (centered)"]
print(f"\nDeltas vs baseline:")
for name, (sp, mo, va) in results.items():
    if name == "Baseline (centered)":
        continue
    print(f"  {name:30s} {sp-baseline[0]:>+9.0%} {mo-baseline[1]:>+9.0%} {va-baseline[2]:>+9.0%}")
