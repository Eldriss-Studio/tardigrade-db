"""Experiment: Latent-space vague query refinement v2.

Tests three training-free techniques individually and stacked:
1. ZCA whitening (extends mean-centering with covariance normalization)
2. Token importance reweighting (IDF-like corpus-mean distance)
3. Multi-layer query fusion (RRF over layers at 50%, 67%, 83% depth)

All preserve TardigradeDB's tensor-native premise — no text retrieval.
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
from tardigrade_hooks.multi_layer_query import MultiLayerQuery

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


def capture_key(text: str) -> np.ndarray:
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


def make_engine_with_facts(tmpdir):
    engine = tardigrade_db.Engine(tmpdir, vamana_threshold=9999)
    pids = []
    for fact in FACTS:
        key = capture_key(fact)
        pid = engine.mem_write_pack(
            OWNER, key, [(0, np.zeros(8, dtype=np.float32))], 80.0, text=fact,
        )
        pids.append(pid)
    return engine, pids


def run_single_layer(engine, queries, pack_ids, label, k=5):
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


def run_multi_layer(engine, queries, pack_ids, label, k=5):
    mlq = MultiLayerQuery(engine)
    hits = 0
    for text, idx in queries:
        results = mlq.query(model, tokenizer, text, k=k, owner=OWNER)
        if pack_ids[idx] in {r["pack_id"] for r in results}:
            hits += 1
        else:
            print(f"    MISS: '{text[:50]}' expected={pack_ids[idx]}")
    r = hits / len(queries) if queries else 0
    print(f"  {label}: R@{k} = {hits}/{len(queries)} = {r:.0%}")
    return r


configs = [
    ("Baseline (centered)", "centered", False, False),
    ("Whitened", "whitened", False, False),
    ("Whitened + reweight", "whitened", True, False),
    ("Whitened + reweight + multi-layer", "whitened", True, True),
]

results = {}

for name, mode, reweight, multi_layer in configs:
    print(f"\n{'=' * 60}")
    print(f"CONFIG: {name}")
    print(f"{'=' * 60}")

    tmpdir = tempfile.mkdtemp()
    engine, pids = make_engine_with_facts(tmpdir)
    engine.set_refinement_mode(mode)
    if reweight:
        engine.set_token_reweighting(True)

    query_fn = run_multi_layer if multi_layer else run_single_layer
    sp = query_fn(engine, SPECIFIC, pids, "Specific")
    mo = query_fn(engine, MODERATE, pids, "Moderate")
    va = query_fn(engine, VAGUE, pids, "Vague")
    results[name] = (sp, mo, va)


print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"{'Config':40s} {'Specific':>10s} {'Moderate':>10s} {'Vague':>10s}")
for name, (sp, mo, va) in results.items():
    print(f"{name:40s} {sp:>9.0%} {mo:>9.0%} {va:>9.0%}")

baseline = results["Baseline (centered)"]
print(f"\nDeltas vs baseline:")
for name, (sp, mo, va) in results.items():
    if name == "Baseline (centered)":
        continue
    print(f"  {name:40s} {sp-baseline[0]:>+9.0%} {mo-baseline[1]:>+9.0%} {va-baseline[2]:>+9.0%}")

print(f"\nModel: {MODEL_NAME}, Device: {DEVICE}, Layer: {query_layer}/{n_layers}")
best = max(results.items(), key=lambda x: x[1][2])
print(f"Best vague config: {best[0]} at {best[1][2]:.0%}")

print("\nSuccess criteria:")
for name, (sp, mo, va) in results.items():
    sp_ok = "PASS" if sp >= 1.0 else "FAIL"
    mo_ok = "PASS" if mo >= 0.8 else "FAIL"
    va_ok = "PASS" if va > 0.6 else "NEED"
    print(f"  {name}: Sp={sp_ok} Mo={mo_ok} Va={va_ok}")
