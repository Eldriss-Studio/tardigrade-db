"""Diagnose why multi-view consolidation hurts moderate recall.

Questions:
1. What do the view texts actually look like vs. canonicals?
2. How similar are view K-vectors to canonical K-vectors? (cosine sim)
3. When a query misses, what IS it hitting — which views, from which facts?
4. Are views scoring higher than their own canonical?
5. Would filtering views from results (only using them as retrieval paths
   back to canonical) fix the problem?
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
from tardigrade_hooks.constants import DEFAULT_CAPTURE_LAYER_RATIO, EDGE_SUPPORTS
from tardigrade_hooks.encoding import encode_per_token
from tardigrade_hooks.view_generator import ViewGenerator

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OWNER = 1

print(f"Loading {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float32, attn_implementation="eager",
).to(DEVICE).eval()

n_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
query_layer = int(n_layers * DEFAULT_CAPTURE_LAYER_RATIO)


def capture_hidden(text: str) -> np.ndarray:
    """Return raw hidden states at query_layer (skip pos 0)."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[query_layer][0][1:].cpu().numpy().astype(np.float32)


def capture_key(text: str) -> np.ndarray:
    """Return encoded per-token retrieval key."""
    h = capture_hidden(text)
    return encode_per_token(h, hidden_size)


def mean_pool(hidden: np.ndarray) -> np.ndarray:
    return hidden.mean(axis=0)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


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

MODERATE_QUERIES = [
    ("What did Sonia do related to translation work?", 0),
    ("Tell me about Sonia's athletic achievements", 2),
    ("What scientific research has Sonia done?", 4),
    ("What engineering projects has Sonia worked on?", 3),
    ("What environmental work has Sonia been involved in?", 6),
]

view_gen = ViewGenerator()

# ---------------------------------------------------------------------------
# 1. What do views look like?
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("1. VIEW TEXT QUALITY")
print("=" * 70)

for i, fact in enumerate(FACTS[:3]):
    views = view_gen.generate(fact)
    print(f"\nFact {i}: {fact[:80]}...")
    for j, v in enumerate(views):
        print(f"  View {j}: {v[:80]}...")

# ---------------------------------------------------------------------------
# 2. Cosine similarity: canonical vs. views vs. cross-fact
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("2. LATENT-SPACE SIMILARITY (mean-pooled hidden states)")
print("=" * 70)

fact_hiddens = [capture_hidden(f) for f in FACTS]
fact_means = [mean_pool(h) for h in fact_hiddens]

for i in range(5):
    views = view_gen.generate(FACTS[i])
    view_hiddens = [capture_hidden(v) for v in views]
    view_means = [mean_pool(h) for h in view_hiddens]

    print(f"\nFact {i}: {FACTS[i][:60]}...")
    for j, (vtext, vmean) in enumerate(zip(views, view_means)):
        sim = cosine_sim(fact_means[i], vmean)
        print(f"  View {j} ({vtext[:40]}...): cos={sim:.4f}")

    sims_to_others = []
    for k in range(len(FACTS)):
        if k == i:
            continue
        sims_to_others.append((k, cosine_sim(fact_means[i], fact_means[k])))
    sims_to_others.sort(key=lambda x: -x[1])
    top3 = sims_to_others[:3]
    print(f"  Nearest other facts: {', '.join(f'fact{k}={s:.4f}' for k, s in top3)}")

# ---------------------------------------------------------------------------
# 3. What does a missed query actually hit?
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("3. MISS ANALYSIS: What do moderate queries retrieve?")
print("=" * 70)

tmpdir = tempfile.mkdtemp()
engine = tardigrade_db.Engine(tmpdir, vamana_threshold=9999)

pack_id_to_fact = {}
pack_ids = []
for i, fact in enumerate(FACTS):
    key = capture_key(fact)
    value = np.zeros(8, dtype=np.float32)
    pid = engine.mem_write_pack(OWNER, key, [(0, value)], 80.0, text=fact)
    pack_ids.append(pid)
    pack_id_to_fact[pid] = i

view_pack_to_canonical = {}
for pid in pack_ids:
    text = engine.pack_text(pid)
    views = view_gen.generate(text)
    for view_text in views:
        view_key = capture_key(view_text)
        value = np.zeros(8, dtype=np.float32)
        vid = engine.mem_write_pack(OWNER, view_key, [(0, value)], 50.0, text=view_text)
        engine.add_pack_edge(vid, pid, EDGE_SUPPORTS)
        view_pack_to_canonical[vid] = pid
        pack_id_to_fact[vid] = pack_id_to_fact[pid]

engine.set_refinement_mode("centered")

print(f"\nTotal packs: {len(engine.list_packs(OWNER))} (10 canonical + 30 views)")

for query_text, expected_fact_idx in MODERATE_QUERIES:
    query_key = capture_key(query_text)
    results = engine.mem_read_pack(query_key, 10, OWNER)

    expected_pid = pack_ids[expected_fact_idx]
    print(f"\nQuery: '{query_text}'")
    print(f"  Expected: fact {expected_fact_idx} (pack {expected_pid})")

    for rank, r in enumerate(results[:10]):
        rpid = r["pack_id"]
        score = r["score"]
        rtext = (engine.pack_text(rpid) or "")[:60]
        is_view = rpid in view_pack_to_canonical
        canonical = view_pack_to_canonical.get(rpid, rpid)
        fact_idx = pack_id_to_fact.get(rpid, "?")
        marker = " VIEW" if is_view else " CANON"
        hit = " <<< TARGET" if canonical == expected_pid else ""
        print(f"    #{rank+1} pack={rpid:3d} fact={fact_idx}{marker} score={score:.4f}{hit}")
        print(f"         {rtext}")

# ---------------------------------------------------------------------------
# 4. Canonical vs view scores
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("4. CANONICAL vs VIEW SCORES (per query)")
print("=" * 70)

for query_text, expected_fact_idx in MODERATE_QUERIES:
    query_key = capture_key(query_text)
    results = engine.mem_read_pack(query_key, 40, OWNER)
    score_map = {r["pack_id"]: r["score"] for r in results}

    expected_pid = pack_ids[expected_fact_idx]
    canonical_score = score_map.get(expected_pid, 0.0)

    view_scores = []
    for vid, cpid in view_pack_to_canonical.items():
        if cpid == expected_pid:
            view_scores.append(score_map.get(vid, 0.0))

    print(f"\nQuery: '{query_text[:50]}...'")
    print(f"  Canonical (pack {expected_pid}) score: {canonical_score:.4f}")
    if view_scores:
        print(f"  View scores: {', '.join(f'{s:.4f}' for s in sorted(view_scores, reverse=True))}")

    # Also show what the top-5 canonicals are
    canonical_scores = []
    for pid in pack_ids:
        s = score_map.get(pid, 0.0)
        canonical_scores.append((pid, pack_id_to_fact[pid], s))
    canonical_scores.sort(key=lambda x: -x[2])
    print(f"  Top-5 canonical scores: {', '.join(f'f{fi}={s:.4f}' for _, fi, s in canonical_scores[:5])}")

# ---------------------------------------------------------------------------
# 5. Simulate fix: view->canonical dedup
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("5. SIMULATED FIX: Resolve views to canonical, dedup, top-5")
print("=" * 70)

hits = 0
for query_text, expected_fact_idx in MODERATE_QUERIES:
    query_key = capture_key(query_text)
    results = engine.mem_read_pack(query_key, 10, OWNER)

    seen = set()
    deduped = []
    for r in results:
        rpid = r["pack_id"]
        canonical = view_pack_to_canonical.get(rpid, rpid)
        if canonical not in seen:
            seen.add(canonical)
            deduped.append({"pack_id": canonical, "score": r["score"]})
        if len(deduped) >= 5:
            break

    expected_pid = pack_ids[expected_fact_idx]
    found = {d["pack_id"] for d in deduped}
    hit = expected_pid in found
    hits += int(hit)
    status = "HIT " if hit else "MISS"
    print(f"  {status}: '{query_text[:50]}...' top-5 canonicals: {[d['pack_id'] for d in deduped]}")

print(f"\n  Moderate R@5 with view-to-canonical dedup: {hits}/{len(MODERATE_QUERIES)} = {hits/len(MODERATE_QUERIES):.0%}")
