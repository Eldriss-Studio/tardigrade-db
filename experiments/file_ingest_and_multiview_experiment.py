"""Experiment: File ingestion + multi-view consolidation — does it actually work?

Tests the core research questions from docs/refs/file-ingest-as-kv-memory.md:
1. Can we ingest text as KV memory and retrieve it with related queries?
2. Does multi-view consolidation improve recall on vague queries?

Uses Qwen3-0.6B on MPS/CPU with real KV capture — no stubs.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import tardigrade_db
from tardigrade_hooks.constants import DEFAULT_CAPTURE_LAYER_RATIO, EDGE_SUPPORTS
from tardigrade_hooks.encoding import encode_per_token
from tardigrade_hooks.chunker import TextChunker
from tardigrade_hooks.consolidator import MemoryConsolidator
from tardigrade_hooks.view_generator import ViewGenerator

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OWNER = 1

print(f"Loading {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32, attn_implementation="eager",
)
model = model.to(DEVICE).eval()

n_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
query_layer = int(n_layers * DEFAULT_CAPTURE_LAYER_RATIO)
print(f"Model: {n_layers} layers, hidden_size={hidden_size}, query_layer={query_layer}")


def capture_kv(text: str) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    """Run a forward pass and capture hidden states as retrieval key."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hidden = out.hidden_states[query_layer][0]  # (seq, hidden_size)
    h_np = hidden[1:].cpu().numpy().astype(np.float32)  # skip pos 0
    retrieval_key = encode_per_token(h_np, hidden_size)
    # Dummy layer payload (we only need retrieval key for this experiment)
    value = np.zeros(8, dtype=np.float32)
    return retrieval_key, [(0, value)]


def query_packs(engine, query_text: str, k: int = 5):
    """Query the engine with real KV from a forward pass, return pack results."""
    inputs = tokenizer(query_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hidden = out.hidden_states[query_layer][0]
    h_np = hidden[1:].cpu().numpy().astype(np.float32)
    query_key = encode_per_token(h_np, hidden_size)
    return engine.mem_read_pack(query_key, k, OWNER)


# ---------------------------------------------------------------------------
# Corpus: 10 novel facts about fictional character "Sonia"
# ---------------------------------------------------------------------------

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
    """Run queries and compute R@k. A hit = expected pack or its views in top-k."""
    hits = 0
    total = len(queries)
    for query_text, fact_idx in queries:
        results = query_packs(engine, query_text, k=k)
        found_packs = {r["pack_id"] for r in results}

        expected = pack_ids[fact_idx]
        # Also count hits on views of the expected pack
        expected_views = set(engine.pack_supports(expected))
        hit = expected in found_packs or bool(expected_views & found_packs)

        if hit:
            hits += 1
        else:
            print(f"    MISS: '{query_text[:55]}' expected={expected}, got={found_packs}")

    r_at_k = hits / total if total > 0 else 0
    print(f"  {label}: R@{k} = {hits}/{total} = {r_at_k:.0%}")
    return r_at_k


# ---------------------------------------------------------------------------
# Experiment 1: Baseline — facts with real KV, centered refinement
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("EXPERIMENT 1: Baseline — 10 facts, centered refinement, no views")
print("=" * 70)

tmpdir = tempfile.mkdtemp()
engine = tardigrade_db.Engine(tmpdir, vamana_threshold=9999)

pack_ids = []
t0 = time.time()
for fact in FACTS:
    key, payloads = capture_kv(fact)
    pid = engine.mem_write_pack(OWNER, key, payloads, 80.0, text=fact)
    pack_ids.append(pid)
t_store = time.time() - t0

print(f"Stored {len(pack_ids)} facts in {t_store:.1f}s")
engine.set_refinement_mode("centered")

r_specific_base = run_queries(engine, SPECIFIC_QUERIES, pack_ids, "Specific")
r_moderate_base = run_queries(engine, MODERATE_QUERIES, pack_ids, "Moderate")
r_vague_base = run_queries(engine, VAGUE_QUERIES, pack_ids, "Vague")


# ---------------------------------------------------------------------------
# Experiment 2: Multi-view consolidation with real KV per view
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("EXPERIMENT 2: Same facts + multi-view consolidation (real KV)")
print("=" * 70)

tmpdir2 = tempfile.mkdtemp()
engine2 = tardigrade_db.Engine(tmpdir2, vamana_threshold=9999)

pack_ids2 = []
for fact in FACTS:
    key, payloads = capture_kv(fact)
    pid = engine2.mem_write_pack(OWNER, key, payloads, 80.0, text=fact)
    pack_ids2.append(pid)

view_gen = ViewGenerator()
total_views = 0
t0 = time.time()

for pid in pack_ids2:
    text = engine2.pack_text(pid)
    views = view_gen.generate(text)
    for view_text in views:
        view_key, view_payloads = capture_kv(view_text)
        vid = engine2.mem_write_pack(OWNER, view_key, view_payloads, 50.0, text=view_text)
        engine2.add_pack_edge(vid, pid, EDGE_SUPPORTS)
        total_views += 1

t_consolidate = time.time() - t0
print(f"Created {total_views} views in {t_consolidate:.1f}s ({total_views // len(pack_ids2)} per fact)")
engine2.set_refinement_mode("centered")

r_specific_mv = run_queries(engine2, SPECIFIC_QUERIES, pack_ids2, "Specific")
r_moderate_mv = run_queries(engine2, MODERATE_QUERIES, pack_ids2, "Moderate")
r_vague_mv = run_queries(engine2, VAGUE_QUERIES, pack_ids2, "Vague")


# ---------------------------------------------------------------------------
# Experiment 3: File ingestion — chunked document with real KV
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("EXPERIMENT 3: File ingestion — multi-paragraph document")
print("=" * 70)

DOCUMENT = (
    "Sonia's Career and Life\n\n"
    "Sonia began her career as a technical translator, specializing in pharmaceutical patents. "
    "She translated complex German legal and scientific documents into English for biotech "
    "companies based in Berlin. Her language skills were honed during childhood when her "
    "grandmother in Kazan taught her multiple languages alongside balalaika lessons every Sunday.\n\n"
    "Beyond languages, Sonia has a strong engineering background. She designed a waste-heat "
    "recovery system for a cement kiln in Rajasthan that reduced emissions by 18 percent. "
    "She also built laboratory equipment from surplus parts, including a mass spectrometer "
    "calibration jig for her university's chemistry department.\n\n"
    "Sonia is passionate about environmental sustainability. She organized a neighborhood "
    "composting cooperative in the Kreuzberg district of Berlin that now processes two tonnes "
    "of organic waste every month. Her academic work includes a peer-reviewed paper on "
    "mycorrhizal networks in boreal peatlands, published in the Journal of Ecology.\n\n"
    "In her personal life, Sonia is an endurance athlete who completed a 42-kilometer "
    "ultramarathon through the Scottish Highlands. She also has a love for vintage machinery, "
    "having restored a 1967 Jawa 350 motorcycle she discovered rusted in a barn outside Brno."
)

tmpdir3 = tempfile.mkdtemp()
engine3 = tardigrade_db.Engine(tmpdir3, vamana_threshold=9999)

chunker = TextChunker(tokenizer, max_tokens=128, overlap_tokens=16)
chunks = chunker.chunk(DOCUMENT)
print(f"Document chunked into {len(chunks)} chunks")

chunk_pack_ids = []
for chunk in chunks:
    key, payloads = capture_kv(chunk.text)
    pid = engine3.mem_write_pack(OWNER, key, payloads, 70.0, text=chunk.text)
    chunk_pack_ids.append(pid)

for i in range(len(chunk_pack_ids) - 1):
    engine3.add_pack_edge(chunk_pack_ids[i], chunk_pack_ids[i + 1], EDGE_SUPPORTS)

engine3.set_refinement_mode("centered")

DOC_QUERIES = [
    ("Who translated pharmaceutical patents?", "translat"),
    ("What did Sonia do in Rajasthan?", "waste-heat"),
    ("Tell me about the composting project", "compost"),
    ("What paper did Sonia publish?", "mycorrhizal"),
    ("What motorcycle did Sonia restore?", "Jawa"),
    ("What musical instrument does Sonia play?", "balalaika"),
    ("What athletic event did Sonia complete?", "ultramarathon"),
    ("What did Sonia build for the chemistry department?", "spectrometer"),
]

print("\nQuerying document chunks...")
doc_hits = 0
for query_text, expected_keyword in DOC_QUERIES:
    results = query_packs(engine3, query_text, k=3)
    found_text = " ".join(engine3.pack_text(r["pack_id"]) or "" for r in results)

    hit = expected_keyword.lower() in found_text.lower()
    doc_hits += int(hit)
    status = "HIT " if hit else "MISS"
    print(f"  {status}: '{query_text}' -> '{expected_keyword}'")

print(f"\n  File ingest R@3: {doc_hits}/{len(DOC_QUERIES)} = {doc_hits/len(DOC_QUERIES):.0%}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'':30s} {'Specific':>10s} {'Moderate':>10s} {'Vague':>10s}")
print(f"{'Baseline (centered)':30s} {r_specific_base:>9.0%} {r_moderate_base:>9.0%} {r_vague_base:>9.0%}")
print(f"{'+ Multi-view consolidation':30s} {r_specific_mv:>9.0%} {r_moderate_mv:>9.0%} {r_vague_mv:>9.0%}")
delta_mod = r_moderate_mv - r_moderate_base
delta_vag = r_vague_mv - r_vague_base
print(f"{'Delta (multi-view)':30s} {'--':>10s} {delta_mod:>+9.0%} {delta_vag:>+9.0%}")
print(f"\nFile ingest document R@3: {doc_hits}/{len(DOC_QUERIES)} = {doc_hits/len(DOC_QUERIES):.0%}")
print(f"\nModel: {MODEL_NAME}, Device: {DEVICE}, Query layer: {query_layer}/{n_layers}")
