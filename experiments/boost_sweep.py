#!/usr/bin/env python3
"""Sweep boost_factor for trace-boosted retrieval."""

import sys, tempfile, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "python"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import tardigrade_db
from tardigrade_hooks.kp_injector import KnowledgePackStore
from multi_memory_scale_test import CROSS_REF_PAIRS
from corpus_100 import MEMORIES

print("Loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype=torch.float32, attn_implementation="eager"
)
model.eval()

tmpdir = tempfile.mkdtemp()
engine = tardigrade_db.Engine(tmpdir)
kps = KnowledgePackStore(engine, model, tokenizer, owner=1)

print("Storing 140 memories...", flush=True)
for mem in MEMORIES:
    kps.store(mem)
for entry in CROSS_REF_PAIRS:
    kps.store_linked(entry["facts"])
print(f"Total: {engine.pack_count()} packs", flush=True)

for bf in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
    correct = 0
    for entry in CROSS_REF_PAIRS:
        text, tokens, had = kps.generate_with_trace(
            entry["query"] + " /no_think", k=1, boost_factor=bf,
            max_new_tokens=100, do_sample=False,
        )
        if entry["expected"].lower() in text.lower():
            correct += 1
    print(f"  boost={bf:.1f}: {correct}/20 ({100*correct//20}%)", flush=True)
    gc.collect()
