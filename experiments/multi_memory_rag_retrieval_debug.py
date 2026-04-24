#!/usr/bin/env python3
"""Phase 30B diagnostic: does standard embedding RAG also fail at multi-hop?

Tests whether e5-small-v2 cosine similarity retrieval finds BOTH
needed facts for cross-referencing queries.

If RAG also fails -> multi-hop is universal, TardigradeDB is no worse.
If RAG succeeds -> TardigradeDB's latent retrieval is the bottleneck.

Usage:
    source .venv/bin/activate
    python experiments/multi_memory_rag_retrieval_debug.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "python"))

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from multi_memory_corpus import MULTI_FACTS


def load_embedding_model():
    model_name = "intfloat/e5-small-v2"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def embed(model, tokenizer, texts, prefix="passage: "):
    prefixed = [prefix + t for t in texts]
    encoded = tokenizer(prefixed, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**encoded)
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled.numpy()


def main():
    model, tokenizer = load_embedding_model()

    all_facts = []
    fact_to_idx = {}
    for entry in MULTI_FACTS:
        for fact in entry["facts"]:
            fact_to_idx[fact] = len(all_facts)
            all_facts.append(fact)

    print(f"Embedding {len(all_facts)} facts...")
    fact_embeddings = embed(model, tokenizer, all_facts, prefix="passage: ")

    print()
    print("=" * 70)
    print("RAG RETRIEVAL DIAGNOSTIC (e5-small-v2)")
    print("=" * 70)
    print()

    all_found = 0
    partial = 0
    none_found = 0

    for i, entry in enumerate(MULTI_FACTS):
        query = entry["query"]
        expected_indices = {fact_to_idx[f] for f in entry["facts"]}
        k = len(entry["facts"])

        query_emb = embed(model, tokenizer, [query], prefix="query: ")
        scores = (fact_embeddings @ query_emb.T).flatten()

        top_k_indices = set(np.argsort(scores)[-k:][::-1].tolist())
        top_5_indices = np.argsort(scores)[-5:][::-1]

        found = expected_indices & top_k_indices
        missing = expected_indices - top_k_indices

        if len(missing) == 0:
            status = "ALL FOUND"
            all_found += 1
        elif len(found) > 0:
            status = "PARTIAL"
            partial += 1
        else:
            status = "NONE FOUND"
            none_found += 1

        print(f"Q{i+1} [{status}]: {query[:60]}...")
        print(f"  Expected facts ({k}):")
        for idx in sorted(expected_indices):
            marker = "found" if idx in top_k_indices else "MISSING"
            print(f"    [{marker}] #{idx}: {all_facts[idx][:60]}...")
        wrong = top_k_indices - expected_indices
        if wrong:
            print(f"  Wrong facts retrieved:")
            for idx in sorted(wrong):
                print(f"    [wrong] #{idx}: {all_facts[idx][:60]}...")

        print(f"  Top-5 ranking:")
        for rank, idx in enumerate(top_5_indices):
            score = scores[idx]
            is_expected = "***" if idx in expected_indices else "   "
            print(f"    {is_expected} #{rank+1}: (score={score:.4f}) {all_facts[idx][:50]}...")
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  All facts retrieved:  {all_found}/{len(MULTI_FACTS)}")
    print(f"  Partial retrieval:    {partial}/{len(MULTI_FACTS)}")
    print(f"  No facts retrieved:   {none_found}/{len(MULTI_FACTS)}")


if __name__ == "__main__":
    main()
