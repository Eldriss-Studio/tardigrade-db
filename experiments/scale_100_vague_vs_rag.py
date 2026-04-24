#!/usr/bin/env python3
# Vague query test: hidden states + Top5Avg vs traditional RAG.
#
# Tests whether latent retrieval works with natural broad queries
# that a real agent would ask.

import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(".").resolve() / "python"))
sys.path.insert(0, str(Path(".").resolve() / "experiments"))

import tardigrade_db
from corpus_100 import (
    ALL_QUERIES_WITH_VAGUE,
    CROSS_DOMAIN_QUERIES,
    MEMORIES,
    NEGATIVE_QUERIES,
    VAGUE_QUERIES,
    WITHIN_DOMAIN_QUERIES,
)
from tardigrade_hooks.hf_kv_hook import HuggingFaceKVHook

MODEL_NAME = "Qwen/Qwen3-0.6B"


def run_latent(model, tokenizer, query_layer):
    db_dir = Path(tempfile.mkdtemp(prefix="tardigrade_vague_latent_"))
    engine = tardigrade_db.Engine(str(db_dir))
    hook = HuggingFaceKVHook(
        engine, owner=1, model_config=model.config,
        model=model, use_hidden_states=True,
    )
    for memory in MEMORIES:
        inputs = tokenizer(memory, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = model(**inputs, use_cache=True, output_hidden_states=True)
        d = hook.on_generate(
            layer=query_layer,
            past_key_values=out.past_key_values,
            model_hidden_states=out.hidden_states[query_layer],
        )
        if d.should_write:
            engine.mem_write(1, query_layer, d.key, d.value, d.salience, None)

    results = {}
    for query_text, expected, qtype in ALL_QUERIES_WITH_VAGUE:
        inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = model(**inputs, use_cache=True, output_hidden_states=True)
        handles = hook.on_prefill(
            layer=query_layer,
            past_key_values=out.past_key_values,
            model_hidden_states=out.hidden_states[query_layer],
        )
        retrieved = [h.cell_id for h in handles]
        results[query_text] = {
            "top5": retrieved[:5],
            "expected": expected,
            "qtype": qtype,
            "hit": any(m in expected for m in retrieved[:5]) if expected else None,
        }
    shutil.rmtree(db_dir)
    return results


def run_rag():
    from transformers import AutoModel

    rag_name = "intfloat/e5-small-v2"
    rag_tok = AutoTokenizer.from_pretrained(rag_name)
    rag_model = AutoModel.from_pretrained(rag_name)
    rag_model.eval()

    def embed(texts, prefix):
        prefixed = [f"{prefix}: {t}" for t in texts]
        inputs = rag_tok(prefixed, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            out = rag_model(**inputs)
        embs = out.last_hidden_state * inputs["attention_mask"].unsqueeze(-1)
        embs = embs.sum(dim=1) / inputs["attention_mask"].sum(dim=1, keepdim=True)
        return torch.nn.functional.normalize(embs, p=2, dim=1).numpy()

    mem_embs = embed(MEMORIES, "passage")
    results = {}
    for query_text, expected, qtype in ALL_QUERIES_WITH_VAGUE:
        q_emb = embed([query_text], "query")
        scores = (q_emb @ mem_embs.T)[0]
        top_ids = np.argsort(-scores)[:5].tolist()
        results[query_text] = {
            "top5": top_ids,
            "expected": expected,
            "qtype": qtype,
            "hit": any(m in expected for m in top_ids) if expected else None,
        }
    return results


def recall_by_type(results, qtypes):
    rows = {}
    for qt in qtypes:
        entries = [r for r in results.values() if r["qtype"] == qt and r["expected"]]
        if not entries:
            continue
        hits = sum(1 for e in entries if e["hit"])
        rows[qt] = (hits, len(entries))
    all_e = [r for r in results.values() if r["qtype"] != "negative" and r["expected"]]
    rows["overall"] = (sum(1 for e in all_e if e["hit"]), len(all_e))
    return rows


def main():
    print("=" * 70)
    print("Vague Query Test: Hidden States vs Traditional RAG")
    print(f"Memories: {len(MEMORIES)}")
    print(f"Specific: {len(CROSS_DOMAIN_QUERIES)} cross + {len(WITHIN_DOMAIN_QUERIES)} within")
    print(f"Vague: {len(VAGUE_QUERIES)}")
    print(f"Negative: {len(NEGATIVE_QUERIES)}")
    print("=" * 70)

    print(f"\n  Loading {MODEL_NAME}...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, attn_implementation="eager",
    )
    model.eval()
    ql = int(model.config.num_hidden_layers * 0.67)
    print(f"OK (ql={ql})")

    print("\n--- Running latent retrieval ---")
    latent = run_latent(model, tokenizer, ql)
    del model

    print("--- Running RAG baseline ---")
    rag = run_rag()

    # Results
    qtypes = ["cross", "within", "vague"]
    lat_rows = recall_by_type(latent, qtypes)
    rag_rows = recall_by_type(rag, qtypes)

    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}\n")

    print(f"  {'Query Type':<15} {'Latent':>15} {'RAG':>15}")
    print(f"  {'-' * 47}")
    for qt in qtypes + ["overall"]:
        lh, lt = lat_rows.get(qt, (0, 0))
        rh, rt = rag_rows.get(qt, (0, 0))
        lp = f"{lh}/{lt} ({100*lh/lt:.0f}%)" if lt else "N/A"
        rp = f"{rh}/{rt} ({100*rh/rt:.0f}%)" if rt else "N/A"
        label = qt.upper() if qt == "overall" else qt
        print(f"  {label:<15} {lp:>15} {rp:>15}")

    # Vague detail
    print(f"\n  -- Vague Query Detail --\n")
    print(f"  {'Query':<43} {'Latent':>7} {'RAG':>7}")
    print(f"  {'-' * 59}")
    for q, expected, qtype in ALL_QUERIES_WITH_VAGUE:
        if qtype != "vague":
            continue
        l = latent[q]
        r = rag[q]
        lm = "Y" if l["hit"] else "N"
        rm = "Y" if r["hit"] else "N"
        print(f"  {q[:41]:<43} {lm:>7} {rm:>7}")

    # Conclusion
    lv = lat_rows.get("vague", (0, 1))
    rv = rag_rows.get("vague", (0, 1))
    lo = lat_rows.get("overall", (0, 1))
    ro = rag_rows.get("overall", (0, 1))

    print(f"\n{'=' * 70}")
    print(f"  Vague:   Latent {lv[0]}/{lv[1]} ({100*lv[0]/lv[1]:.0f}%) | RAG {rv[0]}/{rv[1]} ({100*rv[0]/rv[1]:.0f}%)")
    print(f"  Overall: Latent {lo[0]}/{lo[1]} ({100*lo[0]/lo[1]:.0f}%) | RAG {ro[0]}/{ro[1]} ({100*ro[0]/ro[1]:.0f}%)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
