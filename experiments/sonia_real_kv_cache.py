#!/usr/bin/env python3
# The REAL KV cache experiment.
#
# Previous experiments stored hidden_states (raw pre-projection activations).
# This stores the actual KV cache -- the projected K tensors that the model
# trained its attention mechanism to use.
#
# This is what TardigradeDB was designed to store.
#
# Character: Sonia, 34, freelance translator, Chicago. 16 diverse memories.
#
# Usage:
#     source .venv/bin/activate
#     python experiments/sonia_real_kv_cache.py

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import tardigrade_db

MEMORIES = [
    "Spent all of Monday translating a pharmaceutical patent from Spanish. Thirty pages of enzyme inhibitor nomenclature. Had to look up farmacocinetica de absorcion three times because the client keeps using non-standard abbreviations.",
    "Lucia teacher called during lunch. She bit another kid on the playground over a swing dispute. Drove to school feeling like a failure. Lucia cried the whole way home and said sorry fourteen times.",
    "Tried making risotto from that Italian cookbook mom gave me. Burned the bottom because I was on a work call and forgot to stir. The smoke alarm went off and Lucia said Mama your food is angry again.",
    "Annual physical on Wednesday. Dr. Huang said my vitamin D is low and my blood pressure is borderline. She asked if I was sleeping enough. I lied and said yes.",
    "Video call with the mediator about adjusting the custody schedule. Eduardo wants every other weekend instead of every weekend. I said fine but my hands were shaking the whole time. Lucia does not know yet.",
    "Ran into Camila at the grocery store buying avocados. Have not seen her since before the divorce. She hugged me for too long and whispered you look tired mi amor. I almost cried in the produce section.",
    "Ran three miles along the lakefront on Thursday morning before Lucia woke up. First time in two months. My knees hurt but the sunrise over Lake Michigan was so orange it looked fake.",
    "Had a dream that I was back in my grandmother kitchen in Guadalajara. She was making tamales and singing but when I looked at her face it was Lucia face thirty years older. Woke up disoriented and sad.",
    "Check engine light came on again. Took the car to the mechanic on Western Avenue. They said the catalytic converter needs replacing. Fourteen hundred dollars. Put it on the credit card I just paid off.",
    "Lucia read an entire picture book out loud to me before bed. Where the Wild Things Are. She sounded out rumpus by herself and looked up at me like she had solved a puzzle. Best moment of the whole month.",
    "The pharmaceutical client rejected my translation of section 4.2 because I used bioavailability instead of biological availability. Called my friend Ana who is a pharma translator in Madrid. She said both are correct but the client is technically right for EU regulatory submissions.",
    "Sat on the kitchen floor at midnight eating cereal out of the box. Lucia was asleep. The apartment was so quiet I could hear the refrigerator humming. Thought about calling Eduardo but did not. Thought about calling my mom but it was 2am in Guadalajara.",
    "First real snowfall of the season on Friday. Lucia pressed her face against the window and fogged up the glass with her breath. She drew a smiley face in the fog. I took a photo but it came out blurry.",
    "Spent Sunday doing invoices. I am owed four thousand dollars from three clients. The pharmaceutical company is sixty days overdue. Sent a polite follow-up email that took me forty minutes to write because I kept deleting the passive-aggressive parts.",
    "Mrs. Kowalski from 3B brought over pierogi because she heard Lucia coughing through the wall. Told me her husband left in 1987 and she raised four kids alone. Said it gets easier but it never gets easy. Cried after she left.",
    "Watched Coco with Lucia for the fifth time. She asked me if abuelita can really see us from heaven. I said yes even though I do not know what I believe anymore. She seemed satisfied and fell asleep on my shoulder.",
]

SPECIFIC_QUERIES = [
    ("The translation project about pharmaceutical patents", [0, 10]),
    ("When Lucia bit someone at school", [1]),
    ("The cooking disaster with the smoke alarm", [2]),
    ("Going to the doctor and lying about sleep", [3]),
    ("The custody schedule discussion with the mediator", [4]),
    ("Running into an old friend at the grocery store", [5]),
    ("Running along the lake at sunrise", [6]),
    ("The dream about my grandmother kitchen", [7]),
    ("Car problems and the expensive repair", [8]),
    ("Lucia reading a book out loud before bed", [9]),
    ("The client who rejected my translation over word choice", [10]),
    ("Eating cereal alone on the kitchen floor at midnight", [11]),
    ("First snow and Lucia drawing on the foggy window", [12]),
    ("Doing invoices and chasing overdue payments", [13]),
    ("The neighbor who brought food and shared advice", [14]),
    ("Watching the movie about the Day of the Dead with Lucia", [15]),
]

NEGATIVE_QUERIES = [
    "Did I go to a concert downtown this week",
    "My job interview at the marketing agency",
    "The plumber who came to fix the bathroom sink",
    "Taking Lucia to her soccer practice on Saturday",
    "The argument with my sister about Thanksgiving plans",
]

MODEL_NAME = "Qwen/Qwen3-0.6B"


def get_kv_key_mean(past_kv, layer_idx):
    """Mean-pool K vectors from the real KV cache at a layer."""
    k = past_kv.layers[layer_idx].keys  # (1, heads, seq, head_dim)
    b, h, s, d = k.shape
    flat = k[0].permute(1, 0, 2).reshape(s, h * d)  # (seq, heads*head_dim)
    return flat.mean(dim=0).numpy().astype(np.float32)


def get_kv_key_tokens(past_kv, layer_idx):
    """Per-token K vectors from the real KV cache at a layer."""
    k = past_kv.layers[layer_idx].keys  # (1, heads, seq, head_dim)
    b, h, s, d = k.shape
    flat = k[0].permute(1, 0, 2).reshape(s, h * d)  # (seq, heads*head_dim)
    return [flat[t].numpy().astype(np.float32) for t in range(s)]


def run_variant(model, tokenizer, query_layer, mode):
    db_dir = Path(tempfile.mkdtemp(prefix=f"tardigrade_sonia_kv_{mode}_"))
    engine = tardigrade_db.Engine(str(db_dir))
    cell_to_mem = {}
    kv_dim = None

    for mem_idx, memory in enumerate(MEMORIES):
        inputs = tokenizer(memory, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = model(**inputs, use_cache=True)

        if mode == "per-token":
            vecs = get_kv_key_tokens(out.past_key_values, query_layer)
            for v in vecs:
                if kv_dim is None:
                    kv_dim = len(v)
                cell_id = engine.mem_write(1, 0, v, v, 80.0, None)
                cell_to_mem[cell_id] = mem_idx
        else:
            v = get_kv_key_mean(out.past_key_values, query_layer)
            if kv_dim is None:
                kv_dim = len(v)
            cell_id = engine.mem_write(1, 0, v, v, 80.0, None)
            cell_to_mem[cell_id] = mem_idx

    print(f"  [{mode}] {engine.cell_count()} cells, kv_dim={kv_dim}")

    results_map = {}
    genuine_scores = []
    negative_scores = []

    for query_text, expected_mems in SPECIFIC_QUERIES:
        inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = model(**inputs, use_cache=True)

        query_vec = get_kv_key_mean(out.past_key_values, query_layer)
        results = engine.mem_read(query_vec, 10, 1)
        rmems = [cell_to_mem.get(r.cell_id, -1) for r in results]

        seen = set()
        unique = []
        for m in rmems:
            if m not in seen:
                seen.add(m)
                unique.append(m)

        hit = any(m in expected_mems for m in unique[:5])
        genuine_scores.extend(r.score for r in results[:5])

        rank = "---"
        if hit:
            for j, m in enumerate(unique[:5]):
                if m in expected_mems:
                    rank = f"#{j+1}"
                    break

        top_mem = unique[0] if unique else -1
        results_map[query_text] = {
            "hit": hit, "rank": rank,
            "top_score": results[0].score if results else 0,
            "top_mem": top_mem, "unique": unique[:5],
        }

    for query_text in NEGATIVE_QUERIES:
        inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        query_vec = get_kv_key_mean(out.past_key_values, query_layer)
        results = engine.mem_read(query_vec, 3, 1)
        if results:
            negative_scores.append(results[0].score)

    shutil.rmtree(db_dir)

    hits = sum(1 for r in results_map.values() if r["hit"])
    total = len(SPECIFIC_QUERIES)
    avg_g = np.mean(genuine_scores) if genuine_scores else 0
    avg_n = np.mean(negative_scores) if negative_scores else 0

    return {
        "mode": mode, "recall": hits / total * 100,
        "hits": hits, "total": total,
        "avg_genuine": avg_g, "avg_negative": avg_n,
        "snr": avg_g - avg_n, "details": results_map,
    }


def main():
    print("=" * 70)
    print("REAL KV CACHE -- Sonia, 16 diverse memories")
    print("Storing actual K projections (past_key_values), not hidden_states")
    print("=" * 70)

    print(f"\n  Loading {MODEL_NAME}...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, attn_implementation="eager",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    query_layer = int(n_layers * 0.67)
    print(f"OK ({n_layers}L, ql={query_layer})")

    # Mean-pool KV
    print(f"\n--- MEAN-POOL (real KV) ---")
    mp = run_variant(model, tokenizer, query_layer, "mean-pool")
    for q, _ in SPECIFIC_QUERIES:
        d = mp["details"][q]
        mark = f"Y{d['rank']}" if d["hit"] else "N"
        print(f"  {mark:>6}  {q[:52]:<52} -> mem {d['top_mem']:>2}")

    # Per-token KV
    print(f"\n--- PER-TOKEN (real KV) ---")
    pt = run_variant(model, tokenizer, query_layer, "per-token")
    for q, _ in SPECIFIC_QUERIES:
        d = pt["details"][q]
        mark = f"Y{d['rank']}" if d["hit"] else "N"
        print(f"  {mark:>6}  {q[:52]:<52} -> mem {d['top_mem']:>2}")

    # Comparison
    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}\n")

    print(f"  {'Query':<50} {'MeanP':>6} {'Token':>6}")
    print(f"  {'-' * 64}")
    for q, _ in SPECIFIC_QUERIES:
        m = mp["details"][q]
        p = pt["details"][q]
        mm = f"Y{m['rank']}" if m["hit"] else "N"
        pm = f"Y{p['rank']}" if p["hit"] else "N"
        print(f"  {q[:48]:<50} {mm:>6} {pm:>6}")

    mp_r = f"{mp['recall']:.1f}%"
    pt_r = f"{pt['recall']:.1f}%"
    mp_s = f"{mp['snr']:+.1f}"
    pt_s = f"{pt['snr']:+.1f}"
    mp_tops = len(set(mp["details"][q]["top_mem"] for q, _ in SPECIFIC_QUERIES))
    pt_tops = len(set(pt["details"][q]["top_mem"] for q, _ in SPECIFIC_QUERIES))

    print(f"\n  {'Metric':<25} {'Mean-Pool':>12} {'Per-Token':>12}")
    print(f"  {'-' * 51}")
    print(f"  {'Recall':<25} {mp_r:>12} {pt_r:>12}")
    print(f"  {'Signal-to-noise':<25} {mp_s:>12} {pt_s:>12}")
    print(f"  {'Unique top-1 memories':<25} {mp_tops:>12} {pt_tops:>12}")

    for r in [mp, pt]:
        misses = [q for q, _ in SPECIFIC_QUERIES if not r["details"][q]["hit"]]
        if misses:
            print(f"\n  {r['mode']} misses ({len(misses)}):")
            for q in misses:
                exp = [e for qt, e in SPECIFIC_QUERIES if qt == q][0]
                d = r["details"][q]
                print(f"    X \"{q[:55]}\"")
                print(f"      Got: {d['unique'][:3]} | Expected: {exp}")

    print(f"\n  -- Previous (hidden_states, same model, same memories) --")
    print(f"  hidden mean-pool: 31.2% | SNR: -267.7")
    print(f"  hidden per-token: 31.2% | SNR: +11096.9")

    delta = pt["recall"] - mp["recall"]
    print(f"\n{'=' * 70}")
    print(f"  KV mean-pool: {mp['hits']}/{mp['total']} ({mp['recall']:.1f}%) | SNR: {mp['snr']:+.1f}")
    print(f"  KV per-token: {pt['hits']}/{pt['total']} ({pt['recall']:.1f}%) | SNR: {pt['snr']:+.1f}")
    print(f"  Delta:        {delta:+.1f}%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
