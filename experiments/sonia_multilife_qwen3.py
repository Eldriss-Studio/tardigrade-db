#!/usr/bin/env python3
"""Multi-domain memory retrieval -- Qwen3-0.6B, per-token KV tensors.

Character: Sonia, 34, freelance translator. Lives in Chicago with her
6-year-old daughter Lucia. Recently divorced. Memories span two weeks
across very different domains: work, parenting, cooking, health, social,
legal, hobbies, errands, dreams, emotions.

The hypothesis: when memories come from genuinely different life domains,
per-token KV retrieval should achieve high recall because the model's
internal representations are naturally spread across distinct regions
of latent space. "Custody hearing" and "burnt risotto" don't cluster.

Usage:
    source .venv/bin/activate
    python experiments/sonia_multilife_qwen3.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import tardigrade_db

# -- Sonia's memories (2 weeks, multiple domains) -----------------------------

MEMORIES = [
    # 0 -- Work (translation deadline)
    "Spent all of Monday translating a pharmaceutical patent from Spanish. Thirty pages of enzyme inhibitor nomenclature. Had to look up farmacocinetica de absorcion three times because the client keeps using non-standard abbreviations.",

    # 1 -- Parenting (school incident)
    "Lucia's teacher called during lunch. She bit another kid on the playground over a swing dispute. Drove to school feeling like a failure. Lucia cried the whole way home and said sorry fourteen times.",

    # 2 -- Cooking (disaster)
    "Tried making risotto from that Italian cookbook mom gave me. Burned the bottom because I was on a work call and forgot to stir. The smoke alarm went off and Lucia said Mama your food is angry again.",

    # 3 -- Health (doctor visit)
    "Annual physical on Wednesday. Dr. Huang said my vitamin D is low and my blood pressure is borderline. She asked if I was sleeping enough. I lied and said yes.",

    # 4 -- Legal (custody)
    "Video call with the mediator about adjusting the custody schedule. Eduardo wants every other weekend instead of every weekend. I said fine but my hands were shaking the whole time. Lucia doesn't know yet.",

    # 5 -- Social (old friend)
    "Ran into Camila at the grocery store buying avocados. Haven't seen her since before the divorce. She hugged me for too long and whispered you look tired mi amor. I almost cried in the produce section.",

    # 6 -- Hobby (running)
    "Ran three miles along the lakefront on Thursday morning before Lucia woke up. First time in two months. My knees hurt but the sunrise over Lake Michigan was so orange it looked fake.",

    # 7 -- Dream
    "Had a dream that I was back in my grandmother's kitchen in Guadalajara. She was making tamales and singing but when I looked at her face it was Lucia's face, thirty years older. Woke up disoriented and sad.",

    # 8 -- Errand (car trouble)
    "Check engine light came on again. Took the car to the mechanic on Western Avenue. They said the catalytic converter needs replacing. Fourteen hundred dollars. Put it on the credit card I just paid off.",

    # 9 -- Parenting (proud moment)
    "Lucia read an entire picture book out loud to me before bed. Where the Wild Things Are. She sounded out rumpus by herself and looked up at me like she had solved a puzzle. Best moment of the whole month.",

    # 10 -- Work (difficult client)
    "The pharmaceutical client rejected my translation of section 4.2 because I used bioavailability instead of biological availability. Called my friend Ana who is a pharma translator in Madrid. She said both are correct but the client is technically right for EU regulatory submissions.",

    # 11 -- Emotional (late night)
    "Sat on the kitchen floor at midnight eating cereal out of the box. Lucia was asleep. The apartment was so quiet I could hear the refrigerator humming. Thought about calling Eduardo but didn't. Thought about calling my mom but it was 2am in Guadalajara.",

    # 12 -- Weather/environment
    "First real snowfall of the season on Friday. Lucia pressed her face against the window and fogged up the glass with her breath. She drew a smiley face in the fog. I took a photo but it came out blurry.",

    # 13 -- Finances
    "Spent Sunday doing invoices. I am owed four thousand dollars from three clients. The pharmaceutical company is sixty days overdue. Sent a polite follow-up email that took me forty minutes to write because I kept deleting the passive-aggressive parts.",

    # 14 -- Neighbor interaction
    "Mrs. Kowalski from 3B brought over pierogi because she heard Lucia coughing through the wall. Told me her husband left in 1987 and she raised four kids alone. Said it gets easier but it never gets easy. Cried after she left.",

    # 15 -- Media/culture
    "Watched Coco with Lucia for the fifth time. She asked me if abuelita can really see us from heaven. I said yes even though I don't know what I believe anymore. She seemed satisfied and fell asleep on my shoulder.",
]

# -- Queries ------------------------------------------------------------------

SPECIFIC_QUERIES = [
    ("The translation project about pharmaceutical patents", [0, 10]),
    ("When Lucia bit someone at school", [1]),
    ("The cooking disaster with the smoke alarm", [2]),
    ("Going to the doctor and lying about sleep", [3]),
    ("The custody schedule discussion with the mediator", [4]),
    ("Running into an old friend at the grocery store", [5]),
    ("Running along the lake at sunrise", [6]),
    ("The dream about my grandmother's kitchen", [7]),
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


def run_experiment(model, tokenizer, n_layers, query_layer, mode="per-token"):
    """Run retrieval experiment in either per-token or mean-pool mode."""
    db_dir = Path(tempfile.mkdtemp(prefix=f"tardigrade_sonia_{mode}_"))
    engine = tardigrade_db.Engine(str(db_dir))

    cell_to_mem = {}

    for mem_idx, memory in enumerate(MEMORIES):
        inputs = tokenizer(memory, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)

        seq_len = inputs["input_ids"].shape[1]

        for layer_idx in range(n_layers):
            hidden = outputs.hidden_states[layer_idx + 1].numpy()[0]

            if mode == "per-token":
                for tok_idx in range(seq_len):
                    tok_vec = hidden[tok_idx].astype(np.float32)
                    salience = min(75.0 + mem_idx, 100.0)
                    cell_id = engine.mem_write(1, layer_idx, tok_vec, tok_vec, salience, None)
                    cell_to_mem[cell_id] = mem_idx
            else:
                mean_vec = hidden.mean(axis=0).astype(np.float32)
                salience = min(75.0 + mem_idx, 100.0)
                cell_id = engine.mem_write(1, layer_idx, mean_vec, mean_vec, salience, None)
                cell_to_mem[cell_id] = mem_idx

    print(f"  [{mode}] Stored: {engine.cell_count()} cells")

    # Retrieve.
    results_map = {}
    genuine_scores = []
    negative_scores = []

    for query_text, expected_mems in SPECIFIC_QUERIES:
        inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)

        query_hidden = outputs.hidden_states[query_layer + 1].numpy()[0]
        query_vec = query_hidden.mean(axis=0).astype(np.float32)

        results = engine.mem_read(query_vec, 10, 1)

        retrieved_mems = [cell_to_mem.get(r.cell_id, -1) for r in results]

        seen = set()
        unique_mems = []
        for m in retrieved_mems:
            if m not in seen:
                seen.add(m)
                unique_mems.append(m)

        hit = any(m in expected_mems for m in unique_mems[:5])
        genuine_scores.extend(r.score for r in results[:5])

        rank = "---"
        if hit:
            for j, m in enumerate(unique_mems[:5]):
                if m in expected_mems:
                    rank = f"#{j+1}"
                    break

        top_mem = unique_mems[0] if unique_mems else -1
        results_map[query_text] = {
            "hit": hit, "rank": rank,
            "top_score": results[0].score if results else 0,
            "top_mem": top_mem, "unique_mems": unique_mems[:5],
        }

    for query_text in NEGATIVE_QUERIES:
        inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
        query_hidden = outputs.hidden_states[query_layer + 1].numpy()[0]
        query_vec = query_hidden.mean(axis=0).astype(np.float32)
        results = engine.mem_read(query_vec, 3, 1)
        if results:
            negative_scores.append(results[0].score)

    shutil.rmtree(db_dir)

    hits = sum(1 for r in results_map.values() if r["hit"])
    total = len(SPECIFIC_QUERIES)
    recall = hits / total * 100
    avg_genuine = np.mean(genuine_scores) if genuine_scores else 0
    avg_negative = np.mean(negative_scores) if negative_scores else 0
    snr = avg_genuine - avg_negative

    return {
        "mode": mode, "recall": recall, "hits": hits, "total": total,
        "avg_genuine": avg_genuine, "avg_negative": avg_negative,
        "snr": snr, "details": results_map,
    }


def main():
    print("=" * 70)
    print("Multi-Domain Memory Retrieval -- Sonia's Two Weeks")
    print("Qwen3-0.6B | Per-token vs Mean-pool | 16 memories, 16 queries")
    print("=" * 70)

    print(f"\n  Loading {MODEL_NAME}...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,

        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    query_layer = int(n_layers * 0.67)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"OK ({param_count/1e6:.0f}M, {n_layers}L, d={d_model}, ql={query_layer})")

    # Mean-pool first (fast).
    print(f"\n{'=' * 70}")
    print("  MEAN-POOL BASELINE")
    print(f"{'=' * 70}")
    mp = run_experiment(model, tokenizer, n_layers, query_layer, mode="mean-pool")

    print()
    for q, _ in SPECIFIC_QUERIES:
        d = mp["details"][q]
        mark = f"Y{d['rank']}" if d["hit"] else "N"
        short_q = q[:50]
        print(f"  {mark:>6}  {short_q:<50} -> mem {d['top_mem']:>2}")

    # Per-token (slow but the real test).
    print(f"\n{'=' * 70}")
    print("  PER-TOKEN (TardigradeDB design)")
    print(f"{'=' * 70}")
    pt = run_experiment(model, tokenizer, n_layers, query_layer, mode="per-token")

    print()
    for q, _ in SPECIFIC_QUERIES:
        d = pt["details"][q]
        mark = f"Y{d['rank']}" if d["hit"] else "N"
        short_q = q[:50]
        print(f"  {mark:>6}  {short_q:<50} -> mem {d['top_mem']:>2}")

    # -- Comparison -------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("  COMPARISON")
    print(f"{'=' * 70}\n")

    print(f"  {'Query':<50} {'MeanP':>6} {'Token':>6}")
    print(f"  {'-' * 64}")
    for q, _ in SPECIFIC_QUERIES:
        m = mp["details"][q]
        p = pt["details"][q]
        mm = f"Y{m['rank']}" if m["hit"] else "N"
        pm = f"Y{p['rank']}" if p["hit"] else "N"
        print(f"  {q[:48]:<50} {mm:>6} {pm:>6}")

    mp_recall = f"{mp['recall']:.1f}%"
    pt_recall = f"{pt['recall']:.1f}%"
    mp_snr = f"{mp['snr']:+.1f}"
    pt_snr = f"{pt['snr']:+.1f}"

    print(f"\n  {'Metric':<25} {'Mean-Pool':>15} {'Per-Token':>15}")
    print(f"  {'-' * 57}")
    print(f"  {'Recall':<25} {mp_recall:>15} {pt_recall:>15}")
    print(f"  {'Signal-to-noise':<25} {mp_snr:>15} {pt_snr:>15}")

    mp_tops = set(mp["details"][q]["top_mem"] for q, _ in SPECIFIC_QUERIES)
    pt_tops = set(pt["details"][q]["top_mem"] for q, _ in SPECIFIC_QUERIES)
    print(f"  {'Unique top-1 memories':<25} {len(mp_tops):>15} {len(pt_tops):>15}")

    # Misses.
    for r in [mp, pt]:
        misses = [q for q, _ in SPECIFIC_QUERIES if not r["details"][q]["hit"]]
        if misses:
            print(f"\n  {r['mode']} misses ({len(misses)}):")
            for q in misses:
                expected = [e for qt, e in SPECIFIC_QUERIES if qt == q][0]
                d = r["details"][q]
                print(f"    X {q[:55]}")
                print(f"      Got: mem {d['unique_mems'][:3]} | Expected: {expected}")

    delta = pt["recall"] - mp["recall"]
    print(f"\n{'=' * 70}")
    print(f"  Mean-pool: {mp['hits']}/{mp['total']} ({mp['recall']:.1f}%) | SNR: {mp['snr']:+.1f}")
    print(f"  Per-token: {pt['hits']}/{pt['total']} ({pt['recall']:.1f}%) | SNR: {pt['snr']:+.1f}")
    print(f"  Delta:     {delta:+.1f}% recall")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
