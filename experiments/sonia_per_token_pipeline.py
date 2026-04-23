#!/usr/bin/env python3
# Per-token pipeline experiment -- Sonia, 16 diverse memories.
#
# Full end-to-end: HuggingFaceKVHook (per-token encoded keys)
# through Engine (PerTokenRetriever in pipeline) with max-sim scoring.
#
# This is the definitive test of the new retrieval architecture.

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(".").resolve() / "python"))
import tardigrade_db
from tardigrade_hooks.hf_kv_hook import HuggingFaceKVHook

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


def main():
    print("=" * 70)
    print("Per-Token Pipeline -- Sonia, 16 diverse memories")
    print("KVHook (per-token) -> Engine (PerTokenRetriever) -> max-sim")
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

    db_dir = Path(tempfile.mkdtemp(prefix="tardigrade_sonia_pt_pipeline_"))
    engine = tardigrade_db.Engine(str(db_dir))
    hook = HuggingFaceKVHook(engine, owner=1, model_config=model.config)

    # Store
    print(f"\n--- STORING {len(MEMORIES)} MEMORIES ---")
    for mem_idx, memory in enumerate(MEMORIES):
        inputs = tokenizer(memory, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        decision = hook.on_generate(layer=query_layer, past_key_values=out.past_key_values)
        if decision.should_write:
            engine.mem_write(1, query_layer, decision.key, decision.value, decision.salience, None)
    print(f"  Cells: {engine.cell_count()}")

    # Retrieve
    print(f"\n--- QUERYING ---")
    results_map = {}
    genuine_scores = []
    negative_scores = []

    for query_text, expected_mems in SPECIFIC_QUERIES:
        inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        handles = hook.on_prefill(layer=query_layer, past_key_values=out.past_key_values)

        retrieved = [h.cell_id for h in handles]
        hit = any(m in expected_mems for m in retrieved[:5])
        genuine_scores.extend(h.score for h in handles[:5])

        rank = "---"
        if hit:
            for j, m in enumerate(retrieved[:5]):
                if m in expected_mems:
                    rank = f"#{j+1}"
                    break

        top = retrieved[0] if retrieved else -1
        results_map[query_text] = {"hit": hit, "rank": rank, "top_mem": top}
        mark = f"Y{rank}" if hit else "N"
        print(f"  {mark:>6}  {query_text[:52]:<52} -> mem {top:>2}")

    for query_text in NEGATIVE_QUERIES:
        inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        handles = hook.on_prefill(layer=query_layer, past_key_values=out.past_key_values)
        if handles:
            negative_scores.append(handles[0].score)

    shutil.rmtree(db_dir)

    # Results
    hits = sum(1 for r in results_map.values() if r["hit"])
    total = len(SPECIFIC_QUERIES)
    recall = hits / total * 100
    avg_g = np.mean(genuine_scores) if genuine_scores else 0
    avg_n = np.mean(negative_scores) if negative_scores else 0
    snr = avg_g - avg_n
    unique_tops = len(set(r["top_mem"] for r in results_map.values()))

    print(f"\n{'=' * 70}")
    print(f"  Recall:              {hits}/{total} ({recall:.1f}%)")
    print(f"  Signal-to-noise:     {snr:+.1f}")
    print(f"  Unique top-1:        {unique_tops}")

    misses = [q for q, r in results_map.items() if not r["hit"]]
    if misses:
        print(f"\n  Misses ({len(misses)}):")
        for q in misses:
            exp = [e for qt, e in SPECIFIC_QUERIES if qt == q][0]
            print(f"    X \"{q[:55]}\"")
            print(f"      Got mem {results_map[q]['top_mem']} | Expected {exp}")

    print(f"\n  -- Progression --")
    print(f"  Hidden states mean-pool:    31.2%")
    print(f"  Real KV mean-pool:          62.5%")
    print(f"  Real KV per-token (manual): 75.0%")
    print(f"  Per-token pipeline (e2e):   {recall:.1f}%  <-- THIS RUN")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
