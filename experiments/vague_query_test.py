#!/usr/bin/env python3
"""Vague query retrieval experiment.

The most important untested assumption: does TardigradeDB work with
the kind of queries real agents actually ask?

Every existing test uses specific queries:
  "The pharmaceutical patent about enzyme inhibitors"
  "Lucia standing in front of the T-Rex at the Field Museum"

Real agents ask vague questions:
  "How has work been lately?"
  "What's going on with Lucia?"
  "Any health issues?"

Hypothesis: vague queries will show significant recall degradation
because per-token dot-product scoring relies on vocabulary overlap
between query and memory hidden states.

Design (Controlled Experiment):
  - Same 100 memories, same engine, same model (Qwen3-0.6B)
  - Three query tiers:
    A) Specific (existing corpus): "The birth certificate translation"
    B) Moderate: "Any translation work recently?"
    C) Vague: "How is work going?"
  - Each tier has queries covering all 10 domains
  - Compare R@5 across tiers

100 memories from corpus_100.py (Sonia's life across 10 domains).
"""

import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(".").resolve() / "python"))
sys.path.insert(0, str(Path(".").resolve() / "experiments"))

import tardigrade_db
from corpus_100 import ALL_QUERIES, MEMORIES
from tardigrade_hooks.hf_kv_hook import HuggingFaceKVHook

MODEL_NAME = "Qwen/Qwen3-0.6B"
DOMAINS = [
    "Work", "Parenting", "Cooking", "Health", "Legal",
    "Social", "Fitness", "Dreams", "Errands", "Media",
]

# ── Vague queries ──────────────────────────────────────────────────────
# Each tuple: (query_text, expected_memory_indices, vagueness_tier)
# Tier "moderate": domain-relevant but no specific details
# Tier "vague": how a real agent/user would actually ask

MODERATE_QUERIES = [
    # Work (10 queries)
    ("Any translation projects recently?", list(range(0, 10)), "moderate"),
    ("How are the freelance clients?", list(range(0, 10)), "moderate"),
    ("Any interpreting gigs at the hospital or courts?", [5, 7], "moderate"),
    ("Working on any legal documents?", [1, 4, 7], "moderate"),
    ("Any new language work come in?", list(range(0, 10)), "moderate"),
    ("How's the translation business?", list(range(0, 10)), "moderate"),
    ("Got any deadlines this week?", [0, 3, 6], "moderate"),
    ("Any interesting documents to translate?", [0, 2, 3, 8], "moderate"),
    ("Taking on any interpreting shifts?", [5, 7], "moderate"),
    ("How are the publishing house clients?", [3, 6], "moderate"),
    # Parenting (10 queries)
    ("What's been happening with Lucia at school?", [11, 17, 19], "moderate"),
    ("How are things at bedtime with Lucia?", [12, 14], "moderate"),
    ("Any milestones or activities with Lucia?", [10, 15, 17, 18], "moderate"),
    ("How is Lucia doing at swimming lessons?", [17], "moderate"),
    ("Any school events coming up?", [11, 19], "moderate"),
    ("What's Lucia been drawing or creating?", [16, 18], "moderate"),
    ("Any parent-teacher updates?", [11], "moderate"),
    ("How is Lucia handling the family changes?", [12, 14, 16, 19], "moderate"),
    ("Did Lucia have any playdates?", [13, 15], "moderate"),
    ("Any funny things Lucia said recently?", [14, 18], "moderate"),
    # Cooking (10 queries)
    ("Have you tried any new recipes?", list(range(20, 30)), "moderate"),
    ("Done any baking recently?", [22, 24, 29], "moderate"),
    ("Any cooking disasters or successes?", [20, 21, 23, 27], "moderate"),
    ("Made anything from your grandmother's recipes?", [20, 25], "moderate"),
    ("What did you cook for Lucia's school?", [22], "moderate"),
    ("Any Mexican dishes lately?", [20, 25, 26], "moderate"),
    ("Tried any Asian recipes?", [21, 23], "moderate"),
    ("How's the sourdough going?", [29], "moderate"),
    ("Any meal prep this week?", list(range(20, 30)), "moderate"),
    ("Cook anything special for anyone?", [22, 24, 27, 28], "moderate"),
    # Health (10 queries)
    ("Any doctor visits or health concerns?", [30, 31, 32, 33, 34, 35], "moderate"),
    ("How is the therapy going?", [38, 39], "moderate"),
    ("Any physical problems lately?", [31, 32, 33, 34], "moderate"),
    ("How's your blood pressure?", [30], "moderate"),
    ("Any dental appointments?", [36], "moderate"),
    ("Are you sleeping well these days?", [35, 37], "moderate"),
    ("How's the vitamin D situation?", [30], "moderate"),
    ("Any medication changes?", [30, 33], "moderate"),
    ("How are the headaches?", [32], "moderate"),
    ("Any anxiety or stress episodes?", [34, 38, 39], "moderate"),
    # Legal (10 queries)
    ("What's the latest with the divorce paperwork?", [40, 42, 45, 47], "moderate"),
    ("Any updates on custody arrangements?", [41, 43, 44], "moderate"),
    ("How are things going with the lawyers?", [42, 45, 47], "moderate"),
    ("Any mediation sessions scheduled?", [41, 46], "moderate"),
    ("How's the child support situation?", [42, 48], "moderate"),
    ("Any court dates coming up?", [40, 45, 47], "moderate"),
    ("Eduardo following the schedule?", [43, 44, 46], "moderate"),
    ("Any property division updates?", [47, 49], "moderate"),
    ("How's communication with Eduardo's lawyer?", [42, 45], "moderate"),
    ("Any changes to the visitation agreement?", [41, 44, 46], "moderate"),
    # Social (10 queries)
    ("Seen any friends lately?", list(range(50, 60)), "moderate"),
    ("Any new people you've met?", [52, 55, 56], "moderate"),
    ("How are things with your friends from work?", [51, 53, 57], "moderate"),
    ("Gone to any social events?", [52, 54, 58, 59], "moderate"),
    ("How's the book club going?", [54], "moderate"),
    ("Heard from Camila or Ana?", [50, 51], "moderate"),
    ("Any dates or romantic interests?", [52, 55], "moderate"),
    ("Gone out in the neighborhood?", [56, 58, 59], "moderate"),
    ("Any gatherings or dinner parties?", [52, 58], "moderate"),
    ("Talked to anyone interesting?", list(range(50, 60)), "moderate"),
    # Fitness (10 queries)
    ("How's the exercise routine going?", list(range(60, 70)), "moderate"),
    ("Any running or swimming lately?", [60, 61, 63, 65], "moderate"),
    ("Have you tried any new workouts?", [62, 64, 66], "moderate"),
    ("Getting to the YMCA?", [61, 62], "moderate"),
    ("How are the morning runs?", [60, 63, 65], "moderate"),
    ("Any races or fitness events?", [65, 67], "moderate"),
    ("Doing any yoga or stretching?", [62, 64], "moderate"),
    ("How's the lakefront running going?", [60, 63], "moderate"),
    ("Any activities with Lucia outdoors?", [67, 68, 69], "moderate"),
    ("Hit any personal records?", [61, 63, 65], "moderate"),
    # Dreams (10 queries)
    ("Any interesting dreams?", list(range(70, 80)), "moderate"),
    ("Any nightmares recently?", [73, 75, 78], "moderate"),
    ("Had any dreams about family?", [70, 72, 74, 79], "moderate"),
    ("Any recurring dreams?", list(range(70, 80)), "moderate"),
    ("Dreams about work or translation?", [71, 76], "moderate"),
    ("Any dreams about Eduardo?", [72, 79], "moderate"),
    ("Had any vivid or unusual dreams?", list(range(70, 80)), "moderate"),
    ("Any dreams about your grandmother?", [70], "moderate"),
    ("Dreams about Lucia?", [73, 74, 79], "moderate"),
    ("Any dreams that felt meaningful?", list(range(70, 80)), "moderate"),
    # Errands (10 queries)
    ("What errands needed doing?", list(range(80, 90)), "moderate"),
    ("Any car or house problems?", [80, 82, 83, 85], "moderate"),
    ("Had to deal with any stores or services?", [81, 84, 86, 88], "moderate"),
    ("Any home repairs needed?", [82, 83, 85], "moderate"),
    ("How's the car situation?", [80, 87], "moderate"),
    ("Any bureaucratic errands?", [81, 84, 89], "moderate"),
    ("Had to return anything?", [88], "moderate"),
    ("Any appointments or waiting in lines?", [81, 84, 89], "moderate"),
    ("Package deliveries or shipping?", [86, 89], "moderate"),
    ("Any grocery or shopping trips?", [84, 86, 88], "moderate"),
    # Media (10 queries)
    ("Watched or read anything good?", list(range(90, 100)), "moderate"),
    ("Any movies or shows with Lucia?", [90, 94, 95], "moderate"),
    ("Read any books recently?", [91, 96, 98], "moderate"),
    ("Listening to any podcasts?", [92, 93], "moderate"),
    ("Any music you're enjoying?", [94, 99], "moderate"),
    ("Seen any art exhibits?", [97], "moderate"),
    ("Any documentaries or news stories?", [93, 95, 99], "moderate"),
    ("What are you reading right now?", [91, 96, 98], "moderate"),
    ("Any concerts or live events?", [94, 99], "moderate"),
    ("Found any good shows to binge?", [90, 95, 96], "moderate"),
]

VAGUE_QUERIES = [
    # Work (10 phrasings)
    ("How is work going?", list(range(0, 10)), "vague"),
    ("Busy with work?", list(range(0, 10)), "vague"),
    ("Anything happening professionally?", list(range(0, 10)), "vague"),
    ("Work keeping you busy?", list(range(0, 10)), "vague"),
    ("How's the career?", list(range(0, 10)), "vague"),
    ("Making money?", list(range(0, 10)), "vague"),
    ("Work stress?", list(range(0, 10)), "vague"),
    ("Job going alright?", list(range(0, 10)), "vague"),
    ("Productive lately?", list(range(0, 10)), "vague"),
    ("Any work updates?", list(range(0, 10)), "vague"),
    # Parenting (10 phrasings)
    ("How is Lucia doing?", list(range(10, 20)), "vague"),
    ("Everything okay with the kid?", list(range(10, 20)), "vague"),
    ("How's parenting going?", list(range(10, 20)), "vague"),
    ("Lucia good?", list(range(10, 20)), "vague"),
    ("How's your daughter?", list(range(10, 20)), "vague"),
    ("The little one okay?", list(range(10, 20)), "vague"),
    ("Mom life treating you well?", list(range(10, 20)), "vague"),
    ("Kids are alright?", list(range(10, 20)), "vague"),
    ("Family doing well?", list(range(10, 20)), "vague"),
    ("How's home life?", list(range(10, 20)), "vague"),
    # Cooking (10 phrasings)
    ("What have you been eating?", list(range(20, 30)), "vague"),
    ("Cooking much?", list(range(20, 30)), "vague"),
    ("What's for dinner these days?", list(range(20, 30)), "vague"),
    ("Eating well?", list(range(20, 30)), "vague"),
    ("Food been good?", list(range(20, 30)), "vague"),
    ("In the kitchen much?", list(range(20, 30)), "vague"),
    ("What are you eating?", list(range(20, 30)), "vague"),
    ("Feeding yourself okay?", list(range(20, 30)), "vague"),
    ("Any good meals?", list(range(20, 30)), "vague"),
    ("Cooking or ordering in?", list(range(20, 30)), "vague"),
    # Health (10 phrasings)
    ("How are you feeling?", list(range(30, 40)), "vague"),
    ("Everything okay health-wise?", list(range(30, 40)), "vague"),
    ("Taking care of yourself?", list(range(30, 40)), "vague"),
    ("Feeling alright?", list(range(30, 40)), "vague"),
    ("How's your health?", list(range(30, 40)), "vague"),
    ("Body holding up?", list(range(30, 40)), "vague"),
    ("Doing okay physically?", list(range(30, 40)), "vague"),
    ("Any aches or pains?", list(range(30, 40)), "vague"),
    ("Feeling healthy?", list(range(30, 40)), "vague"),
    ("How's your wellbeing?", list(range(30, 40)), "vague"),
    # Legal (10 phrasings)
    ("How are things with Eduardo?", list(range(40, 50)), "vague"),
    ("Any drama lately?", list(range(40, 50)), "vague"),
    ("How's the co-parenting situation?", [41, 43, 44, 46, 48], "vague"),
    ("Eduardo being difficult?", list(range(40, 50)), "vague"),
    ("The ex causing problems?", list(range(40, 50)), "vague"),
    ("Legal stuff sorted?", list(range(40, 50)), "vague"),
    ("How's the separation going?", list(range(40, 50)), "vague"),
    ("Divorce stuff?", list(range(40, 50)), "vague"),
    ("Any issues with Eduardo?", list(range(40, 50)), "vague"),
    ("Everything civil with the ex?", list(range(40, 50)), "vague"),
    # Social (10 phrasings)
    ("What's your social life like?", list(range(50, 60)), "vague"),
    ("Getting out much?", list(range(50, 60)), "vague"),
    ("Hanging out with anyone?", list(range(50, 60)), "vague"),
    ("Seeing people?", list(range(50, 60)), "vague"),
    ("Got friends around?", list(range(50, 60)), "vague"),
    ("Socializing?", list(range(50, 60)), "vague"),
    ("Lonely or connected?", list(range(50, 60)), "vague"),
    ("Any fun plans with friends?", list(range(50, 60)), "vague"),
    ("People in your life?", list(range(50, 60)), "vague"),
    ("Community feeling good?", list(range(50, 60)), "vague"),
    # Fitness (10 phrasings)
    ("Getting any exercise?", list(range(60, 70)), "vague"),
    ("Staying active?", list(range(60, 70)), "vague"),
    ("Moving your body?", list(range(60, 70)), "vague"),
    ("Working out?", list(range(60, 70)), "vague"),
    ("Being physical?", list(range(60, 70)), "vague"),
    ("Keeping fit?", list(range(60, 70)), "vague"),
    ("Active lifestyle?", list(range(60, 70)), "vague"),
    ("Exercise happening?", list(range(60, 70)), "vague"),
    ("Getting your steps in?", list(range(60, 70)), "vague"),
    ("Body moving?", list(range(60, 70)), "vague"),
    # Dreams (10 phrasings)
    ("Sleep well lately?", list(range(70, 80)), "vague"),
    ("Any weird dreams?", list(range(70, 80)), "vague"),
    ("How's sleep?", list(range(70, 80)), "vague"),
    ("Sleeping okay?", list(range(70, 80)), "vague"),
    ("Rest well?", list(range(70, 80)), "vague"),
    ("Night time okay?", list(range(70, 80)), "vague"),
    ("Dream anything?", list(range(70, 80)), "vague"),
    ("Getting enough rest?", list(range(70, 80)), "vague"),
    ("Nights been peaceful?", list(range(70, 80)), "vague"),
    ("Waking up rested?", list(range(70, 80)), "vague"),
    # Errands (10 phrasings)
    ("Anything annoying happen?", list(range(80, 90)), "vague"),
    ("Any chores or errands?", list(range(80, 90)), "vague"),
    ("Dealing with anything tedious?", list(range(80, 90)), "vague"),
    ("Life admin piling up?", list(range(80, 90)), "vague"),
    ("Any hassles?", list(range(80, 90)), "vague"),
    ("Boring stuff to do?", list(range(80, 90)), "vague"),
    ("Adulting hard?", list(range(80, 90)), "vague"),
    ("Things to take care of?", list(range(80, 90)), "vague"),
    ("Running around doing stuff?", list(range(80, 90)), "vague"),
    ("Logistical headaches?", list(range(80, 90)), "vague"),
    # Media (10 phrasings)
    ("What are you into these days?", list(range(90, 100)), "vague"),
    ("Consuming any media?", list(range(90, 100)), "vague"),
    ("Anything entertaining going on?", list(range(90, 100)), "vague"),
    ("Watching anything?", list(range(90, 100)), "vague"),
    ("Reading stuff?", list(range(90, 100)), "vague"),
    ("Any entertainment?", list(range(90, 100)), "vague"),
    ("Screen time?", list(range(90, 100)), "vague"),
    ("Bingeing anything?", list(range(90, 100)), "vague"),
    ("Good content lately?", list(range(90, 100)), "vague"),
    ("Finding things to watch?", list(range(90, 100)), "vague"),
]

OPEN_QUERIES = [
    ("Tell me about your week", list(range(100)), "open"),
    ("What's on your mind?", list(range(100)), "open"),
    ("Anything memorable happen recently?", list(range(100)), "open"),
    ("How are things?", list(range(100)), "open"),
    ("What's new?", list(range(100)), "open"),
    ("Catch me up", list(range(100)), "open"),
    ("What's been going on?", list(range(100)), "open"),
    ("How have you been?", list(range(100)), "open"),
    ("Fill me in", list(range(100)), "open"),
    ("Anything I should know about?", list(range(100)), "open"),
]


def main():
    print("=" * 70)
    print("VAGUE QUERY RETRIEVAL EXPERIMENT")
    print(f"Model: {MODEL_NAME}, Memories: {len(MEMORIES)}")
    print("=" * 70)

    print(f"\n  Loading {MODEL_NAME}...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32, attn_implementation="eager",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    query_layer = int(n_layers * 0.67)
    print(f"OK ({n_layers}L, ql={query_layer})")

    db_dir = tempfile.mkdtemp(prefix="tdb_vague_")
    engine = tardigrade_db.Engine(db_dir)
    hook = HuggingFaceKVHook(
        engine, owner=1, model_config=model.config,
        model=model, use_hidden_states=True,
    )

    # Store all 100 memories
    print(f"\n--- STORING {len(MEMORIES)} MEMORIES ---")
    for i, memory in enumerate(MEMORIES):
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
    print(f"  Stored {engine.cell_count()} memories")

    # Query function
    def run_queries(queries, label):
        print(f"\n--- {label} ({len(queries)} queries) ---")
        hits_at_1, hits_at_5, hits_at_10 = 0, 0, 0
        total = len(queries)

        for query_text, expected, tier in queries:
            inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                out = model(**inputs, use_cache=True, output_hidden_states=True)
            handles = hook.on_prefill(
                layer=query_layer,
                past_key_values=out.past_key_values,
                model_hidden_states=out.hidden_states[query_layer],
            )
            retrieved = [h.cell_id for h in handles]
            top1 = retrieved[:1]
            top5 = retrieved[:5]
            top10 = retrieved[:10]

            hit1 = any(m in expected for m in top1)
            hit5 = any(m in expected for m in top5)
            hit10 = any(m in expected for m in top10)

            if hit1: hits_at_1 += 1
            if hit5: hits_at_5 += 1
            if hit10: hits_at_10 += 1

            domain = DOMAINS[expected[0] // 10] if expected and expected[0] < 100 else "Any"
            mark = "Y" if hit5 else "N"
            top_domain = DOMAINS[retrieved[0] // 10] if retrieved and retrieved[0] < 100 else "?"
            print(f"  {mark} [{domain:>10}] → [{top_domain:>10}] \"{query_text[:50]}\"")

        r1 = 100 * hits_at_1 / total if total else 0
        r5 = 100 * hits_at_5 / total if total else 0
        r10 = 100 * hits_at_10 / total if total else 0
        print(f"  R@1: {r1:.1f}%  R@5: {r5:.1f}%  R@10: {r10:.1f}%")
        return r1, r5, r10

    # Run all tiers
    # Tier A: Specific (existing corpus)
    specific = [(q, e, "specific") for q, e, t in ALL_QUERIES if t != "negative"]
    r1_a, r5_a, r10_a = run_queries(specific, "TIER A: SPECIFIC (existing corpus)")

    # Tier B: Moderate
    r1_b, r5_b, r10_b = run_queries(MODERATE_QUERIES, "TIER B: MODERATE (domain-relevant)")

    # Tier C: Vague
    r1_c, r5_c, r10_c = run_queries(VAGUE_QUERIES, "TIER C: VAGUE (how agents actually ask)")

    # Tier D: Open-ended
    r1_d, r5_d, r10_d = run_queries(OPEN_QUERIES, "TIER D: OPEN-ENDED (broadest possible)")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    print(f"  {'Tier':<35} {'R@1':>8} {'R@5':>8} {'R@10':>8}")
    print(f"  {'─' * 63}")
    print(f"  {'A: Specific (benchmark queries)':<35} {r1_a:>7.1f}% {r5_a:>7.1f}% {r10_a:>7.1f}%")
    print(f"  {'B: Moderate (domain-relevant)':<35} {r1_b:>7.1f}% {r5_b:>7.1f}% {r10_b:>7.1f}%")
    print(f"  {'C: Vague (real agent queries)':<35} {r1_c:>7.1f}% {r5_c:>7.1f}% {r10_c:>7.1f}%")
    print(f"  {'D: Open-ended (broadest)':<35} {r1_d:>7.1f}% {r5_d:>7.1f}% {r10_d:>7.1f}%")

    drop_b = r5_a - r5_b
    drop_c = r5_a - r5_c
    drop_d = r5_a - r5_d

    print(f"\n  Degradation from specific baseline:")
    print(f"    Moderate: {drop_b:+.1f}%")
    print(f"    Vague:    {drop_c:+.1f}%")
    print(f"    Open:     {drop_d:+.1f}%")
    print(f"{'=' * 70}")

    shutil.rmtree(db_dir)


if __name__ == "__main__":
    main()
