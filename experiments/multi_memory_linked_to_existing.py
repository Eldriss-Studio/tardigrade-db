#!/usr/bin/env python3
"""Hypothesis test: does linking details to existing memories break the 70% plateau?

The scale test (multi_memory_scale_test.py) stores cross-ref pairs as
*separate* memories that compete with the 100 background narratives.
A real agent wouldn't do that --- it would attach new details to the memory
it already has.

This test restructures the corpus: instead of storing a cross-ref linking
fact as a new memory, we link the detail fact directly to the existing
background memory that already describes the same event. The linking fact
is not stored at all --- the background memory *is* the linking context.

If this breaks the 70% plateau -> the plateau was a corpus design artifact.
If it doesn't -> there's a real architectural limitation.

Usage:
    source .venv/bin/activate
    python experiments/multi_memory_linked_to_existing.py
"""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "python"))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import tardigrade_db
from tardigrade_hooks.kp_injector import KnowledgePackStore
from tardigrade_hooks.multi_composer import NaiveConcatComposer

# Map each cross-ref pair to the background memory index it refers to.
# The detail fact gets linked to this background memory instead of
# competing with it.
#
# Format: (background_memory_index, detail_fact, query, expected)
#
# The linking fact is NOT stored --- the background memory is the link.
LINKED_DETAILS = [
    {
        # Background #17: "Signed Lucia up for swimming lessons at the community center..."
        "background_idx": 17,
        "detail": "Lucia's swimming instructor at the community center is a woman named Tomoko who drives a red Honda Civic",
        "query": "What kind of car does Lucia's swimming instructor drive?",
        "expected": "honda civic",
    },
    {
        # Background #80: "Check engine light... mechanic on Western Avenue... catalytic converter"
        "background_idx": 80,
        "detail": "The mechanic on Western Avenue who fixed the catalytic converter is named Raul and has a tattoo of a compass on his left forearm",
        "query": "What tattoo does the mechanic who fixed my car have?",
        "expected": "compass",
    },
    {
        # Background #11: "Teacher conference at Lucia's school. Ms. Navarro..."
        "background_idx": 11,
        "detail": "Ms. Navarro has been teaching at Lucia's school for seventeen years",
        "query": "How long has Lucia's teacher been at the school?",
        "expected": "seventeen",
    },
    {
        # Background #30: "Annual physical. Dr. Huang said my vitamin D is low..."
        "background_idx": 30,
        "detail": "Dr. Huang graduated from Northwestern Medical School in 2009",
        "query": "Where did the doctor who checked my vitamin D go to medical school?",
        "expected": "northwestern",
    },
    {
        # Background #59: "Went to a poetry reading at a bookstore in Pilsen..."
        "background_idx": 59,
        "detail": "The bookstore in Pilsen where I went to the poetry reading is called Casa Azul and hosts open mic night on the last Friday of every month",
        "query": "When is the open mic at the bookstore where I did the poetry reading?",
        "expected": "last friday",
    },
    {
        # Background #52: "Went to a dinner party... Met a man named David who teaches seventh grade history."
        "background_idx": 52,
        "detail": "David the history teacher teaches at Lane Tech which is on Addison Street near the Western Avenue Brown Line stop",
        "query": "What street is the school where David teaches located on?",
        "expected": "addison",
    },
    {
        # Background #61: "Swam forty laps at the YMCA pool..."
        "background_idx": 61,
        "detail": "The YMCA pool lap lanes are reserved from 6am to 8am on weekday mornings",
        "query": "What time are the lap lanes available at the pool where I swim?",
        "expected": "6am",
    },
    {
        # Background #9: "A tech startup asked me to localize their app into Brazilian Portuguese..."
        "background_idx": 9,
        "detail": "My cousin Renata in Sao Paulo works as a marketing manager at a tech company in the Pinheiros neighborhood",
        "query": "What does my cousin who helped with the Portuguese localization do for work?",
        "expected": "marketing",
    },
    {
        # Background #38: "Went to a therapist for the first time since the divorce..."
        "background_idx": 38,
        "detail": "The therapist is named Dr. Morales and her office is on the third floor of a brownstone on Damen Avenue",
        "query": "Where is the office of the therapist I see since the divorce?",
        "expected": "damen",
    },
    {
        # Background #29: "Bought a sourdough starter kit online. Named it Fernando..."
        "background_idx": 29,
        "detail": "The online shop that sold the sourdough starter kit is called BreadCraft and they are based in Vermont",
        "query": "What state is the company from that sold me the sourdough starter kit?",
        "expected": "vermont",
    },
    {
        # Background #3: "Had a video call with a publishing house in Barcelona..."
        "background_idx": 3,
        "detail": "The publishing house in Barcelona is called Editorial Sirena and was founded in 1987 specializing in Latin American authors",
        "query": "When was the Barcelona publishing house that wants me to translate the novel founded?",
        "expected": "1987",
    },
    {
        # Background #86: "Picked up Lucia's prescription for ear drops... pharmacist remembered her name..."
        "background_idx": 86,
        "detail": "The pharmacist who remembered Lucia's name is named James Chen and has been working at that pharmacy for twelve years",
        "query": "How long has the pharmacist who knows Lucia's name been working there?",
        "expected": "twelve",
    },
    {
        # Background #84: "The washing machine broke..."
        "background_idx": 84,
        "detail": "The repair guy who looked at the broken washing machine is from a company called FixIt Pro which charges a seventy-five dollar diagnostic fee",
        "query": "How much does the company that looked at my washing machine charge for diagnostics?",
        "expected": "seventy-five",
    },
    {
        # Background #41: "Video call with the mediator about adjusting the custody schedule..."
        "background_idx": 41,
        "detail": "The mediator is named Patricia Goldberg who also mediated the original divorce settlement two years ago",
        "query": "How long ago did the custody mediator handle our original divorce?",
        "expected": "two years",
    },
    {
        # Background #69: "Joined a running group that meets at the lakefront on Saturday mornings..."
        "background_idx": 69,
        "detail": "The running group is led by a coach named Marcus who used to run competitively at the University of Illinois",
        "query": "Where did the coach of my Saturday running group go to college?",
        "expected": "illinois",
    },
    {
        # Background #54: "Joined a book club that meets at the library..."
        "background_idx": 54,
        "detail": "The book club is organized by a librarian named Helen Park who has a PhD in comparative literature from the University of Chicago",
        "query": "What degree does the librarian who runs the book club have?",
        "expected": "phd",
    },
    {
        # Background #83: "Dropped off seven bags of old clothes at Goodwill..."
        "background_idx": 83,
        "detail": "The Goodwill on Milwaukee Avenue is open until 9pm on weekdays",
        "query": "What time does the Goodwill where I dropped off clothes close on weekdays?",
        "expected": "9pm",
    },
    {
        # Background #19: "Had to pick Lucia up early from school..." (school context)
        "background_idx": 19,
        "detail": "Lucia's friend Harper at school has a younger brother named Oliver who is four years old",
        "query": "What is the name of Lucia's friend Harper's younger brother?",
        "expected": "oliver",
    },
    {
        # Background #2: "The marketing agency sent me a website localization project... skincare brand..."
        "background_idx": 2,
        "detail": "The skincare brand is called Lumina and is headquartered in Mexico City with plans to expand to Brazil next year",
        "query": "Where is the skincare company whose website I am localizing headquartered?",
        "expected": "mexico city",
    },
    {
        # Background #1: "Got an urgent request to translate a birth certificate... immigration case..."
        "background_idx": 1,
        "detail": "The immigration case birth certificate was for a family named Rojas who has been waiting fourteen months for asylum",
        "query": "How long has the family whose birth certificate I translated been waiting for asylum?",
        "expected": "fourteen",
    },
]


def load_model():
    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    return model, tokenizer


def run_text_rag(model, tokenizer, entry):
    """Baseline: put the detail fact in the prompt as text."""
    facts_text = f"- {entry['detail']}"
    messages = [
        {"role": "system", "content": f"Use these facts to answer:\n{facts_text}"},
        {"role": "user", "content": entry["query"]},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(formatted, return_tensors="pt")
    prompt_tokens = input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=100, do_sample=False)
    response = tokenizer.decode(out[0][prompt_tokens:], skip_special_tokens=True).strip()
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()
    correct = entry["expected"].lower() in response.lower()
    return correct, response, prompt_tokens


def main():
    from corpus_100 import MEMORIES

    model, tokenizer = load_model()

    total_facts = len(MEMORIES) + len(LINKED_DETAILS)
    print(f"\n{'='*70}")
    print("HYPOTHESIS TEST: LINKING DETAILS TO EXISTING MEMORIES")
    print(f"{'='*70}")
    print(f"Background memories: {len(MEMORIES)}")
    print(f"Detail facts linked to existing: {len(LINKED_DETAILS)}")
    print(f"Total packs: {total_facts}")
    print(f"Linking facts stored separately: 0 (this is the difference)")
    print()

    # -- Text RAG baseline (single detail fact only) --
    print("--- TEXT RAG BASELINE (detail fact in prompt) ---")
    rag_correct = 0
    for i, entry in enumerate(LINKED_DETAILS):
        correct, response, tokens = run_text_rag(model, tokenizer, entry)
        rag_correct += int(correct)
        status = "PASS" if correct else "MISS"
        print(f"  [{status}] Q{i+1}: {entry['query'][:55]}...")
        if not correct:
            print(f"    Expected: {entry['expected']}")
            print(f"    Got: {response[:80]}")
    print(f"  Text RAG: {rag_correct}/{len(LINKED_DETAILS)}")

    # -- Store everything --
    print()
    print("--- STORING ---")
    tmpdir = tempfile.mkdtemp()
    engine = tardigrade_db.Engine(tmpdir)
    kps = KnowledgePackStore(engine, model, tokenizer, owner=1)

    # Phase 1: Store all 100 background memories.
    print(f"  Storing {len(MEMORIES)} background memories...")
    t0 = time.time()
    background_pack_ids = []
    for mem in MEMORIES:
        pid = kps.store(mem)
        background_pack_ids.append(pid)
    bg_time = time.time() - t0
    print(f"  Background: {engine.pack_count()} packs in {bg_time:.1f}s")

    # Phase 2: Store each detail fact, then link it to its background memory.
    # The linking fact is NOT stored --- the background memory serves as the link.
    print(f"  Storing {len(LINKED_DETAILS)} detail facts (linked to existing)...")
    t0 = time.time()
    for entry in LINKED_DETAILS:
        bg_pack_id = background_pack_ids[entry["background_idx"]]
        detail_pack_id = kps.store(entry["detail"])
        # Bidirectional link: background <-> detail (via Rust engine)
        engine.add_pack_link(bg_pack_id, detail_pack_id)
    link_time = time.time() - t0
    total = engine.pack_count()
    print(f"  Total: {total} packs in {bg_time + link_time:.1f}s")

    # Count trace links for diagnostic
    linked_bg_count = sum(
        1 for pid in background_pack_ids if len(engine.pack_links(pid)) > 0
    )
    print(f"  Background memories with links: {linked_bg_count}/{len(MEMORIES)}")

    # -- Trace-linked retrieval + injection --
    print()
    print("--- GENERATE WITH TRACE ---")
    trace_correct = 0

    for i, entry in enumerate(LINKED_DETAILS):
        text, tokens, had_memory = kps.generate_with_trace(
            entry["query"] + " /no_think", k=1,
            max_new_tokens=100, do_sample=False,
        )
        correct = entry["expected"].lower() in text.lower()
        trace_correct += int(correct)
        status = "PASS" if correct else "MISS"
        print(f"  [{status}] Q{i+1}: {entry['query'][:55]}...")
        if not correct:
            print(f"    Expected: {entry['expected']}")
            print(f"    Got: {text[:80]}")

    # -- Summary --
    print()
    print("=" * 70)
    print("SUMMARY: LINKED-TO-EXISTING vs COMPETING DUPLICATES")
    print("=" * 70)
    print(f"  Total packs: {total} (vs 140 in original scale test)")
    print(f"  Queries: {len(LINKED_DETAILS)}")
    print()
    print(f"  Text RAG (detail fact only):      {rag_correct}/{len(LINKED_DETAILS)}")
    print(f"  Linked-to-existing (this test):   {trace_correct}/{len(LINKED_DETAILS)} ({100*trace_correct//len(LINKED_DETAILS)}%)")
    print(f"  Original store_linked (baseline):  14/20 (70%)")
    print(f"  Text RAG (both facts, original):   19/20 (95%)")
    print()

    if trace_correct > 14:
        print(f"  HYPOTHESIS CONFIRMED: {trace_correct}/20 > 14/20 (70% plateau broken)")
        print("  The 70% plateau was a corpus design artifact.")
    elif trace_correct == 14:
        print("  HYPOTHESIS INCONCLUSIVE: matched 70% plateau exactly")
    else:
        print(f"  HYPOTHESIS REJECTED: {trace_correct}/20 <= 14/20")
        print("  The limitation is architectural, not corpus design.")


if __name__ == "__main__":
    main()
