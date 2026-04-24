#!/usr/bin/env python3
"""Scale test: trace-linked multi-memory injection at 100+ memories.

Stores 100 existing single-fact memories PLUS 20 cross-referencing
fact pairs (40 additional facts, 140 total). Tests whether trace-linked
retrieval holds at scale with retrieval pressure from 100+ unrelated
memories competing for attention.

Usage:
    source .venv/bin/activate
    python experiments/multi_memory_scale_test.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "python"))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import tardigrade_db
from tardigrade_hooks.kp_injector import KnowledgePackStore
from tardigrade_hooks.multi_composer import NaiveConcatComposer

# 20 cross-referencing fact pairs embedded in Sonia's world
CROSS_REF_PAIRS = [
    {
        "facts": [
            "Lucia's swimming instructor at the community center is a woman named Tomoko",
            "Tomoko drives a red Honda Civic and parks it behind the community center",
        ],
        "query": "What kind of car does Lucia's swimming instructor drive?",
        "expected": "honda civic",
    },
    {
        "facts": [
            "The mechanic on Western Avenue who fixed the catalytic converter is named Raul",
            "Raul has a tattoo of a compass on his left forearm",
        ],
        "query": "What tattoo does the mechanic who fixed my car have?",
        "expected": "compass",
    },
    {
        "facts": [
            "Ms. Navarro is Lucia's teacher who mentioned she reads above grade level",
            "Ms. Navarro has been teaching at that school for seventeen years",
        ],
        "query": "How long has Lucia's teacher been at the school?",
        "expected": "seventeen",
    },
    {
        "facts": [
            "Dr. Huang is the doctor who said my vitamin D is low",
            "Dr. Huang graduated from Northwestern Medical School in 2009",
        ],
        "query": "Where did the doctor who checked my vitamin D go to medical school?",
        "expected": "northwestern",
    },
    {
        "facts": [
            "The bookstore in Pilsen where I did the poetry reading is called Casa Azul",
            "Casa Azul hosts a monthly open mic night on the last Friday",
        ],
        "query": "When is the open mic at the bookstore where I did the poetry reading?",
        "expected": "last friday",
    },
    {
        "facts": [
            "David the history teacher I met at the dinner party teaches at Lane Tech",
            "Lane Tech is on Addison Street near the Western Avenue Brown Line stop",
        ],
        "query": "What street is the school where David teaches located on?",
        "expected": "addison",
    },
    {
        "facts": [
            "The YMCA pool where I swim forty laps has a lap lane schedule",
            "The YMCA lap lanes are reserved from 6am to 8am on weekday mornings",
        ],
        "query": "What time are the lap lanes available at the pool where I swim?",
        "expected": "6am",
    },
    {
        "facts": [
            "My cousin in Sao Paulo who helped with the Brazilian Portuguese app is named Renata",
            "Renata works as a marketing manager at a tech company in the Pinheiros neighborhood",
        ],
        "query": "What does my cousin who helped with the Portuguese localization do for work?",
        "expected": "marketing",
    },
    {
        "facts": [
            "The therapist I started seeing after the divorce is named Dr. Morales",
            "Dr. Morales has her office on the third floor of a brownstone on Damen Avenue",
        ],
        "query": "Where is the office of the therapist I see since the divorce?",
        "expected": "damen",
    },
    {
        "facts": [
            "The sourdough starter I named Fernando was from a kit I bought online",
            "The online shop that sold the sourdough kit is called BreadCraft and they are based in Vermont",
        ],
        "query": "What state is the company from that sold me the sourdough starter kit?",
        "expected": "vermont",
    },
    {
        "facts": [
            "The publishing house in Barcelona that wants the young adult novel translated is called Editorial Sirena",
            "Editorial Sirena was founded in 1987 and specializes in Latin American authors",
        ],
        "query": "When was the Barcelona publishing house that wants me to translate the novel founded?",
        "expected": "1987",
    },
    {
        "facts": [
            "The pharmacist who remembered Lucia's name when I picked up the ear drops is named James Chen",
            "James Chen has been working at that pharmacy for twelve years",
        ],
        "query": "How long has the pharmacist who knows Lucia's name been working there?",
        "expected": "twelve",
    },
    {
        "facts": [
            "The repair guy who looked at the broken washing machine is from a company called FixIt Pro",
            "FixIt Pro charges a seventy-five dollar diagnostic fee before any repairs",
        ],
        "query": "How much does the company that looked at my washing machine charge for diagnostics?",
        "expected": "seventy-five",
    },
    {
        "facts": [
            "The mediator handling our custody adjustment is named Patricia Goldberg",
            "Patricia Goldberg also mediated the original divorce settlement two years ago",
        ],
        "query": "How long ago did the custody mediator handle our original divorce?",
        "expected": "two years",
    },
    {
        "facts": [
            "The running group in the park on Saturday mornings is led by a coach named Marcus",
            "Marcus used to run competitively in college at the University of Illinois",
        ],
        "query": "Where did the coach of my Saturday running group go to college?",
        "expected": "illinois",
    },
    {
        "facts": [
            "The book club at the library is organized by a librarian named Helen Park",
            "Helen Park has a PhD in comparative literature from the University of Chicago",
        ],
        "query": "What degree does the librarian who runs the book club have?",
        "expected": "phd",
    },
    {
        "facts": [
            "The Goodwill where I dropped off Eduardo's old clothes is on Milwaukee Avenue",
            "The Milwaukee Avenue Goodwill is open until 9pm on weekdays",
        ],
        "query": "What time does the Goodwill where I dropped off clothes close on weekdays?",
        "expected": "9pm",
    },
    {
        "facts": [
            "Lucia's friend Harper who she plays with at school has a younger brother",
            "Harper's younger brother is named Oliver and he is four years old",
        ],
        "query": "What is the name of Lucia's friend Harper's younger brother?",
        "expected": "oliver",
    },
    {
        "facts": [
            "The skincare brand website localization project is for a company called Lumina",
            "Lumina is headquartered in Mexico City and plans to expand to Brazil next year",
        ],
        "query": "Where is the skincare company whose website I am localizing headquartered?",
        "expected": "mexico city",
    },
    {
        "facts": [
            "The immigration case birth certificate was for a family named Rojas",
            "The Rojas family is applying for asylum and has been waiting fourteen months",
        ],
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
    facts_text = "\n".join(f"- {f}" for f in entry["facts"])
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

    total_facts = len(MEMORIES) + sum(len(e["facts"]) for e in CROSS_REF_PAIRS)
    print(f"\n{'='*70}")
    print("SCALE TEST: TRACE-LINKED MULTI-MEMORY AT 140 MEMORIES")
    print(f"{'='*70}")
    print(f"Background memories: {len(MEMORIES)}")
    print(f"Cross-ref pairs: {len(CROSS_REF_PAIRS)} ({sum(len(e['facts']) for e in CROSS_REF_PAIRS)} facts)")
    print(f"Total facts: {total_facts}")
    print()

    # -- Text RAG baseline --
    print("--- TEXT RAG (baseline) ---")
    rag_correct = 0
    for i, entry in enumerate(CROSS_REF_PAIRS):
        correct, response, tokens = run_text_rag(model, tokenizer, entry)
        rag_correct += int(correct)
        status = "PASS" if correct else "MISS"
        print(f"  [{status}] Q{i+1}: {entry['query'][:55]}...")
        if not correct:
            print(f"    Expected: {entry['expected']}")
            print(f"    Got: {response[:80]}")
    print(f"  Text RAG: {rag_correct}/{len(CROSS_REF_PAIRS)}")

    # -- Store everything --
    print()
    print("--- STORING ---")
    tmpdir = tempfile.mkdtemp()
    engine = tardigrade_db.Engine(tmpdir)
    kps = KnowledgePackStore(engine, model, tokenizer, owner=1)

    print(f"  Storing {len(MEMORIES)} background memories...")
    import time
    t0 = time.time()
    for mem in MEMORIES:
        kps.store(mem)
    bg_time = time.time() - t0
    print(f"  Background: {engine.pack_count()} packs in {bg_time:.1f}s")

    print(f"  Storing {len(CROSS_REF_PAIRS)} cross-ref pairs (trace-linked)...")
    t0 = time.time()
    for entry in CROSS_REF_PAIRS:
        kps.store_linked(entry["facts"])
    link_time = time.time() - t0
    total = engine.pack_count()
    print(f"  Total: {total} packs in {bg_time + link_time:.1f}s")

    # -- Trace-linked retrieval + injection --
    print()
    print("--- GENERATE WITH TRACE ---")
    trace_correct = 0

    for i, entry in enumerate(CROSS_REF_PAIRS):
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
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total memories: {total}")
    print(f"  Cross-ref queries: {len(CROSS_REF_PAIRS)}")
    print()
    print(f"  Text RAG:     {rag_correct}/{len(CROSS_REF_PAIRS)} ({100*rag_correct//len(CROSS_REF_PAIRS)}%)")
    print(f"  Trace-linked: {trace_correct}/{len(CROSS_REF_PAIRS)} ({100*trace_correct//len(CROSS_REF_PAIRS)}%)")
    print()

    if trace_correct >= 14:
        print("  SCALE TEST: PASS (>= 70%)")
    elif trace_correct >= 10:
        print("  SCALE TEST: PARTIAL (50-70%)")
    else:
        print("  SCALE TEST: FAIL (< 50%)")


if __name__ == "__main__":
    main()
