#!/usr/bin/env python3
"""Example: Agent memory loop with TardigradeDB.

Demonstrates the full lifecycle:
1. Store memories from a conversation
2. Link related facts
3. Retrieve memories for new queries
4. Generate responses with injected KV cache

Usage:
    source .venv/bin/activate
    PYTHONPATH=python python examples/agent_memory.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "python"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import tardigrade_db
from tardigrade_hooks.kp_injector import KnowledgePackStore


def main():
    print("Loading model (Qwen3-0.6B)...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    db_path = tempfile.mkdtemp()
    engine = tardigrade_db.Engine(db_path)
    kps = KnowledgePackStore(engine, model, tokenizer, owner=1)
    print(f"Engine ready at {db_path}\n")

    # -- Simulate a conversation -----------------------------------------------

    print("=" * 60)
    print("CONVERSATION: Agent learns about the user")
    print("=" * 60)
    print()

    print("User: I prefer morning meetings, ideally before 10am")
    pid1 = kps.store("User prefers morning meetings before 10am")
    print(f"  Agent stored memory (pack_id={pid1})\n")

    print("User: I'm working on the Meridian project with a March deadline")
    pid2 = kps.store("User is working on the Meridian project")
    pid3 = kps.store_and_link("Meridian project deadline is March 15th", pid2)
    print(f"  Agent stored project (pack_id={pid2})")
    print(f"  Agent linked deadline (pack_id={pid3})\n")

    print("User: My teammate Sarah is the lead on Meridian")
    pid4 = kps.store_and_link("Sarah is the lead on the Meridian project", pid2)
    print(f"  Agent linked teammate (pack_id={pid4})\n")

    # -- Later: Agent recalls memories -----------------------------------------

    print("=" * 60)
    print("LATER: Agent recalls memories for new queries")
    print("=" * 60)
    print()

    # Reasoning behavior (e.g. Qwen3's `<think>` blocks) is controlled by
    # KnowledgePackStore via the tokenizer's `enable_thinking=False` kwarg —
    # tokenizers that don't support it silently ignore the flag. The example
    # query stays clean: no model-specific tags or post-hoc output stripping.
    print("User: When should we schedule our next sync?")
    text, tokens, had = kps.generate(
        "When should we schedule the next sync?",
        max_new_tokens=120, do_sample=False,
    )
    print(f"  Agent (memory={had}, tokens={tokens}): {text}")
    print()

    print("User: When is the Meridian deadline?")
    text, tokens, had = kps.generate_with_trace(
        "When is the Meridian project deadline?",
        k=1, max_new_tokens=120, do_sample=False,
    )
    print(f"  Agent (memory={had}, tokens={tokens}): {text}")
    print()

    # -- Memory graph ----------------------------------------------------------

    print("Memory graph:")
    for pid in range(1, engine.pack_count() + 1):
        links = engine.pack_links(pid)
        fact_text = engine.pack_text(pid) or "?"
        link_str = f" -> links to {links}" if links else ""
        print(f"  [{pid}] {fact_text[:50]}...{link_str}")

    total_links = sum(
        len(engine.pack_links(p)) for p in range(1, engine.pack_count() + 1)
    ) // 2
    print(f"\nTotal: {engine.pack_count()} memories, {total_links} links")


if __name__ == "__main__":
    main()
