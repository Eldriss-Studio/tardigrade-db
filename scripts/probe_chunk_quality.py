"""Forensic on the fixed-chunker diagnostic output.

Tests three hypotheses about the persistent LongMemEval hub-cell
catastrophe:

H1 — Hub cells are fragments at BOTH start AND end (not just start).
H2 — Hub cells are a small minority; most chunks are clean but a
     few fragments still dominate. (Anisotropy hypothesis.)
H3 — The boundary trim is failing silently on LongMemEval because
     the lookback window doesn't contain a clean break (no \\n\\n,
     no sentence punctuation, no whitespace within window).

For each, prints quantitative evidence.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from tdb_bench.adapters.tardigrade import TardigradeAdapter
from tdb_bench.models import BenchmarkItem


REPORT = "target/rank-diagnostic-fixed-chunker.json"
DATASET = "benchmarks/datasets/phase1_oracle/longmemeval_phase1.jsonl"
N_ITEMS = 500
N_TOP_HUBS = 20


def looks_like_fragment_start(text: str) -> bool:
    """Heuristic: chunk starts mid-word if the first non-whitespace
    character is a letter or digit AND the preceding character in
    the chunk's stored text was none (i.e., chunk truly starts here)."""
    s = text.lstrip()
    if not s:
        return False
    first = s[0]
    # If first char is a digit, almost certainly mid-number.
    if first.isdigit():
        return True
    # If first char is a lowercase letter, likely mid-sentence
    # continuation. Capital starts are likely sentence-initial or
    # proper-noun-initial — usually fine.
    return first.islower()


def looks_like_fragment_end(text: str) -> bool:
    """Heuristic: chunk ends mid-word if the last character is a
    letter or digit (not punctuation/whitespace)."""
    s = text.rstrip()
    if not s:
        return False
    last = s[-1]
    return last.isalnum() and last not in {".", "!", "?"}


def main():
    d = json.load(open(REPORT))
    records = d["datasets"]["longmemeval"]["records"]

    # Identify hub cells from the diagnostic
    counter: collections.Counter[int] = collections.Counter()
    for r in records:
        counter.update(r["retrieved_top10"])
    top_hubs = {cell for cell, _ in counter.most_common(N_TOP_HUBS)}

    # Re-ingest to recover chunk texts
    items = []
    with open(DATASET) as f:
        for line in f:
            j = json.loads(line)
            items.append(BenchmarkItem(
                item_id=j["id"], dataset=j["dataset"], context=j["context"],
                question=j["question"], ground_truth=j["ground_truth"],
            ))
            if len(items) >= N_ITEMS:
                break

    adapter = TardigradeAdapter()
    adapter.enable_chunk_text_tracking()
    adapter.ingest(items)

    cell_to_text = adapter._cell_to_chunk_text
    print(f"Total cells: {len(cell_to_text)}")

    # --- H1: are hub cells fragments at BOTH ends? ---
    print(f"\n=== H1: hub-cell fragment quality (top {N_TOP_HUBS} hubs) ===")
    hub_frag_starts = 0
    hub_frag_ends = 0
    hub_both = 0
    for cell in top_hubs:
        text = cell_to_text.get(cell, "")
        s = looks_like_fragment_start(text)
        e = looks_like_fragment_end(text)
        hub_frag_starts += int(s)
        hub_frag_ends += int(e)
        if s and e:
            hub_both += 1
    print(f"  hub cells with fragment START: {hub_frag_starts}/{N_TOP_HUBS}")
    print(f"  hub cells with fragment END:   {hub_frag_ends}/{N_TOP_HUBS}")
    print(f"  hub cells with BOTH:           {hub_both}/{N_TOP_HUBS}")

    # --- H2: how does this compare to NON-hub cells? ---
    print(f"\n=== H2: fragment rate, hubs vs non-hubs ===")
    non_hub_frag_starts = 0
    non_hub_frag_ends = 0
    non_hub_both = 0
    non_hub_total = 0
    for cell_id, text in cell_to_text.items():
        if cell_id in top_hubs:
            continue
        non_hub_total += 1
        s = looks_like_fragment_start(text)
        e = looks_like_fragment_end(text)
        non_hub_frag_starts += int(s)
        non_hub_frag_ends += int(e)
        if s and e:
            non_hub_both += 1
    print(f"  non-hub fragment START: {non_hub_frag_starts}/{non_hub_total} "
          f"({non_hub_frag_starts/non_hub_total:.1%})")
    print(f"  non-hub fragment END:   {non_hub_frag_ends}/{non_hub_total} "
          f"({non_hub_frag_ends/non_hub_total:.1%})")
    print(f"  non-hub BOTH:           {non_hub_both}/{non_hub_total} "
          f"({non_hub_both/non_hub_total:.1%})")
    print(f"\n  hub fragment rate:     {hub_both}/{N_TOP_HUBS} "
          f"= {hub_both/N_TOP_HUBS:.1%}")
    print(f"  non-hub fragment rate: {non_hub_both}/{non_hub_total} "
          f"= {non_hub_both/non_hub_total:.1%}")

    # --- H3: what does a LongMemEval context look like for the chunker? ---
    print(f"\n=== H3: LongMemEval text structure analysis ===")
    sample_ctx = items[0].context
    print(f"  sample context len: {len(sample_ctx)} chars")
    print(f"  '\\n\\n' count:    {sample_ctx.count(chr(10) + chr(10))}")
    print(f"  '\\n' count:      {sample_ctx.count(chr(10))}")
    print(f"  sentence-ending punctuation (.?!):  "
          f"{sum(sample_ctx.count(c) for c in '.?!')}")
    # Average distance between paragraph boundaries
    para_positions = [i for i, c in enumerate(sample_ctx)
                       if sample_ctx[i:i+2] == "\n\n"]
    if para_positions:
        avg_para_gap = (
            sum(b - a for a, b in zip(para_positions[:-1], para_positions[1:]))
            / max(1, len(para_positions) - 1)
        )
        print(f"  avg distance between '\\n\\n': {avg_para_gap:.0f} chars")
    else:
        print(f"  no '\\n\\n' found at all!")
    line_positions = [i for i, c in enumerate(sample_ctx) if c == "\n"]
    if line_positions:
        avg_line_gap = (
            sum(b - a for a, b in zip(line_positions[:-1], line_positions[1:]))
            / max(1, len(line_positions) - 1)
        )
        print(f"  avg distance between '\\n':   {avg_line_gap:.0f} chars")


if __name__ == "__main__":
    main()
