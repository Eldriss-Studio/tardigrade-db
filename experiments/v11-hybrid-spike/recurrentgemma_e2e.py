"""End-to-end RecurrentGemma validation: calibrate → store → retrieve → inject → generate.

Two-phase validation of the v0.3.2-v0.3.4 stack on a real hybrid-attention
model. The point: prove the full pipeline works on RecurrentGemma, not just
the storage path that v0.3.4 unblocked.

Phase 1 — Calibration
    Run ``select_query_layer(model, tok, registry=reg)`` to discover the
    best ``(strategy, layer)`` pair on the bundled paraphrased corpus.
    The result is persisted to ``~/.tardigrade/calibration.json`` so a
    second run skips the sweep.

Phase 2 — End-to-end recall
    Construct ``KnowledgePackStore`` with the same registry. The
    constructor picks up the cached calibration, instantiates the
    right strategy at the right layer. Store 20 facts. For each:
    retrieve + inject + generate. Score:
        retrieval-hit : was the correct pack returned by mem_read_pack?
        generation-hit: did the answer substring appear in the generation?

Run:

    HYBRID_MODEL=google/recurrentgemma-2b-it python recurrentgemma_e2e.py

To force a re-calibration (skip the registry cache), wipe it first:

    rm ~/.tardigrade/calibration.json && python recurrentgemma_e2e.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tardigrade_db import Engine
from tardigrade_hooks import (
    CalibrationRegistry,
    is_supported,
    select_query_layer,
)
from tardigrade_hooks.kp_injector import KnowledgePackStore

# Surface calibration progress so the sweep is visible (otherwise looks hung).
logging.basicConfig(
    level=logging.INFO,
    format="  [calibrate] %(message)s",
    stream=sys.stderr,
)

MODEL_ID = os.environ.get("HYBRID_MODEL", "google/recurrentgemma-2b-it")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LIMIT = int(os.environ.get("HYBRID_SPIKE_LIMIT", "20"))
MAX_NEW_TOKENS = 40
FACTS_PATH = Path(__file__).parent / "facts.json"


def main() -> int:
    print(f"Loading {MODEL_ID} on {DEVICE}…", flush=True)
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(DEVICE)
    model.train(False)

    # Pre-flight: refuse to proceed if the model isn't supported.
    report = is_supported(model, tok)
    print(f"\nCompatibility report:")
    print(f"  is_supported       : {report.is_supported}")
    print(f"  architecture       : {report.architecture}")
    print(f"  n_hidden_layers    : {report.n_hidden_layers}")
    print(f"  n_softmax_layers   : {report.n_softmax_layers}")
    print(f"  recommended        : {report.recommended_strategy}")
    print(f"  adapter            : {report.adapter_type}")
    for note in report.notes:
        print(f"  note               : {note}")
    if not report.is_supported:
        for blocker in report.blockers:
            print(f"  BLOCKER            : {blocker}")
        return 1

    # Phase 1: calibrate (or load cached calibration).
    registry = CalibrationRegistry()
    print(f"\nPhase 1: select_query_layer(registry=...) — this may run a sweep.")
    print(f"  (registry path: {registry.path})")
    result = select_query_layer(model, tok, registry=registry)
    print(f"  best_strategy      : {result.best_strategy}")
    print(f"  best_layer         : {result.best_layer}")
    print(f"  top1 / top5 scores : "
          f"({result.scores[0].top1 if result.scores else '?'}"
          f" / {result.scores[0].top5 if result.scores else '?'})"
          if result.scores else "  (loaded from cache)")

    # Phase 2: end-to-end recall via registry.
    facts = json.loads(FACTS_PATH.read_text())[:LIMIT]
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = Engine(tmpdir)
        kps = KnowledgePackStore(
            engine, model, tok, owner=1, calibration_registry=registry,
        )
        print(f"\nPhase 2: KnowledgePackStore built from registry")
        print(f"  strategy class     : {type(kps.retrieval_key_strategy).__name__}")
        print(f"  layer index        : {kps.query_layer}")

        # Store all facts.
        print(f"\nStoring {len(facts)} facts…")
        fact_to_pack: dict[int, int] = {}
        for i, item in enumerate(facts):
            pid = kps.store(item["fact"], auto_link=False)
            fact_to_pack[i] = pid
        print(f"  stored {len(facts)} packs, count={engine.pack_count()}")

        # Per-query retrieve + inject + generate.
        print(f"\nRunning retrieve + inject + generate for {len(facts)} queries…\n")
        retrieval_hits = 0
        generation_hits = 0
        for i, item in enumerate(facts):
            query = item["query"]
            answer = item["answer"]
            expected_pack = fact_to_pack[i]

            cache, query_ids, attention_mask = kps.retrieve_and_inject(query)
            if cache is None:
                ret_ok = False
                gen = "<NO PACK RETRIEVED>"
            else:
                # Diagnostic peek for retrieval hit — go through the
                # strategy-aware path so we get the same key the
                # store path used.
                qk, _ = kps._compute_query_key(query)
                packs = engine.mem_read_pack(qk, 1, kps.owner)
                retrieved_pack = packs[0]["pack_id"] if packs else -1
                ret_ok = (retrieved_pack == expected_pack)

                with torch.no_grad():
                    out = model.generate(
                        query_ids,
                        past_key_values=cache,
                        attention_mask=attention_mask,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=tok.eos_token_id,
                    )
                gen = tok.decode(out[0, query_ids.shape[1]:], skip_special_tokens=True)

            gen_ok = answer.lower() in gen.lower()
            retrieval_hits += int(ret_ok)
            generation_hits += int(gen_ok)
            r = "✓" if ret_ok else "·"
            g = "✓" if gen_ok else "·"
            short = gen.replace("\n", " ")[:65]
            print(f"  [{i+1:>2}] ret:{r} gen:{g} {answer!r:<25s} → {short!r}",
                  flush=True)

        print()
        print("=" * 72)
        print(f"End-to-end on {MODEL_ID}")
        print(f"  strategy={result.best_strategy}, layer={result.best_layer}")
        print("-" * 72)
        print(f"Engine retrieval (right pack):           "
              f"{retrieval_hits:>3d}/{len(facts):<3d}  "
              f"({100*retrieval_hits/len(facts):>3.0f}%)")
        print(f"Generation hit (answer in output):       "
              f"{generation_hits:>3d}/{len(facts):<3d}  "
              f"({100*generation_hits/len(facts):>3.0f}%)")
        print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
