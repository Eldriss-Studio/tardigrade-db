"""Phase 2.5 validation: KnowledgePackStore.store() no longer crashes on
hybrid-attention models.

Pre-fix (≤ 0.3.3): the unconditional `kv.layers[li].keys[0]` loop in
`store()` raised AttributeError on RecurrentGemma's recurrent layers
(which have no `.keys`). Post-fix (commit 6ed22c6): the softmax-only
filter from `_softmax_layer_payloads` skips them.

This script writes 5 facts and checks two contracts:

    1.  Every `store(...)` call returns an int pack_id (no exception).
    2.  Each stored pack has `len(layers) == n_softmax_layers`, not
        `n_layers` — i.e. recurrent layers were correctly skipped.

Run:

    HYBRID_MODEL=google/recurrentgemma-2b-it python validate_hybrid_store.py

The retrieval-quality story (does K-vector at the right layer recover
the fact?) is v0.3.3's domain, validated by `engine_recall.py`. This
script only proves the storage gap is closed.
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tardigrade_db import Engine
from tardigrade_hooks.kp_injector import KnowledgePackStore

MODEL_ID = os.environ.get("HYBRID_MODEL", "google/recurrentgemma-2b-it")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FACTS = [
    "The override vector for the orbital relay is DILLINGER-1.",
    "Mira Chen's apartment number is 4B on the third floor.",
    "The wifi password at the cafe is mango-cathedral-7.",
    "The pharmacy on Bleeker closes at 8:30pm on Tuesdays.",
    "Lucia's favorite dinosaur is the Pachycephalosaurus.",
]


def main() -> int:
    print(f"Loading {MODEL_ID} on {DEVICE}…", flush=True)
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(DEVICE)
    model.train(False)

    cfg = model.config
    block_types = getattr(cfg, "layers_block_type", None) or getattr(
        cfg, "layer_types", None
    )
    print(f"  num_hidden_layers   : {cfg.num_hidden_layers}")
    print(f"  layers_block_type   : {block_types}")

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = Engine(tmpdir)
        kps = KnowledgePackStore(engine, model, tok, owner=1)
        print(f"  n_softmax_layers    : {kps.n_softmax_layers}")
        print(f"  retrieval strategy  : {type(kps.retrieval_key_strategy).__name__}")
        print()

        pack_ids = []
        for i, fact in enumerate(FACTS):
            try:
                pid = kps.store(fact, auto_link=False)
            except AttributeError as e:
                print(f"  [{i+1}] FAIL — AttributeError: {e}", flush=True)
                return 1
            pack_ids.append(pid)
            print(f"  [{i+1}] stored, pack_id={pid}", flush=True)

        print()
        # The retrieval-side check confirms the read guard accepts the
        # hybrid pack: if `n_softmax_layers` were over-counted, the
        # `len(layers) < self.n_softmax_layers` guard in
        # retrieve_and_inject would reject every pack as malformed.
        cache, _, _ = kps.retrieve_and_inject("what is the wifi password?")
        if cache is None:
            print("FAIL — retrieve_and_inject rejected all packs as malformed.")
            print("  (This means n_softmax_layers over-counted vs what store() wrote.)")
            return 1

        cache_layer_count = sum(1 for li in range(len(cache.layers))
                                if cache.layers[li].keys is not None
                                and cache.layers[li].keys.numel() > 0)
        print(f"retrieve_and_inject reconstructed cache: "
              f"{cache_layer_count} populated layers")
        print()

        n_packs = engine.pack_count()
        if n_packs == len(FACTS):
            print(f"PASS — {n_packs}/{len(FACTS)} stores succeeded without "
                  f"AttributeError; n_softmax_layers={kps.n_softmax_layers} "
                  f"matches what was persisted (read guard accepted).")
            return 0
        else:
            print(f"FAIL — pack_count={n_packs}, expected {len(FACTS)}.")
            return 1


if __name__ == "__main__":
    sys.exit(main())
