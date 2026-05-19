"""End-to-end memory recall: store → retrieve → inject → generate.

Exercises the full KnowledgePackStore pipeline that production
consumers use. Goes through every layer of the stack:

    1.  KnowledgePackStore.store(fact)   — writes K/V pack to engine
    2.  KnowledgePackStore.retrieve_and_inject(query)
                                        — engine retrieval (read_pack)
                                        — DynamicCache reconstruction
                                        — chat-template-adapter splice
    3.  model.generate(query_ids, past_key_values=cache, ...)
                                        — model continuation
    4.  substring match of expected answer in the generated text

Compare across query_layer choices to see how the layer pick affects
the *behavior the consumer actually observes* (the model's generated
output) rather than just engine retrieval quality.

Run:

    HYBRID_SPIKE_QUERY_LAYER=0  python end_to_end_recall.py
    HYBRID_SPIKE_QUERY_LAYER=17 python end_to_end_recall.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tardigrade_db import Engine
from tardigrade_hooks.constants import DEFAULT_STORE_SALIENCE
from tardigrade_hooks.kp_injector import KnowledgePackStore

MODEL_ID = os.environ.get("HYBRID_SPIKE_MODEL", "Qwen/Qwen3-1.7B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LIMIT = int(os.environ.get("HYBRID_SPIKE_LIMIT", "20"))
MAX_NEW_TOKENS = 40
FACTS_PATH = Path(__file__).parent / "facts.json"


def load_model():
    print(f"Loading {MODEL_ID} on {DEVICE}…", flush=True)
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    model = model.to(DEVICE)
    model.train(False)
    return model, tok


def main():
    facts = json.loads(FACTS_PATH.read_text())[:LIMIT]
    model, tok = load_model()

    qlayer_env = os.environ.get("HYBRID_SPIKE_QUERY_LAYER")
    qlayer = int(qlayer_env) if qlayer_env else None

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = Engine(tmpdir)
        kps = KnowledgePackStore(engine, model, tok, owner=1, query_layer=qlayer)
        print(f"\nKnowledgePackStore using query_layer={kps.query_layer}")
        print(f"Adapter: {type(kps.adapter).__name__}\n")

        # Store all facts
        print(f"Storing {len(facts)} facts via kps.store()…")
        fact_to_pack: dict[int, int] = {}
        for i, item in enumerate(facts):
            pack_id = kps.store(item["fact"], salience=DEFAULT_STORE_SALIENCE, auto_link=False)
            fact_to_pack[i] = pack_id
            if (i + 1) % 5 == 0:
                print(f"  stored {i+1}/{len(facts)}", flush=True)

        # End-to-end retrieve + inject + generate
        print(f"\nRetrieving + injecting + generating for {len(facts)} queries…\n", flush=True)
        retrieval_hits = 0    # was the right pack retrieved?
        generation_hits = 0   # did the generated text contain the answer?
        for i, item in enumerate(facts):
            query = item["query"]
            answer = item["answer"]
            expected_pack = fact_to_pack[i]

            cache, query_ids, attention_mask = kps.retrieve_and_inject(query)
            if cache is None:
                gen = "<NO PACK RETRIEVED>"
                retrieved_pack = -1
            else:
                # Find which pack was retrieved.
                # retrieve_and_inject only returns the top-1 cache; we
                # need another peek for the diagnostic. Re-query for
                # the pack metadata directly.
                import numpy as np
                from tardigrade_hooks.encoding import encode_per_token
                q_in = tok.encode(query, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    q_out = model(q_in, output_hidden_states=True)
                h = q_out.hidden_states[kps.query_layer][0]
                h_tokens = h[1:].float().cpu().numpy().astype(np.float32)
                qk = encode_per_token(h_tokens, kps.hidden_size)
                packs = engine.mem_read_pack(qk, 1, kps.owner)
                retrieved_pack = packs[0]["pack_id"] if packs else -1

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

            ret_ok = (retrieved_pack == expected_pack)
            gen_ok = answer.lower() in gen.lower()
            retrieval_hits += int(ret_ok)
            generation_hits += int(gen_ok)
            ret_mark = "✓" if ret_ok else "·"
            gen_mark = "✓" if gen_ok else "·"
            short = gen.replace("\n", " ")[:65]
            print(f"  [{i+1:>2}] ret:{ret_mark} gen:{gen_mark} {answer!r:<25s} → {short!r}", flush=True)

        print()
        print("=" * 72)
        print(f"End-to-end recall on {MODEL_ID} @ query_layer={kps.query_layer}")
        print("-" * 72)
        print(f"Engine retrieval (right pack returned):  {retrieval_hits:>3d}/{len(facts):<3d}  ({100*retrieval_hits/len(facts):>3.0f}%)")
        print(f"Generation hit (answer in output text):  {generation_hits:>3d}/{len(facts):<3d}  ({100*generation_hits/len(facts):>3.0f}%)")
        print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
