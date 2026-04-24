#!/usr/bin/env python3
# Phase 26: Stage-by-stage KV pipeline diagnostic.
#
# Direct injection works (8/10). Through TardigradeDB: 0/10.
# This script tests each stage to find where fidelity breaks.

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, str(Path(".").resolve() / "python"))
import tardigrade_db

MODEL = "Qwen/Qwen3-0.6B"
FACT = "Sonia's wifi password is mango-cathedral-7"
QUERY = "What is Sonia's wifi password?"
EXPECTED = "mango-cathedral-7"


def clone_cache(kv):
    c = DynamicCache()
    for li in range(len(kv.layers)):
        layer = kv.layers[li]
        c.update(layer.keys.clone(), layer.values.clone(), li)
    return c


def generate_with_kv(model, tokenizer, kv_cache, fact_text):
    messages = [
        {"role": "system", "content": fact_text},
        {"role": "user", "content": QUERY + " /no_think"},
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    fact_msgs = [{"role": "system", "content": fact_text}]
    fact_fmt = tokenizer.apply_chat_template(fact_msgs, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer.encode(full_text, return_tensors="pt")
    fact_len = len(tokenizer.encode(fact_fmt))
    query_ids = full_ids[:, fact_len:]
    kv_len = kv_cache.get_seq_length()
    q_len = query_ids.shape[1]
    attn_mask = torch.ones(1, kv_len + q_len, dtype=torch.long)
    with torch.no_grad():
        out = model.generate(
            query_ids, past_key_values=kv_cache,
            attention_mask=attn_mask, max_new_tokens=80, do_sample=False,
        )
    return tokenizer.decode(out[0][q_len:], skip_special_tokens=True).strip()


def cosine(a, b):
    af = a.float().flatten()
    bf = b.float().flatten()
    return float(torch.nn.functional.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)))


def main():
    print("=" * 60)
    print("KV Pipeline Diagnostic")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager")
    model.eval()
    n_layers = model.config.num_hidden_layers

    # Compute reference KV with chat template
    fact_msgs = [{"role": "system", "content": FACT}]
    fact_text = tokenizer.apply_chat_template(fact_msgs, tokenize=False, add_generation_prompt=False)
    fact_ids = tokenizer.encode(fact_text, return_tensors="pt")
    with torch.no_grad():
        fact_out = model(fact_ids, use_cache=True)
    orig_kv = fact_out.past_key_values
    seq_len = orig_kv.get_seq_length()
    h, s, d = orig_kv.layers[0].keys[0].shape
    print(f"  {n_layers} layers, seq={seq_len}, heads={h}, head_dim={d}")

    # Stage 0: direct
    print("\n--- Stage 0: Direct injection ---")
    gen0 = generate_with_kv(model, tokenizer, clone_cache(orig_kv), FACT)
    hit0 = EXPECTED.lower() in gen0.lower()
    print(f"  {'Y' if hit0 else 'N'}  \"{gen0[:60]}\"")

    # Stage 1: extract + reconstruct (no storage)
    print("\n--- Stage 1: Extract + reconstruct (no Q4) ---")
    payloads = []
    for li in range(n_layers):
        k = orig_kv.layers[li].keys[0]
        v = orig_kv.layers[li].values[0]
        k_np = k.permute(1, 0, 2).reshape(s, h * d).numpy().astype(np.float32)
        v_np = v.permute(1, 0, 2).reshape(s, h * d).numpy().astype(np.float32)
        payloads.append(np.concatenate([k_np.ravel(), v_np.ravel()]))

    kv1 = DynamicCache()
    for li in range(n_layers):
        p = payloads[li]
        half = len(p) // 2
        kt = torch.tensor(p[:half]).reshape(1, s, h, d).permute(0, 2, 1, 3)
        vt = torch.tensor(p[half:]).reshape(1, s, h, d).permute(0, 2, 1, 3)
        kv1.update(kt, vt, li)

    ck1 = cosine(orig_kv.layers[0].keys, kv1.layers[0].keys)
    gen1 = generate_with_kv(model, tokenizer, kv1, FACT)
    hit1 = EXPECTED.lower() in gen1.lower()
    print(f"  Cosine K: {ck1:.6f}")
    print(f"  {'Y' if hit1 else 'N'}  \"{gen1[:60]}\"")

    # Stage 2: through engine Q4
    print("\n--- Stage 2: Through TardigradeDB (Q4 round-trip) ---")
    db_dir = tempfile.mkdtemp()
    engine = tardigrade_db.Engine(db_dir)
    kv_dim = h * d
    dummy_key = np.zeros(kv_dim, dtype=np.float32)

    for li in range(n_layers):
        engine.mem_write(1, li, dummy_key, payloads[li], 80.0, None)

    # Read back all cells
    all_cells = {}
    results = engine.mem_read(dummy_key, n_layers * 2, None)
    for r in results:
        all_cells[r.layer] = np.array(r.value(), dtype=np.float32)

    kv2 = DynamicCache()
    recovered = 0
    for li in range(n_layers):
        if li in all_cells:
            val = all_cells[li]
            half = len(val) // 2
            kt = torch.tensor(val[:half]).reshape(1, s, h, d).permute(0, 2, 1, 3)
            vt = torch.tensor(val[half:]).reshape(1, s, h, d).permute(0, 2, 1, 3)
            kv2.update(kt, vt, li)
            recovered += 1

    if recovered == n_layers:
        ck2 = cosine(orig_kv.layers[0].keys, kv2.layers[0].keys)
        mse = float(((orig_kv.layers[0].keys.float() - kv2.layers[0].keys.float()) ** 2).mean())
        gen2 = generate_with_kv(model, tokenizer, kv2, FACT)
        hit2 = EXPECTED.lower() in gen2.lower()
        print(f"  Cosine K: {ck2:.6f} | MSE: {mse:.6f}")
        print(f"  {'Y' if hit2 else 'N'}  \"{gen2[:60]}\"")
    else:
        print(f"  Only recovered {recovered}/{n_layers} layers")
        hit2 = False
        ck2 = 0

    shutil.rmtree(db_dir)

    # Summary
    print(f"\n{'=' * 60}")
    print("  Stage 0 (direct):            " + ("PASS" if hit0 else "FAIL"))
    print("  Stage 1 (extract+rebuild):   " + ("PASS" if hit1 else "FAIL") + f" (cos={ck1:.4f})")
    if recovered == n_layers:
        print("  Stage 2 (Q4 round-trip):     " + ("PASS" if hit2 else "FAIL") + f" (cos={ck2:.4f})")
    if hit0 and hit1 and hit2:
        print("\n  Pipeline works. Previous failures were hook/format issues.")
    elif hit0 and hit1 and not hit2:
        print("\n  Q4 quantization corrupts KV. Need higher precision storage.")
    elif hit0 and not hit1:
        print("\n  Extract/reshape is wrong. Check permute ordering.")
    print("=" * 60)


if __name__ == "__main__":
    main()
