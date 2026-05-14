"""Look at the raw stored key + the raw query key for one LoCoMo item.

Verifies the encoder format is what the Rust decoder expects, and that
ingest-key and query-key are similar enough that retrieval should work.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import tardigrade_db
from tardigrade_hooks.constants import DEFAULT_CAPTURE_LAYER_RATIO
from tardigrade_hooks.hf_kv_hook import HuggingFaceKVHook

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "cpu"
DATASET = str(Path(__file__).resolve().parents[1]
              / "benchmarks/datasets/phase1_oracle/locomo_phase1.jsonl")


def main():
    with open(DATASET) as f:
        items = [json.loads(next(f)) for _ in range(3)]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32, attn_implementation="eager",
    ).to(DEVICE).eval()
    n_layers = model.config.num_hidden_layers
    query_layer = int(n_layers * DEFAULT_CAPTURE_LAYER_RATIO)
    print(f"query_layer={query_layer} hidden_size={model.config.hidden_size}")

    data_dir = tempfile.mkdtemp(prefix="probe_keys_")
    engine = tardigrade_db.Engine(data_dir)
    hook = HuggingFaceKVHook(
        engine, owner=1, k=5,
        model_config=model.config, model=model,
        use_hidden_states=True,
    )

    print(f"\n=== Constants ===")
    print(f"ENCODING_HEADER_SIZE: {tardigrade_db.ENCODING_HEADER_SIZE}")
    print(f"ENCODING_SENTINEL: {tardigrade_db.ENCODING_SENTINEL}")
    print(f"ENCODING_N_TOKENS_IDX: {tardigrade_db.ENCODING_N_TOKENS_IDX}")
    print(f"ENCODING_DIM_IDX: {tardigrade_db.ENCODING_DIM_IDX}")

    for i, item in enumerate(items):
        print(f"\n=== Item {i}: {item['id']} ===")
        print(f"q: {item['question']!r}")
        print(f"ctx[:80]: {item['context'][:80]!r}")

        # INGEST PATH
        inputs = tokenizer(item["context"], return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, use_cache=True, output_hidden_states=True)
        decision = hook.on_generate(
            layer=query_layer,
            past_key_values=out.past_key_values,
            model_hidden_states=out.hidden_states[query_layer],
        )
        key = decision.key
        print(f"  ingest key.shape: {key.shape}")
        print(f"  ingest key[0] (sentinel): {key[0]}")
        print(f"  ingest key[33] (DIM_IDX, expect 1024): {key[33]}")
        print(f"  ingest key[32] (N_TOKENS_IDX): {key[32]}")
        n_inferred = (len(key) - 64) // int(round(key[33]))
        print(f"  ingest inferred n: {n_inferred}")
        print(f"  ingest data norm: {np.linalg.norm(key[64:]):.4f}")

        # QUERY PATH
        inputs = tokenizer(item["question"], return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, use_cache=True, output_hidden_states=True)
        # Same hook flow but get raw key as on_prefill builds it
        h = out.hidden_states[query_layer][0]
        q_tokens = h[1:].detach().cpu().numpy().astype(np.float32)
        from tardigrade_hooks.encoding import encode_per_token
        q_key = encode_per_token(q_tokens, q_tokens.shape[1])
        print(f"  query key.shape: {q_key.shape}")
        print(f"  query key[33]: {q_key[33]}")
        print(f"  query data norm: {np.linalg.norm(q_key[64:]):.4f}")

        # Compute dot products manually: each query token vs each ingest token
        n_q = (len(q_key) - 64) // int(round(q_key[33]))
        dim = int(round(q_key[33]))
        n_k = (len(key) - 64) // dim
        q_mat = q_key[64:].reshape(n_q, dim)
        k_mat = key[64:].reshape(n_k, dim)
        # Top5Avg per query token: max-pool query token against all k tokens, then avg top-5
        dot = q_mat @ k_mat.T  # (n_q, n_k)
        print(f"  dot.shape: {dot.shape}, max: {dot.max():.2f}, mean: {dot.mean():.2f}")
        top_per_q = dot.max(axis=1)
        print(f"  top-1-per-q-token mean: {top_per_q.mean():.2f}")

        # Now write to engine and query
        cell_id = engine.mem_write(
            1, query_layer, decision.key, decision.value, decision.salience, None,
        )
        print(f"  cell_id: {cell_id}")

    # Now query the engine with item 0's question and see what it returns
    print(f"\n=== Engine query for item 0's question ===")
    inputs = tokenizer(items[0]["question"], return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, use_cache=True, output_hidden_states=True)
    handles = hook.on_prefill(
        layer=query_layer,
        past_key_values=out.past_key_values,
        model_hidden_states=out.hidden_states[query_layer],
    )
    print(f"  returned {len(handles)} handles")
    for h in handles[:5]:
        print(f"    cell_id={h.cell_id} score={h.score:.4f}")


if __name__ == "__main__":
    main()
