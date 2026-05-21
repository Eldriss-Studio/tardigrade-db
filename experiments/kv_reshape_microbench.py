"""Microbench: Python flat_to_blocks/blocks_to_flat vs Rust engine path.

Phase 8 bench gate: ≥15× per-layer speedup over numpy at Qwen3-0.6B dims.
"""

import tempfile
import time

import numpy as np

import tardigrade_db
from tardigrade_vllm.format import blocks_to_flat, flat_to_blocks

NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 16
SEQ_LEN = 263  # not a multiple of BLOCK_SIZE — forces real padding work
WARMUP = 20
ITERS = 500


def bench(label, fn, iters):
    for _ in range(WARMUP):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - start
    per = elapsed / iters * 1e6
    print(f"  {label:32s} {per:8.1f} us/call  ({iters} iters)")
    return per


def main():
    rng = np.random.default_rng(0)
    kv_dim = NUM_KV_HEADS * HEAD_DIM
    flat = rng.standard_normal(2 * SEQ_LEN * kv_dim).astype(np.float32)
    num_blocks = (SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
    k_pre_blocks, v_pre_blocks = flat_to_blocks(flat, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE)
    k_flat = k_pre_blocks.ravel()
    v_flat = v_pre_blocks.ravel()

    with tempfile.TemporaryDirectory() as tmp:
        eng = tardigrade_db.Engine(tmp)
        print(f"seq_len={SEQ_LEN}, kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}, "
              f"block_size={BLOCK_SIZE}, num_blocks={num_blocks}, iters={ITERS}\n")

        print("flat_to_blocks / flat_to_paged")
        py_us = bench(
            "python (numpy)",
            lambda: flat_to_blocks(flat, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE),
            ITERS,
        )
        rs_us = bench(
            "rust (engine.flat_to_paged)",
            lambda: eng.flat_to_paged(flat, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE),
            ITERS,
        )
        print(f"  speedup: {py_us / rs_us:.1f}x\n")

        print("blocks_to_flat / paged_to_flat")
        py_us = bench(
            "python (numpy)",
            lambda: blocks_to_flat(k_pre_blocks, v_pre_blocks, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM),
            ITERS,
        )
        rs_us = bench(
            "rust (engine.paged_to_flat)",
            lambda: eng.paged_to_flat(k_flat, v_flat, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM),
            ITERS,
        )
        print(f"  speedup: {py_us / rs_us:.1f}x\n")


if __name__ == "__main__":
    main()
