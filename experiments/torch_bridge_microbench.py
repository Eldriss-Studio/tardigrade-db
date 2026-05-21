"""Microbench: torch -> numpy -> flat_to_paged vs torch -> flat_to_paged_torch.

Phase 12 reality check. The "win" of the torch bridge is avoiding the
intermediate numpy materialisation — `tensor.numpy()` allocates a
fresh numpy ndarray and memcpys the buffer. The Rust torch path reads
data_ptr() directly, skipping that copy.
"""

import time

import numpy as np
import torch

import tardigrade_db

KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 16
SEQ_LEN = 263  # non-aligned, forces padding work
KV_DIM = KV_HEADS * HEAD_DIM
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
    print(f"  {label:48s} {per:8.1f} us/call  ({iters} iters)")
    return per


def main():
    flat_torch = torch.randn(2 * SEQ_LEN * KV_DIM, dtype=torch.float32)
    print(f"seq_len={SEQ_LEN}, kv_dim={KV_DIM}, iters={ITERS}\n")

    numpy_path_us = bench(
        "torch -> numpy -> flat_to_paged",
        lambda: tardigrade_db.flat_to_paged(
            flat_torch.numpy(), KV_HEADS, HEAD_DIM, BLOCK_SIZE,
        ),
        ITERS,
    )
    torch_path_us = bench(
        "torch -> flat_to_paged_torch",
        lambda: tardigrade_db.flat_to_paged_torch(
            flat_torch, KV_HEADS, HEAD_DIM, BLOCK_SIZE,
        ),
        ITERS,
    )
    print(f"\n  speedup: {numpy_path_us / torch_path_us:.2f}x")


if __name__ == "__main__":
    main()
