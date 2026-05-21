"""Microbench: Python rrf_fuse over N mem_read_pack calls vs one Rust mem_read_multi_layer.

Phase 11 plan target: 5x. Realistic expectation: parity-plus, since
engine query work dominates and RRF over 30 items in Python is trivial.
"""

import tempfile
import time

import numpy as np

import tardigrade_db
from tardigrade_hooks.multi_layer_query import rrf_fuse

DIM = 128
NUM_PACKS = 5000
NUM_LAYERS = 3
K = 5
ITERS = 100
WARMUP = 10


def bench(label, fn, iters):
    for _ in range(WARMUP):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - start
    per = elapsed / iters * 1e6
    print(f"  {label:42s} {per:8.1f} us/call  ({iters} iters)")
    return per


def main():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        eng = tardigrade_db.Engine(tmp)
        for _ in range(NUM_PACKS):
            key = rng.standard_normal(DIM).astype(np.float32)
            data = rng.standard_normal(DIM).astype(np.float32)
            eng.mem_write_pack(
                owner=1, retrieval_key=key,
                layer_payloads=[(0, data)], salience=50.0,
            )

        queries = [rng.standard_normal(DIM).astype(np.float32) for _ in range(NUM_LAYERS)]
        print(f"packs={NUM_PACKS}, layers={NUM_LAYERS}, k={K}, iters={ITERS}\n")

        def python_path():
            ranked = [eng.mem_read_pack(q, k=K * 2, owner=1) for q in queries]
            fused = rrf_fuse(ranked, k=60)
            return fused[:K]

        def rust_path():
            return eng.mem_read_multi_layer(queries, k=K, owner=1)

        py_us = bench("python rrf_fuse over N mem_read_pack", python_path, ITERS)
        rs_us = bench("rust engine.mem_read_multi_layer", rust_path, ITERS)
        print(f"\n  speedup: {py_us / rs_us:.2f}x")


if __name__ == "__main__":
    main()
