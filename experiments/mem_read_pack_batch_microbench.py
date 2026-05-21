"""Microbench: N sequential mem_read_pack calls vs one mem_read_pack_batch.

The batch API exists to amortise PyO3 crossing cost; the actual retrieval
work is identical. Useful especially for multi-agent / multi-NPC tick
patterns where N owners query at once.
"""

import tempfile
import time

import numpy as np

import tardigrade_db

DIM = 128
NUM_PACKS = 256
BATCH = 16
K = 5
ITERS = 200
WARMUP = 20


def bench(label, fn, iters):
    for _ in range(WARMUP):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - start
    per = elapsed / iters * 1e6
    print(f"  {label:38s} {per:8.1f} us/call  ({iters} iters)")
    return per


def main():
    rng = np.random.default_rng(0)
    queries = [rng.standard_normal(DIM).astype(np.float32) for _ in range(BATCH)]

    with tempfile.TemporaryDirectory() as tmp:
        eng = tardigrade_db.Engine(tmp)
        for _ in range(NUM_PACKS):
            key = rng.standard_normal(DIM).astype(np.float32)
            data = rng.standard_normal(DIM).astype(np.float32)
            eng.mem_write_pack(
                owner=1, retrieval_key=key,
                layer_payloads=[(0, data)], salience=50.0,
            )

        print(f"packs={NUM_PACKS}, batch={BATCH}, k={K}, iters={ITERS}\n")

        print("no refinement")
        per_us = bench(
            "N sequential mem_read_pack",
            lambda: [eng.mem_read_pack(q, k=K, owner=1) for q in queries],
            ITERS,
        )
        batch_us = bench(
            "one mem_read_pack_batch",
            lambda: eng.mem_read_pack_batch(queries, k=K, owner=1),
            ITERS,
        )
        print(f"  speedup: {per_us / batch_us:.2f}x\n")

        eng.set_refinement_mode("centered")
        print("centered refinement")
        per_us = bench(
            "N sequential mem_read_pack",
            lambda: [eng.mem_read_pack(q, k=K, owner=1) for q in queries],
            ITERS,
        )
        batch_us = bench(
            "one mem_read_pack_batch",
            lambda: eng.mem_read_pack_batch(queries, k=K, owner=1),
            ITERS,
        )
        print(f"  speedup: {per_us / batch_us:.2f}x")


if __name__ == "__main__":
    main()
