"""Microbenchmark: Python retrieval-key strategy vs Rust Engine path.

Phase 7 bench gate: ≥30× speedup at prompt_len=1024, hidden=1024.

Both paths produce identical vectors. The Python path runs the strategy
in numpy; the Rust path calls Engine.compute_retrieval_key after the
embedding table has been loaded once on the engine.
"""

import time
import tempfile
import numpy as np

import tardigrade_db
from tardigrade_vllm.retrieval_key import (
    LastTokenEmbeddingStrategy,
    MeanPoolEmbeddingStrategy,
)

VOCAB = 152064  # Qwen3-0.6B vocab
HIDDEN = 1024
PROMPT_LEN = 1024
WARMUP = 20
ITERS = 200


def bench(label, fn, iters):
    for _ in range(WARMUP):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - start
    per = elapsed / iters * 1e6
    print(f"  {label:30s} {per:8.1f} us/call  ({iters} iters)")
    return per


def main():
    rng = np.random.default_rng(0)
    table = rng.standard_normal((VOCAB, HIDDEN)).astype(np.float32)
    token_ids_np = rng.integers(0, VOCAB, size=PROMPT_LEN, dtype=np.int64)
    token_ids_list = token_ids_np.tolist()

    with tempfile.TemporaryDirectory() as tmp:
        eng = tardigrade_db.Engine(tmp)
        eng.load_embedding_table(table)

        print(f"prompt_len={PROMPT_LEN}, hidden={HIDDEN}, vocab={VOCAB}, iters={ITERS}\n")

        print("LastToken")
        py_last = LastTokenEmbeddingStrategy()
        py_us = bench("python (numpy)", lambda: py_last.compute(token_ids_list, table), ITERS)
        rs_us = bench(
            "rust (engine.compute_retrieval_key)",
            lambda: eng.compute_retrieval_key(token_ids_list, "last_token"),
            ITERS,
        )
        print(f"  speedup: {py_us / rs_us:.1f}x\n")

        print("MeanPool")
        py_mean = MeanPoolEmbeddingStrategy()
        py_us = bench("python (numpy)", lambda: py_mean.compute(token_ids_list, table), ITERS)
        rs_us = bench(
            "rust (engine.compute_retrieval_key)",
            lambda: eng.compute_retrieval_key(token_ids_list, "mean_pool"),
            ITERS,
        )
        print(f"  speedup: {py_us / rs_us:.1f}x\n")


if __name__ == "__main__":
    main()
