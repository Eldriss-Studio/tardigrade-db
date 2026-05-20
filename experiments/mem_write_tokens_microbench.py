"""Microbenchmark for the write path at (seq_len=256, dim=1024).

Asserts an absolute per-write latency budget on `mem_write_pack` under a
streaming write buffer. The budget reflects that the engine must build
cells, Q4-quantize them into the block pool, fsync once per batch, and
index each pack into `pipeline`, `slb`, `governance`, and the pack
directory — without paying the O(dim²) whitening covariance cost on
every insert.

Both write paths (legacy pre-encoded + new token-matrix API) are timed
side-by-side as a sanity check on API symmetry. The headline budget is
on the token-matrix path because that's the recommended API for
consumers that have a `(n_tokens, dim)` numpy matrix in hand.

Run from the repo root:
    source .venv/bin/activate
    python experiments/mem_write_tokens_microbench.py

Writes a JSON record to docs/perf/mem-write-tokens.json on success.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np

import tardigrade_db
from tardigrade_hooks.encoding import encode_per_token

SEQ_LEN = 256
DIM = 1024
N_WRITES_PER_RUN = 20
N_RUNS = 3
OWNER = 1


def _make_tokens(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((SEQ_LEN, DIM)).astype(np.float32)


def _make_layer() -> tuple[int, np.ndarray]:
    rng = np.random.default_rng(0)
    return (0, rng.standard_normal(64).astype(np.float32))


def _time_encoded_path(engine: tardigrade_db.Engine, tokens_batch: list[np.ndarray]) -> float:
    layer = _make_layer()
    start = time.perf_counter()
    for tokens in tokens_batch:
        encoded = encode_per_token(tokens, DIM)
        engine.mem_write_pack(OWNER, encoded, [layer], 70.0)
    engine.flush_buffer()
    return time.perf_counter() - start


def _time_token_path(engine: tardigrade_db.Engine, tokens_batch: list[np.ndarray]) -> float:
    layer = _make_layer()
    start = time.perf_counter()
    for tokens in tokens_batch:
        engine.mem_write_pack_tokens(OWNER, tokens, [layer], 70.0)
    engine.flush_buffer()
    return time.perf_counter() - start


def _run_once(seed_offset: int) -> tuple[float, float]:
    tokens_batch = [_make_tokens(seed=seed_offset + i) for i in range(N_WRITES_PER_RUN)]

    with tempfile.TemporaryDirectory() as tmp_enc:
        engine_enc = tardigrade_db.Engine.open_with_write_buffer(
            tmp_enc, max_batch_size=N_WRITES_PER_RUN, max_idle_ms=10_000
        )
        encoded_secs = _time_encoded_path(engine_enc, tokens_batch)

    with tempfile.TemporaryDirectory() as tmp_tok:
        engine_tok = tardigrade_db.Engine.open_with_write_buffer(
            tmp_tok, max_batch_size=N_WRITES_PER_RUN, max_idle_ms=10_000
        )
        token_secs = _time_token_path(engine_tok, tokens_batch)

    return encoded_secs, token_secs


def main() -> None:
    print(
        f"timing {N_WRITES_PER_RUN} writes at (seq_len={SEQ_LEN}, dim={DIM}) × {N_RUNS} runs...",
        flush=True,
    )

    encoded_runs: list[float] = []
    token_runs: list[float] = []
    for r in range(N_RUNS):
        enc, tok = _run_once(seed_offset=r * 10_000)
        encoded_runs.append(enc)
        token_runs.append(tok)
        print(
            f"  run {r + 1}: encoded={enc * 1000:.1f} ms  tokens={tok * 1000:.1f} ms  "
            f"({(enc / tok if tok > 0 else float('inf')):.2f}× faster)",
            flush=True,
        )

    encoded_runs.sort()
    token_runs.sort()
    encoded_median = encoded_runs[len(encoded_runs) // 2]
    token_median = token_runs[len(token_runs) // 2]
    speedup = encoded_median / token_median if token_median > 0 else float("inf")
    reduction_pct = 100.0 * (1.0 - token_median / encoded_median) if encoded_median > 0 else 0.0

    token_per_write_ms = token_median * 1000 / N_WRITES_PER_RUN
    encoded_per_write_ms = encoded_median * 1000 / N_WRITES_PER_RUN

    result = {
        "topic": "mem_write_pack absolute per-write latency",
        "seq_len": SEQ_LEN,
        "dim": DIM,
        "n_writes_per_run": N_WRITES_PER_RUN,
        "n_runs": N_RUNS,
        "encoded_seconds_median": encoded_median,
        "token_seconds_median": token_median,
        "encoded_per_write_ms": encoded_per_write_ms,
        "token_per_write_ms": token_per_write_ms,
        "encoded_to_token_ratio": speedup,
        "pass_criterion_per_write_ms": 10.0,
        "passed": token_per_write_ms < 10.0,
    }

    out_path = Path("docs/perf/mem-write-tokens.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")

    print()
    print(f"  encoded path:   {encoded_per_write_ms:.2f} ms / write")
    print(f"  token path:     {token_per_write_ms:.2f} ms / write")
    print(f"  ratio (enc/tok): {speedup:.2f}×")
    print(f"  pass (token < 10 ms / write): {'YES' if result['passed'] else 'NO'}")
    print(f"  wrote: {out_path}")

    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
