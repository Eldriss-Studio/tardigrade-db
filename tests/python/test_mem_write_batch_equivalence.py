"""AT-B2 — `mem_write_batch` produces engine state equivalent to a
serial `mem_write` loop.

Behavioral test (Kent Dodds): two engine instances ingest the *same*
synthetic cells, one via serial `mem_write` per chunk, the other via
a single `mem_write_batch` call. Both engines are then queried with
the same probe; the top-K results must match.

Closes the test gap left when the bench adapter was migrated to
`mem_write_batch` in the batched-write path without a focused equivalence test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db  # noqa: E402
from tardigrade_hooks.encoding import encode_per_token  # noqa: E402

# ---------- fixture constants ----------

FIXTURE_DIM = 32
TOKENS_PER_CELL = 4
CELL_COUNT = 50
QUERY_TOP_K = 5
OWNER_ID = 1
LAYER_ID = 0
SALIENCE = 1.0
NO_PARENT = None
DUMMY_VALUE_LEN = 16
SIGNAL_MAGNITUDE = 0.95
NOISE_HALF_RANGE = 0.02
SEED = 17


def _cell_signal_dim(cell_id: int) -> int:
    return cell_id % FIXTURE_DIM


def _cell_tokens(rng: np.random.Generator, cell_id: int) -> np.ndarray:
    tokens = rng.uniform(-NOISE_HALF_RANGE, NOISE_HALF_RANGE,
                         size=(TOKENS_PER_CELL, FIXTURE_DIM)).astype(np.float32)
    tokens[:, _cell_signal_dim(cell_id)] = SIGNAL_MAGNITUDE
    return tokens


def _encoded_key_for(rng: np.random.Generator, cell_id: int) -> np.ndarray:
    tokens = _cell_tokens(rng, cell_id)
    return encode_per_token(tokens, FIXTURE_DIM).astype(np.float32)


def _ingest_serial(engine, payloads):
    for key, value in payloads:
        engine.mem_write(OWNER_ID, LAYER_ID, key, value, SALIENCE, NO_PARENT)


def _ingest_batched(engine, payloads):
    requests = [
        (OWNER_ID, LAYER_ID, key, value, SALIENCE, NO_PARENT)
        for key, value in payloads
    ]
    engine.mem_write_batch(requests)


@pytest.fixture
def payloads():
    rng = np.random.default_rng(SEED)
    return [
        (
            _encoded_key_for(rng, cell_id),
            np.zeros(DUMMY_VALUE_LEN, dtype=np.float32),
        )
        for cell_id in range(CELL_COUNT)
    ]


def _probe_top_k(engine, probe_cell_id):
    rng = np.random.default_rng(SEED + probe_cell_id)
    probe_tokens = _cell_tokens(rng, probe_cell_id)
    return engine.mem_read_tokens(probe_tokens, QUERY_TOP_K, OWNER_ID)


def test_serial_and_batched_writes_produce_equivalent_engine_state(payloads, tmp_path):
    serial_dir = tmp_path / "serial"
    batched_dir = tmp_path / "batched"
    serial_engine = tardigrade_db.Engine(str(serial_dir))
    batched_engine = tardigrade_db.Engine(str(batched_dir))

    _ingest_serial(serial_engine, payloads)
    _ingest_batched(batched_engine, payloads)

    assert serial_engine.cell_count() == batched_engine.cell_count() == CELL_COUNT

    # Probe both engines with the same set of queries — top-K results
    # must match, modulo INT8 round-trip noise.
    for probe_cell_id in (0, CELL_COUNT // 2, CELL_COUNT - 1):
        serial_results = _probe_top_k(serial_engine, probe_cell_id)
        batched_results = _probe_top_k(batched_engine, probe_cell_id)

        serial_ids = [r.cell_id for r in serial_results]
        batched_ids = [r.cell_id for r in batched_results]
        overlap = len(set(serial_ids) & set(batched_ids))
        assert overlap >= QUERY_TOP_K - 1, (
            f"top-{QUERY_TOP_K} for probe {probe_cell_id}: serial={serial_ids} "
            f"batched={batched_ids}; overlap {overlap} < {QUERY_TOP_K - 1}"
        )
