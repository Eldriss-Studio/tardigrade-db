"""Acceptance tests for `Engine.mem_write_batch_packs`.

The eager batched write API: caller hands the engine N packs at once and
the engine persists them all under one fsync, returning pack ids aligned
to the input order. Distinct from `open_with_write_buffer`, which is the
streaming opportunistic batching path.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

import tardigrade_db


DIM = 8
OWNER = 1


def _engine(path):
    return tardigrade_db.Engine(str(path), vamana_threshold=9999)


def _pack(seed: int, text: str | None = None, salience: float = 70.0):
    rng = np.random.default_rng(seed)
    key = rng.standard_normal(DIM).astype(np.float32)
    layer = rng.standard_normal(64).astype(np.float32)
    return (OWNER, key, [(0, layer)], salience, text)


def test_returns_pack_ids_in_input_order(tmp_path):
    engine = _engine(tmp_path)
    batch = [_pack(i, text=f"pack {i}") for i in range(8)]
    ids = engine.mem_write_batch_packs(batch)
    assert len(ids) == 8
    assert ids == sorted(ids), f"ids not in increasing order: {ids}"
    assert len(set(ids)) == 8, f"ids not unique: {ids}"


def test_empty_batch_returns_empty_list(tmp_path):
    engine = _engine(tmp_path)
    assert engine.mem_write_batch_packs([]) == []
    meta = engine.list_packs_metadata(OWNER)
    assert meta["pack_ids"].size == 0


def test_batch_packs_visible_immediately_without_flush(tmp_path):
    engine = _engine(tmp_path)
    ids = engine.mem_write_batch_packs([_pack(i, text=f"p{i}") for i in range(5)])
    meta = engine.list_packs_metadata(OWNER)
    assert sorted(meta["pack_ids"].tolist()) == sorted(ids)


def test_batch_text_persisted_for_each_pack(tmp_path):
    engine = _engine(tmp_path)
    ids = engine.mem_write_batch_packs(
        [_pack(0, text="alpha"), _pack(1, text=None), _pack(2, text="gamma")]
    )
    rows = engine.list_packs(OWNER, fetch_text=True)
    by_id = {r["pack_id"]: r["text"] for r in rows}
    assert by_id[ids[0]] == "alpha"
    assert by_id[ids[1]] is None
    assert by_id[ids[2]] == "gamma"


def test_concurrent_batch_writes_do_not_corrupt(tmp_path):
    """8 threads × 5 batches of 10 packs each → 400 packs, no torn ids."""
    engine = _engine(tmp_path)
    errors: list[BaseException] = []

    def worker(thread_id: int) -> None:
        try:
            for batch_no in range(5):
                batch = [
                    _pack(thread_id * 1000 + batch_no * 10 + i, text=f"t{thread_id} b{batch_no} p{i}")
                    for i in range(10)
                ]
                ids = engine.mem_write_batch_packs(batch)
                assert len(ids) == 10
                assert len(set(ids)) == 10
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"concurrent batch writes raised: {errors}"
    meta = engine.list_packs_metadata(OWNER)
    assert int(meta["pack_ids"].size) == 400


def test_batch_writes_persist_across_engine_reopen(tmp_path):
    """Recovery: a batch written eagerly must survive engine reopen."""
    engine = _engine(tmp_path)
    ids = engine.mem_write_batch_packs([_pack(i, text=f"p{i}") for i in range(6)])
    engine.flush()
    del engine

    engine2 = _engine(tmp_path)
    meta = engine2.list_packs_metadata(OWNER)
    assert sorted(meta["pack_ids"].tolist()) == sorted(ids)
