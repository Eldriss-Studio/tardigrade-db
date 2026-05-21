"""Acceptance tests for `Engine.list_packs_metadata` and `Engine.list_packs`.

Two enumeration APIs cover the metadata-vs-text axis:

* `Engine.list_packs_metadata(owner)` returns a dict of four parallel numpy
  arrays — `pack_ids`, `owners`, `tiers`, `importances` — built in one
  Rust→Python crossing. No per-row Python dict, no text fetch.
* `Engine.list_packs(owner, fetch_text=True)` returns the list-of-dicts
  shape with `text` included, for callers that need text inline.
  `fetch_text` is a required keyword argument.
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


def _write_pack(engine, owner, salience, text=None, marker=0.0):
    rng = np.random.default_rng(int(marker * 1000) + 1)
    key = rng.standard_normal(DIM).astype(np.float32)
    value = rng.standard_normal(DIM).astype(np.float32)
    return engine.mem_write_pack(owner, key, [(0, value)], salience, text=text)


# ---- empty owner -----------------------------------------------------------

def test_list_packs_metadata_returns_empty_arrays_when_owner_has_no_packs(tmp_path):
    engine = _engine(tmp_path)
    meta = engine.list_packs_metadata(99)
    assert set(meta.keys()) == {"pack_ids", "owners", "tiers", "importances"}
    for col in meta.values():
        assert col.size == 0


# ---- columnar shape, no text field ----------------------------------------

def test_list_packs_metadata_returns_four_aligned_numpy_arrays(tmp_path):
    engine = _engine(tmp_path)
    _write_pack(engine, OWNER, 70.0, text="alpha", marker=0.1)
    _write_pack(engine, OWNER, 70.0, text=None, marker=0.2)

    meta = engine.list_packs_metadata(OWNER)

    assert set(meta.keys()) == {"pack_ids", "owners", "tiers", "importances"}, (
        f"metadata dict has unexpected keys: {set(meta.keys())}"
    )
    assert "text" not in meta

    assert meta["pack_ids"].dtype == np.uint64
    assert meta["owners"].dtype == np.uint64
    assert meta["tiers"].dtype == np.uint8
    assert meta["importances"].dtype == np.float32

    n = meta["pack_ids"].size
    assert n == 2
    for col in meta.values():
        assert col.size == n, f"column size mismatch: {col.size} != {n}"


# ---- ordering invariant ----------------------------------------------------

def test_list_packs_metadata_sorted_by_importance_descending(tmp_path):
    """Preserves the existing `list_packs` contract: importance descending.

    Callers like `prefix_builder.py` rely on this ordering implicitly to
    surface high-importance packs first.
    """
    engine = _engine(tmp_path)
    _write_pack(engine, OWNER, 40.0, marker=0.1)
    _write_pack(engine, OWNER, 90.0, marker=0.2)
    _write_pack(engine, OWNER, 60.0, marker=0.3)
    _write_pack(engine, OWNER, 75.0, marker=0.4)

    importances = engine.list_packs_metadata(OWNER)["importances"]
    assert importances.size == 4
    assert np.all(importances[:-1] >= importances[1:]), (
        f"importance not sorted descending: {importances.tolist()}"
    )


# ---- legacy path still returns text ---------------------------------------

def test_list_packs_with_fetch_text_true_returns_stored_text_or_none(tmp_path):
    engine = _engine(tmp_path)
    pid_with = _write_pack(engine, OWNER, 70.0, text="alpha", marker=0.1)
    pid_without = _write_pack(engine, OWNER, 70.0, text=None, marker=0.2)

    rows = engine.list_packs(OWNER, fetch_text=True)
    by_id = {r["pack_id"]: r for r in rows}

    assert by_id[pid_with]["text"] == "alpha"
    assert by_id[pid_without]["text"] is None


# ---- fetch_text is a required keyword argument ----------------------------

def test_list_packs_requires_fetch_text_kwarg(tmp_path):
    """`list_packs` rejects callers that omit `fetch_text=` — the kwarg is
    required so the caller's intent (pay the per-pack text lookup or not)
    is explicit at every call site."""
    engine = _engine(tmp_path)
    _write_pack(engine, OWNER, 70.0, text="alpha", marker=0.1)

    with pytest.raises(TypeError):
        engine.list_packs(OWNER)  # type: ignore[call-arg]


# ---- consolidate latency at 10K (slow) ------------------------------------

@pytest.mark.slow
def test_consolidate_one_pack_completes_under_50ms_at_10k_packs(tmp_path):
    """Behavioural latency assertion on `MemoryConsolidator.consolidate`.

    At 10K packs `_pack_info` walks the columnar metadata once with
    `np.where`. The 50ms budget reflects an enumeration that answers from
    `PackDirectory`'s in-memory indices, not from per-pack pool reads.

    Setup writes 10K packs (≈80s on SSD) so this only runs under `-m slow`.
    """
    import time

    from tardigrade_hooks.consolidator import MemoryConsolidator

    n_packs = 10_000
    engine = _engine(tmp_path)
    target_pid = _write_pack(engine, OWNER, 70.0, text="canonical fact to consolidate", marker=0.001)
    for i in range(1, n_packs):
        _write_pack(engine, OWNER, 70.0, text=f"filler pack {i}", marker=0.001 + i * 0.0001)

    consolidator = MemoryConsolidator(engine, owner=OWNER)
    start = time.perf_counter()
    consolidator.consolidate(target_pid)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Also time the bare metadata call to compare with what we measured in
    # the microbench — should reveal whether consolidate is amortising it
    # somehow or whether the microbench overcounted.
    meta_runs = []
    for _ in range(5):
        ms_start = time.perf_counter()
        engine.list_packs_metadata(OWNER)
        meta_runs.append((time.perf_counter() - ms_start) * 1000)
    meta_runs.sort()
    meta_median_ms = meta_runs[len(meta_runs) // 2]

    print(
        f"\n  consolidate({target_pid}) @ {n_packs} packs: {elapsed_ms:.2f} ms"
        f"\n  list_packs_metadata median (5 runs):       {meta_median_ms:.2f} ms",
        flush=True,
    )

    assert elapsed_ms < 50.0, (
        f"consolidate({target_pid}) at {n_packs} packs took {elapsed_ms:.1f}ms; "
        f"expected < 50ms with the columnar metadata path"
    )


# ---- concurrent read/write safety -----------------------------------------

def test_list_packs_metadata_consistent_under_concurrent_writes(tmp_path):
    """Metadata enumeration must not crash or yield torn columns while a
    concurrent writer is appending. Engine's `Arc<Mutex<>>` serialises access;
    every list call sees either pre-write or post-write state.
    """
    engine = _engine(tmp_path)
    for i in range(20):
        _write_pack(engine, OWNER, 70.0, text=f"seed {i}", marker=0.001 + i * 0.0001)

    errors = []
    stop = threading.Event()

    def writer():
        i = 100
        while not stop.is_set():
            try:
                _write_pack(engine, OWNER, 70.0, text=f"hot {i}", marker=0.1 + i * 0.0001)
                i += 1
            except Exception as exc:
                errors.append(("writer", exc))
                return

    def reader():
        try:
            for _ in range(50):
                meta = engine.list_packs_metadata(OWNER)
                assert set(meta.keys()) == {"pack_ids", "owners", "tiers", "importances"}
                n = meta["pack_ids"].size
                for col in meta.values():
                    assert col.size == n
        except Exception as exc:
            errors.append(("reader", exc))

    w = threading.Thread(target=writer)
    readers = [threading.Thread(target=reader) for _ in range(4)]
    w.start()
    for r in readers:
        r.start()
    for r in readers:
        r.join()
    stop.set()
    w.join()

    assert errors == [], f"concurrent access produced errors: {errors}"
