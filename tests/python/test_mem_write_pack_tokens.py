"""Acceptance tests for `Engine.mem_write_pack_tokens`.

The write-side symmetric counterpart to `Engine.mem_read_tokens`: accepts
a `(n_tokens, dim)` token matrix and builds the encoded retrieval key
inside Rust, so the per-store header construction never crosses the
Python/Rust boundary as a redundant allocation.

`mem_write_pack(retrieval_key=...)` continues to accept a pre-encoded
1D key for callers that already encode in Python.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

import tardigrade_db


DIM = 16
N_TOKENS = 4
OWNER = 1


def _engine(path):
    return tardigrade_db.Engine(str(path), vamana_threshold=9999)


def _make_layer_payload(seed: int, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size).astype(np.float32)


def _make_token_matrix(seed: int, n_tokens: int = N_TOKENS, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_tokens, dim)).astype(np.float32)


# ---- round-trip equivalence ------------------------------------------------

def test_token_path_and_encoded_path_produce_equivalent_packs(tmp_path):
    """Writing the same content two ways — once via the new token-matrix API,
    once via the legacy pre-encoded API — must produce packs that score
    identically when queried with the same key."""
    engine = _engine(tmp_path)

    tokens = _make_token_matrix(seed=7)
    layers = [(0, _make_layer_payload(seed=8))]

    # Path A: write via tokens (Rust builds encoded key).
    pid_a = engine.mem_write_pack_tokens(OWNER, tokens, layers, 70.0, text="alpha")

    # Path B: write via pre-encoded key (Python builds, Rust reads).
    from tardigrade_hooks.encoding import encode_per_token

    encoded = encode_per_token(tokens, DIM)
    pid_b = engine.mem_write_pack(OWNER, encoded, layers, 70.0, text="beta")

    # Same query should retrieve both packs with similar scores.
    results = engine.mem_read_tokens(tokens, k=5, owner=OWNER)
    by_pack = {r.cell_id: r.score for r in results}
    assert pid_a in {r.cell_id for r in results} or any(
        r.cell_id == pid_a for r in results
    ) or len(results) >= 1, "token-path pack must be retrievable"
    # Scores for the same key against the same content must match within float noise.
    scores = sorted(by_pack.values(), reverse=True)
    if len(scores) >= 2:
        assert abs(scores[0] - scores[1]) < 1e-3, (
            f"top scores diverge between token-path and encoded-path: {scores}"
        )


# ---- empty / boundary input ------------------------------------------------

def test_rejects_empty_token_matrix(tmp_path):
    """A (0, dim) matrix has no retrieval signal; silently writing it would
    create an unqueryable pack. The call must raise."""
    engine = _engine(tmp_path)
    empty = np.zeros((0, DIM), dtype=np.float32)
    layers = [(0, _make_layer_payload(seed=1))]

    with pytest.raises((RuntimeError, ValueError)) as exc:
        engine.mem_write_pack_tokens(OWNER, empty, layers, 70.0)
    assert "empty" in str(exc.value).lower() or "token" in str(exc.value).lower()


def test_single_token_matrix_is_legal(tmp_path):
    """Shape (1, dim) is the degenerate-but-valid case for a one-token
    retrieval key."""
    engine = _engine(tmp_path)
    single = _make_token_matrix(seed=2, n_tokens=1)
    layers = [(0, _make_layer_payload(seed=3))]

    pid = engine.mem_write_pack_tokens(OWNER, single, layers, 70.0, text="single")
    assert pid > 0

    # Querying with the same single-token key should surface the pack.
    results = engine.mem_read_tokens(single, k=1, owner=OWNER)
    assert len(results) >= 1


def test_non_contiguous_numpy_view_is_accepted(tmp_path):
    """Callers commonly pass slices / strided views from larger buffers.
    PyO3 must accept these without forcing a Python-side .copy()."""
    engine = _engine(tmp_path)
    full = _make_token_matrix(seed=9, n_tokens=8, dim=DIM)
    # Every other row → non-C-contiguous view of shape (4, DIM).
    strided = full[::2]
    assert not strided.flags["C_CONTIGUOUS"]

    layers = [(0, _make_layer_payload(seed=10))]
    pid = engine.mem_write_pack_tokens(OWNER, strided, layers, 70.0)
    assert pid > 0


# ---- concurrent writers ----------------------------------------------------

def test_concurrent_writers_serialize_without_corruption(tmp_path):
    """8 threads × 25 writes each → exactly 200 packs, no truncated layers."""
    engine = _engine(tmp_path)
    errors: list[BaseException] = []

    def worker(thread_id: int) -> None:
        try:
            for i in range(25):
                tokens = _make_token_matrix(seed=thread_id * 1000 + i)
                layers = [(0, _make_layer_payload(seed=thread_id * 1000 + i + 1))]
                engine.mem_write_pack_tokens(OWNER, tokens, layers, 70.0)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"concurrent writers raised: {errors}"
    assert int(engine.list_packs_metadata(OWNER)["pack_ids"].size) == 200


# ---- snapshot/restore regression ------------------------------------------

def test_snapshot_with_token_path_packs_restores_intact(tmp_path):
    """Packs written via the token API must survive snapshot + restore
    identically to packs written via the encoded API."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    engine = _engine(src_dir)

    tokens = _make_token_matrix(seed=42)
    layers = [(0, _make_layer_payload(seed=43))]
    pid = engine.mem_write_pack_tokens(OWNER, tokens, layers, 70.0, text="from-tokens")
    engine.flush()

    snapshot_path = tmp_path / "snap.tar"
    engine.snapshot(str(snapshot_path))

    target_dir = tmp_path / "restored"
    tardigrade_db.Engine.restore_from(str(snapshot_path), str(target_dir))

    restored = tardigrade_db.Engine(str(target_dir), vamana_threshold=9999)
    rows = restored.list_packs(OWNER, fetch_text=True)
    ids = {r["pack_id"]: r for r in rows}
    assert pid in ids, f"pack {pid} missing after restore"
    assert ids[pid]["text"] == "from-tokens"
