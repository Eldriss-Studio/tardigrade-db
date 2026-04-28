"""ATDD tests for SynapticBank Python bindings (Repository + Facade)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db

OWNER_A = 42
OWNER_B = 99
OWNER_NONEXISTENT = 999
RANK_SMALL = 2
D_MODEL_SMALL = 4
RANK_LARGE = 16
D_MODEL_LARGE = 512
SCALE = 0.1
QUALITY = 0.85
LAST_USED = 1234567890


@pytest.fixture
def engine(tmp_path):
    return tardigrade_db.Engine(str(tmp_path))


def test_store_and_load_round_trip(engine):
    """GIVEN a fresh engine,
    WHEN a synaptic entry is stored and loaded,
    THEN all fields round-trip correctly."""
    lora_a = np.ones(RANK_SMALL * D_MODEL_SMALL, dtype=np.float32)
    lora_b = np.full(RANK_SMALL * D_MODEL_SMALL, 0.5, dtype=np.float32)

    engine.store_synapsis(
        id=0, owner=OWNER_A, lora_a=lora_a, lora_b=lora_b,
        scale=SCALE, rank=RANK_SMALL, d_model=D_MODEL_SMALL,
    )

    entries = engine.load_synapsis(OWNER_A)
    assert len(entries) == 1

    entry = entries[0]
    assert entry["id"] == 0
    assert entry["owner"] == OWNER_A
    assert entry["rank"] == RANK_SMALL
    assert entry["d_model"] == D_MODEL_SMALL
    np.testing.assert_allclose(entry["lora_a"], lora_a, atol=0.01)
    np.testing.assert_allclose(entry["lora_b"], lora_b, atol=0.01)
    assert abs(entry["scale"] - SCALE) < 0.01


def test_owner_isolation(engine):
    """GIVEN entries for owners A, B, and A again,
    WHEN loading by owner,
    THEN only that owner's entries are returned."""
    arr = np.ones(RANK_SMALL * D_MODEL_SMALL, dtype=np.float32)

    engine.store_synapsis(id=0, owner=1, lora_a=arr, lora_b=arr, scale=0.1, rank=RANK_SMALL, d_model=D_MODEL_SMALL)
    engine.store_synapsis(id=1, owner=2, lora_a=arr, lora_b=arr, scale=0.1, rank=RANK_SMALL, d_model=D_MODEL_SMALL)
    engine.store_synapsis(id=2, owner=1, lora_a=arr, lora_b=arr, scale=0.1, rank=RANK_SMALL, d_model=D_MODEL_SMALL)

    assert len(engine.load_synapsis(1)) == 2
    assert len(engine.load_synapsis(2)) == 1
    assert len(engine.load_synapsis(OWNER_NONEXISTENT)) == 0


def test_persists_across_reopen(tmp_path):
    """GIVEN a stored entry,
    WHEN the engine is reopened,
    THEN the entry is still loadable."""
    arr = np.ones(RANK_SMALL * D_MODEL_SMALL, dtype=np.float32)
    engine = tardigrade_db.Engine(str(tmp_path))
    engine.store_synapsis(id=0, owner=OWNER_A, lora_a=arr, lora_b=arr, scale=0.1, rank=RANK_SMALL, d_model=D_MODEL_SMALL)
    del engine

    engine2 = tardigrade_db.Engine(str(tmp_path))
    entries = engine2.load_synapsis(OWNER_A)
    assert len(entries) == 1
    assert entries[0]["owner"] == OWNER_A


def test_dimension_validation(engine):
    """GIVEN mismatched lora_a dimensions,
    WHEN storing,
    THEN raises an error."""
    wrong_size = np.ones(5, dtype=np.float32)
    correct_size = np.ones(RANK_SMALL * D_MODEL_SMALL, dtype=np.float32)

    with pytest.raises(Exception):
        engine.store_synapsis(
            id=0, owner=OWNER_A, lora_a=wrong_size, lora_b=correct_size,
            scale=0.1, rank=RANK_SMALL, d_model=D_MODEL_SMALL,
        )


def test_quality_and_last_used_fields(engine):
    """GIVEN an entry with quality and last_used set,
    WHEN loaded back,
    THEN those fields match."""
    arr = np.ones(RANK_SMALL * D_MODEL_SMALL, dtype=np.float32)

    engine.store_synapsis(
        id=0, owner=OWNER_A, lora_a=arr, lora_b=arr,
        scale=0.1, rank=RANK_SMALL, d_model=D_MODEL_SMALL,
        quality=QUALITY, last_used=LAST_USED,
    )

    entry = engine.load_synapsis(OWNER_A)[0]
    assert abs(entry["quality"] - QUALITY) < 0.01
    assert entry["last_used"] == LAST_USED


def test_large_matrices(engine):
    """GIVEN realistic LoRA dimensions (rank=16, d_model=512),
    WHEN stored and loaded,
    THEN arrays round-trip within f16 precision."""
    size = RANK_LARGE * D_MODEL_LARGE
    lora_a = np.random.randn(size).astype(np.float32) * 0.01
    lora_b = np.random.randn(size).astype(np.float32) * 0.01

    engine.store_synapsis(
        id=0, owner=OWNER_A, lora_a=lora_a, lora_b=lora_b,
        scale=0.001, rank=RANK_LARGE, d_model=D_MODEL_LARGE,
    )

    entry = engine.load_synapsis(OWNER_A)[0]
    assert len(entry["lora_a"]) == size
    assert len(entry["lora_b"]) == size
    np.testing.assert_allclose(entry["lora_a"], lora_a, atol=0.001)
