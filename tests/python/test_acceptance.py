"""ATDD acceptance tests for TardigradeDB Python bindings."""

import numpy as np
import pytest
import tardigrade_db


@pytest.fixture
def engine(tmp_path):
    """Create a fresh engine in a temporary directory."""
    return tardigrade_db.Engine(str(tmp_path))


def test_python_write_read_round_trip(engine):
    """ATDD Test 1: Write a cell from Python with numpy arrays, read it back."""
    key = np.sin(np.arange(64, dtype=np.float32) * 0.1)
    value = np.cos(np.arange(64, dtype=np.float32) * 0.2)

    cell_id = engine.mem_write(42, 12, key, value, 50.0)
    assert cell_id == 0

    results = engine.mem_read(key, 1, None)
    assert len(results) == 1

    r = results[0]
    assert r.cell_id == cell_id
    assert r.owner == 42
    assert r.layer == 12
    assert len(r.key()) == 64
    assert len(r.value()) == 64


def test_mem_read_topk_from_python(engine):
    """ATDD Test 2: Write 50 cells, query, assert correct cell in top-5."""
    dim = 32

    for i in range(50):
        key = np.full(dim, 0.01, dtype=np.float32)
        key[i % dim] = 1.0
        value = np.zeros(dim, dtype=np.float32)
        engine.mem_write(1, 0, key, value, 50.0)

    # Query for cell #10's pattern.
    query = np.full(dim, 0.01, dtype=np.float32)
    query[10] = 1.0

    results = engine.mem_read(query, 5, None)
    assert len(results) == 5
    ids = [r.cell_id for r in results]
    assert 10 in ids, f"Cell #10 not in top-5. Got: {ids}"


def test_owner_filtering_from_python(engine):
    """ATDD Test 3: Owner filtering works from Python."""
    dim = 16

    # Owner 1 cells.
    for i in range(10):
        key = np.full(dim, float(i), dtype=np.float32)
        engine.mem_write(1, 0, key, np.zeros(dim, dtype=np.float32), 50.0)

    # Owner 2 cells.
    for i in range(10, 20):
        key = np.full(dim, float(i), dtype=np.float32)
        engine.mem_write(2, 0, key, np.zeros(dim, dtype=np.float32), 50.0)

    query = np.full(dim, 5.0, dtype=np.float32)
    results = engine.mem_read(query, 5, 1)  # owner=1

    for r in results:
        assert r.owner == 1, f"Expected owner 1, got {r.owner}"


def test_governance_from_python(engine):
    """ATDD Test 4: Tier promotion via reads observed from Python."""
    dim = 32
    key = np.ones(dim, dtype=np.float32)
    cell_id = engine.mem_write(1, 0, key, np.zeros(dim, dtype=np.float32), 50.0)

    # Initial: 50 + 5 (write) = 55 → Draft.
    assert engine.cell_tier(cell_id) == 0  # Draft

    # 4 reads: 55 + 12 = 67 → Validated.
    for _ in range(4):
        engine.mem_read(key, 1, None)

    assert engine.cell_importance(cell_id) >= 65.0
    assert engine.cell_tier(cell_id) == 1  # Validated


def test_decay_from_python(engine):
    """ATDD Test 5: Decay and demotion observable from Python."""
    dim = 32
    key = np.ones(dim, dtype=np.float32)
    cell_id = engine.mem_write(1, 0, key, np.zeros(dim, dtype=np.float32), 90.0)

    # Initial: 90 + 5 = 95 → Core.
    assert engine.cell_tier(cell_id) == 2  # Core

    # 100 days of decay: 95 × 0.995^100 ≈ 57.5 → Validated.
    engine.advance_days(100.0)
    importance = engine.cell_importance(cell_id)
    assert importance < 60.0, f"Importance {importance:.1f} should be <60"
    assert engine.cell_tier(cell_id) == 1  # Validated
