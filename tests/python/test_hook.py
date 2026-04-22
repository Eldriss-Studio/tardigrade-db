"""ATDD acceptance tests for TardigradeDB LLM inference hooks."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add the Python package to the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db
from tardigrade_hooks.hf_hook import HuggingFaceHook
from tardigrade_hooks.hook import MemoryCellHandle, TardigradeHook, WriteDecision


@pytest.fixture
def engine(tmp_path):
    """Create a fresh engine in a temporary directory."""
    return tardigrade_db.Engine(str(tmp_path))


# ── ATDD Test 1: ABC enforcement ──────────────────────────────────────────


def test_hook_abc_enforced():
    """Subclass TardigradeHook without implementing on_generate → TypeError."""

    class IncompleteHook(TardigradeHook):
        def on_prefill(self, layer, query_states):
            return []

        # Missing on_generate

    with pytest.raises(TypeError, match="abstract method"):
        IncompleteHook()


# ── ATDD Test 2: WriteDecision controls write ─────────────────────────────


def test_write_decision_controls_write(engine):
    """Mock hook returns WriteDecision with salience=80. Engine receives it."""
    dim = 32
    hidden = np.ones((1, 10, dim), dtype=np.float32) * 2.0  # high norm

    hook = HuggingFaceHook(engine, owner=1, norm_threshold=0.1)
    decision = hook.on_generate(layer=0, hidden_states=hidden)

    assert decision.should_write is True
    assert 0.0 <= decision.salience <= 100.0
    assert decision.key is not None
    assert len(decision.key) == dim

    # Apply the decision to the engine.
    if decision.should_write:
        cell_id = engine.mem_write(
            1, 0, decision.key, decision.value, decision.salience, None
        )
        importance = engine.cell_importance(cell_id)
        assert importance > 0.0


# ── ATDD Test 3: Prefill returns handles ──────────────────────────────────


def test_prefill_returns_handles(engine):
    """Write 5 cells, on_prefill returns MemoryCellHandle list."""
    dim = 32

    for i in range(5):
        key = np.zeros(dim, dtype=np.float32)
        key[i % dim] = 1.0
        engine.mem_write(1, 0, key, np.zeros(dim, dtype=np.float32), 50.0, None)

    hook = HuggingFaceHook(engine, owner=1, k=3)
    query = np.ones((1, 4, dim), dtype=np.float32)  # (batch, seq, hidden)
    handles = hook.on_prefill(layer=0, query_states=query)

    assert len(handles) > 0
    assert len(handles) <= 3

    for h in handles:
        assert isinstance(h, MemoryCellHandle)
        assert isinstance(h.cell_id, int)
        assert isinstance(h.key, np.ndarray)
        assert h.key.dtype == np.float32
        assert len(h.key) == dim


# ── ATDD Test 4: HF hook salience heuristic ──────────────────────────────


def test_hf_hook_salience_heuristic(engine):
    """High-norm hidden states → should_write=True, salience ∈ [0,100]."""
    dim = 64
    hook = HuggingFaceHook(engine, owner=1, norm_threshold=0.5)

    # High norm input.
    high_norm = np.random.randn(10, dim).astype(np.float32) * 5.0
    decision_high = hook.on_generate(layer=0, hidden_states=high_norm)
    assert decision_high.should_write is True
    assert 0.0 <= decision_high.salience <= 100.0

    # Low norm input (below threshold).
    low_norm = np.random.randn(10, dim).astype(np.float32) * 0.01
    decision_low = hook.on_generate(layer=0, hidden_states=low_norm)
    assert decision_low.should_write is False


# ── ATDD Test 5: MemoryCellHandle access ─────────────────────────────────


def test_memory_cell_handle_access():
    """MemoryCellHandle.key returns numpy array of correct length."""
    dim = 16
    handle = MemoryCellHandle(
        cell_id=42,
        owner=1,
        layer=12,
        score=0.95,
        key=np.ones(dim, dtype=np.float32),
        value=np.zeros(dim, dtype=np.float32),
    )

    assert handle.cell_id == 42
    assert handle.owner == 1
    assert handle.layer == 12
    assert handle.score == pytest.approx(0.95)
    assert isinstance(handle.key, np.ndarray)
    assert len(handle.key) == dim
    assert handle.key.dtype == np.float32
