"""ATDD tests for multi-agent isolation (Fixture-Based Organization)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db
from tardigrade_hooks.encoding import encode_per_token

AGENT_ALPHA = 100
AGENT_BETA = 200
AGENT_GAMMA = 300
PACKS_PER_AGENT = 5
PACK_SALIENCE = 80.0
LOW_SALIENCE = 10.0
EVICTION_THRESHOLD = 30.0


def _store_pack(engine, owner, seed):
    key = encode_per_token(
        np.array([[seed, 0.0, 0.0, 0.0]], dtype=np.float32), dim=4
    )
    return engine.mem_write_pack(
        owner, key, [(0, np.array([seed] * 16, dtype=np.float32))], PACK_SALIENCE
    )


def _multi_agent_fixture(tmp_path):
    engine = tardigrade_db.Engine(str(tmp_path))
    alpha_ids, beta_ids, gamma_ids = [], [], []
    for i in range(PACKS_PER_AGENT):
        alpha_ids.append(_store_pack(engine, AGENT_ALPHA, float(i + 1)))
        beta_ids.append(_store_pack(engine, AGENT_BETA, float(i + 10)))
        gamma_ids.append(_store_pack(engine, AGENT_GAMMA, float(i + 20)))
    return engine, alpha_ids, beta_ids, gamma_ids


def test_multi_agent_pack_isolation(tmp_path):
    """GIVEN 3 agents x 5 packs,
    WHEN listing packs by owner,
    THEN each agent sees only their own packs."""
    engine, _, _, _ = _multi_agent_fixture(tmp_path)

    alpha_packs = engine.list_packs(AGENT_ALPHA)
    beta_packs = engine.list_packs(AGENT_BETA)
    all_packs = engine.list_packs()

    assert len(alpha_packs) == PACKS_PER_AGENT
    assert len(beta_packs) == PACKS_PER_AGENT
    assert len(all_packs) == PACKS_PER_AGENT * 3
    assert all(p["owner"] == AGENT_ALPHA for p in alpha_packs)


def test_multi_agent_eviction_scoped(tmp_path):
    """GIVEN ALPHA has a low-salience Draft, BETA has a low-salience Draft,
    WHEN evicting ALPHA's Drafts,
    THEN only ALPHA's low-salience Draft is removed."""
    engine = tardigrade_db.Engine(str(tmp_path))

    key = encode_per_token(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), dim=4)
    payload = [(0, np.array([1.0] * 16, dtype=np.float32))]

    alpha_low = engine.mem_write_pack(AGENT_ALPHA, key, payload, LOW_SALIENCE)
    _alpha_high = engine.mem_write_pack(AGENT_ALPHA, key, payload, PACK_SALIENCE)
    beta_low = engine.mem_write_pack(AGENT_BETA, key, payload, LOW_SALIENCE)

    evicted = engine.evict_draft_packs(EVICTION_THRESHOLD, AGENT_ALPHA)

    assert evicted == 1
    assert not engine.pack_exists(alpha_low)
    assert engine.pack_exists(beta_low)


def test_multi_agent_delete_isolation(tmp_path):
    """GIVEN 3 agents x 5 packs,
    WHEN ALPHA deletes one pack,
    THEN BETA still has all 5."""
    engine, alpha_ids, beta_ids, _ = _multi_agent_fixture(tmp_path)

    engine.delete_pack(alpha_ids[0])

    assert len(engine.list_packs(AGENT_ALPHA)) == PACKS_PER_AGENT - 1
    assert len(engine.list_packs(AGENT_BETA)) == PACKS_PER_AGENT


def test_multi_agent_trace_isolation(tmp_path):
    """GIVEN ALPHA links packs A0-A1, BETA links packs B0-B1,
    WHEN querying ALPHA's links,
    THEN no BETA packs appear."""
    engine, alpha_ids, beta_ids, _ = _multi_agent_fixture(tmp_path)

    engine.add_pack_link(alpha_ids[0], alpha_ids[1])
    engine.add_pack_link(beta_ids[0], beta_ids[1])

    alpha_links = engine.pack_links(alpha_ids[0])
    assert alpha_ids[1] in alpha_links
    assert not any(bid in alpha_links for bid in beta_ids)


def test_multi_agent_retrieval_isolation(tmp_path):
    """GIVEN 3 agents x 5 packs,
    WHEN querying with owner=ALPHA and k=10,
    THEN only ALPHA's packs are returned."""
    engine, _, _, _ = _multi_agent_fixture(tmp_path)

    key = encode_per_token(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), dim=4)
    results = engine.mem_read_pack(key, 10, AGENT_ALPHA)

    assert len(results) <= PACKS_PER_AGENT
    assert all(r["owner"] == AGENT_ALPHA for r in results)
