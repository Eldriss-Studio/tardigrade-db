"""ATDD tests for semantic edge types (Supports/Contradicts)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db

EDGE_FOLLOWS = 1
EDGE_CONTRADICTS = 2
EDGE_SUPPORTS = 3

CHAT_TEMPLATE = '{% for message in messages %}{{ message["content"] }}{% endfor %}'


@pytest.fixture
def engine(tmp_path):
    return tardigrade_db.Engine(str(tmp_path))


def _store_pack(engine, owner, seed):
    import numpy as np
    from tardigrade_hooks.encoding import encode_per_token
    key = encode_per_token(
        np.array([[seed, 0.0, 0.0, 0.0]], dtype=np.float32), dim=4
    )
    return engine.mem_write_pack(owner, key, [(0, np.array([seed] * 16, dtype=np.float32))], 80.0)


def test_add_pack_edge_round_trip(engine):
    """GIVEN two packs,
    WHEN linked via Supports edge,
    THEN pack_supports returns the linked pack."""
    pack_a = _store_pack(engine, 1, 1.0)
    pack_b = _store_pack(engine, 1, 2.0)

    engine.add_pack_edge(pack_a, pack_b, EDGE_SUPPORTS)

    assert pack_b in engine.pack_supports(pack_a)
    assert pack_a in engine.pack_supports(pack_b)


def test_pack_supports_and_contradicts(engine):
    """GIVEN a pack with both Supports and Contradicts edges,
    WHEN querying each type,
    THEN the correct packs are returned separately."""
    pack_a = _store_pack(engine, 1, 1.0)
    pack_b = _store_pack(engine, 1, 2.0)
    pack_c = _store_pack(engine, 1, 3.0)

    engine.add_pack_edge(pack_a, pack_b, EDGE_SUPPORTS)
    engine.add_pack_edge(pack_a, pack_c, EDGE_CONTRADICTS)

    assert pack_b in engine.pack_supports(pack_a)
    assert pack_c not in engine.pack_supports(pack_a)

    assert pack_c in engine.pack_contradicts(pack_a)
    assert pack_b not in engine.pack_contradicts(pack_a)

    assert len(engine.pack_links(pack_a)) == 2


def test_store_supporting_convenience(tmp_path):
    """GIVEN a KnowledgePackStore,
    WHEN store_supporting is called,
    THEN the new pack is linked via Supports edge."""
    pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tardigrade_hooks.kp_injector import KnowledgePackStore

    model = AutoModelForCausalLM.from_pretrained("gpt2", output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.pad_token = tokenizer.eos_token

    engine = tardigrade_db.Engine(str(tmp_path))
    kps = KnowledgePackStore(engine, model, tokenizer, owner=1)

    pack_a = kps.store("Nyx lives in a lighthouse")
    pack_b = kps.store_supporting("The lighthouse is on the eastern cliffs", pack_a)

    assert pack_a in engine.pack_supports(pack_b)
    assert pack_b in engine.pack_supports(pack_a)


def test_store_contradicting_convenience(tmp_path):
    """GIVEN a KnowledgePackStore,
    WHEN store_contradicting is called,
    THEN the new pack is linked via Contradicts edge."""
    pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tardigrade_hooks.kp_injector import KnowledgePackStore

    model = AutoModelForCausalLM.from_pretrained("gpt2", output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.pad_token = tokenizer.eos_token

    engine = tardigrade_db.Engine(str(tmp_path))
    kps = KnowledgePackStore(engine, model, tokenizer, owner=1)

    pack_a = kps.store("Nyx's favorite color is blue")
    pack_b = kps.store_contradicting("Nyx's favorite color is actually green", pack_a)

    assert pack_a in engine.pack_contradicts(pack_b)
    assert pack_b in engine.pack_contradicts(pack_a)
