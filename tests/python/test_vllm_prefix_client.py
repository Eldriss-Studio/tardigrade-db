# ATDD acceptance tests for VLLMMemoryClient — governed memory prefix
# injection for vLLM serving.
#
# Design pattern: Facade (VLLMMemoryClient wraps MemoryPrefixBuilder +
# prompt formatting for vLLM's generate API and OpenAI chat API).
#
# CPU-only, no vLLM or GPU needed — tests the prompt composition logic.

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db
from tardigrade_vllm.prefix_client import VLLMMemoryClient
from tardigrade_hooks.prefix_format import TierAnnotatedFormat

KEY_DIM = 16


@pytest.fixture
def engine(tmp_path):
    return tardigrade_db.Engine(str(tmp_path))


def _store(engine, text, owner=1, salience=80.0):
    key = np.random.randn(KEY_DIM).astype(np.float32)
    payload = np.zeros(KEY_DIM, dtype=np.float32)
    return engine.mem_write_pack(owner, key, [(0, payload)], salience, text)


# -- 1: empty engine produces bare prompt ------------------------------------


def test_empty_engine_returns_bare_prompt(engine):
    client = VLLMMemoryClient(engine, owner=1)
    prompt = client.prepare_prompt("What is the vault code?")
    assert prompt == "What is the vault code?"


# -- 2: prefix prepended to prompt -------------------------------------------


def test_prefix_prepended_to_prompt(engine):
    _store(engine, "The vault code is 9-Quornth-44")
    client = VLLMMemoryClient(engine, owner=1)
    prompt = client.prepare_prompt("What is the vault code?")
    assert prompt.startswith("Memory context:")
    assert "9-Quornth-44" in prompt
    assert prompt.endswith("What is the vault code?")


# -- 3: separator between prefix and prompt ----------------------------------


def test_custom_separator(engine):
    _store(engine, "some fact")
    client = VLLMMemoryClient(engine, owner=1, separator="\n---\n")
    prompt = client.prepare_prompt("query")
    assert "\n---\n" in prompt


# -- 4: prepare_messages inserts system message --------------------------------


def test_prepare_messages_inserts_system(engine):
    _store(engine, "The capital of Vrenthar is Zyphlox-9")
    client = VLLMMemoryClient(engine, owner=1)
    messages = [{"role": "user", "content": "What is the capital?"}]
    result = client.prepare_messages(messages)
    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert "Zyphlox-9" in result[0]["content"]
    assert result[1]["role"] == "user"


# -- 5: prepare_messages merges with existing system message -------------------


def test_prepare_messages_merges_system(engine):
    _store(engine, "some memory")
    client = VLLMMemoryClient(engine, owner=1)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]
    result = client.prepare_messages(messages)
    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert "some memory" in result[0]["content"]
    assert "You are a helpful assistant." in result[0]["content"]


# -- 6: empty prefix doesn't modify messages ----------------------------------


def test_empty_prefix_passthrough_messages(engine):
    client = VLLMMemoryClient(engine, owner=1)
    messages = [{"role": "user", "content": "Hello"}]
    result = client.prepare_messages(messages)
    assert result == messages


# -- 7: version tracks prefix changes ----------------------------------------


def test_version_changes_on_new_memory(engine):
    _store(engine, "fact one")
    client = VLLMMemoryClient(engine, owner=1)
    v1 = client.version
    _store(engine, "fact two")
    assert client.has_changed(v1)


# -- 8: format strategy propagates -------------------------------------------


def test_format_strategy_propagates(engine):
    _store(engine, "annotated fact")
    client = VLLMMemoryClient(engine, owner=1, format=TierAnnotatedFormat())
    prompt = client.prepare_prompt("query")
    assert "[Core]" in prompt


# -- 9: owner isolation -------------------------------------------------------


def test_owner_isolation(engine):
    _store(engine, "owner 1 memory", owner=1)
    _store(engine, "owner 2 memory", owner=2)
    client1 = VLLMMemoryClient(engine, owner=1)
    client2 = VLLMMemoryClient(engine, owner=2)
    p1 = client1.prepare_prompt("query")
    p2 = client2.prepare_prompt("query")
    assert "owner 1 memory" in p1
    assert "owner 2 memory" not in p1
    assert "owner 2 memory" in p2
    assert "owner 1 memory" not in p2


# -- 10: token budget limits prefix ------------------------------------------


def test_token_budget_limits_prefix(engine):
    for i in range(20):
        _store(engine, f"memory number {i} with enough words to consume tokens")
    client_unlimited = VLLMMemoryClient(engine, owner=1)
    client_limited = VLLMMemoryClient(engine, owner=1, token_budget=50)
    p_all = client_unlimited.prepare_prompt("query")
    p_limited = client_limited.prepare_prompt("query")
    assert len(p_limited) < len(p_all)


# -- 11: draft memories excluded from prefix ----------------------------------


def test_draft_excluded_from_prefix(engine):
    _store(engine, "draft memory", salience=20.0)
    _store(engine, "core memory", salience=80.0)
    client = VLLMMemoryClient(engine, owner=1)
    prompt = client.prepare_prompt("query")
    assert "core memory" in prompt
    assert "draft memory" not in prompt


# -- 12: prefix_pack_ids accessible ------------------------------------------


def test_prefix_pack_ids(engine):
    pid = _store(engine, "tracked memory")
    client = VLLMMemoryClient(engine, owner=1)
    client.build_prefix()
    assert pid in client.prefix_pack_ids


# -- 13: prepare_messages doesn't mutate input --------------------------------


def test_prepare_messages_no_mutation(engine):
    _store(engine, "some memory")
    client = VLLMMemoryClient(engine, owner=1)
    original = [{"role": "user", "content": "Hello"}]
    original_copy = [dict(m) for m in original]
    client.prepare_messages(original)
    assert original == original_copy
