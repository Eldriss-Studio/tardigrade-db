# ATDD tests for MCP tool functions (Item 4: release-readiness).
#
# Facade Testing pattern: test the MCP tool functions as plain callables
# by pre-setting module-level globals (_kps, _engine) with GPT-2 fixtures.
# No MCP server process needed.

import sys
from pathlib import Path

import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db
from tardigrade_hooks.kp_injector import KnowledgePackStore

CHAT_TEMPLATE = '{% for message in messages %}{{ message["content"] }}{% endfor %}'


@pytest.fixture
def gpt2():
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()
    return model


@pytest.fixture
def tokenizer():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.chat_template = CHAT_TEMPLATE
    tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def mcp_env(tmp_path, gpt2, tokenizer):
    """Pre-set MCP server globals with GPT-2 for direct tool testing."""
    import tardigrade_mcp.server as srv

    engine = tardigrade_db.Engine(str(tmp_path))
    kps = KnowledgePackStore(engine, gpt2, tokenizer, owner=1)
    srv._engine = engine
    srv._kps = kps
    yield srv
    srv._kps = None
    srv._engine = None


# -- Store tests ---------------------------------------------------------------


def test_store_returns_pack_id(mcp_env):
    """GIVEN the MCP server,
    WHEN tardigrade_store(text),
    THEN returns dict with pack_id and status 'stored'."""
    result = mcp_env.tardigrade_store("Nyx's favorite constellation is Cassiopeia")
    assert isinstance(result, dict)
    assert "pack_id" in result
    assert isinstance(result["pack_id"], int)
    assert result["status"] == "stored"


def test_store_and_link_creates_connection(mcp_env):
    """GIVEN an existing memory,
    WHEN tardigrade_store_and_link(text, existing_pack_id),
    THEN returns dict with linked_to and the engine confirms the edge."""
    first = mcp_env.tardigrade_store("Nyx teaches archery on Thursdays")
    second = mcp_env.tardigrade_store_and_link(
        "Her bow is named Stormstring", first["pack_id"]
    )

    assert second["linked_to"] == first["pack_id"]
    assert second["status"] == "stored_and_linked"
    assert first["pack_id"] in mcp_env._engine.pack_links(second["pack_id"])


def test_store_persists_text(mcp_env):
    """GIVEN a stored fact,
    THEN the text appears in the text registry."""
    text = "Nyx brews rosemary tea every morning"
    result = mcp_env.tardigrade_store(text)
    assert mcp_env._kps._text_registry.get(result["pack_id"]) == text


# -- Recall tests --------------------------------------------------------------


def test_recall_finds_stored_fact(mcp_env):
    """GIVEN a stored fact about a character,
    WHEN recalling with a related query,
    THEN returns at least one result with matching text and positive score."""
    mcp_env.tardigrade_store("Nyx's workshop is in a lighthouse on the eastern cliffs")

    results = mcp_env.tardigrade_recall("Where is Nyx's workshop?")
    assert isinstance(results, list)
    assert len(results) >= 1
    assert results[0]["score"] > 0
    assert "lighthouse" in results[0]["text"]


def test_recall_empty_store(mcp_env):
    """GIVEN an empty memory store,
    WHEN recalling,
    THEN returns an empty list."""
    results = mcp_env.tardigrade_recall("What color is the sky?")
    assert results == []


def test_recall_k_parameter(mcp_env):
    """GIVEN 3 stored facts,
    WHEN recalling with k=2,
    THEN returns at most 2 results."""
    mcp_env.tardigrade_store("Nyx collects vintage compasses")
    mcp_env.tardigrade_store("Nyx's cat is named Parallax")
    mcp_env.tardigrade_store("Nyx plays the theremin at sunset")

    results = mcp_env.tardigrade_recall("Tell me about Nyx", k=2)
    assert len(results) <= 2


# -- Trace recall tests --------------------------------------------------------


def test_recall_with_trace_follows_links(mcp_env):
    """GIVEN two linked facts,
    WHEN recall_with_trace queries for one,
    THEN both facts appear in results with linked_packs populated."""
    first = mcp_env.tardigrade_store("Nyx has a mentor named Corvus")
    mcp_env.tardigrade_store_and_link(
        "Corvus lives in the old observatory", first["pack_id"]
    )

    results = mcp_env.tardigrade_recall_with_trace("Who is Nyx's mentor?")
    assert len(results) >= 2

    texts = [r["text"] for r in results]
    assert any("Corvus" in t and "mentor" in t for t in texts)
    assert any("observatory" in t for t in texts)


# -- List tests ----------------------------------------------------------------


def test_list_links_shows_connections(mcp_env):
    """GIVEN two linked packs,
    WHEN list_links on one,
    THEN the other appears."""
    first = mcp_env.tardigrade_store("Nyx found an old map in the attic")
    second = mcp_env.tardigrade_store_and_link(
        "The map leads to a sunken library", first["pack_id"]
    )

    links = mcp_env.tardigrade_list_links(first["pack_id"])
    linked_ids = [entry["pack_id"] for entry in links]
    assert second["pack_id"] in linked_ids


def test_list_all_returns_all_memories(mcp_env):
    """GIVEN 3 stored facts,
    WHEN list_all,
    THEN returns 3 entries each with pack_id, text, and links count."""
    mcp_env.tardigrade_store("Nyx speaks four languages")
    mcp_env.tardigrade_store("Nyx's favorite number is 7")
    mcp_env.tardigrade_store("Nyx was born during an eclipse")

    results = mcp_env.tardigrade_list_all()
    assert len(results) == 3
    for entry in results:
        assert "pack_id" in entry
        assert "text" in entry
        assert "links" in entry


def test_list_all_empty_store(mcp_env):
    """GIVEN an empty store,
    WHEN list_all,
    THEN returns an empty list."""
    assert mcp_env.tardigrade_list_all() == []


# -- Forget tests --------------------------------------------------------------


def test_tardigrade_forget_removes_from_recall(mcp_env):
    """GIVEN a stored fact,
    WHEN tardigrade_forget(pack_id),
    THEN recall returns empty."""
    result = mcp_env.tardigrade_store("Nyx's secret lab is under the clock tower")
    mcp_env.tardigrade_forget(result["pack_id"])

    results = mcp_env.tardigrade_recall("Where is Nyx's secret lab?")
    assert results == []
    assert mcp_env._engine.pack_count() == 0


def test_tardigrade_forget_removes_from_list_all(mcp_env):
    """GIVEN 2 stored facts,
    WHEN forgetting one,
    THEN list_all returns only the other."""
    a = mcp_env.tardigrade_store("Nyx knows calligraphy")
    b = mcp_env.tardigrade_store("Nyx studies astrophysics")

    mcp_env.tardigrade_forget(a["pack_id"])

    results = mcp_env.tardigrade_list_all()
    assert len(results) == 1
    assert results[0]["pack_id"] == b["pack_id"]


# -- Migration tests -----------------------------------------------------------


def test_legacy_text_registry_migrates_to_rust(tmp_path, gpt2, tokenizer):
    """GIVEN a database with text only in the JSON sidecar (legacy state),
    WHEN a new KnowledgePackStore opens it,
    THEN the text appears in the durable Rust text_store."""
    import os

    # Step 1: create a pack the normal way (writes to both JSON and Rust).
    engine = tardigrade_db.Engine(str(tmp_path))
    kps = KnowledgePackStore(engine, gpt2, tokenizer, owner=1)
    kps.store("Legacy memory")
    pack_id = next(iter(kps._text_registry.keys()))

    # Step 2: simulate a pre-Item-3 database — wipe the Rust text store.
    text_store_path = tmp_path / "text_store.bin"
    if text_store_path.exists():
        os.remove(text_store_path)

    # Confirm the pre-migration state: Rust text_store is empty.
    engine2 = tardigrade_db.Engine(str(tmp_path))
    assert engine2.pack_text(pack_id) is None

    # Step 3: opening a KnowledgePackStore migrates legacy entries.
    kps2 = KnowledgePackStore(engine2, gpt2, tokenizer, owner=1)
    assert kps2.engine.pack_text(pack_id) == "Legacy memory"


def test_migration_skips_stale_sidecar_entries(tmp_path, gpt2, tokenizer):
    """GIVEN a JSON sidecar entry for a pack that has been deleted,
    WHEN migration runs,
    THEN the stale entry is silently skipped (no exception)."""
    engine = tardigrade_db.Engine(str(tmp_path))
    kps = KnowledgePackStore(engine, gpt2, tokenizer, owner=1)
    pack_id = kps.store("Will be deleted")
    kps.forget(pack_id)

    # Manually re-add a stale entry to the sidecar (simulates a crash between
    # forget and _save_text_registry).
    kps._text_registry[pack_id] = "Stale text"
    kps._save_text_registry()

    # Re-opening should not raise — migration silently skips deleted packs.
    engine2 = tardigrade_db.Engine(str(tmp_path))
    kps2 = KnowledgePackStore(engine2, gpt2, tokenizer, owner=1)
    assert kps2.engine.pack_text(pack_id) is None


def test_migration_is_idempotent(tmp_path, gpt2, tokenizer):
    """GIVEN an already-migrated database,
    WHEN KnowledgePackStore is re-opened,
    THEN migration runs without errors and leaves data unchanged."""
    engine = tardigrade_db.Engine(str(tmp_path))
    kps = KnowledgePackStore(engine, gpt2, tokenizer, owner=1)
    pack_id = kps.store("Already migrated")

    # Re-open multiple times — should be a no-op.
    for _ in range(3):
        engine_n = tardigrade_db.Engine(str(tmp_path))
        kps_n = KnowledgePackStore(engine_n, gpt2, tokenizer, owner=1)
        assert kps_n.engine.pack_text(pack_id) == "Already migrated"


# -- Direct engine.set_pack_text tests (Gap #2) --------------------------------


def test_engine_set_pack_text_direct(tmp_path, gpt2, tokenizer):
    """GIVEN an existing pack,
    WHEN engine.set_pack_text() is called directly from Python,
    THEN engine.pack_text() reflects the new value (last-writer-wins)."""
    engine = tardigrade_db.Engine(str(tmp_path))
    kps = KnowledgePackStore(engine, gpt2, tokenizer, owner=1)
    pack_id = kps.store("Original")

    engine.set_pack_text(pack_id, "Replaced")
    assert engine.pack_text(pack_id) == "Replaced"


def test_engine_set_pack_text_errors_on_missing_pack(tmp_path):
    """GIVEN no pack with the given ID,
    WHEN engine.set_pack_text() is called,
    THEN it raises (CellNotFound)."""
    engine = tardigrade_db.Engine(str(tmp_path))
    with pytest.raises(Exception):
        engine.set_pack_text(9999, "anything")


# -- Deleted-pack-text invariant after reopen (Gap #3) -------------------------


def test_forgotten_pack_text_gone_after_reopen(tmp_path, gpt2, tokenizer):
    """GIVEN a stored pack that was forgotten,
    WHEN the engine is reopened,
    THEN engine.pack_text() returns None for the deleted pack_id.

    Regression guard: the deletion log must clear text_store entries
    on Engine::open, not just pack_directory."""
    engine = tardigrade_db.Engine(str(tmp_path))
    kps = KnowledgePackStore(engine, gpt2, tokenizer, owner=1)
    pack_id = kps.store("Will be deleted")
    kps.forget(pack_id)

    engine2 = tardigrade_db.Engine(str(tmp_path))
    assert engine2.pack_text(pack_id) is None


# -- Corrupted sidecar handling (Gap #6) ---------------------------------------


def test_corrupted_text_registry_does_not_crash_init(tmp_path, gpt2, tokenizer, capsys):
    """GIVEN a corrupted text_registry.json on disk,
    WHEN KnowledgePackStore opens the database,
    THEN init succeeds, falls back to engine text_store, and logs a warning."""
    engine = tardigrade_db.Engine(str(tmp_path))
    kps = KnowledgePackStore(engine, gpt2, tokenizer, owner=1)
    pack_id = kps.store("Survives via Rust")

    sidecar = tmp_path / "text_registry.json"
    sidecar.write_text("{not valid json")

    engine2 = tardigrade_db.Engine(str(tmp_path))
    kps2 = KnowledgePackStore(engine2, gpt2, tokenizer, owner=1)

    assert kps2._text_registry == {}
    assert kps2.engine.pack_text(pack_id) == "Survives via Rust"

    captured = capsys.readouterr()
    assert "corrupted text_registry.json" in captured.err
