"""Acceptance tests for TardigradeClient — high-level API facade.

Tests verify the wiring between engine, ingestor, consolidator, and
query path through a single object. Uses stub KV capture (no real LLM).
"""

import numpy as np
import pytest

import tardigrade_db

from tardigrade_hooks.client import TardigradeClient
from tardigrade_hooks.constants import DEFAULT_VIEW_FRAMINGS

DIM = 8
OWNER = 1

FACT = "Sonia translated a pharmaceutical patent from German to English in March 2024."
DOC = "Alice moved to Berlin in 2023. She works at a biotech startup. Her team builds diagnostic tools for rare diseases."


class _WordTokenizer:
    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


def _stub_kv(chunk_text, _tokenizer):
    rng = np.random.default_rng(abs(hash(chunk_text)) % (2**31))
    key = rng.standard_normal(DIM).astype(np.float32)
    value = rng.standard_normal(DIM).astype(np.float32)
    return key, [(0, value)]


@pytest.fixture
def client(tmp_path):
    return TardigradeClient(
        db_path=str(tmp_path),
        tokenizer=_WordTokenizer(),
        owner=OWNER,
        kv_capture_fn=_stub_kv,
    )


class TestClientStore:
    def test_store_returns_pack_id(self, client):
        pid = client.store(FACT)
        assert isinstance(pid, int)
        assert pid > 0

    def test_stored_text_retrievable(self, client):
        pid = client.store(FACT)
        text = client.engine.pack_text(pid)
        assert FACT in text


class TestClientIngest:
    def test_ingest_text_returns_result(self, client):
        result = client.ingest_text(DOC)
        assert result.chunk_count >= 1
        assert len(result.pack_ids) == result.chunk_count

    def test_ingest_wires_edges(self, client):
        result = client.ingest_text(DOC, chunk_size=20)
        if result.chunk_count > 1:
            assert result.edge_count == result.chunk_count - 1


class TestClientConsolidate:
    def test_consolidate_creates_views(self, client):
        pid = client.store(FACT, salience=70.0)
        views = client.consolidate(pid)
        assert len(views) == len(DEFAULT_VIEW_FRAMINGS)

    def test_consolidate_all_processes_stored(self, client):
        client.store("Fact about Alice.", salience=70.0)
        client.store("Fact about Bob.", salience=70.0)
        result = client.consolidate_all()
        assert len(result) == 2


class TestClientQuery:
    def test_query_returns_list(self, client):
        client.store(FACT)
        results = client.query("pharmaceutical patent", k=3)
        assert isinstance(results, list)

    def test_query_returns_results_after_store(self, client):
        client.store(FACT)
        results = client.query("translation", k=3)
        assert len(results) >= 1


class TestClientLifecycle:
    def test_list_packs(self, client):
        client.store("Fact one.")
        client.store("Fact two.")
        packs = client.list_packs()
        assert len(packs) == 2

    def test_pack_count(self, client):
        assert client.pack_count() == 0
        client.store("Something.")
        assert client.pack_count() == 1
