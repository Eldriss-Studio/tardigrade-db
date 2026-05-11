"""Acceptance tests for FileIngestor — file → KV memory pipeline.

Uses a minimal synthetic setup (random retrieval keys, no real LLM)
to verify the chunking → storage → edge-wiring pipeline topology.
"""

import numpy as np
import pytest

import tardigrade_db

from tardigrade_hooks.constants import DEFAULT_FILE_INGEST_SALIENCE, EDGE_SUPPORTS
from tardigrade_hooks.chunker import TextChunker
from tardigrade_hooks.file_ingestor import FileIngestor, IngestResult

DIM = 8
OWNER = 1

SHORT_DOC = "Alice moved to Berlin. She works at a biotech startup. Her team builds diagnostic tools."

LONG_DOC = " ".join(f"Sentence number {i} contains some words about topic {i % 5}." for i in range(80))


class _WordTokenizer:
    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


def _stub_kv_fn(chunk_text, _tokenizer):
    """Returns a random retrieval key and dummy layer payload."""
    rng = np.random.default_rng(abs(hash(chunk_text)) % (2**31))
    key = rng.standard_normal(DIM).astype(np.float32)
    value = rng.standard_normal(DIM).astype(np.float32)
    return key, [(0, value)]


@pytest.fixture
def env(tmp_path):
    engine = tardigrade_db.Engine(str(tmp_path), vamana_threshold=9999)
    tokenizer = _WordTokenizer()
    ingestor = FileIngestor(
        engine, tokenizer=tokenizer, owner=OWNER, kv_capture_fn=_stub_kv_fn,
    )
    return engine, ingestor


class TestIngestContract:
    def test_returns_ingest_result(self, env):
        engine, ingestor = env
        result = ingestor.ingest(SHORT_DOC)
        assert isinstance(result, IngestResult)

    def test_pack_ids_nonempty(self, env):
        engine, ingestor = env
        result = ingestor.ingest(SHORT_DOC)
        assert len(result.pack_ids) > 0

    def test_chunk_count_matches_pack_ids(self, env):
        engine, ingestor = env
        result = ingestor.ingest(SHORT_DOC)
        assert result.chunk_count == len(result.pack_ids)

    def test_empty_text_returns_empty(self, env):
        engine, ingestor = env
        result = ingestor.ingest("")
        assert result.pack_ids == []
        assert result.chunk_count == 0

    def test_document_id_stored(self, env):
        engine, ingestor = env
        result = ingestor.ingest(SHORT_DOC, document_id="doc-001")
        assert result.document_id == "doc-001"


class TestEdgeWiring:
    def test_consecutive_chunks_linked_via_supports(self, env):
        engine, ingestor = env
        result = ingestor.ingest(LONG_DOC)
        assert len(result.pack_ids) >= 2
        for i in range(len(result.pack_ids) - 1):
            supporters = engine.pack_supports(result.pack_ids[i])
            assert result.pack_ids[i + 1] in supporters, (
                f"Pack {result.pack_ids[i]} not linked to next {result.pack_ids[i+1]}"
            )

    def test_edge_count_matches_expected(self, env):
        engine, ingestor = env
        result = ingestor.ingest(LONG_DOC)
        expected_edges = max(0, result.chunk_count - 1)
        assert result.edge_count == expected_edges


class TestChunkText:
    def test_each_pack_has_text(self, env):
        engine, ingestor = env
        result = ingestor.ingest(SHORT_DOC)
        for pid in result.pack_ids:
            text = engine.pack_text(pid)
            assert text is not None and len(text.strip()) > 0

    def test_pack_text_covers_document(self, env):
        engine, ingestor = env
        result = ingestor.ingest(SHORT_DOC)
        all_text = " ".join(engine.pack_text(pid) for pid in result.pack_ids)
        for word in ("Alice", "Berlin", "biotech", "diagnostic"):
            assert word in all_text


class TestLongDocument:
    def test_long_doc_produces_multiple_chunks(self, tmp_path):
        engine = tardigrade_db.Engine(str(tmp_path), vamana_threshold=9999)
        tokenizer = _WordTokenizer()
        ingestor = FileIngestor(
            engine, tokenizer=tokenizer, owner=OWNER,
            kv_capture_fn=_stub_kv_fn,
            chunker=TextChunker(tokenizer, max_tokens=50, overlap_tokens=0),
        )
        result = ingestor.ingest(LONG_DOC)
        assert result.chunk_count >= 5

    def test_all_chunks_indexed(self, env):
        engine, ingestor = env
        result = ingestor.ingest(LONG_DOC)
        packs = engine.list_packs(OWNER)
        pack_ids_in_engine = {p["pack_id"] for p in packs}
        for pid in result.pack_ids:
            assert pid in pack_ids_in_engine


class TestSalience:
    def test_default_salience_used(self, env):
        engine, ingestor = env
        result = ingestor.ingest(SHORT_DOC)
        packs = engine.list_packs(OWNER)
        for p in packs:
            if p["pack_id"] in result.pack_ids:
                assert p["importance"] >= DEFAULT_FILE_INGEST_SALIENCE - 1.0
