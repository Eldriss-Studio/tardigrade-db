"""ATDD tests for chunked benchmark adapter ingestion."""

import tempfile

import numpy as np
import pytest

import tardigrade_db

from tardigrade_hooks.chunker import TextChunker


class _WordTokenizer:
    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


LONG_CONTEXT = " ".join(
    f"Sentence {i} about topic {i % 5} with specific details number {i}."
    for i in range(200)
)


class TestChunkedIngestion:
    def test_long_context_produces_multiple_chunks(self):
        tokenizer = _WordTokenizer()
        chunker = TextChunker(tokenizer, max_tokens=128, overlap_tokens=16)
        chunks = chunker.chunk(LONG_CONTEXT)
        assert len(chunks) >= 5, f"Expected >=5 chunks, got {len(chunks)}"

    def test_scores_differentiate_with_distinct_data(self):
        engine = tardigrade_db.Engine(tempfile.mkdtemp(), vamana_threshold=9999)
        engine.set_refinement_mode("centered")

        texts = [
            "Alice moved to Berlin and works at a biotech startup",
            "Bob plays guitar every evening in his apartment in Munich",
            "Claire studies astrophysics at the university in Hamburg",
        ]
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**31))
            key = rng.standard_normal(8).astype(np.float32)
            engine.mem_write_pack(
                1, key, [(0, np.zeros(8, dtype=np.float32))], 80.0, text=text,
            )

        query_key = np.random.default_rng(42).standard_normal(8).astype(np.float32)
        results = engine.mem_read_pack(query_key, 3, 1)
        scores = [r["score"] for r in results]

        if len(scores) >= 2:
            assert not all(
                s == scores[0] for s in scores
            ), "Scores should NOT all be identical"
