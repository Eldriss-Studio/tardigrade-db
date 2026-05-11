"""File ingestion pipeline — document → KV memory cells.

Facade pattern: hides chunker + KV capture + pack storage + edge wiring
behind a single ``ingest()`` method.  Pipeline pattern: chunk → capture
→ store → wire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from .chunker import TextChunker
from .constants import DEFAULT_CHUNK_TOKENS, DEFAULT_FILE_INGEST_SALIENCE, EDGE_SUPPORTS

if TYPE_CHECKING:
    pass


@dataclass
class IngestResult:
    """Outcome of ingesting a document."""

    pack_ids: list[int] = field(default_factory=list)
    chunk_count: int = 0
    edge_count: int = 0
    document_id: str | None = None


class FileIngestor:
    """Ingests text documents as TardigradeDB KV memory packs.

    Each chunk becomes a separate pack.  Consecutive chunks are linked
    via Supports edges so retrieval can walk from one to the next.

    Args:
        engine: A ``tardigrade_db.Engine`` instance.
        tokenizer: Tokenizer with ``.encode()`` / ``.decode()`` methods.
        owner: Agent/user owner ID for the ingested packs.
        chunker: Optional pre-configured ``TextChunker``.
        salience: Importance hint for ingested chunks.
        kv_capture_fn: Callable ``(chunk_text, tokenizer) -> (key, layer_payloads)``
            for computing retrieval key and layer data from a chunk.
            If None, uses a random-key stub (testing only).
    """

    def __init__(
        self,
        engine,
        *,
        tokenizer=None,
        owner: int = 1,
        chunker: TextChunker | None = None,
        salience: float = DEFAULT_FILE_INGEST_SALIENCE,
        kv_capture_fn: Callable | None = None,
    ):
        self._engine = engine
        self._tokenizer = tokenizer
        self._owner = owner
        self._chunker = chunker or TextChunker(
            tokenizer, max_tokens=DEFAULT_CHUNK_TOKENS,
        )
        self._salience = salience
        self._kv_capture_fn = kv_capture_fn or self._random_kv_stub

    def ingest(
        self,
        text: str,
        document_id: str | None = None,
    ) -> IngestResult:
        """Chunk text, capture KV per chunk, store packs, wire edges."""
        if not text or not text.strip():
            return IngestResult(document_id=document_id)

        chunks = self._chunker.chunk(text)
        if not chunks:
            return IngestResult(document_id=document_id)

        pack_ids: list[int] = []
        for chunk in chunks:
            key, layer_payloads = self._kv_capture_fn(chunk.text, self._tokenizer)
            pid = self._engine.mem_write_pack(
                self._owner, key, layer_payloads, self._salience, text=chunk.text,
            )
            pack_ids.append(pid)

        edge_count = self._wire_sequential_edges(pack_ids)

        return IngestResult(
            pack_ids=pack_ids,
            chunk_count=len(chunks),
            edge_count=edge_count,
            document_id=document_id,
        )

    def ingest_file(
        self,
        path: str | Path,
        document_id: str | None = None,
    ) -> IngestResult:
        """Read a file and ingest its contents."""
        text = Path(path).read_text(encoding="utf-8")
        doc_id = document_id or Path(path).name
        return self.ingest(text, document_id=doc_id)

    def _wire_sequential_edges(self, pack_ids: list[int]) -> int:
        """Add Supports edges between consecutive chunks."""
        count = 0
        for i in range(len(pack_ids) - 1):
            self._engine.add_pack_edge(pack_ids[i], pack_ids[i + 1], EDGE_SUPPORTS)
            count += 1
        return count

    @staticmethod
    def _random_kv_stub(chunk_text: str, _tokenizer) -> tuple:
        """Fallback for testing: random retrieval key, no real KV capture."""
        rng = np.random.default_rng(abs(hash(chunk_text)) % (2**31))
        key = rng.standard_normal(8).astype(np.float32)
        value = rng.standard_normal(8).astype(np.float32)
        return key, [(0, value)]
