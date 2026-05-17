"""TardigradeClient — high-level facade for the TardigradeDB Python API.

Combines Engine + tokenizer + KV capture into a single object with
store / query / ingest / consolidate methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

import tardigrade_db

from .chunker import TextChunker
from .consolidator import MemoryConsolidator
from .constants import DEFAULT_CHUNK_TOKENS, DEFAULT_FILE_INGEST_SALIENCE, EDGE_SUPPORTS
from .file_ingestor import FileIngestor, IngestResult
from .view_generator import ViewGenerator

if TYPE_CHECKING:
    pass


class TardigradeClient:
    """Single entry point for TardigradeDB memory operations.

    Facade pattern: wraps Engine, FileIngestor, MemoryConsolidator,
    and query path behind a unified API.

    Args:
        db_path: Directory for the engine's persistent storage.
        tokenizer: Tokenizer with ``.encode()`` / ``.decode()``.
        owner: Agent/user owner ID.
        kv_capture_fn: ``(chunk_text, tokenizer) -> (key, layer_payloads)``
            for computing retrieval key from text.  If None, uses a
            random-key stub (testing only).
        vamana_threshold: Engine's Vamana activation threshold.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        tokenizer=None,
        owner: int = 1,
        kv_capture_fn: Callable | None = None,
        vamana_threshold: int = 9999,
    ):
        self._db_path = str(db_path)
        self._engine = tardigrade_db.Engine(self._db_path, vamana_threshold=vamana_threshold)
        self._tokenizer = tokenizer
        self._owner = owner
        self._kv_fn = kv_capture_fn or self._random_kv_stub
        self._view_gen = ViewGenerator()
        self._consolidator = MemoryConsolidator(
            self._engine, owner=owner, view_generator=self._view_gen,
        )

    @classmethod
    def builder(cls):
        """Return a fluent :class:`TardigradeClientBuilder`.

        Use the builder when more than two non-default options are
        in play — fields self-document via ``.with_<name>(value)``
        instead of positional-arg cargo culting. See M3.4 in the
        foundation plan and ``tests/python/test_client_builder.py``
        for usage examples.
        """
        from .builder import TardigradeClientBuilder
        return TardigradeClientBuilder()

    @property
    def engine(self):
        """Direct access to the underlying ``tardigrade_db.Engine``."""
        return self._engine

    # -- Store ---------------------------------------------------------------

    def store(self, fact_text: str, *, salience: float = 80.0) -> int:
        """Store a single fact as a KV memory pack.  Returns pack_id."""
        key, layer_payloads = self._kv_fn(fact_text, self._tokenizer)
        return self._engine.mem_write_pack(
            self._owner, key, layer_payloads, salience, text=fact_text,
        )

    # -- Ingest --------------------------------------------------------------

    def ingest_text(
        self,
        text: str,
        *,
        document_id: str | None = None,
        chunk_size: int = DEFAULT_CHUNK_TOKENS,
    ) -> IngestResult:
        """Chunk and ingest a text document.  Returns ``IngestResult``."""
        chunker = TextChunker(self._tokenizer, max_tokens=chunk_size)
        ingestor = FileIngestor(
            self._engine,
            tokenizer=self._tokenizer,
            owner=self._owner,
            chunker=chunker,
            kv_capture_fn=self._kv_fn,
        )
        return ingestor.ingest(text, document_id=document_id)

    def ingest_file(
        self,
        path: str | Path,
        *,
        document_id: str | None = None,
        chunk_size: int = DEFAULT_CHUNK_TOKENS,
    ) -> IngestResult:
        """Read and ingest a file.  Returns ``IngestResult``."""
        text = Path(path).read_text(encoding="utf-8")
        doc_id = document_id or Path(path).name
        return self.ingest_text(text, document_id=doc_id, chunk_size=chunk_size)

    # -- Consolidate ---------------------------------------------------------

    def consolidate(self, pack_id: int) -> int:
        """Attach multi-view retrieval keys to a canonical memory. Returns views attached."""
        return self._consolidator.consolidate(pack_id)

    def consolidate_all(self) -> dict[int, int]:
        """Consolidate all eligible packs. Returns {pack_id: views_attached}."""
        return self._consolidator.consolidate_all(owner=self._owner)

    # -- Query ---------------------------------------------------------------

    def encode_query(self, text: str) -> np.ndarray:
        """Compute the per-token retrieval key for *text*.

        Returns just the query key (no layer payloads) — useful
        when a consumer wants to cache the key, log it, or feed it
        directly into :py:meth:`Engine.mem_read_pack` /
        :py:meth:`Engine.mem_read_tokens` without going through
        :meth:`query`.
        """
        key, _ = self._kv_fn(text, self._tokenizer)
        return key

    def query(self, query_text: str, *, k: int = 5) -> list[dict]:
        """Retrieve the top-k packs matching *query_text*."""
        key = self.encode_query(query_text)
        return self._engine.mem_read_pack(key, k, self._owner)

    # -- Lifecycle -----------------------------------------------------------

    def list_packs(self) -> list[dict]:
        """List all packs for this owner."""
        return self._engine.list_packs(self._owner)

    def pack_count(self) -> int:
        """Number of packs for this owner."""
        return len(self._engine.list_packs(self._owner))

    # -- Internals -----------------------------------------------------------

    @staticmethod
    def _random_kv_stub(text: str, _tokenizer) -> tuple:
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        key = rng.standard_normal(8).astype(np.float32)
        value = rng.standard_normal(8).astype(np.float32)
        return key, [(0, value)]
