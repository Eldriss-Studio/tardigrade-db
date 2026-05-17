"""Pydantic request/response models for the HTTP bridge.

Kept in a separate module so the OpenAPI schema is generated from
typed contracts rather than ad-hoc dict shapes. Every payload here
is the public wire contract — breaking changes need a version bump.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .constants import (
    DEFAULT_QUERY_K,
    MAX_QUERY_K,
    MIN_OWNER_ID,
    MIN_QUERY_K,
)


class StoreRequest(BaseModel):
    """Body of ``POST /mem/store``."""

    owner: int = Field(..., ge=MIN_OWNER_ID, description="Owner ID for the memory.")
    fact_text: str = Field(..., min_length=1, description="Fact to remember.")
    salience: float = Field(
        80.0,
        ge=0.0,
        le=100.0,
        description="Initial importance score in [0, 100].",
    )


class StoreResponse(BaseModel):
    pack_id: int


class QueryRequest(BaseModel):
    """Body of ``POST /mem/query``."""

    owner: int = Field(..., ge=MIN_OWNER_ID)
    query_text: str = Field(..., min_length=1)
    k: int = Field(DEFAULT_QUERY_K, ge=MIN_QUERY_K, le=MAX_QUERY_K)


class QueryResult(BaseModel):
    pack_id: int
    score: float
    text: str | None = None


class QueryResponse(BaseModel):
    results: list[QueryResult]


class SaveRequest(BaseModel):
    """Body of ``POST /mem/save``.

    Snapshots are engine-wide — they capture the whole working
    directory, not a single owner. ``snapshot_path`` must be
    outside the engine's directory (the bridge propagates the
    engine's foot-gun guard).
    """

    snapshot_path: str = Field(..., min_length=1)


class Manifest(BaseModel):
    """Mirror of :class:`tdb_engine::snapshot::SnapshotManifest`.

    Field names match the Rust struct so consumers can rely on a
    stable wire contract across languages.
    """

    magic: str
    format_version: int
    created_at: str
    pack_count: int
    owner_count: int
    sha256: str
    quantization_codec: str
    key_codec: str


class SaveResponse(BaseModel):
    manifest: Manifest


class RestoreRequest(BaseModel):
    """Body of ``POST /mem/restore``.

    Restores the snapshot at ``snapshot_path`` into ``target_dir``;
    the bridge does *not* hot-swap its running engine. To use the
    restored engine the consumer should relaunch the bridge with
    ``TARDIGRADE_HTTP_DB=target_dir``.
    """

    snapshot_path: str = Field(..., min_length=1)
    target_dir: str = Field(..., min_length=1)


class RestoreResponse(BaseModel):
    ok: bool
    pack_count: int


class OwnersResponse(BaseModel):
    owners: list[int]


class StatusResponse(BaseModel):
    """Mirror of :meth:`Engine.status` dict."""

    cell_count: int
    pack_count: int
    segment_count: int
    slb_occupancy: int
    slb_capacity: int
    vamana_active: bool
    pipeline_stages: int
    governance_entries: int
    trace_edges: int
    arena_bytes: int
    arena_bytes_per_cell: float


class ProblemDetail(BaseModel):
    """RFC 7807 error envelope (``application/problem+json``)."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
