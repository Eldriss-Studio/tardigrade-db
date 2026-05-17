"""Fluent builder for :class:`TardigradeClient` (M3.4).

The positional :class:`TardigradeClient` constructor stays for the
simple case; the builder kicks in when a consumer needs more than
two non-default options. Inspired by SpacetimeDB's
``DbConnectionBuilder``.

Usage::

    client = (
        TardigradeClient.builder()
        .with_engine_dir("./engine")
        .with_owner(npc_id)
        .with_tokenizer(tokenizer)
        .with_capture_fn(qwen_capture)
        .build()
    )

Missing the required ``engine_dir`` field raises
:class:`BuilderIncomplete` with the field name in the message —
fail-fast at build time, not at first method call.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .client import TardigradeClient


DEFAULT_OWNER: int = 1
DEFAULT_VAMANA_THRESHOLD: int = 9999


class BuilderIncomplete(ValueError):
    """Raised when :meth:`TardigradeClientBuilder.build` is called
    before all required fields have been set. The message names
    every missing field so the consumer can fix all of them in one
    pass instead of one-error-per-build."""


class TardigradeClientBuilder:
    """Fluent builder for :class:`TardigradeClient`.

    Mutates internal state and returns ``self`` from each setter
    so calls chain. Build is the only validating step — setters
    accept whatever you pass and complaints come from
    :meth:`build`.
    """

    def __init__(self) -> None:
        self._engine_dir: Path | None = None
        self._engine = None
        self._tokenizer = None
        self._owner: int = DEFAULT_OWNER
        self._kv_capture_fn: Callable | None = None
        self._vamana_threshold: int = DEFAULT_VAMANA_THRESHOLD

    # -- Setters -------------------------------------------------------------

    def with_engine_dir(self, engine_dir: str | Path) -> TardigradeClientBuilder:
        """Open a fresh engine at ``engine_dir``. Mutually exclusive
        with :meth:`with_engine`."""
        self._engine_dir = Path(engine_dir)
        return self

    def with_engine(self, engine) -> TardigradeClientBuilder:
        """Reuse an existing engine.

        Critical for multi-agent setups where N clients (one per
        owner) must share one underlying engine — opening the same
        ``engine_dir`` from N separate ``Engine`` instances yields
        N isolated states, not one shared one. Mutually exclusive
        with :meth:`with_engine_dir`.
        """
        self._engine = engine
        return self

    def with_tokenizer(self, tokenizer) -> TardigradeClientBuilder:
        """Tokenizer with ``.encode()`` / ``.decode()``."""
        self._tokenizer = tokenizer
        return self

    def with_owner(self, owner: int) -> TardigradeClientBuilder:
        """Owner ID for memories. Defaults to ``DEFAULT_OWNER``."""
        self._owner = int(owner)
        return self

    def with_capture_fn(self, kv_capture_fn: Callable) -> TardigradeClientBuilder:
        """``(chunk_text, tokenizer) -> (key, layer_payloads)``.

        When omitted, the resulting client falls back to its
        deterministic random-key stub — fine for tests, useless in
        production.
        """
        self._kv_capture_fn = kv_capture_fn
        return self

    def with_vamana_threshold(self, threshold: int) -> TardigradeClientBuilder:
        """Cell count at which Vamana ANN index activates."""
        self._vamana_threshold = int(threshold)
        return self

    # -- Build ---------------------------------------------------------------

    def build(self) -> TardigradeClient:
        """Instantiate the client.

        Raises :class:`BuilderIncomplete` if neither
        ``engine_dir`` nor ``engine`` is set, or if both are set
        (they're mutually exclusive — exactly one source for the
        engine).
        """
        has_dir = self._engine_dir is not None
        has_engine = self._engine is not None
        if not has_dir and not has_engine:
            raise BuilderIncomplete(
                "missing required builder field(s): engine_dir or engine "
                "— call .with_engine_dir(path) or .with_engine(eng) before .build()",
            )
        if has_dir and has_engine:
            raise BuilderIncomplete(
                "engine_dir and engine are mutually exclusive — set exactly "
                "one (call .with_engine_dir or .with_engine, not both)",
            )

        from .client import TardigradeClient
        if has_engine:
            return TardigradeClient(
                engine=self._engine,
                tokenizer=self._tokenizer,
                owner=self._owner,
                kv_capture_fn=self._kv_capture_fn,
            )
        return TardigradeClient(
            self._engine_dir,
            tokenizer=self._tokenizer,
            owner=self._owner,
            kv_capture_fn=self._kv_capture_fn,
            vamana_threshold=self._vamana_threshold,
        )
