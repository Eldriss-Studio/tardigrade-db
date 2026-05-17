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
        self._tokenizer = None
        self._owner: int = DEFAULT_OWNER
        self._kv_capture_fn: Callable | None = None
        self._vamana_threshold: int = DEFAULT_VAMANA_THRESHOLD

    # -- Setters -------------------------------------------------------------

    def with_engine_dir(self, engine_dir: str | Path) -> TardigradeClientBuilder:
        """Required. Directory for the engine's persistent storage."""
        self._engine_dir = Path(engine_dir)
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

        Raises :class:`BuilderIncomplete` if any required field is
        unset.
        """
        missing: list[str] = []
        if self._engine_dir is None:
            missing.append("engine_dir")
        if missing:
            raise BuilderIncomplete(
                f"missing required builder field(s): {', '.join(missing)} "
                "— call the matching .with_<field>() setter before .build()",
            )

        from .client import TardigradeClient
        return TardigradeClient(
            self._engine_dir,
            tokenizer=self._tokenizer,
            owner=self._owner,
            kv_capture_fn=self._kv_capture_fn,
            vamana_threshold=self._vamana_threshold,
        )
