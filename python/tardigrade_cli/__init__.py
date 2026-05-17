"""TardigradeDB command-line interface.

Façade over :class:`tardigrade_hooks.TardigradeClient` exposing
task-oriented subcommands: ``init``, ``store``, ``query``,
``status``, ``consolidate``.

The first-touch experience for new consumers. Inspired by
SpacetimeDB's ``spacetime`` CLI but scoped to TardigradeDB's
actual surface area (no cloud, no codegen).
"""

from tardigrade_cli.main import main

__all__ = ["main"]
