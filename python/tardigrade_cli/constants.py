"""CLI-wide named constants. No magic values in subcommand code."""

from __future__ import annotations

# --- Templates ---

# Default template chosen when ``tardigrade init`` is invoked without
# an explicit ``--template`` flag. Mirrors SpacetimeDB's pattern of
# picking a sensible default (python-basic) instead of forcing the
# user to choose on their first run.
DEFAULT_TEMPLATE = "python-basic"

# Templates shipped inside the CLI package. Each value is the
# directory name under ``tardigrade_cli/templates/``.
TEMPLATES: dict[str, str] = {
    "python-basic": "python_basic",
    "rust-basic": "rust_basic",
}

# --- Scaffold layout ---

# Name of the engine-state subdirectory inside a scaffolded project.
# Mirrors the starter files; consumers can change this in their own
# code but the default keeps documentation simple.
ENGINE_DIR_NAME = "engine_dir"

# --- Query defaults ---

# Default top-k for ``tardigrade query`` when ``--top-k`` is omitted.
# Picked to match the default top_k used by TardigradeClient.query.
DEFAULT_QUERY_TOP_K = 5

# --- Owner defaults ---

# Default owner id when ``--owner`` is omitted. Mirrors
# TardigradeClient's default constructor parameter so the CLI's
# behaviour matches the library.
DEFAULT_OWNER = 1
