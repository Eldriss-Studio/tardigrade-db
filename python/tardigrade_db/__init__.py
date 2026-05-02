# TardigradeDB — LLM-native memory kernel.
#
# The native Rust extension is at tardigrade_db.tardigrade_db (PyO3).
# This __init__.py re-exports it so `from tardigrade_db import Engine` works.
#
# Python packages (hooks, vllm, mcp) are shipped alongside as
# tardigrade_hooks, tardigrade_vllm, tardigrade_mcp — they are
# separate top-level packages included in the wheel via maturin.

from tardigrade_db.tardigrade_db import *  # noqa: F401,F403
from tardigrade_db.tardigrade_db import __version__  # noqa: F401
