"""Constants for the tardigrade-chat product.

Per the project's no-magic-values rule, every literal that the chat product depends on lives here.
"""

from __future__ import annotations

from pathlib import Path

# -- Engine + persona defaults -----------------------------------------------

DEFAULT_CHAT_STATE_DIR: Path = Path.home() / ".tardigrade" / "chat"
DEFAULT_PERSONA: str = "default"
ENGINE_SUBDIR: str = "engine"
PERSONA_REGISTRY_FILE: str = "personas.json"
LAST_PERSONA_FILE: str = "last_persona.txt"
PERSONA_OWNER_BASE: int = 1000  # owner ids count up from here so they don't collide with manual owner ids
DEFAULT_RECALL_K: int = 5
DEFAULT_INITIAL_SALIENCE: float = 80.0

# -- Latency / stats ---------------------------------------------------------

LATENCY_P50_PERCENTILE: int = 50
LATENCY_P95_PERCENTILE: int = 95
LATENCY_P99_PERCENTILE: int = 99
MIN_SAMPLES_FOR_PERCENTILES: int = 1

# -- Real-model defaults -----------------------------------------------------

DEFAULT_QWEN_MODEL: str = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_NEW_TOKENS: int = 120
DEFAULT_TEMPERATURE: float = 0.0

# -- REPL ux -----------------------------------------------------------------

REPL_PROMPT: str = "you> "
REPL_AGENT_PREFIX: str = "agent> "
REPL_LATENCY_TAG: str = "recalled {n} memories in {ms:.2f} ms"
SLASH_COMMAND_PREFIX: str = "/"
