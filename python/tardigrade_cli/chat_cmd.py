"""``tardigrade chat`` — interactive memory-aware chat REPL.

Backend choice:

- ``--backend persistent`` (default): real :class:`tardigrade_db.Engine` on disk, deterministic echo responses. Demonstrates persistence + multi-persona + sub-ms recall without paying for a model. Useful for the user-facing "I can pick this up and try it" experience and for verifying the engine works correctly outside the test suite.
- ``--backend qwen``: real Qwen3-0.6B for both KV capture and response generation. Heavy (model download ~600 MB on first run). Lazy-imports torch + transformers only when selected.
- ``--backend memory``: in-process only, deterministic, fastest. For smoke-testing the CLI without touching the engine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tardigrade_cli.subcommand import Subcommand

DEFAULT_BACKEND: str = "persistent"
SUPPORTED_BACKENDS: tuple[str, ...] = ("persistent", "memory", "qwen")


class ChatCommand(Subcommand):
    """Run the memory-aware chat REPL."""

    name = "chat"
    help = "Open the memory-aware chat REPL."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        from tardigrade_chat.constants import DEFAULT_CHAT_STATE_DIR, DEFAULT_PERSONA

        parser.description = (
            "Open an interactive REPL that stores every message as a persona-scoped "
            "memory and recalls relevant ones on each new turn.\n\n"
            "Example:\n"
            "  tardigrade chat --persona alice\n\n"
            "Slash commands inside the REPL: /memories /forget <text> /personas "
            "/switch <persona> /stats /exit /help."
        )
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.add_argument(
            "--persona",
            default=None,
            help=(
                "Persona to start with. Defaults to the last persona used "
                f"(or {DEFAULT_PERSONA!r} on first run)."
            ),
        )
        parser.add_argument(
            "--state-dir",
            default=str(DEFAULT_CHAT_STATE_DIR),
            help=f"Where the chat state lives (default: {DEFAULT_CHAT_STATE_DIR}).",
        )
        parser.add_argument(
            "--backend",
            choices=SUPPORTED_BACKENDS,
            default=DEFAULT_BACKEND,
            help=(
                "Storage + generation backend. 'persistent' uses a real engine on "
                "disk with echo responses (default). 'memory' is in-process only. "
                "'qwen' uses Qwen3-0.6B for both capture and generation."
            ),
        )

    def run(self, args: argparse.Namespace) -> int:
        from tardigrade_chat import MemoryChatSession
        from tardigrade_chat.constants import DEFAULT_PERSONA, LAST_PERSONA_FILE
        from tardigrade_chat.repl import Repl

        state_dir = Path(args.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        backend = self._build_backend(args.backend, state_dir)
        persona = args.persona or self._resolve_last_persona(state_dir) or DEFAULT_PERSONA

        session = MemoryChatSession(backend=backend, persona=persona, state_dir=state_dir)
        repl = Repl(session)
        return repl.run()

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_backend(name: str, state_dir: Path):
        if name == "memory":
            from tardigrade_chat.backend import InMemoryBackend
            return InMemoryBackend()
        if name == "persistent":
            from tardigrade_chat.backend import PersistentBackend
            return PersistentBackend(state_dir=state_dir)
        if name == "qwen":
            # Lazy import — torch + transformers should not be required
            # for the other backends.
            try:
                from tardigrade_chat.qwen_backend import QwenChatBackend
            except ImportError as exc:
                sys.stderr.write(
                    f"error: 'qwen' backend needs torch + transformers installed.\n"
                    f"       pip install 'tardigrade-db[transformers]'\n"
                    f"       (underlying error: {exc})\n",
                )
                raise SystemExit(2) from exc
            return QwenChatBackend(state_dir=state_dir)
        raise ValueError(f"unknown backend: {name}")  # argparse should have caught this

    @staticmethod
    def _resolve_last_persona(state_dir: Path) -> str | None:
        pointer = state_dir / "last_persona.txt"
        if pointer.exists():
            value = pointer.read_text(encoding="utf-8").strip()
            return value or None
        return None
