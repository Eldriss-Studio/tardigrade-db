"""tardigrade-chat — a memory-aware chat product over TardigradeDB.

The public surface is intentionally small:

- :class:`MemoryChatSession` — the consumer-facing per-persona session that the REPL drives.
- :class:`tardigrade_chat.backend.ChatBackend` — the swappable Strategy. Built-ins: :class:`InMemoryBackend` (tests), :class:`PersistentBackend` (real engine, no LLM), and (lazy-loaded) :class:`QwenChatBackend` (real model + real engine).

The CLI subcommand ``tardigrade chat`` wraps the REPL around a session; that's the runnable product. See ``docs/guide/chat.md`` for the consumer story.
"""

from .session import ChatTurn, MemoryChatSession, SessionStats

__all__ = ["ChatTurn", "MemoryChatSession", "SessionStats"]
