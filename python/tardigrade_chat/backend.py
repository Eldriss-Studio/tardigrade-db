"""Chat backends — pluggable response generators with memory recall.

Pattern: **Strategy**. ``ChatBackend`` is the abstract contract; concrete implementations are:

- :class:`InMemoryBackend` — deterministic, no LLM, no persistence; for fast tests of the session/REPL surface.
- :class:`PersistentBackend` — a real engine on disk, deterministic echo responses; exercises the persistence story without the cost of loading a model.
- :class:`QwenChatBackend` (in :mod:`tardigrade_chat.qwen_backend`) — Qwen3-0.6B for real recall + response; what the CLI defaults to.

The session class is the consumer-facing surface; backends are the swap-out.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .constants import (
    DEFAULT_RECALL_K,
    ENGINE_SUBDIR,
    PERSONA_OWNER_BASE,
    PERSONA_REGISTRY_FILE,
)


@dataclass(frozen=True)
class RecallResult:
    """Output of a backend's memory recall step. Drives the latency tag in the REPL."""
    memories: list[dict]
    elapsed_ms: float


@dataclass(frozen=True)
class GenerationResult:
    """Output of a backend's response-generation step."""
    text: str


class ChatBackend(ABC):
    """The Strategy. Owns memory storage + response generation."""

    @abstractmethod
    def owner_for(self, persona: str) -> int: ...

    @abstractmethod
    def store(self, persona: str, text: str) -> int:
        """Store ``text`` as a memory under ``persona``. Returns a memory id."""

    @abstractmethod
    def recall(self, persona: str, query: str, k: int = DEFAULT_RECALL_K) -> RecallResult:
        """Return relevant memories for ``query`` plus the measured retrieval time."""

    @abstractmethod
    def generate(self, persona: str, prompt: str, recalled: list[dict]) -> GenerationResult:
        """Generate a response. May ignore ``recalled`` (mock backends) or weave it in (real ones)."""

    @abstractmethod
    def list_memories(self, persona: str) -> list[dict]:
        """All stored memories for ``persona``."""

    @abstractmethod
    def forget(self, persona: str, pattern: str) -> int:
        """Delete every memory whose text contains ``pattern``. Returns the count removed."""

    @abstractmethod
    def known_personas(self) -> list[str]:
        """Every persona this backend has seen (for ``/personas``)."""


# ─── In-memory backend (tests) ──────────────────────────────────────────────


class InMemoryBackend(ChatBackend):
    """A backend that keeps everything in process memory.

    Deterministic: ``generate`` returns a templated string mentioning the recalled memory count and the most recent persona message. Useful for testing the session/REPL surface without paying for an LLM or hitting disk.
    """

    def __init__(self) -> None:
        self._memories: dict[str, list[dict]] = {}
        self._next_id = 1
        self._owners: dict[str, int] = {}

    def owner_for(self, persona: str) -> int:
        if persona not in self._owners:
            self._owners[persona] = PERSONA_OWNER_BASE + len(self._owners)
        return self._owners[persona]

    def store(self, persona: str, text: str) -> int:
        self.owner_for(persona)
        mem = {"id": self._next_id, "text": text}
        self._next_id += 1
        self._memories.setdefault(persona, []).append(mem)
        return mem["id"]

    def recall(self, persona: str, query: str, k: int = DEFAULT_RECALL_K) -> RecallResult:
        t0 = time.perf_counter()
        mems = self._memories.get(persona, [])
        # Trivial "scoring": memories whose text shares any whole-word with the
        # query rank above the rest. Stable ordering for deterministic tests.
        q_words = {w.lower() for w in query.split()}
        scored = [
            (sum(1 for w in m["text"].lower().split() if w in q_words), m)
            for m in mems
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        relevant = [m for score, m in scored[:k] if score > 0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return RecallResult(memories=relevant, elapsed_ms=elapsed_ms)

    def generate(self, persona: str, prompt: str, recalled: list[dict]) -> GenerationResult:
        word = "memory" if len(recalled) == 1 else "memories"
        return GenerationResult(
            text=f"[{persona} echoes with {len(recalled)} relevant {word}]: {prompt}",
        )

    def list_memories(self, persona: str) -> list[dict]:
        return list(self._memories.get(persona, []))

    def forget(self, persona: str, pattern: str) -> int:
        before = self._memories.get(persona, [])
        kept = [m for m in before if pattern not in m["text"]]
        self._memories[persona] = kept
        return len(before) - len(kept)

    def known_personas(self) -> list[str]:
        return sorted(self._owners.keys())


# ─── Persistent backend (real engine, no LLM) ───────────────────────────────


class PersistentBackend(ChatBackend):
    """A real :class:`tardigrade_db.Engine` for storage; deterministic echo for generation.

    Demonstrates the persistence story end-to-end without paying for an LLM. Used in the cross-session AT to prove memories actually survive a process restart via the engine — not via the in-memory dict.

    Personas → owner ids via a JSON registry persisted next to the engine.
    """

    def __init__(self, state_dir: Path) -> None:
        from tardigrade_hooks import TardigradeClient

        self._state_dir = state_dir
        self._engine_dir = state_dir / ENGINE_SUBDIR
        self._registry_path = state_dir / PERSONA_REGISTRY_FILE
        state_dir.mkdir(parents=True, exist_ok=True)
        # One engine, many owners.
        self._client_factory = lambda owner: TardigradeClient.builder() \
            .with_engine_dir(self._engine_dir) \
            .with_owner(owner) \
            .build() if owner < 0 else None  # unused alt branch keeps type-checker quiet
        # Open a shared engine via the first client; reuse it across personas via with_engine.
        bootstrap = TardigradeClient.builder() \
            .with_engine_dir(self._engine_dir) \
            .with_owner(PERSONA_OWNER_BASE) \
            .build()
        self._engine = bootstrap.engine
        self._clients: dict[str, object] = {}
        self._registry: dict[str, int] = self._load_registry()

    def _load_registry(self) -> dict[str, int]:
        if self._registry_path.exists():
            return json.loads(self._registry_path.read_text(encoding="utf-8"))
        return {}

    def _save_registry(self) -> None:
        self._registry_path.write_text(json.dumps(self._registry, indent=2), encoding="utf-8")

    def owner_for(self, persona: str) -> int:
        if persona not in self._registry:
            self._registry[persona] = PERSONA_OWNER_BASE + len(self._registry)
            self._save_registry()
        return self._registry[persona]

    def _client(self, persona: str):
        if persona not in self._clients:
            from tardigrade_hooks import TardigradeClient
            self._clients[persona] = TardigradeClient.builder() \
                .with_engine(self._engine) \
                .with_owner(self.owner_for(persona)) \
                .build()
        return self._clients[persona]

    def store(self, persona: str, text: str) -> int:
        return self._client(persona).store(text)

    def recall(self, persona: str, query: str, k: int = DEFAULT_RECALL_K) -> RecallResult:
        t0 = time.perf_counter()
        results = self._client(persona).query(query, k=k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return RecallResult(memories=[dict(r) for r in results], elapsed_ms=elapsed_ms)

    def generate(self, persona: str, prompt: str, recalled: list[dict]) -> GenerationResult:
        word = "memory" if len(recalled) == 1 else "memories"
        return GenerationResult(
            text=f"[{persona} echoes with {len(recalled)} relevant {word}]: {prompt}",
        )

    def list_memories(self, persona: str) -> list[dict]:
        client = self._client(persona)
        packs = client.list_packs()
        out = []
        for pack in packs:
            text = pack.get("text") or self._engine.pack_text(pack["pack_id"]) or ""
            out.append({"id": pack["pack_id"], "text": text})
        return out

    def forget(self, persona: str, pattern: str) -> int:
        # Pattern-match by text substring. Uses the engine's delete_pack API.
        client = self._client(persona)
        removed = 0
        for mem in self.list_memories(persona):
            if pattern in mem["text"]:
                self._engine.delete_pack(mem["id"])
                removed += 1
        return removed

    def known_personas(self) -> list[str]:
        return sorted(self._registry.keys())
