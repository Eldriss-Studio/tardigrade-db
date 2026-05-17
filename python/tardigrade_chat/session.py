"""Per-persona, latency-tracking chat session.

The session is the consumer-facing surface that the REPL (and any future UI) drives. It owns:

- The current persona (an owner id under the hood).
- A rolling history of turns with their measured retrieval latency, so ``stats()`` can report percentiles.
- The slash-command primitives — ``memories``, ``forget``, ``switch_persona``, ``personas``.

Storage and response generation live in a swappable :class:`tardigrade_chat.backend.ChatBackend`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from .backend import ChatBackend
from .constants import (
    DEFAULT_RECALL_K,
    LATENCY_P50_PERCENTILE,
    LATENCY_P95_PERCENTILE,
    LATENCY_P99_PERCENTILE,
    MIN_SAMPLES_FOR_PERCENTILES,
)


@dataclass(frozen=True)
class ChatTurn:
    """One round-trip through the session. Returned by :meth:`MemoryChatSession.send`."""

    response: str
    retrieval_ms: float
    recalled_count: int
    stored_id: int


@dataclass(frozen=True)
class SessionStats:
    """Latency percentiles + turn count for the current session. Returned by :meth:`MemoryChatSession.stats`."""

    turn_count: int
    retrieval_p50_ms: float
    retrieval_p95_ms: float
    retrieval_p99_ms: float


@dataclass
class MemoryChatSession:
    """Persona-aware chat session over a :class:`ChatBackend`.

    Construction does no I/O beyond what the backend does. ``state_dir`` is kept for future use (REPL writes the last-persona pointer there).
    """

    backend: ChatBackend
    persona: str
    state_dir: Path
    recall_k: int = DEFAULT_RECALL_K
    _latencies_ms: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        # Resolve the persona once so it shows up in known_personas() even
        # before the first send().
        self.backend.owner_for(self.persona)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    # ── Core turn flow ──────────────────────────────────────────────────────

    def send(self, message: str) -> ChatTurn:
        """Run one full round-trip: recall → generate → store."""
        recall = self.backend.recall(self.persona, message, k=self.recall_k)
        gen = self.backend.generate(self.persona, message, recall.memories)
        stored_id = self.backend.store(self.persona, message)
        self._latencies_ms.append(recall.elapsed_ms)
        return ChatTurn(
            response=gen.text,
            retrieval_ms=recall.elapsed_ms,
            recalled_count=len(recall.memories),
            stored_id=stored_id,
        )

    # ── Persona management ──────────────────────────────────────────────────

    def switch_persona(self, persona: str) -> None:
        """Switch the session to a different persona. Past latency samples are kept for the cumulative stats view; the new persona's own samples accumulate from here."""
        self.persona = persona
        self.backend.owner_for(persona)

    def personas(self) -> list[str]:
        return self.backend.known_personas()

    # ── Memory inspection ───────────────────────────────────────────────────

    def memories(self) -> list[dict]:
        return self.backend.list_memories(self.persona)

    def forget(self, pattern: str) -> int:
        return self.backend.forget(self.persona, pattern)

    # ── Stats ───────────────────────────────────────────────────────────────

    def stats(self) -> SessionStats:
        if len(self._latencies_ms) < MIN_SAMPLES_FOR_PERCENTILES:
            return SessionStats(turn_count=0, retrieval_p50_ms=0.0, retrieval_p95_ms=0.0, retrieval_p99_ms=0.0)
        return SessionStats(
            turn_count=len(self._latencies_ms),
            retrieval_p50_ms=_percentile(self._latencies_ms, LATENCY_P50_PERCENTILE),
            retrieval_p95_ms=_percentile(self._latencies_ms, LATENCY_P95_PERCENTILE),
            retrieval_p99_ms=_percentile(self._latencies_ms, LATENCY_P99_PERCENTILE),
        )


def _percentile(samples: list[float], pct: int) -> float:
    if not samples:
        return 0.0
    sorted_samples = sorted(samples)
    idx = max(0, min(len(sorted_samples) - 1, math.ceil(pct / 100.0 * len(sorted_samples)) - 1))
    return sorted_samples[idx]
