"""ATDD: MemoryChatSession — the heart of the chat-with-memory product.

The session wraps a ``ChatBackend`` (real model in prod, mock in tests) and adds the per-persona, per-session, latency-tracking surface the REPL needs.

These tests use a deterministic in-memory backend so the suite is fast and CPU-only. The live-model integration test sits separately and is gated on the Qwen3 weights being available.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    return tmp_path / "chat"


class TestMemoryChatSessionBasics:
    def test_send_returns_chat_turn_with_response_and_latency(self, session_dir):
        from tardigrade_chat import MemoryChatSession
        from tardigrade_chat.backend import InMemoryBackend
        session = MemoryChatSession(InMemoryBackend(), persona="alice", state_dir=session_dir)
        turn = session.send("hello world")
        assert isinstance(turn.response, str) and turn.response
        assert turn.retrieval_ms >= 0.0
        assert turn.recalled_count >= 0

    def test_memories_lists_what_was_stored(self, session_dir):
        from tardigrade_chat import MemoryChatSession
        from tardigrade_chat.backend import InMemoryBackend
        session = MemoryChatSession(InMemoryBackend(), persona="alice", state_dir=session_dir)
        session.send("I like sushi")
        session.send("my dog is Cooper")
        memories = session.memories()
        texts = [m["text"] for m in memories]
        # Both user messages should appear among stored memories for this persona.
        assert any("sushi" in t for t in texts)
        assert any("Cooper" in t for t in texts)

    def test_personas_are_isolated(self, session_dir):
        from tardigrade_chat import MemoryChatSession
        from tardigrade_chat.backend import InMemoryBackend
        backend = InMemoryBackend()
        alice = MemoryChatSession(backend, persona="alice", state_dir=session_dir)
        bob = MemoryChatSession(backend, persona="bob", state_dir=session_dir)
        alice.send("I am alice's memory")
        bob.send("I am bob's memory")
        alice_texts = " ".join(m["text"] for m in alice.memories())
        bob_texts = " ".join(m["text"] for m in bob.memories())
        assert "alice's memory" in alice_texts and "bob's memory" not in alice_texts
        assert "bob's memory" in bob_texts and "alice's memory" not in bob_texts

    def test_switch_persona_changes_visible_memories(self, session_dir):
        from tardigrade_chat import MemoryChatSession
        from tardigrade_chat.backend import InMemoryBackend
        backend = InMemoryBackend()
        session = MemoryChatSession(backend, persona="alice", state_dir=session_dir)
        session.send("alice fact")
        session.switch_persona("bob")
        session.send("bob fact")
        bob_texts = [m["text"] for m in session.memories()]
        assert any("bob fact" in t for t in bob_texts)
        assert not any("alice fact" in t for t in bob_texts)

    def test_forget_removes_matching_memories(self, session_dir):
        from tardigrade_chat import MemoryChatSession
        from tardigrade_chat.backend import InMemoryBackend
        session = MemoryChatSession(InMemoryBackend(), persona="alice", state_dir=session_dir)
        session.send("forgettable")
        session.send("keep this")
        removed = session.forget("forgettable")
        assert removed >= 1
        remaining = [m["text"] for m in session.memories()]
        assert not any("forgettable" in t for t in remaining)
        assert any("keep this" in t for t in remaining)

    def test_stats_reports_retrieval_latency_percentiles(self, session_dir):
        from tardigrade_chat import MemoryChatSession
        from tardigrade_chat.backend import InMemoryBackend
        session = MemoryChatSession(InMemoryBackend(), persona="alice", state_dir=session_dir)
        for i in range(10):
            session.send(f"turn {i}")
        stats = session.stats()
        assert stats.turn_count == 10
        assert stats.retrieval_p50_ms >= 0.0
        assert stats.retrieval_p95_ms >= stats.retrieval_p50_ms

    def test_personas_list_includes_every_seen_persona(self, session_dir):
        from tardigrade_chat import MemoryChatSession
        from tardigrade_chat.backend import InMemoryBackend
        backend = InMemoryBackend()
        for name in ("alice", "bob", "carol"):
            MemoryChatSession(backend, persona=name, state_dir=session_dir).send("hi")
        observer = MemoryChatSession(backend, persona="alice", state_dir=session_dir)
        listed = set(observer.personas())
        assert {"alice", "bob", "carol"}.issubset(listed)


class TestCrossSessionPersistence:
    def test_memories_survive_session_recreation_via_state_dir(self, session_dir):
        from tardigrade_chat import MemoryChatSession
        from tardigrade_chat.backend import PersistentBackend
        backend = PersistentBackend(state_dir=session_dir)
        first = MemoryChatSession(backend, persona="alice", state_dir=session_dir)
        first.send("I should survive a restart")
        # Drop the session + backend, recreate from the same state_dir.
        del first
        del backend
        backend2 = PersistentBackend(state_dir=session_dir)
        second = MemoryChatSession(backend2, persona="alice", state_dir=session_dir)
        texts = [m["text"] for m in second.memories()]
        assert any("survive a restart" in t for t in texts)
