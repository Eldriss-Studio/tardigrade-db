"""ATDD: the chat REPL drives correctly over a list of input lines.

Uses the in-memory backend for speed. Live-model integration sits separately and is gated on the Qwen weights being available.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    return tmp_path / "chat"


def _build_repl(session_dir):
    from tardigrade_chat import MemoryChatSession
    from tardigrade_chat.backend import InMemoryBackend
    from tardigrade_chat.repl import Repl
    session = MemoryChatSession(InMemoryBackend(), persona="alice", state_dir=session_dir)
    return Repl(session), session


class TestReplCommands:
    def test_free_text_turn_emits_agent_and_latency(self, session_dir):
        repl, _ = _build_repl(session_dir)
        code, out = repl.drive(["I like sushi"])
        assert code == 0
        assert "agent>" in out
        assert "recalled" in out and "ms" in out

    def test_exit_command_returns_zero(self, session_dir):
        repl, _ = _build_repl(session_dir)
        code, out = repl.drive(["/exit"])
        assert code == 0

    def test_help_lists_known_commands(self, session_dir):
        repl, _ = _build_repl(session_dir)
        _, out = repl.drive(["/help"])
        for cmd in ("/memories", "/forget", "/personas", "/switch", "/stats", "/exit"):
            assert cmd in out

    def test_memories_lists_after_turns(self, session_dir):
        repl, _ = _build_repl(session_dir)
        _, out = repl.drive(["I like sushi", "/memories"])
        assert "sushi" in out

    def test_forget_removes_and_reports_count(self, session_dir):
        repl, _ = _build_repl(session_dir)
        _, out = repl.drive(["I like sushi", "/forget sushi", "/memories"])
        assert "forgot 1 memory" in out
        assert "sushi" not in out.split("/memories")[1] if "/memories" in out else True

    def test_switch_persona_isolates_memories(self, session_dir):
        repl, _ = _build_repl(session_dir)
        _, out = repl.drive([
            "alice fact",
            "/switch bob",
            "bob fact",
            "/memories",
        ])
        # /memories was run while persona was bob — only bob's fact should appear.
        listing = out.split("/memories")[-1] if False else out  # noqa: SIM108
        # Simpler check: after switching to bob, alice's memory must not appear in
        # the listing that follows.
        last_listing_start = out.rfind("memories for persona")
        assert last_listing_start != -1
        listing = out[last_listing_start:]
        assert "bob fact" in listing
        assert "alice fact" not in listing

    def test_personas_lists_known(self, session_dir):
        repl, _ = _build_repl(session_dir)
        _, out = repl.drive([
            "first message",
            "/switch bob",
            "second message",
            "/personas",
        ])
        assert "alice" in out and "bob" in out

    def test_stats_after_turns(self, session_dir):
        repl, _ = _build_repl(session_dir)
        _, out = repl.drive([f"turn {i}" for i in range(5)] + ["/stats"])
        assert "5 turn" in out
        assert "p50" in out and "p95" in out

    def test_unknown_command_does_not_crash(self, session_dir):
        repl, _ = _build_repl(session_dir)
        _, out = repl.drive(["/wat"])
        assert "unknown command" in out

    def test_empty_line_is_no_op(self, session_dir):
        repl, _ = _build_repl(session_dir)
        code, _ = repl.drive(["", "  ", "/exit"])
        assert code == 0
