"""Interactive REPL over :class:`MemoryChatSession`.

Pattern: **Command** — each slash command is a method on :class:`Repl`; the dispatcher maps the slash string to the method. Keeps the parser simple and the methods individually testable.

The REPL is the smallest UI on top of the session. It accepts free-text turns and a handful of slash commands. Designed so the message-handling logic can be driven from a list of input lines for testing (no actual TTY required).
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from io import StringIO
from typing import TextIO

from .constants import (
    LAST_PERSONA_FILE,
    REPL_AGENT_PREFIX,
    REPL_LATENCY_TAG,
    REPL_PROMPT,
    SLASH_COMMAND_PREFIX,
)
from .session import MemoryChatSession


@dataclass
class ReplExit(Exception):
    """Raised when the REPL should terminate cleanly."""
    code: int = 0


class Repl:
    """A single-session interactive loop.

    The constructor takes a session and an optional ``stdin``/``stdout`` pair so tests can drive the loop with a string buffer instead of a TTY.
    """

    def __init__(
        self,
        session: MemoryChatSession,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        self.session = session
        self._stdin = stdin if stdin is not None else sys.stdin
        self._stdout = stdout if stdout is not None else sys.stdout
        # Map command name → handler. Kept on the instance so subclasses
        # can add commands without touching parsing logic.
        self._commands: dict[str, Callable[[list[str]], None]] = {
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
            "memories": self._cmd_memories,
            "forget": self._cmd_forget,
            "personas": self._cmd_personas,
            "switch": self._cmd_switch,
            "stats": self._cmd_stats,
            "help": self._cmd_help,
        }

    # ── Public entry points ────────────────────────────────────────────────

    def run(self) -> int:
        """Block until the user exits. Returns a shell exit code."""
        self._write(f"persona: {self.session.persona} — type /help for commands\n")
        try:
            while True:
                line = self._read_line()
                if line is None:  # EOF
                    return 0
                code = self.handle_line(line)
                if code is not None:
                    return code
        except ReplExit as exit_signal:
            return exit_signal.code

    def handle_line(self, line: str) -> int | None:
        """Process one input line. Returns an exit code if the REPL should stop, ``None`` otherwise."""
        line = line.rstrip("\n").strip()
        if not line:
            return None
        if line.startswith(SLASH_COMMAND_PREFIX):
            return self._dispatch_command(line)
        turn = self.session.send(line)
        tag = REPL_LATENCY_TAG.format(n=turn.recalled_count, ms=turn.retrieval_ms)
        self._write(f"{REPL_AGENT_PREFIX}{turn.response}\n[{tag}]\n")
        # Update last-persona pointer so the next CLI invocation can resume.
        (self.session.state_dir / LAST_PERSONA_FILE).write_text(
            self.session.persona, encoding="utf-8",
        )
        return None

    def drive(self, lines: Iterable[str]) -> tuple[int, str]:
        """Run the REPL against a list of input lines (for tests).

        Returns ``(exit_code, captured_stdout)``. EOF after the last line.
        """
        buf = StringIO()
        old_out = self._stdout
        self._stdout = buf
        try:
            code = 0
            for line in lines:
                result = self.handle_line(line)
                if result is not None:
                    code = result
                    break
            return code, buf.getvalue()
        finally:
            self._stdout = old_out

    # ── Slash commands ──────────────────────────────────────────────────────

    def _dispatch_command(self, line: str) -> int | None:
        parts = line[len(SLASH_COMMAND_PREFIX):].split()
        if not parts:
            self._write("(empty command — try /help)\n")
            return None
        cmd, args = parts[0], parts[1:]
        handler = self._commands.get(cmd)
        if handler is None:
            self._write(f"unknown command: /{cmd} (try /help)\n")
            return None
        try:
            handler(args)
        except ReplExit as exit_signal:
            return exit_signal.code
        return None

    def _cmd_exit(self, _args: list[str]) -> None:
        raise ReplExit(code=0)

    def _cmd_memories(self, _args: list[str]) -> None:
        mems = self.session.memories()
        if not mems:
            self._write("(no memories for this persona)\n")
            return
        self._write(f"{len(mems)} memories for persona {self.session.persona!r}:\n")
        for m in mems:
            text = m.get("text", "<no text>")
            self._write(f"  [{m['id']}] {text}\n")

    def _cmd_forget(self, args: list[str]) -> None:
        if not args:
            self._write("usage: /forget <substring>\n")
            return
        pattern = " ".join(args)
        removed = self.session.forget(pattern)
        word = "memory" if removed == 1 else "memories"
        self._write(f"forgot {removed} {word} matching {pattern!r}\n")

    def _cmd_personas(self, _args: list[str]) -> None:
        names = self.session.personas()
        if not names:
            self._write("(no personas yet)\n")
            return
        marker = lambda n: " <- current" if n == self.session.persona else ""  # noqa: E731
        self._write(f"{len(names)} persona(s):\n")
        for name in names:
            self._write(f"  {name}{marker(name)}\n")

    def _cmd_switch(self, args: list[str]) -> None:
        if not args:
            self._write("usage: /switch <persona>\n")
            return
        target = args[0]
        self.session.switch_persona(target)
        self._write(f"switched to persona {target!r}\n")

    def _cmd_stats(self, _args: list[str]) -> None:
        s = self.session.stats()
        if s.turn_count == 0:
            self._write("(no turns yet)\n")
            return
        self._write(
            f"{s.turn_count} turn(s) — retrieval p50={s.retrieval_p50_ms:.2f} ms, "
            f"p95={s.retrieval_p95_ms:.2f} ms, p99={s.retrieval_p99_ms:.2f} ms\n",
        )

    def _cmd_help(self, _args: list[str]) -> None:
        self._write(
            "commands:\n"
            "  /memories             list memories for the current persona\n"
            "  /forget <substring>   delete memories whose text contains <substring>\n"
            "  /personas             list known personas\n"
            "  /switch <persona>     switch to a different persona\n"
            "  /stats                show retrieval-latency stats for this session\n"
            "  /exit, /quit          leave\n"
            "  /help                 this message\n",
        )

    # ── I/O ─────────────────────────────────────────────────────────────────

    def _read_line(self) -> str | None:
        self._write(REPL_PROMPT)
        line = self._stdin.readline()
        if not line:
            return None
        return line

    def _write(self, text: str) -> None:
        self._stdout.write(text)
        self._stdout.flush()
