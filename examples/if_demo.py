"""Interactive-fiction NPC memory demo — subjective memory pattern.

Two NPCs (a guarded blacksmith and a cheerful tavern keeper) react
to the same scripted player events. Each NPC interprets the events
through its own persona and stores a different memory — same input,
different recall.

Owner-scoping is what makes the pattern work: every NPC has its
own owner id, so a query under that id only retrieves that NPC's
memories. Two NPCs ingesting the same event each get their own
pack; neither query ever crosses the boundary.

This is intentionally minimal — no inkjs dependency, no story
engine, no UI. The point is to demonstrate the engine primitive
that game-style consumers will build on.

Run manually:

    python examples/if_demo.py
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import tardigrade_db
from tardigrade_hooks import TardigradeClient


@dataclass(frozen=True)
class Npc:
    """A single non-player character with an owner id and a
    persona-coloured way of remembering events."""

    name: str
    owner: int
    persona: str

    def interpret(self, event: str) -> str:
        """Re-phrase ``event`` in this NPC's voice.

        A real consumer would call an LLM here; we keep it
        deterministic for the demo so the test output is stable.
        """
        return f"[{self.name} | {self.persona}] {event}"


BLACKSMITH = Npc(name="Brom", owner=1, persona="suspicious blacksmith")
TAVERN_KEEPER = Npc(name="Lila", owner=2, persona="cheerful tavern keeper")
NPCS = (BLACKSMITH, TAVERN_KEEPER)

SCRIPTED_EVENTS = (
    "the player entered town at dusk",
    "the player asked about the missing caravan",
    "the player paid for ale with a strange coin",
    "the player left wearing the cloak from yesterday",
)


def demo_dir() -> Path:
    """Working directory for the demo. Honors ``TARDIGRADE_DEMO_DIR``
    so the integration test can isolate runs in a tmpdir."""
    env = os.environ.get("TARDIGRADE_DEMO_DIR")
    if env:
        path = Path(env)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path(tempfile.mkdtemp(prefix="tardigrade-if-demo-"))


def main() -> int:
    root = demo_dir()
    engine_dir = root / "engine"

    # One engine, two clients sharing it — same pattern as the
    # multi-agent demo.
    engine = tardigrade_db.Engine(str(engine_dir))
    clients = {
        npc: (
            TardigradeClient.builder()
            .with_engine(engine)
            .with_owner(npc.owner)
            .build()
        )
        for npc in NPCS
    }

    # Each NPC stores its own persona-coloured version of every
    # event. The engine sees N × M packs but owner-scoping keeps
    # each NPC's recall pure.
    for event in SCRIPTED_EVENTS:
        for npc in NPCS:
            clients[npc].store(npc.interpret(event))

    # Per-NPC recall summary. Each NPC sees only its own packs.
    counts: dict[str, int] = {}
    for npc in NPCS:
        recalled = clients[npc].list_packs()
        counts[npc.name] = len(recalled)
        if npc is BLACKSMITH:
            label = "blacksmith"
        else:
            label = "tavern keeper"
        print(f"{label} recalls {len(recalled)} event(s)")

    # Subjectivity check — query the *same* phrase under each
    # NPC's owner id. The top result for each comes back tagged
    # with that NPC's persona, never the other's.
    probe = "strange coin"
    summaries: list[str] = []
    for npc in NPCS:
        results = clients[npc].query(probe, k=1)
        if not results:
            summaries.append(f"{npc.name}=<no recall>")
            continue
        top = results[0]
        text = top.get("text") or engine.pack_text(top["pack_id"]) or ""
        assert npc.persona in text, (
            f"owner-scoping leak: {npc.name} got text without their persona: {text!r}"
        )
        summaries.append(f"{npc.name}={text}")

    print("subjective divergence: " + " | ".join(summaries))

    # Sanity assertions that catch regressions even if the print
    # lines change.
    assert all(count == len(SCRIPTED_EVENTS) for count in counts.values()), counts
    assert engine.list_owners() == sorted(npc.owner for npc in NPCS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
