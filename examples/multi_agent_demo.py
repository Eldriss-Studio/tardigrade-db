"""Generic multi-agent reference consumer.

Three independent agents observe events, recall them, save/restore
across a checkpoint, and the host enumerates them via the owner
registry. The demo doubles as a foundation-feature acceptance gate
— if any of the underlying primitives (owner registry, write path,
encode_query, sweep, checkpoint, builder) regresses, this demo
breaks and ``tests/python/test_multi_agent_demo.py`` fails.

Each printed line names the primitive it exercises. The test
greps for those names; rename a line here, update the test there.

Run manually:

    python examples/multi_agent_demo.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import tardigrade_db
from tardigrade_hooks import TardigradeClient


AGENT_IDS = (1, 7, 42)
AGENT_OBSERVATIONS = {
    1: ["I am Agent A and I observed a sunrise"],
    7: ["I am Agent G and I observed a thunderstorm"],
    42: ["I am Agent X and I observed a quiet library"],
}
CHECKPOINT_LABEL = "demo-checkpoint"


def demo_dir() -> Path:
    """Working directory for the demo. Honors TARDIGRADE_DEMO_DIR
    so the integration test can isolate runs in a tmpdir."""
    env = os.environ.get("TARDIGRADE_DEMO_DIR")
    if env:
        path = Path(env)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path(tempfile.mkdtemp(prefix="tardigrade-demo-"))


def main() -> int:
    root = demo_dir()
    engine_dir = root / "engine"
    repo_dir = root / "checkpoints"
    restore_dir = root / "restored"

    # One engine, three clients — opening engine_dir from three
    # Engine instances yields three isolated states, not one
    # shared state. ``with_engine`` is the multi-agent pattern.
    engine = tardigrade_db.Engine(str(engine_dir))
    agents = {
        owner: (
            TardigradeClient.builder()
            .with_engine(engine)
            .with_owner(owner)
            .build()
        )
        for owner in AGENT_IDS
    }
    print("builder: constructed 3 agents sharing one engine")

    # Each agent observes its own events. Owner-scoping means
    # agent A's pack never appears in agent G's query.
    for owner, observations in AGENT_OBSERVATIONS.items():
        client = agents[owner]
        for fact in observations:
            client.store(fact)

    owners = engine.list_owners()
    print(f"owners: {owners}")
    assert owners == list(AGENT_IDS), f"owner registry drift: {owners}"

    # encode_query: agent A computes its own retrieval key and
    # reads via the engine directly, bypassing the convenience
    # query() path.
    client_a = agents[AGENT_IDS[0]]
    key = client_a.encode_query("sunrise")
    direct_results = client_a.engine.mem_read_pack(key, 5, AGENT_IDS[0])
    print(f"encode_query: {len(direct_results)} result(s) for agent A")
    assert direct_results, "expected at least one match for the stored fact"

    # Subjectivity check — agent G has no sunrise memory.
    cross_results = agents[AGENT_IDS[1]].query("sunrise", k=5)
    cross_texts = {(r.get("text") or "") for r in cross_results}
    assert "sunrise" not in " ".join(cross_texts), (
        "owner scoping leaked: agent G saw agent A's memory"
    )

    # Force a no-op sweep. The agents share the underlying engine,
    # so a single call covers all three. Returns the eviction
    # count (0 here, since all packs have salience well above the
    # threshold).
    evicted = engine.sweep_now(0.0, 15.0)
    print(f"sweep_now: {evicted} pack(s) evicted")

    # Labeled checkpoint + restore into a fresh dir.
    repo = tardigrade_db.CheckpointRepository(str(repo_dir))
    entry = repo.save(engine, CHECKPOINT_LABEL)
    print(
        f"checkpoint: label={entry['label']} seq={entry['seq']} "
        f"packs={entry['manifest']['pack_count']}"
    )

    restored = repo.restore_latest(CHECKPOINT_LABEL, str(restore_dir))
    assert restored.list_owners() == list(AGENT_IDS), (
        f"restored owner set drift: {restored.list_owners()}"
    )
    assert restored.pack_count() == sum(len(v) for v in AGENT_OBSERVATIONS.values())
    print("restored ok — owners + pack count match the source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
