# Building on TardigradeDB

TardigradeDB is a general-purpose KV-native memory engine. The
foundation surface exposes six verbs — `store`, `query`,
`list_owners`, `status`, `save`, `restore` — and three pluggable
extension points (KV-capture function, write buffer, consolidation
sweep). Everything else is consumer code.

This guide gives three reference patterns for consumers. Each is
under 40 lines of glue, no engine internals.

The Python surface (`tardigrade_hooks.TardigradeClient`) and the
HTTP surface (`POST /mem/*`) expose the same six verbs — pick
whichever fits the consumer's runtime. The patterns below use
HTTP so they read the same in Python, Node, Deno, or Go.

---

## Pattern 1 — Turn-by-turn agent memory

A long-running agent observes events, recalls relevant ones before
each action, and persists state across restarts.

```python
import httpx

class AgentMemory:
    def __init__(self, base_url: str, agent_id: int):
        self.client = httpx.Client(base_url=base_url, timeout=10.0)
        self.agent_id = agent_id

    def observe(self, fact: str) -> int:
        r = self.client.post("/mem/store", json={
            "owner": self.agent_id, "fact_text": fact,
        })
        return r.json()["pack_id"]

    def recall(self, query: str, k: int = 5) -> list[dict]:
        r = self.client.post("/mem/query", json={
            "owner": self.agent_id, "query_text": query, "k": k,
        })
        return r.json()["results"]

    def checkpoint(self, path: str) -> dict:
        r = self.client.post("/mem/save", json={"snapshot_path": path})
        return r.json()["manifest"]
```

Owner-scoping isolates each agent's memory; the engine never
returns one agent's pack to another agent's query. Checkpointing
is durable — the resulting tar archive carries magic, version,
codec identifiers, and a SHA-256 over the payload.

---

## Pattern 2 — Multi-NPC / multi-tenant isolation

Many short-lived consumers (game NPCs, per-user agents, per-tenant
shards) share one engine. Each consumer is an owner id; the engine
keeps memories disjoint at the storage layer.

```js
// Node — same idea in any HTTP-speaking runtime
async function npcMemory(bridgeUrl, npcId) {
  const post = (path, body) =>
    fetch(`${bridgeUrl}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => r.json());

  return {
    observe: (fact) => post("/mem/store", { owner: npcId, fact_text: fact }),
    recall:  (q, k = 5) =>
      post("/mem/query", { owner: npcId, query_text: q, k }).then((r) => r.results),
  };
}
```

The subjectivity property — same event, different memory per
owner — is structural. Two NPCs observing the same event store
two separate packs scoped to their own owner ids; neither query
ever crosses the boundary. See `examples/nodejs_consumer/` for a
runnable end-to-end demo of this pattern.

---

## Pattern 3 — Document ingest as long-context memory

A consumer wants to ask questions over a corpus that's larger than
its model's context window. Ingest each document as one or more
packs; later, a query retrieves the relevant packs and the
consumer composes its own prompt.

```python
import httpx, pathlib

def ingest_corpus(base_url: str, owner: int, docs_dir: pathlib.Path):
    client = httpx.Client(base_url=base_url)
    for path in docs_dir.glob("**/*.md"):
        text = path.read_text(encoding="utf-8")
        for chunk in chunk_by_paragraphs(text, max_chars=800):
            client.post("/mem/store", json={
                "owner": owner, "fact_text": f"[{path.name}] {chunk}",
            })

def chunk_by_paragraphs(text: str, max_chars: int) -> list[str]:
    out, buf = [], ""
    for para in text.split("\n\n"):
        if len(buf) + len(para) > max_chars and buf:
            out.append(buf); buf = ""
        buf += para + "\n\n"
    if buf:
        out.append(buf)
    return out
```

For richer ingestion (token-bounded chunking, overlap, view
generation), the Python API exposes
`tardigrade_hooks.TardigradeClient.ingest_file` directly — no
need to reimplement chunking via HTTP if the consumer is in
Python.

---

## Determinism contract

The engine is deterministic under the following conditions:

- Same seed and same input sequence ⇒ same stored cells.
- Same query key ⇒ same retrieval result order.
- Quantization (Q4 group quant) has no randomness.
- Vamana graph construction is seeded.
- Importance scoring is deterministic.

Consumers that rely on replay testing (snapshot a session, run
again, expect bitwise-identical retrievals) can lean on this.

---

## What lives where

- HTTP contract: `python/tardigrade_http/schema.yaml` (OpenAPI 3.1)
- TypeScript types: `python/tardigrade_http/types.ts` (generated)
- Python facade: `tardigrade_hooks.TardigradeClient`
- CLI: `tardigrade init|store|query|status|consolidate`
- Reference Node.js consumer: `examples/nodejs_consumer/`
- End-to-end Python demo: `examples/e2e_demo.py`
