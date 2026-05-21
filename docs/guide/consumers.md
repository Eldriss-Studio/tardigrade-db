# Building on TardigradeDB

You want to add persistent memory to your agent, your multiplayer NPC, your per-user assistant, or your document-Q&A app. This guide gives you three reference patterns that cover the cases most consumers actually face — turn-by-turn agent recall, multi-tenant isolation, and document ingest. Each is under 40 lines of glue. The engine does the work; you just wire it.

A few facts before the code:

- **Two surfaces, same operations.** The Python surface (`tardigrade_hooks.TardigradeClient`) and the HTTP surface (`POST /mem/*`) expose the same six operations: `store`, `query`, `list_owners`, `status`, `save`, `restore`. Use Python if your consumer is Python; use HTTP from Node, Deno, Go, Rust, or anything else. The patterns below use HTTP so they read the same regardless of your runtime.
- **Owner-scoping is the isolation primitive.** Every operation takes an owner id. The engine guarantees, at the storage layer, that one owner's `query` never returns another owner's pack — even when both owners stored the same input text. If you have two agents, give them owner ids `1` and `2`; if you have a hundred NPCs in a game, give each one a unique id; if you have a multi-tenant SaaS, owner id maps to tenant. There is no shared namespace to leak from.
- **The bridge runs locally.** All patterns below assume a `base_url` like `http://localhost:5000` pointing at a `tardigrade_http` server. You start it with `python -m tardigrade_http.main`; it embeds the Rust engine and serves the REST endpoints. No Docker, no separate database process.

For advanced extensibility (custom KV-capture functions, write buffers, consolidation sweeps), see [`docs/architecture.md`](../architecture.md). The patterns here use the foundation API only.

---

## Pattern 1 — Turn-by-turn agent memory

A long-running agent observes events, recalls relevant ones before each action, and persists state across restarts.

```python
import httpx

class AgentMemory:
    def __init__(self, base_url: str, agent_id: int):
        # base_url is wherever tardigrade_http is listening; agent_id is
        # the owner under which all this agent's memories will be stored.
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
        # Save the engine's full state to a tar archive at `path`.
        # The archive carries magic, version, codec identifiers, and a
        # SHA-256 over the payload, so corruption is detectable.
        r = self.client.post("/mem/save", json={"snapshot_path": path})
        return r.json()["manifest"]
```

The agent calls `observe()` after every action it takes, `recall()` before deciding the next one, and `checkpoint()` whenever it wants a durable restore point. Owner-scoping (the `agent_id` passed on every call) does the isolation work — two agents using the same engine never see each other's packs, because the storage layer never returns a pack to a query whose owner doesn't match.

---

## Pattern 2 — Multi-NPC / multi-tenant isolation

Many short-lived consumers — game NPCs, per-user agents, per-tenant shards — share a single engine. Each consumer gets its own owner id, and the engine handles the rest.

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

The subjectivity property — same event, different memory per owner — is structural rather than something you have to enforce in glue code. Two NPCs observing the same event store two separate packs scoped to their own owner ids; neither query ever crosses the boundary, even if both NPCs called `observe()` with the exact same `fact` string. A runnable end-to-end demo of this pattern lives at `examples/nodejs_consumer/` in the tardigrade-db repository.

---

## Pattern 3 — Document ingest as long-context memory

You have a corpus that doesn't fit in your model's context window — a code repository, a set of meeting notes, a documentation tree. You want the model to answer questions over it without re-tokenizing the entire corpus on every query. The pattern is to ingest each document as one or more *packs* (the engine's unit of stored memory), let the engine handle similarity-based retrieval, and have your consumer compose the final prompt from whichever packs come back.

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

For richer ingestion (token-bounded chunking, overlap, view generation), the Python API exposes `tardigrade_hooks.TardigradeClient.ingest_file` directly — no need to reimplement chunking via HTTP if the consumer is in Python.

---

## Determinism contract

The engine is deterministic under the following conditions:

- Same seed and same input sequence ⇒ same stored cells.
- Same query key ⇒ same retrieval result order.
- Quantization (Q4 group quant) has no randomness.
- Vamana graph construction is seeded.
- Importance scoring is deterministic.

Consumers that rely on replay testing (snapshot a session, run again, expect bitwise-identical retrievals) can lean on this.

---

## What lives where

- HTTP contract: `python/tardigrade_http/schema.yaml` (OpenAPI 3.1)
- TypeScript types: `python/tardigrade_http/types.ts` (generated)
- Python facade: `tardigrade_hooks.TardigradeClient`
- CLI: `tardigrade init|store|query|status|consolidate`
- Reference Node.js consumer: `examples/nodejs_consumer/`
- End-to-end Python demo: `examples/e2e_demo.py`
