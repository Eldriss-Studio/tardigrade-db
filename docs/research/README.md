# Research log

External research conducted for this project. Each entry credits the sources it leaned on and links forward to where the findings landed (ADRs, code, docs). See the `/research-catalog` skill for the workflow, or just read an entry to see the shape.

## Sessions

| Date | Topic | Triggered by | Informed |
|------|-------|--------------|----------|
| 2026-05-21 | [Learnings from production KV-cache projects (LMCache, vLLM, SGLang, Mooncake, CacheGen, KVTC, Dynamo, DistServe)](./2026-05-21-production-kv-cache-learnings.md) | User question in chat: "Is there anything tardigrade can learn today from production-grade kv cache projects?" | Plan memo `~/.claude/plans/is-there-anything-dreamy-pike.md`; future ATDD plans when top-3 recommendations are picked up |
| 2026-05-19 | [Qwen3-Next hybrid attention and KV-cache persistence](./2026-05-19-qwen3-next-hybrid-attention.md) | `AttributeError: 'LinearAttentionLayer' object has no attribute 'keys'` observed while swapping the casper-spike to Qwen3.5; question raised on whether tardigrade-db should support hybrid-attention models | `CHANGELOG.md` v0.3.2 "Known Limitations" entry; scope decision for tardigrade-db's supported-architecture surface (pending) |

## Adding a session

When you research something external (vendor docs, papers, blog posts, open-source repos), the `/research-catalog` skill produces a file in this directory and a row in this table. Run it before the synthesis goes back to the requester.
