# TardigradeDB Documentation

This directory is the long-form home for everything that doesn't fit on the project's landing page. The README is the storefront; this is the rest of the shop.

## Start here

| If you want to … | Read |
|------------------|------|
| Understand why TardigradeDB exists and what it is *not* | [`positioning.md`](positioning.md) |
| See the four-layer architecture | [`architecture.md`](architecture.md) |
| Run it from Python (high-level facade) | [`guide/python-api.md`](guide/python-api.md) |
| Inject KV into a HuggingFace model | [`guide/knowledge-pack-store.md`](guide/knowledge-pack-store.md) |
| Pick the right retrieval-key layer for your model | [`guide/calibration.md`](guide/calibration.md) |
| Use the engine as a vLLM connector | [`guide/vllm-setup.md`](guide/vllm-setup.md) |
| Embed it in your consumer (agent, NPC, RAG, …) | [`guide/consumers.md`](guide/consumers.md) |
| Wire it into an MCP server | [`guide/mcp-setup.md`](guide/mcp-setup.md) |
| Understand the core concepts | [`guide/concepts.md`](guide/concepts.md) |

## Performance and positioning

- [`positioning/latency_first.md`](positioning/latency_first.md) — the authoritative measured numbers (sub-millisecond p99 retrieval at 5K cells, 751 B per cell on disk).
- [`bench/`](bench/) — Bench V1 harness results, smoke and full runs.
- [`perf/`](perf/) — per-phase performance JSON snapshots.

## Research and history

- [`research-log.md`](research-log.md) — narrative across the experiments that shaped the retrieval pipeline.
- [`experiments/`](experiments/) — every dated experiment writeup, including the 2026-05-14 bench audit.
- [`refs/`](refs/) — industry references, prior art, competitor analyses.
- [`competitors/`](competitors/) — direct competitor matrices (MemArt, ByteRover, Letta, LMCache, …).

## Project planning

- [`roadmap.md`](roadmap.md) — what's shipped, what's next, future paths to production.
- [`technical/tdd.md`](technical/tdd.md) — full technical design document.
- [`technical/spec.md`](technical/spec.md) — condensed four-layer specification.

## Contributing

For everything related to building, testing, and contributing code, see [`../CONTRIBUTING.md`](../CONTRIBUTING.md) at the repo root.
