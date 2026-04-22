# Benchmark External Services

This directory contains container orchestration for optional live comparison runs.
The stack uses public images, with a local Mem0 image patch for missing DB drivers.

## Start services

```bash
docker compose -f benchmarks/docker-compose.external.yml up -d
```

To pull/build first (recommended):

```bash
docker compose -f benchmarks/docker-compose.external.yml build mem0
docker compose -f benchmarks/docker-compose.external.yml pull letta mem0_db mem0_neo4j qdrant
```

## Environment for adapters

```bash
export MEM0_BASE_URL="http://localhost:8888"
# Optional for custom auth frontends; local OSS stacks usually do not require it.
export MEM0_API_KEY=""
export LETTA_BASE_URL="http://localhost:8283"
export LETTA_API_KEY=""
```

Mem0 REST requires `OPENAI_API_KEY` to be set in the container environment because it uses
OpenAI by default for extraction and embeddings.

If services are not reachable/configured, benchmark runs emit explicit `skipped` or `failed`
outcomes (never silent drops).

## Suggested trustworthy run profile

```bash
LETTA_BASE_URL=http://localhost:8283 \
MEM0_BASE_URL=http://localhost:8888 \
PYTHONPATH=python .venv/bin/python -m tdb_bench run \
  --mode smoke \
  --repeat 3 \
  --config python/tdb_bench/config/default.json \
  --output target/bench-v1/smoke-r3.json
```

Repeat runs add `score_stddev` and `score_ci95` aggregate fields for uncertainty reporting.
