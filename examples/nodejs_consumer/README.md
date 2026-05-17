# Node.js consumer (foundation reference)

Smallest possible non-Python consumer of TardigradeDB — pure Node
22+, no dependencies beyond the built-in `fetch`. Proves the HTTP
bridge is enough on its own for an outside-Python integration.

The demo:

1. stores 5 facts under owner 1
2. queries one back, asserts the round-trip
3. snapshots the engine to a tar archive
4. restores into a fresh directory and asserts pack-count parity
5. lists owners and asserts the expected set

## Run it manually

```bash
# 1. start the bridge in one shell
source .venv/bin/activate
python -m tardigrade_http.server   # binds 127.0.0.1:8765 by default

# 2. in another shell, run the consumer
node examples/nodejs_consumer/index.mjs
```

You should see five `stored …` lines, a `query …` match, a
snapshot SHA-256 prefix, the restore confirmation, and the
`owners = [1]` final check.

## TypeScript types

The HTTP contract is captured by the OpenAPI schema at
`python/tardigrade_http/schema.yaml`. Pre-generated TypeScript
type definitions live next to it at
`python/tardigrade_http/types.ts` — import them from a TypeScript
consumer to get IDE autocomplete on every request/response shape:

```ts
import type { paths, components } from "../../python/tardigrade_http/types";

type StoreReq = components["schemas"]["StoreRequest"];
type QueryRes = components["schemas"]["QueryResponse"];
```

Regenerate after any change to the bridge's models:

```bash
python -m tardigrade_http.export_schema    # writes schema.yaml
npx openapi-typescript python/tardigrade_http/schema.yaml \
  -o python/tardigrade_http/types.ts        # writes types.ts
```

## CI coverage

`tests/python/test_nodejs_consumer.py` runs this script against a
uvicorn server in a thread on every Python test pass — if the
script breaks, that test fails.
