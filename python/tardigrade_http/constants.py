"""Constants for the HTTP bridge.

Per the project's no-magic-values rule, every literal value used by
the bridge — port defaults, validation bounds, content types, env
var names — lives here.
"""

from __future__ import annotations

# -- Validation bounds -------------------------------------------------------

MIN_OWNER_ID: int = 0
MIN_QUERY_K: int = 1
MAX_QUERY_K: int = 100
DEFAULT_QUERY_K: int = 5

# -- HTTP / RFC 7807 ---------------------------------------------------------

PROBLEM_JSON_CONTENT_TYPE: str = "application/problem+json"
PROBLEM_TYPE_ABOUT_BLANK: str = "about:blank"
PROBLEM_TYPE_READ_TIMEOUT: str = "tdb:durability:read_timeout"
PROBLEM_TYPE_INVALID_REQUEST: str = "tdb:request:invalid_argument"
HTTP_STATUS_VALIDATION_ERROR: int = 422
HTTP_STATUS_BAD_REQUEST: int = 400
HTTP_STATUS_GATEWAY_TIMEOUT: int = 504

# -- Confirmed-read durability gate ------------------------------------------

# Server-side cap on ``?timeout=<ms>`` when a client requests
# ``?wait=durable``. Five minutes is comfortably longer than any
# realistic fsync latency while bounding worst-case server-held
# connection lifetimes — a client cannot DoS the bridge by issuing
# many requests with infinite timeouts. See
# ``docs/guide/performance-contracts.md`` for the rationale.
MAX_CONFIRMED_TIMEOUT_MS: int = 300_000
WAIT_DURABLE: str = "durable"

# -- Environment variables ---------------------------------------------------

ENV_DB_PATH: str = "TARDIGRADE_HTTP_DB"
ENV_HOST: str = "TARDIGRADE_HTTP_HOST"
ENV_PORT: str = "TARDIGRADE_HTTP_PORT"

DEFAULT_DB_PATH: str = "./tardigrade-http-engine"
DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 8765

# -- KV stub (no model) ------------------------------------------------------

STUB_KEY_DIM: int = 8
STUB_VALUE_DIM: int = 8
STUB_LAYER_INDEX: int = 0
HASH_SEED_MODULUS: int = 2**31

# -- Server metadata ---------------------------------------------------------

APP_TITLE: str = "TardigradeDB HTTP Bridge"
APP_VERSION: str = "1.0.0"
APP_DESCRIPTION: str = (
    "REST adapter over the TardigradeDB Engine. Endpoints under "
    "`/mem/*` expose store/query/owners/status/snapshot/restore."
)
