# SpacetimeDB Research Report (April 21, 2026)

Research scope:
- Online-first validation from official SpacetimeDB sources.
- Local exploration of `/Users/storylight/Dev/labs/SpacetimeDB`.
- Practical lessons for TardigradeDB architecture decisions.

---

## 1. Executive Summary

SpacetimeDB is a database-runtime system where application logic runs inside the database as modules and clients subscribe directly to state changes. The most relevant technical lessons for TardigradeDB are not relational semantics, but operational patterns:

- Explicit durability abstractions.
- Snapshot + WAL/commitlog replay pipelines.
- Strong client-facing consistency contracts ("confirmed reads").
- Incremental update engines instead of naive recomputation.
- Heavy investment in test/benchmark harnesses and multi-language parity checks.

For TardigradeDB, these patterns map cleanly onto tensor-native memory infrastructure and should be adopted early.

---

## 2. Online Research Highlights

From official docs and project materials:

- SpacetimeDB positions itself as "database + server" with module-hosted logic and real-time subscriptions.
- It provides transactional reducer execution semantics (atomicity and rollback behavior).
- Runtime architecture keeps hot state in memory and uses durable logging/snapshots for recovery.
- Modern protocol behavior defaults toward durability-aware read visibility (confirmed reads for newer protocol versions).

Latest validated release context:
- `v2.1.0` release line observed in official project release metadata (March 24, 2026).

---

## 3. Local Repo Exploration (`/Users/storylight/Dev/labs/SpacetimeDB`)

Local checkout metadata:
- Commit: `d5c1738c1`
- Commit date: `2026-04-21 08:08:57 -0500`

Workspace structure shows clear subsystem boundaries:
- `commitlog`
- `durability`
- `snapshot`
- `datastore`
- `execution`
- `subscription`
- `client-api`
- `standalone`

This decomposition is a strong indicator of production-focused operational design.

---

## 4. Key Architectural Findings

## 4.1 Durability Is a First-Class Boundary

SpacetimeDB explicitly models durability via traits and handles:
- Non-blocking append path.
- Durable offset tracking.
- History/replay traversal as a separate concern.

Reference:
- `crates/durability/src/lib.rs`

Why this matters:
- It decouples transaction acceptance from flush timing.
- It creates a clean contract for "what is visible" vs "what is durable."

## 4.2 Commitlog + Actor-Based Local Persister

Local durability implementation uses:
- Bounded async queue.
- Background actor.
- Batched writes + flush/sync progression.

Reference:
- `crates/durability/src/imp/local.rs`

Why this matters:
- Backpressure and queue depth become observable.
- Durability work does not block front-path request ingestion.

## 4.3 Snapshot + Replay Is Explicit and Defensive

Observed startup flow:
1. Try latest usable snapshot.
2. Restore snapshot.
3. Replay commitlog suffix.
4. Rebuild in-memory derived structures.

References:
- `crates/core/src/db/relational_db.rs`
- `crates/core/src/db/snapshot.rs`
- `crates/snapshot/src/lib.rs`
- `crates/datastore/src/locking_tx_datastore/datastore.rs`
- `crates/datastore/src/locking_tx_datastore/replay.rs`

Important implementation detail:
- Replay is fail-fast in production mode to avoid reconstructing an incorrect state.

## 4.4 Confirmed Reads Consistency Contract

SpacetimeDB wires durable-offset waiting into client delivery paths:
- Client can request/receive confirmed-read behavior.
- Server can defer update/result delivery until `durable_offset >= tx_offset`.

References:
- `crates/client-api/src/lib.rs`
- `crates/client-api/src/routes/subscribe.rs`
- `crates/core/src/client/client_connection.rs`

Why this matters:
- Consistency semantics are explicit at API level, not implicit in implementation details.

## 4.5 Incremental Subscription Engine

Subscription logic compiles plans/fragments and maintains views incrementally (including join delta behavior), rather than full recompute.

Reference:
- `crates/subscription/src/lib.rs`

Why this matters:
- High-frequency updates remain tractable.
- Complexity is paid once in planner/executor logic, not repeatedly at runtime.

## 4.6 Engineering Discipline: Parity, Harnesses, Benchmarks

SpacetimeDB includes:
- Cross-language schema parity checks.
- SDK and standalone integration harnesses.
- Dedicated benchmark suites and profiling guidance.

References:
- `crates/schema/tests/ensure_same_schema.rs`
- `crates/testing/src/sdk.rs`
- `crates/bench/README.md`
- `TESTING.md`

---

## 5. What TardigradeDB Can Learn

## 5.1 Adopt Now

1. Introduce a durability interface in `tdb-storage` with:
   - `append(tx)`
   - `durable_offset()`
   - `history_from(offset)`
2. Add snapshot + WAL suffix replay workflow in Phase 1 storage work.
3. Define explicit read-visibility modes:
   - Fast/unconfirmed reads.
   - Confirmed reads (wait for durability threshold).
4. Instrument startup/replay metrics from day one.
5. Build replay tests for corrupted/incomplete log scenarios early.

## 5.2 Adapt for Tensor-Native Context

1. Keep replay/restore contracts, but map data units to KV blocks + sidecar metadata instead of rows.
2. Rebuild derived state post-replay:
   - Atlas/Vamana-related structures.
   - SLB hot indexes.
   - Governance materialized views/scores (if partially cached).
3. Add "confirmed retrieval" semantics for KV block visibility, analogous to confirmed reads.

## 5.3 Avoid Copying Directly

1. Do not copy relational abstractions (SQL tables/reducers) into the tensor-native core.
2. Do not tie architecture to one runtime protocol; preserve internal contracts independent of API shape.
3. Do not delay durability semantics until after retrieval/index layers; that raises migration risk.

---

## 6. Suggested Next Actions for TardigradeDB

1. Write a `Durability` trait proposal for `tdb-storage` (with async queue + durable offset semantics).
2. Add a `storage-recovery.md` design note defining:
   - Snapshot format/versioning.
   - WAL segment lifecycle.
   - Replay failure policy.
3. Add acceptance tests for:
   - Crash during append.
   - Crash between append and snapshot.
   - Snapshot load + WAL suffix replay correctness.
4. Add metrics skeleton:
   - replay total time
   - snapshot read/restore time
   - WAL replay count/time
   - append queue depth

---

## 7. Sources

Official/public:
- https://github.com/clockworklabs/spacetimedb
- https://github.com/clockworklabs/SpacetimeDB/releases
- https://spacetimedb.com/docs/intro/what-is-spacetimedb
- https://spacetimedb.com/docs/databases/transactions-atomicity/
- https://spacetimedb.com/docs/upgrade/
- https://spacetimedb.com/docs/tables/event-tables/
- https://spacetimedb.com/docs/http/database/
- https://spacetimedb.com/docs/how-to/deploy/self-hosting/

Local code exploration:
- `/Users/storylight/Dev/labs/SpacetimeDB/crates/durability/src/lib.rs`
- `/Users/storylight/Dev/labs/SpacetimeDB/crates/durability/src/imp/local.rs`
- `/Users/storylight/Dev/labs/SpacetimeDB/crates/commitlog/src/lib.rs`
- `/Users/storylight/Dev/labs/SpacetimeDB/crates/commitlog/src/repo/fs.rs`
- `/Users/storylight/Dev/labs/SpacetimeDB/crates/snapshot/src/lib.rs`
- `/Users/storylight/Dev/labs/SpacetimeDB/crates/core/src/db/snapshot.rs`
- `/Users/storylight/Dev/labs/SpacetimeDB/crates/core/src/db/relational_db.rs`
- `/Users/storylight/Dev/labs/SpacetimeDB/crates/datastore/src/locking_tx_datastore/datastore.rs`
- `/Users/storylight/Dev/labs/SpacetimeDB/crates/datastore/src/locking_tx_datastore/replay.rs`
- `/Users/storylight/Dev/labs/SpacetimeDB/crates/client-api/src/lib.rs`
- `/Users/storylight/Dev/labs/SpacetimeDB/crates/client-api/src/routes/subscribe.rs`
- `/Users/storylight/Dev/labs/SpacetimeDB/crates/core/src/client/client_connection.rs`
- `/Users/storylight/Dev/labs/SpacetimeDB/crates/subscription/src/lib.rs`
- `/Users/storylight/Dev/labs/SpacetimeDB/crates/schema/tests/ensure_same_schema.rs`
- `/Users/storylight/Dev/labs/SpacetimeDB/crates/testing/src/sdk.rs`
- `/Users/storylight/Dev/labs/SpacetimeDB/crates/bench/README.md`
- `/Users/storylight/Dev/labs/SpacetimeDB/TESTING.md`
