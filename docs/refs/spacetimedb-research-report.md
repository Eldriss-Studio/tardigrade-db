# SpacetimeDB Research Report (April 21, 2026)

Research scope:
- Online-first validation from official SpacetimeDB sources.
- Local exploration of the SpacetimeDB tree (commit pinned in §3).
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

## 3. Local Repo Exploration (SpacetimeDB tree, commit pinned below)

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

Local code exploration (paths within the SpacetimeDB tree at the commit pinned in §3):
- `crates/durability/src/lib.rs`
- `crates/durability/src/imp/local.rs`
- `crates/commitlog/src/lib.rs`
- `crates/commitlog/src/repo/fs.rs`
- `crates/snapshot/src/lib.rs`
- `crates/core/src/db/snapshot.rs`
- `crates/core/src/db/relational_db.rs`
- `crates/datastore/src/locking_tx_datastore/datastore.rs`
- `crates/datastore/src/locking_tx_datastore/replay.rs`
- `crates/client-api/src/lib.rs`
- `crates/client-api/src/routes/subscribe.rs`
- `crates/core/src/client/client_connection.rs`
- `crates/subscription/src/lib.rs`
- `crates/schema/tests/ensure_same_schema.rs`
- `crates/testing/src/sdk.rs`
- `crates/bench/README.md`
- `TESTING.md`

---

## 8. Status Update — 2026-05-21

Re-audited against SpacetimeDB commit `93a68ade0` (2026-05-21). The April §5.1 "Adopt Now" list has aged unevenly — two items shipped cleanly, two are partial, one is policy-without-implementation — and the broader workspace survey raised gaps the original report didn't enumerate. A second-pass meta-audit of engineering style and philosophy also revealed that on most of those axes TardigradeDB now meets or exceeds SpacetimeDB's discipline, so the action items are tighter than they would have been a year ago.

### 8.1 Original "Adopt Now" list — status

1. **Durability trait** (`append` / `durable_offset` / `history_from`) — *partial*. WAL and segment-based append behavior exist, but the trait abstraction proposed in §5.1 never crystallized. `rg durable_offset` finds one hit, in `crates/tdb-engine/src/snapshot.rs`. The contract lives in code but isn't a formalized boundary.
2. **Snapshot + WAL suffix replay** — **shipped**. Crash-recovery ATs cover segment + WAL truncation and multi-component atomicity.
3. **Confirmed vs unconfirmed read modes** — **policy without implementation**. CLAUDE.md's "Reliability & Consistency Rules" mandates that "Any API or externally visible read/update behavior must explicitly declare `confirmed` vs `unconfirmed` semantics," but no API surface exposes the choice today. This is the most consequential rule-vs-reality gap in the foundation.
4. **Startup/replay metrics** — **not shipped as a metrics layer**. `Engine::status()` provides coarse monitoring; there's no Prometheus-style instrumentation. `rg prometheus|histogram!|counter!` returns zero matches across `crates/`.
5. **Replay tests for corrupted/incomplete logs** — **shipped** in `tdb-engine` crash-recovery ATs.

### 8.2 Broader workspace gaps — features the April audit didn't raise

A wider sweep of the SpacetimeDB workspace surfaced patterns worth knowing about even though they sit outside the original durability-focused brief:

- **Energy / budget model** — `EnergyMonitor` trait with pluggable backends tracks per-call execution time, disk, memory. Relevant for multi-owner or per-tenant rate limiting in a memory engine.
- **Async snapshot worker** — snapshotting runs as a background actor with watch-channel completion. Ours is synchronous against the engine mutex; not a problem today but a latency cliff at scale.
- **Incremental subscription / live-query engine** — compiled plan fragments, view maintenance with join-delta semantics. We have no push/subscribe surface; HTTP is strictly request-response. Worth revisiting if downstream consumers want "tell me when owner X gets a new pack."
- **DurableOffset watch-channel for clients** — companion to the confirmed-read gap above. Lets clients `await` durability without polling.
- **Algebraic Type System + codegen (`sats`)** — schema-driven Rust/C#/TS codegen. We've gone OpenAPI for the HTTP bridge; `sats`-style codegen would matter if we ship typed client SDKs in additional languages.
- **MVCC-style transaction isolation** (`locking_tx_datastore`) — separate locks for committed vs active state. Our `Arc<Mutex<>>` is coarser; readers block writers. Probably fine for the current call frequency, becomes relevant if we add concurrent retrieval paths.

### 8.3 Engineering philosophy — where TardigradeDB already exceeds SpacetimeDB

The April report focused on what to adopt. The follow-up audit looked one layer down — at lint policy, error handling, documentation discipline, operational rules — and the honest read is that the foundation-completion phase pushed past SpacetimeDB on several of these. Important to record so future readers don't reflexively copy patterns we've already improved on:

- **Lint policy.** SpacetimeDB allows `result_large_err` workspace-wide (FIXME). TardigradeDB enforces `-D warnings` and requires a `// Reason:` comment on every site-level `#[allow]`.
- **Error handling.** They hybrid-use `thiserror` + `anyhow`. We use `miette::Diagnostic` with stable `tdb::<area>::<name>` codes and `#[help(...)]` annotations on every variant.
- **Reliability rules as written policy.** They keep durability invariants implicit in the code. We have CLAUDE.md's "Reliability & Consistency Rules (Canonical)" as explicit canonical policy.
- **Documentation standards.** They write pragmatic SAFETY comments but skip rustdoc examples. We mandate first-class crate-level `//!` docs with diagrams and worked examples.
- **API surface size.** They expose a large generated multi-language API. We deliberately keep the embedded engine surface small.

### 8.4 Engineering philosophy — what's worth borrowing

Four items the deeper audit surfaced that aren't on the §5.1 list and would tighten the engine independently of the formal durability gaps:

- **Cross-language schema parity test** — `crates/schema/tests/ensure_same_schema.rs` in the SpacetimeDB tree is a single test that fails if Rust schemas drift from generated client schemas. We enforce Python↔Rust parity via parallel test suites, which is harder to bypass than a single gate but also easier to forget to add when a new binding surface appears.
- **Property-based tests** (proptest/quickcheck) — SpacetimeDB has the dependencies but uses them sparingly. Our ATDD coverage is example-based; storage code (Q4 quantization round-trips, segment compaction invariants, snapshot restore) is the natural fit for property tests.
- **Replay-determinism as a property** — they're built so the same log always produces the same state. We have crash-recovery ATs but no test that asserts replay determinism over a corpus of generated logs.
- **Versioned schema modules** (their `raw_def::v9` namespace) — our snapshot format carries a `format_version` integer but doesn't version the public type modules. Becomes load-bearing the first time we change the on-disk pack layout and want old snapshots to load on new binaries.

A fifth observation, less actionable but worth recording: SpacetimeDB's code culture attributes magic constants to a person and a date (`// chosen completely arbitrarily by pgoldman 2025-04-10`). That kind of honesty makes future archaeology possible. We have constants in the workspace whose values aren't justified anywhere — a low-cost discipline gap.

### 8.5 Updated punch list

Carrying forward from §5.1, dropping shipped items, adding §8.2 and §8.4 entries:

**Formal foundation gaps (high impact):**

1. Close the confirmed-vs-unconfirmed read API gap — the rule exists in CLAUDE.md, no implementation enforces it.
2. Formalize the `Durability` trait (`append` / `durable_offset` / `history_from`) in `tdb-storage`.
3. Add a Prometheus-style metrics layer (startup time, replay time, queue depth, snapshot timings).

**Opportunistic discipline borrows (each independent, each cheap):**

4. Cross-language schema parity gate (Rust↔Python single test).
5. Property-based tests on storage round-trips.
6. Replay-determinism property test.
7. Versioned schema modules for forward-compat on the on-disk pack layout.

**Lower priority / depends-on-consumer-demand:**

8. Energy/budget trait for per-owner rate limiting.
9. Async background snapshot worker.
10. Subscription / live-query surface.
11. MVCC-style reader/writer isolation.

Items 1–7 are the realistic v0.8.x scope. Items 8–11 wait for a real consumer to ask for them.
