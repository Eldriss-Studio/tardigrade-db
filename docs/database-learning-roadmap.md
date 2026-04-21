# Database Engineering Learning Roadmap

A structured guide for becoming a strong database engineer — from data structures and storage internals to distributed systems, operations, and modern AI-adjacent data workloads. Each level builds on the previous and maps to practical engineering capability.

> **Core mental model:** A database is a correctness engine first, and a performance engine second. If you optimize before you can prove safety and invariants, you build fast corruption.

---

## Level 0 — Engineering Foundations (The Stuff You Can't Skip)

Before database-specific concepts, you need the primitives databases are built from.

### What to Learn

- **Memory and CPU basics** — cache lines, branch prediction, NUMA, memory latency vs throughput.
- **Data structures under load** — arrays, hash tables, B-trees, skip lists, heaps, bloom filters.
- **Complexity in real systems** — asymptotics plus constants, locality, allocator effects.
- **Binary formats & serialization** — varints, endianness, alignment, checksums.
- **OS fundamentals** — virtual memory, page cache, mmap, fsync semantics, file descriptors.

### Resources

- *Computer Systems: A Programmer's Perspective* (CS:APP)
- *The Linux Programming Interface*
- *Designing Data-Intensive Applications* (Ch. 1–3 as baseline framing)

### You're Ready When

You can explain why a B-tree with worse asymptotics can outperform a hash structure in real systems because of cache and disk locality constraints.

---

## Level 1 — Storage Engines (How Bytes Become State)

Now learn how durable state is physically represented and mutated.

### What to Learn

- **Page-oriented vs log-structured storage** — B-tree page stores vs LSM trees.
- **Write path mechanics** — WAL/commit log, flush, sync, group commit.
- **Read path mechanics** — point lookup, range scan, prefetching, read amplification.
- **Compaction/GC** — tombstones, segment merge, space amplification tradeoffs.
- **Checksums and corruption detection** — per-page/per-block integrity strategies.

### Resources

- PostgreSQL storage/WAL internals docs
- RocksDB architecture + tuning guides
- SQLite file format and pager docs

### You're Ready When

You can draw a write lifecycle from client request -> memtable/page mutation -> WAL append -> flush/checkpoint and identify every crash boundary.

---

## Level 2 — Indexing and Query Access Paths

This is where performance engineering starts to become query-shape dependent.

### What to Learn

- **B+ trees deeply** — fanout, fill factor, split/merge, latch coupling.
- **LSM indexes** — SSTables, bloom filters, fence pointers, compaction impact.
- **Secondary indexes** — covering indexes, write amplification costs.
- **Specialized indexes** — bitmap, inverted, R-tree, ANN/Vamana/HNSW basics.
- **Statistics and selectivity** — cardinality estimates and why they fail.

### Resources

- *Database Internals* (Alex Petrov), indexing chapters
- CMU 15-445 lectures on indexing
- DiskANN and modern ANN papers (for vector/latent workloads)

### You're Ready When

Given a query and schema, you can reason about the likely access path and its latency profile before running EXPLAIN.

---

## Level 3 — Transactions, Isolation, and Concurrency Control

Correctness under concurrency is non-negotiable.

### What to Learn

- **ACID in practice** — where guarantees come from in implementation.
- **Isolation levels** — read committed, repeatable read, serializable.
- **Anomalies** — dirty read, non-repeatable read, phantom, write skew.
- **CC strategies** — 2PL, MVCC, OCC, SSI.
- **Locking and deadlock handling** — lock hierarchy, wait-die/wound-wait, detection.

### Resources

- PostgreSQL MVCC + SSI docs
- FoundationDB transaction model docs
- Jepsen analyses (for failure-mode realism)

### You're Ready When

You can identify which anomalies are permitted by a given isolation level and design tests that reproduce them.

---

## Level 4 — Recovery, Replication, and Failure Semantics

Databases are judged by what happens when things break.

### What to Learn

- **Crash recovery** — ARIES-style principles, redo/undo, checkpoints.
- **Replication models** — leader-follower, multi-leader, quorum-based.
- **Consensus basics** — Raft/Paxos at engineering level (not just whiteboard).
- **Durability acknowledgments** — commit index, confirmed reads, read-after-write scopes.
- **Backup/restore and PITR** — snapshots, incremental backup, replay windows.

### Resources

- Raft paper + production postmortems
- PostgreSQL physical/logical replication docs
- CockroachDB architecture docs

### You're Ready When

You can specify exactly what "committed" means in your system and what a client may observe after node crash and leader failover.

---

## Level 5 — Query Processing and Execution Engines

Parsing SQL is easy; executing at scale is hard.

### What to Learn

- **Parsing and planning pipeline** — parser -> logical plan -> physical plan.
- **Cost-based optimization** — join order search, rule/cost hybrid strategies.
- **Execution models** — volcano iterator, vectorized, pipelined operators.
- **Join algorithms** — nested loop, hash join, merge join, index join.
- **Incremental view/subscription maintenance** — delta processing.

### Resources

- CMU 15-445 query processing lectures
- DuckDB papers/docs (vectorized execution)
- Volcano/Cascades optimizer papers

### You're Ready When

You can explain why an optimizer picked a bad plan and what stats/index changes would make it choose a better one.

---

## Level 6 — Distributed Systems for Databases

Single-node correctness is table stakes; distributed correctness is harder.

### What to Learn

- **Sharding/partitioning** — key design, hotspot avoidance, rebalancing.
- **Cross-shard transactions** — 2PC, saga patterns, determinism constraints.
- **Consistency models** — strong, causal, eventual, session guarantees.
- **Clock/time semantics** — NTP drift, hybrid logical clocks, commit timestamps.
- **Geo-distribution tradeoffs** — latency, locality, compliance, failure domains.

### Resources

- Spanner paper (TrueTime)
- Calvin paper (deterministic ordering)
- Jepsen and production incident writeups

### You're Ready When

You can articulate CAP tradeoffs for a real workload and pick a consistency target that matches product requirements.

---

## Level 7 — Performance Engineering and Capacity Planning

Optimization without measurement is superstition.

### What to Learn

- **Benchmark design** — micro vs macro, workload representativeness.
- **Profiling** — CPU, memory, lock contention, I/O, network.
- **Tail latency** — p95/p99/p999 behaviors and queueing effects.
- **Capacity models** — throughput envelopes, headroom, burst behavior.
- **Regression prevention** — performance CI gates, baseline diffs, flamegraphs.

### Resources

- Brendan Gregg's systems performance materials
- RocksDB/Scylla performance tuning docs
- Internal benchmark harness design docs in mature DB projects

### You're Ready When

You can produce a reproducible benchmark, explain regression root cause, and show before/after profiles with confidence.

---

## Level 8 — Operations, Reliability, and Security

Production quality is an engineering discipline, not an afterthought.

### What to Learn

- **Observability** — metrics, logs, traces, health signals, saturation indicators.
- **SLOs and error budgets** — reliability targets tied to user impact.
- **Runbooks and incident response** — escalation paths, mitigation templates.
- **Data security** — authn/authz, row/table-level controls, encryption in transit/at rest.
- **Compliance basics** — backup retention, auditability, data lifecycle policies.

### Resources

- Google's SRE book (reliability patterns)
- NIST and OWASP storage/security guidance
- PostgreSQL and cloud-provider security best practices

### You're Ready When

You can define actionable SLOs, wire alerts to symptoms not noise, and run a controlled recovery drill.

---

## Level 9 — Modern Specializations (Vector, Streaming, AI-Native)

Now extend fundamentals into contemporary data systems.

### What to Learn

- **Vector/ANN retrieval systems** — HNSW, IVF, DiskANN/Vamana, recall/latency tradeoffs.
- **Streaming state stores** — changelog topics, event-time semantics, exactly-once caveats.
- **Hybrid transactional/analytical patterns (HTAP)** — freshness vs contention.
- **GPU-aware data paths** — DMA, pinned memory, batch scheduling.
- **Tensor-native and model-native memory systems** — KV-cache persistence, latent retrieval, memory lifecycle governance.

### Resources

- DiskANN + FAISS docs
- Kafka Streams / Flink state internals
- LLM serving papers (PagedAttention, prefix/KV reuse)

### You're Ready When

You can compare a classical DB, vector DB, and tensor-native memory kernel for the same product requirement and justify the architecture choice.

---

## Level 10 — Engineering Maturity: From Builder to Architect

This level is about long-term engineering leverage.

### What to Learn

- **Design docs that survive implementation** — invariants, failure matrix, migration plans.
- **ATDD for systems code** — acceptance tests first for crash/replay/concurrency edge cases.
- **Change safety** — feature flags, rollout plans, compatibility testing.
- **Code review standards for storage/concurrency code** — proof obligations and explicit assumptions.
- **Technical communication** — explain tradeoffs crisply to product, infra, and security stakeholders.

### Resources

- Internal architecture review templates
- Postmortems from mature database projects
- TLA+/PlusCal intros for critical state-machine modeling

### You're Ready When

You can lead a storage/recovery design from proposal to production rollout with measurable correctness and performance outcomes.

---

## Suggested Practical Track (12-Month Cadence)

1. **Months 1–2:** Levels 0–1 (foundations + storage internals).
2. **Months 3–4:** Levels 2–3 (indexing + transactions/concurrency).
3. **Months 5–6:** Levels 4–5 (recovery/replication + query engine).
4. **Months 7–8:** Level 6 (distributed database patterns).
5. **Months 9–10:** Levels 7–8 (performance + operations/security).
6. **Months 11–12:** Levels 9–10 (modern specializations + architecture leadership).

---

## How This Maps to TardigradeDB

- **Immediate relevance:** Levels 0–4 and 7–8 are directly required to build robust storage, replay, and durability semantics.
- **Near-term relevance:** Levels 5–6 support scalable orchestration and retrieval-serving evolution.
- **Strategic relevance:** Level 9 aligns with TardigradeDB's tensor-native direction; Level 10 is required for safe long-term system evolution.

