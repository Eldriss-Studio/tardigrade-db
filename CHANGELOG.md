# Changelog

All notable changes to TardigradeDB are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project follows semver-ish during the 0.x.y series: minor bumps mark new user-facing surface, patch bumps mark fixes and internal changes. The release procedure lives in [`RELEASE.md`](RELEASE.md).

---

## [Unreleased]

_no changes yet_

## [0.3.4] — 2026-05-19

Hybrid-attention models now work end-to-end. v0.3.3 unblocked retrieval on RecurrentGemma / Jamba / Qwen3-Next / Granite-4 via the K-vector strategy, but `KnowledgePackStore.store()` still crashed on those models because the layer-payload loop assumed every layer carried K/V. Patched.

### Public API

- **`tardigrade_hooks._hidden_states.softmax_layer_count(cfg)`**: new pure-config helper. Counts attention-typed layers via the existing per-layer label probe; returns `num_hidden_layers` for uniform-softmax configs.
- **`KnowledgePackStore.n_softmax_layers`**: new instance attribute, derived once at init. On uniform-softmax models this equals `n_layers` (existing pack-integrity guard's strength is preserved).

### Bug Fixes

- **`KnowledgePackStore.store()`**: no longer raises `AttributeError` on hybrid-attention models. The layer-payload loop now filters via the same `_softmax_layer_payloads` helper calibration already uses; recurrent layers (which have no `.keys`) are skipped at write time.
- **`KnowledgePackStore.retrieve_and_inject()`**: pack-integrity guard now compares `len(layers)` against `n_softmax_layers` instead of `n_layers`, so hybrid packs (which legitimately persist fewer layers than `num_hidden_layers`) are no longer rejected as malformed.

## [0.3.3] — 2026-05-19

Hybrid-attention models now work. Per-model retrieval-key calibration discovers the right encoding empirically; the new K-vector strategy unblocks RecurrentGemma, Qwen3-Next, Jamba, Zamba, and the rest of the hybrid cohort. Plus calibrated per-model layer selection for uniform-softmax models that previously relied on a static 0.67 ratio heuristic.

### General

- **Hybrid-attention models**: tardigrade-db's retrieval engine works on RecurrentGemma, Qwen3-Next, Jamba, Zamba, Falcon-Mamba, Granite-4, MiniMax, Hunyuan-T1, IBM Bamba, Nemotron-H for the first time. Before: every hidden-state layer flatlined at 0–15% top-1 (Michalak & Abreu 2025 explains why — retrieval lives in attention heads, not in the mean-pooled residual stream). After: K-vector encoding at the best softmax layer hits 95–100% top-1 on the bundled 20-fact corpus.

### Public API

- **`tardigrade_hooks.RetrievalKeyStrategy`**: new Strategy ABC with two shipped concrete strategies — `HiddenStateKeyStrategy` (default for uniform-softmax models) and `KVectorKeyStrategy` (default for hybrid models when calibration picks it). `compute(hidden_states, kv, hidden_size) → np.ndarray` takes pre-captured forward-pass outputs so calibration sweeps don't pay the N×M forward-pass cost.
- **`KnowledgePackStore(..., retrieval_key_strategy=...)`**: new optional kwarg. When omitted, the constructor consults the `calibration_registry` (if provided) to pick the right `(strategy, layer)` per model; falls through to `HiddenStateKeyStrategy` at the static-ratio layer if no cached calibration exists. Storage and all four retrieval sites (`store`, `retrieve_and_inject`, `retrieve_with_trace`, `retrieve_and_inject_multi`) route through `self.retrieval_key_strategy.compute()` via a shared `_compute_query_key` helper.
- **`tardigrade_hooks.select_query_layer(...)`**: now returns a `CalibrationResult` with `best_strategy: str` (`"hidden_state"` or `"k_vector"`). The sweep enumerates both strategies × all candidate layers and picks the winner empirically.
- **`LayerScore`**: gains a `strategy: str` field. JSON-cached records without it deserialize with `strategy="hidden_state"` — backwards-compatible.
- **`tardigrade_hooks.CalibrationRegistry`**: persistent JSON-backed store at `~/.tardigrade/calibration.json` (override via `$TARDIGRADE_CALIBRATION_PATH`). One calibration per `model_id`. Atomic writes (temp-file + rename), version-mismatch warnings, corrupted-file tolerance.
- **`tardigrade_hooks.CalibrationStrategy`** + **`LinearSweepStrategy`**: extension point for future sweep algorithms (attention-only, binary search, learned prior).
- **`tardigrade_hooks.constants.DEFAULT_STORE_SALIENCE = 80.0`** and **`CALIBRATION_SALIENCE = 50.0`**: named constants replacing magic numeric literals previously hardcoded in `KnowledgePackStore.store()` and the calibration sweep.

### Behaviour

- **`KnowledgePackStore` default layer selection**: when neither `retrieval_key_strategy` nor `query_layer` nor a `calibration_registry` is supplied, behavior is unchanged from v0.3.2 — `HiddenStateKeyStrategy(int(n_layers × 0.67))`. Calibration is opt-in; existing consumers see zero change.
- **Calibration tiebreak prefers depth.** When multiple layers tie at the corpus ceiling, the deepest tied layer wins (was: first encountered). Shallow layers — especially the embedding — can ace small synthetic corpora purely on surface-token discrimination; deeper layers encode semantic meaning that survives paraphrasing.
- **Bundled calibration corpus** uses paraphrased queries that share proper-name entities with their facts but rewrite the surrounding language. Prevents the embedding layer from winning calibration ties purely on full-sentence surface-token overlap.
- **Calibration progress logging**: `LinearSweepStrategy.run()` emits one INFO-level line per forward-pass milestone and one per `(strategy, layer)` evaluation. Quiet by default (logger at `WARNING`); enable via `logging.getLogger("tardigrade_hooks.calibrate").setLevel(logging.INFO)`.
- **`select_query_layer(model, tokenizer=None, ...)`** (hosted-API mode): no sweep run, no model load; returns a `CalibrationResult` whose `best_layer` matches the static default ratio and whose `scores` is empty. Lets consumers with no local tokenizer call the Factory uniformly without branching.

### Bug Fixes

- **BF16 in `KnowledgePackStore.store()`** no longer crashes the K/V numpy conversion. The v0.3.2 release notes claimed all four BF16 → numpy crash sites were fixed; the `.store()` path was missed and only the retrieval/inject paths were patched. Added `.float()` before `.cpu().numpy()` in the missed site.
- **Cache dtype mismatch on reinject**: reconstructed K/V tensors now cast to `next(model.parameters()).dtype` so a BF16-loaded model's `scaled_dot_product_attention` doesn't reject the FP32 cache with `"query, key, value to have the same dtype"`.


## [0.3.2] — 2026-05-19

Patch release. Two user-facing additions: a real `tardigrade chat` CLI that exercises the engine end-to-end, and a pluggable chat-template Adapter that unblocks every modern HuggingFace instruction-tuned tokenizer (Llama-3, Gemma-2, Mistral, Phi-3.5, …), not just Qwen3.

### General

- **`tardigrade chat` is a real product you can pick up and use.** Persistent backend stores into a real `tardigrade_db.Engine` at `~/.tardigrade/chat/engine` by default; memories survive process restart. Each turn prints `recalled N memories in X ms` inline (measured 0.36–2.48 ms on the smoke run). Multi-persona subjectivity: `--persona NAME` selects, `/switch NAME` changes mid-session, `/personas` lists, and memories don't cross persona boundaries (engine-enforced via owner scoping). A `last_persona.txt` pointer auto-resumes the last persona when run with no flags. Three backends behind the same REPL: `persistent` (default — real engine, echo responses), `memory` (in-process, for tests), `qwen` (real LLM, lazy-imported).

### Public API

- **`tardigrade chat`** subcommand: `tardigrade chat [--persona NAME] [--backend persistent|memory|qwen]`. Slash commands: `/memories`, `/forget <text>`, `/personas`, `/switch <name>`, `/stats`, `/exit`, `/help`.
- **`KnowledgePackStore(..., adapter=ChatTemplateAdapter | None)`**: new optional kwarg. When omitted, a Factory probes the tokenizer with a system-only `apply_chat_template` and picks the right adapter. Existing Qwen3 callers see no behaviour change.
- **`SequentialRecomputeComposer(..., adapter=...)`**: same kwarg, same Factory default.
- **`tardigrade_hooks.ChatTemplateAdapter`**, **`UserMessageAdapter`**, **`LegacySystemAdapter`**, **`select_chat_template_adapter`**: new exports for consumers who want to pin an adapter explicitly across sessions.
- **`tardigrade_chat`** package (`MemoryChatSession`, `ChatBackend` ABC, `InMemoryBackend`, `PersistentBackend`, `Repl`): public surface for embedding the chat REPL outside the CLI.

### Behaviour

- **Strict-template tokenizers** (Llama-3, Gemma-2, Mistral-Instruct, Phi-3.5, …) previously raised `jinja2.TemplateError: No user query found in messages` on store/retrieve. The default `UserMessageAdapter` wraps facts as user-role messages with an empty assistant separator, accepted by every modern instruction-tuned template.
- **Cross-adapter retrieval on a single pack**: unsupported. KV state on disk encodes the wrap used at store time — store and retrieve must use the same adapter, which the Factory enforces de facto per session. Migrating an existing `tardigrade_data/` to a different adapter requires rebuilding the affected packs.

### Bug Fixes

- BF16 hidden states no longer crash `.cpu().numpy()` — `kp_injector` now casts to FP32 before the NumPy conversion, fixing a `TypeError: Got unsupported ScalarType BFloat16` on models that return BF16 even when loaded as FP32.

### Known Limitations

- Hybrid linear/standard attention architectures (e.g. Qwen3.5's `LinearAttentionLayer`) remain incompatible with the KV-capture step — the adapter fixes the chat-template error but `kv.layers[i].keys` still raises `AttributeError` on a linear-attention layer. Separate library concern.

## [0.3.1] — 2026-05-17

Patch release. Polish work plus an honest fix to the latency-bench artifact. No breaking changes.

### Added

- **Typed errors with diagnostic codes.** `TardigradeError` now derives `miette::Diagnostic` alongside `thiserror::Error`. Every variant gets a stable `#[diagnostic(code = "tdb::<area>::<name>")]` and a one-sentence `#[help(...)]` annotation that surfaces in `cargo doc` and miette's pretty output. Three new context-rich variants added alongside the existing string-bag ones: `EmptyCorpus { owner }` (distinguishes "no results" from "queried something that never had data"), `NoQueryKey { hint }` (surfaces query-key derivation failures), `FlushFailed { failed_offset, detail }` (structured failure mode for the streaming write buffer). Existing variants keep their `Display` shape verbatim — code that grep'd "cell not found: 42" or pattern-matched on variants from earlier versions still works (`5b0e230`).

### Changed

- **Simulated bench comparators removed.** `experiments/shared_llm_latency.py` previously reported cold-path and text-RAG baseline latencies, both via `time.sleep()` busy-waits using guessed values (~50 ms for Qwen3-0.6B prefill, ~50 ms for OpenAI embeddings, ~5 ms for a vector-DB lookup). None of those numbers were measured against real systems. Retracted both columns; the script now reports only the warm-path engine retrieval measurement (which is real). `docs/positioning/latency_first.md` updated with a retraction note (`2452640`).
- **Pre-1.0 SemVer rule corrected in `RELEASE.md`.** The previous rule said "minor bump for new user-facing surface, patch bump for fixes" — wrong for the Cargo / Python-packaging convention where `^0.2` means `>=0.2.0, <0.3.0` and `0.X.0` signals incompatibility. Corrected to: patch for anything additive or backward-compatible (including milestone-sized additive batches), minor only for actual breaking changes. Adds an explicit "common mistake to avoid" callout. By the corrected rule, v0.2.0 and v0.3.0 were both over-bumped — but published versions are immutable, so the pattern corrects from here (`0f2e968`).
- **Plan-phase identifiers stripped from bench code, tests, and the positioning doc.** Follow-up to the foundation-phase strip in v0.2.0 era: removed older Phase NN, Track A/B, Slice X, Step Na-LN.M codes that survived in `python/tdb_bench/*`, ~40 test files, and `docs/positioning/latency_first.md`. Dated audit citations preserved (e.g., "bench audit 2026-05-16") since dates are stable historical anchors; plan codes rot when plans are rewritten. Doc/experiment write-ups under `docs/experiments/*` and `docs/refs/*` left untouched — the phase codes there are the historical record (`032a6ea`).

## [0.3.0] — 2026-05-17

Scheduler release. Adds durable scheduled actions (the last big foundation primitive on the roadmap) plus a batch of post-v0.2.0 release-hygiene fixes.

### Added

- **Durable action scheduler.** `Engine::schedule(fires_at, action) -> ScheduledId`, `cancel_scheduled(id)`, `list_scheduled()`, `fire_due_scheduled(now)`. Initial built-in action is `ScheduledAction::EvictDraft { owner, threshold }`; the enum is serde-tagged so future variants extend the wire format. Schedule persists to a JSON sidecar at `<engine_dir>/scheduled_actions.json` and survives engine restart. Delivery is **at-least-once + idempotent actions = fire-once-in-effect across crashes** — entries stay durable until execute returns `Ok`, so a crash mid-fire replays the action safely. Background `MaintenanceWorker` fires due actions on each tick; consumers can also call `fire_due_scheduled()` directly for immediate firing after a known event. Documented contract on `ScheduledAction` so future variants must remain idempotent (`4bd9fe4`).
- **PyO3 bindings for the scheduler.** `engine.schedule_evict_draft(fires_at_unix_secs, owner, threshold)`, `cancel_scheduled(id)`, `list_scheduled() -> list[dict]`, `fire_due_scheduled() -> int`. Dict shape is forward-compatible with future action types.
- **`[llama]` optional extra.** `pip install tardigrade-db[llama]` pulls in `llama-cpp-python>=0.2` for consumers using the local-GGUF embedding path (`b336aa9`).

### Changed

- **`examples/llama_memory_test.py` is now portable.** Replaced the hardcoded macOS Ollama path with a `TARDIGRADE_LLAMA_GGUF` environment variable. Fails fast with actionable errors when the env var is unset, the file doesn't exist, or `llama-cpp-python` isn't installed. The `llama_cpp` import is deferred so `info` / usage paths run without the package installed (`b336aa9`).
- **Release procedure docs.** RELEASE.md now requires: a fact-check pass against the published-artifact list before writing release notes; and flowing-markdown body (one paragraph per line, no hard wrap) so the rendered GitHub release page reads cleanly. Both rules added after v0.2.0 shipped with a wrong "single-platform wheel" claim and hard-wrapped body text (`955c2c2`, `d1c9b8c`).

### Fixed

- **Scheduler firing-model docstring.** Removed a stale "at-most-once" line that contradicted the at-least-once design documented two paragraphs lower (`dfcb9f1`).

### Notes

- Reflowed prose paragraphs in `RELEASE.md`, `CHANGELOG.md` ([0.2.0] block), `docs/guide/consumers.md`, and `examples/nodejs_consumer/README.md` from fixed-column hard wrap to one-paragraph-per-line. Cleaner GitHub rendering and lower diff churn on future edits.

## [0.2.0] — 2026-05-17

Foundation-completion release. Adds the surface non-Python consumers need to plug in (CLI, HTTP/REST bridge, Node.js example), the persistence primitives consumers need to ship real workloads (owner registry, portable snapshot/restore, labeled checkpoints, streaming write buffer, synchronous sweep trigger), and ergonomic upgrades to the Python facade (builder pattern, encode_query convenience, multi-agent shared-engine support). Two runnable demos validate the whole surface.

### Added

- **CLI** (`tardigrade init|store|query|status|consolidate`): Strategy + Template Method per subcommand, `python-basic` and `rust-basic` starter templates (`0282db2`).
- **Owner registry** on `Engine`: `list_owners`, `owner_exists`, `delete_owner` with cascade (`36680e3`).
- **Portable snapshot/restore** (`Engine::snapshot`, `Engine::restore_from`): tar archive with magic `"tdb!"`, format_version, codec ids (`q4` / `top5avg`), SHA-256 payload digest, pack/owner stats. Bounded reads cap manifest at 1 MiB and payload entries at 16 GiB. Foot-gun guard refuses `out_path` inside `engine_dir` (`dbf77a0`).
- **Streaming write buffer** (`Engine::open_with_write_buffer`): coalesces `mem_write_pack` calls behind a single `append_batch` fsync. Pack ids return synchronously; visibility deferred to `flush_buffer()` or auto-flush at threshold (`99c1482`).
- **Synchronous governance trigger** (`Engine::sweep_now(hours, threshold) -> usize`): runs one tick of the maintenance worker inline (`46286ee`).
- **Labeled checkpoints** (`CheckpointRepository`, Repository pattern): label-scoped slots `<root>/<label>/<NNNN>.tar`, `save_from` / `list` / `latest` / `restore_latest` (`1198c9a`).
- **HTTP/REST bridge** (`python/tardigrade_http/`): FastAPI Adapter exposing `POST /mem/store`, `POST /mem/query`, `GET /mem/owners`, `GET /mem/status`, `POST /mem/save`, `POST /mem/restore`. RFC 7807 `application/problem+json` errors. Checked-in OpenAPI contract at `python/tardigrade_http/schema.yaml` with drift-guard test (`4fe00b5`).
- **Node.js consumer + generated TypeScript types + consumer guide** (`examples/nodejs_consumer/index.mjs`, `python/tardigrade_http/types.ts`, `docs/guide/consumers.md`): dependency-free Node 22+ script exercises the foundation surface end-to-end; TypeScript types regenerated from the OpenAPI schema; three documented integration patterns (`6658eed`).
- **Client builder** (`TardigradeClient.builder()`): fluent `with_engine_dir` / `with_engine` (mutually exclusive — the latter is the multi-agent shared-engine pattern) / `with_owner` / `with_tokenizer` / `with_capture_fn` / `with_vamana_threshold`. `BuilderIncomplete` raised on ambiguous or incomplete state (`6232fe5`, `c1fb890`).
- **`TardigradeClient.encode_query(text) -> ndarray`**: returns the raw retrieval key without a query pass (`c92d584`).
- **Multi-agent demo** (`examples/multi_agent_demo.py`): three agents share one engine, exercise every foundation primitive as an acceptance gate (`c1fb890`).
- **Interactive-fiction demo** (`examples/if_demo.py`): two NPCs with distinct personas observe the same scripted events, store divergent memories under owner scoping (`6f00354`).
- **Pre-commit lint gate** (`scripts/lint_lazy_imports.py`): catches CI failures caused by import-time loading of the native extension (`0e84920`).
- **Bench instrumentation**: per-(dataset, category) score breakdown, retrieval-only headline (Recall@k / NDCG@k), answer-text metric sidecar, shared-LLM latency benchmark scaffold (warm-path real; cold-path and text-RAG comparators simulated and tracked as a follow-up).

### Changed

- `TardigradeClient.__init__` now accepts `engine=` as an alternative to `db_path` (mutually exclusive). Enables one engine + N clients in multi-agent workloads (`c1fb890`).
- `TardigradeClient.query()` routes through `encode_query()`, unifying the encode-then-read code path.
- `tardigrade_hooks` package switched to PEP 562 `__getattr__` for `TardigradeClient` so importing `tardigrade_hooks.constants` no longer loads the native extension (`0e84920`).
- Plan-phase identifiers stripped from code, public docstrings, demo print lines, and `CLAUDE.md` (`61e7d09`).

### Fixed

- Node.js consumer no longer double-reads the response body (undici `"Body has already been read"` guard); single `unwrap(path, response)` helper handles both OK and error branches.
- TardigradeAdapter's evidence path returns chunk text rather than the parent item's full context (`64be483`).
- TardigradeAdapter dedups retrieved evidence by chunk text (`31e711e`).
- Per-instance lock around `CrossEncoder.predict` removes the parallel-score drift observed under multi-threaded eval (`bfd248a`).
- LongMemEval bench-prep emits per-turn gold (not per-session) and raw turn text (not `[date]`-wrapped) (`f0fb531`, `e443dd8`).

### Notes

- A bench audit retracted the historical "68.2% LoCoMo / 90.9% LongMemEval" baseline numbers (they measured the lexical fallback adapter on a corpus corrupted by a dataset-prep bug); honest native-engine number on the clean dataset is ~36% R@1 at 50-item scale. Synthetic-corpus measurements (100% recall at 5K memories, vague-query refinement, KV injection on gibberish facts) are unaffected.
- Positioning pivoted to **latency / footprint / KV-native API** (sub-millisecond p99 retrieval at 5K cells, ~751 B per cell on disk) — see `docs/positioning/latency_first.md`.

## [0.1.5] — 2026-05-02

Last patch in the 0.1.x line. The features below were under `[Unreleased]` at tag time; recorded here so the history is not lost.

### Added — DX & Repo Quality
- First-class crate-level `//!` documentation across all 6 library crates, following tokio/serde/crossbeam quality bar
- `Documentation Standards` section added to `CLAUDE.md` and `AGENTS.md`
- `just ci` now mirrors GitHub CI exactly: fmt → lint → typos → test → deny → doc
- `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` propagated to all lint/test recipes (Python 3.14 compatibility)
- `lefthook` replaces manual `.git/hooks/pre-push`; pre-commit runs fmt/clippy/typos in parallel
- `LICENSE` (MIT), `CHANGELOG.md`, `CONTRIBUTING.md`, GitHub issue templates, PR checklist template
- `crates/tdb-engine/examples/basic_usage.rs` — runnable Rust demo

### Added — Python Bindings
- `tdb-python` crate: PyO3 bindings exposing `Engine` and `ReadResult` as Python classes
- `mem_write` / `mem_read` accepting NumPy `float32` arrays via zero-copy slice access
- `cell_importance`, `cell_tier`, `cell_count`, `trace_ancestors`, `has_vamana`, `advance_days` Python methods
- `maturin` build integration (`PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop`)

### Added — Causal Trace Graph + WAL
- `tdb-index`: `TraceGraph` — directed episodic graph with `CausedBy`, `Follows`, `Supports`, `Contradicts` edge types
- `Wal` — append-only binary write-ahead log for `TraceGraph` mutations; 26-byte fixed-size records
- WAL replay on `Engine::open`; crash recovery restores causal graph from durable log
- `Engine::trace_ancestors` — transitive ancestor queries over causal edges
- `Engine::mem_write` accepts optional `parent_cell_id` for causal edge recording

### Added — Vamana ANN Index
- `tdb-index`: `VamanaIndex` — DiskANN-style single-layer graph for cold-path ANN retrieval
- Medoid seeding for greedy beam-search queries; `max_degree` configurable out-degree
- Automatic Vamana activation at 10,000 cells per engine instance
- `Engine::has_vamana` — reflects whether the graph index is active
- Benchmarks: `vamana_build`, `vamana_query` in `tdb-index`

### Added — Governance — Recency Decay + Tier Promotion
- `tdb-governance`: `recency_decay(days)` — exponential decay r = exp(−Δt/τ), τ = 30 days
- Recency multiplier applied to retrieval scores during `mem_read`
- `Engine::advance_days` — simulate time passage for testing and reproducible benchmarks
- `Engine::cell_tier` — expose current maturity tier from Python and integration tests

### Added — Governance — Importance Scoring
- `tdb-governance`: `ImportanceScorer` — ι ∈ [0,100]; +3 on access, +5 on update, ×0.995 daily decay
- `TierStateMachine` — three tiers (Draft/Validated/Core) with hysteresis gaps (30 and 25 pts)
- On-access and on-update governance hooks wired into `Engine::mem_read` / `Engine::mem_write`
- `Engine::cell_importance` — query current importance score by cell ID

### Added — Semantic Lookaside Buffer (SLB)
- `tdb-retrieval`: `SemanticLookasideBuffer` — LRU hot cache with INT8 scalar quantization
- Sub-5μs attention-score hits for recently-accessed cells (NEON SDOT on Apple Silicon)
- SLB inserts on every `mem_write`; checked first in the `mem_read` pipeline

### Added — Brute-Force Retrieval + Attention Scoring
- `tdb-retrieval`: `BruteForceRetriever` — exact scaled dot-product attention (q · k / √d_k)
- Owner-filtered partitions for per-agent isolation
- `BruteForceRetriever::query` returns top-k by score, supporting `mem_read`

### Added — Block Pool + Q4 Quantization
- `tdb-storage`: `BlockPool` — append-only mmap'd segment files (64 MiB default) with `fsync` durability
- Group-wise Q4 (GGML Q4_0) quantization: 8 × f32 → 4 × u8 + f16 scale, ~5.3x compression
- `SynapticStore` — secondary per-owner lookup index for O(1) cell retrieval by owner ID
- Binary record format: `magic(4) | version(1) | dim(4) | cell_id(8) | ... | key_q4 | val_q4`

### Added — Core Types
- `tdb-core`: `MemoryCell` + `MemoryCellBuilder` — KV-cache tensor pair with metadata
- `Tier` enum (Draft/Validated/Core) and `CellMeta` (tier, importance, timestamps, token span)
- `TardigradeError` + `Result<T>` — shared error type across all crates
- `SynapticBankEntry` — lightweight index record for owner-keyed lookups

### Added — Workspace Scaffold
- Cargo workspace: `tdb-core`, `tdb-storage`, `tdb-retrieval`, `tdb-index`, `tdb-governance`, `tdb-engine`, `tdb-python`
- `justfile` with `fmt`, `lint`, `test`, `bench`, `coverage`, `doc`, `ci`, `setup` recipes
- `lefthook.yml` — pre-commit (fmt/clippy/typos parallel) + pre-push (`just ci`)
- `cargo-deny` configuration (license allow-list, advisory deny, wildcard ban)
- GitHub Actions CI: check+lint, test matrix (stable/nightly), coverage, MSRV, doc deploy, doc-check
- `nextest` CI profile with retry on failure and extended timeouts

### Added — Project Foundation
- Initial repository structure and `CLAUDE.md` with Aeon Architecture specification
- Research corpus: technical design document, spec, competitor analysis, industry analysis
- Spec corrections based on external research: custom mmap arena (not safetensors), Vamana graph (not HNSW), Rust+CUDA (not pure Rust)
