# Changelog

All notable changes to TardigradeDB are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project follows semver-ish during the 0.x.y series: minor bumps mark new user-facing surface, patch bumps mark fixes and internal changes. The release procedure lives in [`RELEASE.md`](RELEASE.md).

---

## [Unreleased]

### Public API

- **`CalibrationResult.best_score()`**: returns the `LayerScore` for `(best_strategy, best_layer)`. Use this instead of `result.scores[result.best_layer]` — `scores` is enumeration-ordered, not keyed by layer index, so positional indexing silently returns the wrong strategy's entry whenever multi-strategy enumeration runs (the recurrentgemma hybrid case).

## [0.7.6] — 2026-05-22

Finishes the vLLM 0.19 connector port that started in v0.7.5 — two save-path crashes now eliminated.

### Bug Fixes

- **vLLM connector save path**: under prefix-cache hits, was capturing only the new-token KV slice instead of the full request KV — the on-disk pack lost everything the cache had already served. Now threads `cumulative_seq_len` from `attn_metadata.seq_lens` through the worker so the saved slice covers the entire request.
- **vLLM connector matched-token count**: `get_num_new_matched_tokens` could return a value larger than the request's remaining token budget, hitting `assert num_computed_tokens <= request.num_tokens` in vLLM's scheduler on short queries after the save-path fix above. Now clamps to `len(prompt_ids) - num_computed_tokens - 1`.

### Test Infrastructure

- **`tests/python/_gpu_test_utils.do_gpu_cleanup()`**: canonical helper for releasing GPU model fixtures (`del + gc.collect() + torch.cuda.empty_cache()`). Module-scoped `llm` fixtures in `test_vllm_integration.py` and `test_vllm_prefix_e2e.py` use it via `try/yield/finally`.
- **`test_vllm_cross_session.py`**: `gpu_memory_utilization` lowered to `0.3` for the subprocess vLLM init, leaving headroom for whatever VRAM the parent pytest process is still holding from earlier modules — `empty_cache()` returns blocks to PyTorch's allocator pool, not to the CUDA driver, so only process exit fully releases. Fully eliminating cross-module VRAM leaks requires running heavy-GPU test files in separate `pytest` invocations in CI; tracked for follow-up.

## [0.7.5] — 2026-05-22

vLLM 0.19 connector compatibility restored, plus a wheel matrix cleanup that retires three never-validated Python ABIs from PyPI.

### Bug Fixes

- **vLLM 0.19 connector**: three stacked API drifts since vLLM 0.18. `update_state_after_alloc(blocks=…)` now receives a `KVCacheBlocks` dataclass instead of a list of block IDs — was crashing on `TypeError: unhashable type`. `ForwardContext.kv_caches` was removed; the connector now stashes per-layer caches via the new `register_kv_caches` lifecycle hook. The connector also advertised async-load semantics it never delivered — vLLM 0.19 enforced that contract by waiting forever for a completion signal. Returns sync semantics now. Result: 5 of 7 vLLM integration tests pass reliably (up from 0 of 7 on 0.7.4); test suite wall time 25-min hang → 45s. Two remaining vLLM tests fail on different bug classes (silent injection skip on the worker, fixture isolation) tracked for follow-up.

### Compatibility

- **Wheel matrix pinned**: `pip install tardigrade-db` no longer ships wheels for Python 3.14 RC, Python 3.15 alpha, or PyPy 3.11. v0.7.4 had shipped those by accident — `maturin --find-interpreter` discovered them on the GitHub runner without our ever validating against them. Consumers on those interpreters now fall through to `sdist` build-from-source. Validated set: cp310, cp311, cp312, cp313 (abi3) + cp313t (free-threaded, separate wheel).

## [0.7.4] — 2026-05-22

Free-threaded Python 3.13t (PEP 703) support — a `cp313-cp313t` wheel now ships alongside the abi3 wheels.

### Compatibility

- **Free-threaded Python 3.13t**: `pip install tardigrade-db` under `python3.13t` now resolves to a `cp313-cp313t` wheel built against the no-GIL CPython interpreter. Under 3.10 / 3.11 / 3.12 the abi3 wheel keeps serving as before. No consumer code changes. Lifts the GIL ceiling on concurrent reads: GIL Python plateaus at **~42k qps** for single-call workloads at 8+ threads; the cp313t wheel keeps scaling, reaching **~86k qps** at 16 threads on the proof-of-concept host. Full measurement record: [`docs/experiments/2026-05-22-freethreaded-python-proof.md`](docs/experiments/2026-05-22-freethreaded-python-proof.md).

## [0.7.3] — 2026-05-21

Concurrent pack reads. Multiple Python threads can now call `mem_read_pack` against a shared engine in parallel — **1.74× aggregate throughput at 8 threads** measured on a 192-pack corpus (33,682 queries/sec vs 19,309 single-threaded). Single-thread drops ~10% from RwLock acquire overhead vs the prior Mutex; the win arrives from real parallel reads.

### Concurrency

- **`mem_read_pack`** and the trace-boost variants: lock-free across Python threads. Hold a Python `Engine` reference in N threads and call concurrently — engine work parallelizes. No API change; the win is purely structural.
- **`mem_read_pack_batch`** and **`mem_read_multi_layer`**: same. One read guard per call; multiple batches across threads run in parallel.
- **`mem_read`** (the older direct-cell API): stays serialized. It performs SLB warm-promotion on every returned cell, which is a structural mutation. Consumers that need lock-free concurrent reads must use the pack-read API.

### Performance

- **8-thread aggregate throughput**: 15,784 qps → 33,682 qps (`examples/concurrent_reads_demo.py`, 192 packs, RTX 3070 Ti host).
- **Single-thread throughput**: 21,331 qps → 19,309 qps (RwLock acquire overhead vs Mutex; honest tradeoff for the parallel-read win).
- **8-thread speedup vs single-thread**: 0.71× → 1.74×.

### Public API (Rust)

- **`Engine::mem_read_pack`**, `mem_read_multi_layer`, `mem_read_pack_with_trace_boost`, `mem_read_pack_with_trace_boost_and_follow`, `load_pack_by_id`: now take `&self`. Rust consumers can wrap the engine in `Arc<RwLock<>>` and get parallel reads through `.read()`.
- **`Retriever::query`** / **`Retriever::query_with_source`**: take `&self`. SLB LRU counters moved to `AtomicU64`; `PerTokenRetriever.last_scored_cell_count` moved to `AtomicUsize`. Storage choice documented at the trait + module level.
- **`MaintenanceWorker::start`**: now takes `Arc<RwLock<Engine>>` (was `Arc<Mutex<Engine>>`). Internally acquires the write guard for sweep + compaction.

### New Tests

- **`examples/concurrent_reads_demo.py`**: N reader threads, throughput + consistency reporting. Doubles as the comparator that measured this release's speedup.
- **SLB property test**: 8 threads × 50 queries against a quiescent SLB return identical `(cell_id, score)` tuples.
- **Engine acceptance test**: 8 threads × 25 queries against a shared seeded engine, asserts the retrieved pack-id set matches a quiescent baseline on every call.

## [0.7.2] — 2026-05-21

Warm-tier writes now take 2.66× less disk space; README + all eight `docs/guide/` pages rewritten through the `/write-docs` craft layer; publish workflow guarded against version/tag drift.

### Storage

- **Warm-tier compression**: Validated/Core cells now take **2.66× less disk space**. Measured at 1024-dim KV: 1372 B/cell → 516 B/cell, 62 % saved on the warm tail. Method and caveats: [`docs/experiments/2026-05-21-warm-tier-codec.md`](docs/experiments/2026-05-21-warm-tier-codec.md).
- **Codec is tier-gated**: zstd level 3 on Validated/Core writes, raw Q4 on Draft. Per-cell dispatch — mixed-tier batches land in one segment.
- **No Python API change.** The codec is transparent end-to-end.
- **Segment format bumped to v2**: one new `codec_id` byte per record. Legacy v1 segments remain readable as `UniformQ4`.
- **New module** `tdb_storage::compression::CompressionCodec` (`UniformQ4 = 0`, `ZstdQ4 = 1`). Strategy seat for a future variable-bitrate-per-channel codec (~6–8× target) with no further on-disk changes.

### Documentation

- **README**: revised seven sections through the new `/write-docs` skill's
  craft layer (curse of knowledge, classic style, context-where-it-lands).
  The opening pitch leads with the reader's pain (three round trips per
  fact recalled, context window fills as the agent learns) before naming
  the implementation. The hero example pre-announces the CUDA prereq and
  points CPU users at `examples/e2e_demo.py`. The "Why TardigradeDB?"
  section spells out what a "text round-trip" actually costs instead of
  assuming the reader has already named the inefficiency. The "Isn't this
  just a KV cache?" FAQ entry was a 60-word run-on listing six
  distinctions in one sentence; broken into two paragraphs that each
  carry one idea. The "Why 'Tardigrade'?" entry traded four brand-pitch
  bullets for a paragraph that actually tells the tardigrade-biology
  parallel. The Project Status second bullet now leads with the verdict
  ("HuggingFace direct injection works today; vLLM is partial because
  KV Connector v1 is prefix-cache only") instead of with the framing.
  Quick Start now opens with "Is this a Rust library or a Python
  library?" — the integration story (Rust engine, Python API, why the
  split, no Rust crate to depend on from a third-party Rust app today)
  before the install commands. The subsection headings switched from
  audience labels ("Python users", "Rust contributors") to intent labels
  ("Using TardigradeDB from Python", "Building TardigradeDB from source")
  so a reader who wants to *use* it from Rust isn't accidentally routed
  into the contributor path.
- **All eight `docs/guide/` pages** revised through the same `/write-docs`
  craft layer, in impact-over-effort order. The recurring fix across
  the tree was the same one the README went through: opening sentences
  now name the reader's situation rather than the page's topic, jargon
  gets defined inline at first use, critical caveats moved up where
  readers can't miss them, and procedure-as-bullet-list sections that
  flattened causality became prose. Specifically: `calibration.md` got
  three inline jargon expansions (mean-pool vs K projection, "flatline"
  → explicit "produce identical scores", incomplete citation completed).
  `consumers.md` opening rewrote to reader-situation, owner-scoping
  explained before Pattern 1 uses it, the cryptic "three pluggable
  extension points" line moved to a See-Also pointer. `mcp-setup.md`
  surfaced the text-delivery-token-cost tradeoff at the top instead
  of hiding it at line 5, and explains `PYTHONPATH` and config-file
  locations properly. `quickstart.md` rebuilt around the "MCP path
  vs Python path" decision the reader actually arrives with, the
  `kv_capture_fn` stub warning moved to a precondition rather than a
  buried mid-page note, the RLS algorithm prose-ified. `knowledge-pack-store.md`
  opening rewrote to reader-situation, Q4 quantization and retrieval-key
  strategy defined inline, the hybrid-attention section reframed by
  outcome instead of implementation jargon (`_softmax_layer_payloads`,
  `Griffin 3:1 spacing`). `concepts.md` opens with what a KV cache
  actually is before using the term, the RLS retraction moved from
  buried-in-paragraph to a warning callout at the top of the section,
  the 5-step algorithm became a paragraph. `vllm-setup.md` got the
  biggest restructure: a "what this integration does and doesn't do
  today" subsection at the top that honestly names the prefix-cache-vs-
  cross-prompt-injection distinction (matching the README's Project
  Status framing rather than the old guide's overpromising), and the
  status section reorganised into three honest categories (works
  end-to-end, partial, not supported) instead of a flat feature table.
  `python-api.md` got an opening that distinguishes it from the
  quickstart, then surgical inline definitions for `layer_payloads`,
  `kv_capture_fn`, `query_layer` heuristic, `confidence_threshold`
  ratio, `MemoryCellHandle`, and `mark-sweep GC`.

### Internal

- **`publish.yml` guarded against version/tag drift.** Three new
  asserts in the publish workflow that catch the silent failures
  that lost v0.6.0 and v0.7.0 from PyPI: Cargo.toml version must
  match the release tag (fail-fast, before the matrix build),
  every built wheel filename's version must match the tag, and
  the sdist filename's version must match the tag. The previous
  failure mode (workflow reports success but PyPI returns 400
  File-already-exists against the previous version) is now a
  loud build-stage error with an actionable hint.

### Notes

- **v0.6.0, v0.7.0, v0.7.1 backfilled to PyPI.** Those releases
  were tagged but never installable — the original publish
  workflows uploaded wheels carrying the previous version's
  name (a pyproject.toml-vs-Cargo.toml drift on v0.6.0, a
  GitHub-Release-vs-tag-commit drift on v0.7.0 and v0.7.1).
  All three are now live (`pip install tardigrade-db==0.7.0`
  works; same for 0.6.0 and 0.7.1). The v0.6.0 tag was
  force-pushed to a fix-up commit that corrects pyproject.toml;
  the original v0.6.0 commit (`1e78d691`) remains in history
  via main's branch lineage.

## [0.7.1] — 2026-05-21

### Documentation

- **README**: restructured from 756 lines to ~229 lines following the
  modern OSS landing-page pattern (SpacetimeDB / Tokio / RocksDB /
  Qdrant). Above-the-fold content is now title + badges + status +
  pitch + hero example. Honest LoCoMo retraction lives in the
  Project Status section with a one-sentence link to the audit doc,
  not as a top-of-page wall. FAQ added at the end (name origin,
  "isn't this just a KV cache?", compact comparison table, design
  principles).
- **New `docs/` tree**: `architecture.md`, `positioning.md`,
  `research-log.md`, `roadmap.md`, `guide/calibration.md`,
  `guide/knowledge-pack-store.md`, `docs/README.md` (index).
  Detailed content moved out of the README so it stops scaling
  with the project's history.
- **CONTRIBUTING.md**: expanded with Requirements, full
  Rust + Python test recipes, Bench V1 harness commands, CI gates
  table, Reliability & Consistency Contracts, and a "Working with
  Claude" section pointing at `CLAUDE.md`.

### Community Standards

- **`CODE_OF_CONDUCT.md`** (new): Contributor Covenant 2.1, contact
  via the repo's GitHub private security advisory flow.
- **`SECURITY.md`** (new): supported versions (0.7.x), private
  reporting flow, scope (memory safety / persistence / owner
  isolation / PyO3 boundaries), 30-day disclosure timeline.
- **`CITATION.cff`** (new): CFF v1.2.0 — GitHub renders a "Cite this
  repository" button.
- **`.github/ISSUE_TEMPLATE/config.yml`** (new): blank issues
  disabled; security reports routed to private advisories.
- **`.github/pull_request_template.md`**: added Commit-hygiene
  section (gitmoji + CHANGELOG entry checklist).

### Public API

- **`tardigrade_hooks.KnowledgePackStore`**: now in the public
  package namespace via lazy `__getattr__` (matches the same
  pattern used by `TardigradeClient`). Previously only reachable
  via `from tardigrade_hooks.kp_injector import KnowledgePackStore`.
- **`tardigrade_vllm.retrieval_key.PROJECTED_EMBEDDING`**: removed
  (was orphaned after the v0.7.0 deprecation sweep deleted
  `get_strategy` and `_STRATEGIES`).

### Internal

- **AGENTS.md**: status block refreshed to v0.7.0 reality; old
  "Phases 0–11 complete. 663 tests passing" line removed in favour
  of a pointer to `CLAUDE.md` as the canonical source.
- **`.github/dependabot.yml`**: commit-message template updated to
  match the project's gitmoji + conventional-commits convention.

## [0.7.0] — 2026-05-20

Python ↔ Rust boundary migration complete. Six new engine APIs lift
state and computation out of Python; one deprecation sweep clears
transition scaffolding the project no longer needs.

### Public API

- **`Engine.compute_retrieval_key(token_ids, strategy)`**: new — returns
  the retrieval key for a token sequence via `"last_token"`,
  `"mean_pool"`, or `"projected"`. Pair with
  `Engine.load_embedding_table(weights)` to push the token-embedding
  table to the engine once.
- **`Engine.set_projection_matrix(matrix)`**: new — install the
  projection used by `"projected"` for `hidden_size != kv_dim` models.
- **`tardigrade_db.flat_to_paged(flat, kv_heads, head_dim, block_size)`**
  and **`paged_to_flat(k, v, seq_len, kv_heads, head_dim)`**: new
  module-level functions for vLLM KV block-layout conversion.
  Stateless — no engine instance required.
- **`tardigrade_db.flat_to_paged_torch(tensor, ...)`**: new — accepts a
  `torch.Tensor` directly via `data_ptr()`. No `tch-rs`, no libtorch
  link; the default wheel includes the function and the consumer's
  own torch handles the C++ side.
- **`Engine.set_fingerprint_capacity(n)`** + **`fingerprint_get` /
  `fingerprint_put` / `fingerprint_release` / `fingerprint_len`**:
  new — LRU cache for vLLM request fingerprints, owned by the engine.
- **`Engine.mem_read_pack_batch(queries, k, owner)`**: new — N
  retrievals in one engine call. `k` and `owner` accept a scalar
  (broadcast) or a per-query list; `owner` list entries may be `None`
  to skip the filter for that one query.
- **`Engine.mem_read_multi_layer(query_keys, k, owner=None, rrf_k=60)`**:
  new — multi-layer query fanout + Reciprocal Rank Fusion in one
  engine call. Score-ties break on `pack_id` ascending.
- **`Engine.list_packs(owner=..., *, fetch_text)`**: `fetch_text` is
  now a required keyword-only argument (was optional with
  `DeprecationWarning`). Pass `fetch_text=False` for metadata-only,
  `fetch_text=True` to populate `text`, or call
  `Engine.list_packs_metadata()` for the columnar shape.
- **`tardigrade_vllm.retrieval_key.get_strategy`**: removed. Use
  `Engine.compute_retrieval_key(token_ids, "last_token" | "mean_pool")`
  after `Engine.load_embedding_table(...)`.
- **`tardigrade_hooks.sweep.GovernanceSweepThread`**: removed (module
  deleted). Use `engine.start_maintenance()` /
  `engine.stop_maintenance()` / `engine.maintenance_status()`.
- **`TardigradeError::InvalidArgument(String)`**: new Rust error
  variant with stable code `tdb::api::invalid_argument`. Surfaces as
  `ValueError` on the Python side.

### Performance

- **`Engine.compute_retrieval_key('last_token')`**: 90.3 us → 6.5 us
  at vocab=152064, hidden=1024, prompt_len=1024 (13.9x).
- **`Engine.compute_retrieval_key('mean_pool')`**: 411.4 us → 138.5 us
  at the same dims (3.0x).
- **`Engine.fingerprint_get` / `fingerprint_put`**: 236 ns → 119 ns vs
  Python `OrderedDict` at capacity=256 (2.0x).
- **`tardigrade_db.paged_to_flat`**: 143 us → 68 us at Qwen3-0.6B
  dims, seq_len=263 (2.1x).
- **`tardigrade_db.flat_to_paged`**: parity with numpy (76 us vs 71
  us). numpy reshape + concat already runs at memory bandwidth; the
  Rust path matches it.

### Behaviour

- **`MultiLayerQuery.query()`**: now routes through
  `Engine.mem_read_multi_layer` — one engine call, one mutex
  acquisition on `Arc<Mutex<Engine>>`. Ranking is unchanged.
- **vLLM connector lifecycle**: `request_finished` calls
  `Engine.fingerprint_release(fp)` proactively, so a finished
  request's block id cannot return a stale pack to a new request
  that reuses the same block.
- **Connector save path**: per-layer KV block-layout conversion now
  goes through `tardigrade_db.flat_to_paged` (Rust) instead of the
  Python `format.py` helper. Numerical output unchanged.

## [0.6.0] — 2026-05-20

Write-path rewrite. `mem_write_pack` at typical LLM dimensions is now ~7× faster end-to-end, `list_packs` ~10× faster at 10K packs, and a new batch-write API collapses N fsyncs into one.

### Public API

- **`Engine.mem_write_pack_tokens(owner, token_matrix, layers, salience, text=None, salience_mode=None)`**: new — write-side counterpart to `mem_read_tokens`. Accepts a `(n_tokens, dim)` numpy matrix directly; Rust builds the encoded retrieval key in one allocation. No Python-side `encode_per_token` round-trip.
- **`Engine.mem_write_batch_packs([pack, ...])`**: new — persist N packs with one coalesced fsync, returning assigned pack ids in input order. Eager counterpart to the streaming `open_with_write_buffer`.
- **`mem_write_pack` / `mem_write_pack_tokens`**: new `salience_mode` kwarg. `"l2"` and `"max"` derive salience from the encoded retrieval key inside Rust using `tdb_core::SalienceMode`; `"none"` (default) preserves the caller's explicit `salience`.
- **`tardigrade_db.find_chunk_boundary(text, max_pos, strategy)`**: new module-level function. `strategy` is `"whitespace"`, `"sentence"`, or `"paragraph"`. CJK sentence terminators (`。`, `！`, `？`) recognised natively. Backs the `BoundaryStrategy` ABC in `tardigrade_hooks.chunker`, which is now an Adapter.
- **`tardigrade_db.SALIENCE_SCALE` / `SALIENCE_CAP`**: module-level constants mirroring the Rust derivation; replaces hardcoded `50.0` / `100.0` literals on the Python side.

### Behaviour

- **`Engine.list_packs()`** and **`list_packs_metadata()`** now answer entirely from in-memory indices on the new `PackDirectory` owner map — no per-pack `BlockPool::get`, no Q4 cell decompression. Recovery contract: the index is rebuilt from the segment scan on engine open, so the segment trail remains authoritative.
- **Whitening covariance accumulation is now lazy.** The per-token outer-product `dim²` accumulation only runs when `WhiteningStrategy` is actually used; first call to `whitening_matrix()` flips the flag and backfills from the existing token store. Correctness unchanged for the whitening path.

### Performance

- **`Engine.list_packs_metadata(owner)`**: 57 ms → 5.5 ms median at 10K packs. ~10×.
- **`Engine.mem_write_pack`**: 27.5 ms → 4 ms per write at `(seq_len=256, dim=1024)` under the streaming write buffer. ~7×. The win traces to the now-gated whitening covariance accumulation inside `PerTokenRetriever::insert`.
- **`Engine.mem_write_batch_packs([×50])`**: 2 ms median vs 72 ms median for 50 sequential `mem_write_pack` calls (dim=64, NVMe SSD). 36.7×.

### Bug Fixes

- **HTTP `/mem/query`**: secondary `pack_text()` lookup removed. `mem_read_pack` already populates the `text` field; the fallback was dead defensiveness that added latency without ever changing the answer.

## [0.5.0] — 2026-05-20

Faster pack enumeration via columnar metadata API; multi-pack injection now works on quantized models.

### Public API

- **`Engine.list_packs_metadata(owner)`**: new — returns a dict of four parallel numpy arrays (`pack_ids`, `owners`, `tiers`, `importances`). Single Rust→Python crossing, no text fetch, no per-row dict allocation. ~1.3× faster than `list_packs` at 10K packs. Use when iterating / filtering pack metadata without needing inline text.
- **`Engine.list_packs(owner)`**: now emits `DeprecationWarning` when called without `fetch_text=True`. Pass `fetch_text=True` to keep the legacy list-of-dicts shape silently; switch to `list_packs_metadata` if you don't need inline text.

### Bug Fixes

- **`KnowledgePackStore.retrieve_and_inject_multi()` / `retrieve_with_trace()`**: no longer raise `RuntimeError: Expected query, key, and value to have the same dtype` on bf16 / fp16 / 4-bit-quantized models. Internal cache builder was producing fp32 regardless of model dtype; `_move_cache_to_device` now casts to the model's compute dtype when called by these paths. Single-pack `retrieve_and_inject` had this fix since v0.3.2; the multi-pack path was missed.

### Performance

- **`Engine.list_packs(owner)`**: 1.30× faster at 10K packs.

## [0.4.1] — 2026-05-19

Patch release: multimodal HuggingFace models now work end-to-end. Gemma 3 (1B / 4B / 12B / 27B), Llama 3.2 Vision, Qwen-VL, Phi-4-MM, and any other text-decoder bundled with a vision encoder are now supported across `is_supported`, `KnowledgePackStore`, and the calibration framework.

### Bug Fixes

- **Multimodal HuggingFace models** (Gemma 3, Llama 3.2 Vision, Qwen-VL, Phi-4-MM) now work end-to-end with `is_supported`, `KnowledgePackStore`, `select_query_layer`, and the rest of `tardigrade_hooks`. Pre-fix, multimodal configs hold language attributes inside `cfg.text_config` rather than at the root, and `is_supported` rejected them with `num_hidden_layers is missing`. Tardigrade-db now routes config-attribute reads through HF's canonical `cfg.get_text_config()` accessor at every intake site (7 sites across `_hidden_states`, `compatibility`, `kp_injector`, `calibrate`, `hf_kv_hook`, `multi_layer_query`). Text-only models are unaffected (conditional drill — only fires when the top-level cfg genuinely lacks `num_hidden_layers`).

## [0.4.0] — 2026-05-19

Minor release adding consumer-side pre-flight (`is_supported`) and closing the second half of the hybrid-attention story v0.3.3/v0.3.4 opened. The full pipeline — calibration, storage, retrieval, injection, generation — now works end-to-end on RecurrentGemma, Gemma 2, Jamba, and other architectures with non-trivial layer-type vocabularies.

### Public API

- **`tardigrade_hooks.is_supported(model, tokenizer) → CompatibilityReport`**: pre-flight model compatibility checker. Returns a frozen dataclass with `is_supported`, `architecture` (`"uniform_softmax"` | `"hybrid"` | `"unknown"`), `n_hidden_layers`, `n_softmax_layers`, `recommended_strategy`, `adapter_type`, `notes`, and `blockers`. Pure-config — no forward pass, no device hop. Lets consumers refuse incompatible models at boot instead of crashing inside `KnowledgePackStore.store()` at runtime.
- **`tardigrade_hooks.CompatibilityReport`**: frozen + hashable Value Object returned by `is_supported`. Stashable in sets / dict keys; cacheable across boots.

### Bug Fixes

- **`KnowledgePackStore.generate()` / `generate_with_trace()`**: no longer crashes mid-turn on hybrid-attention models. v0.3.4 fixed the `store()` path's sparse-layer iteration but three cache-clone loops in the generate-side methods still called `layer.keys.clone()` unconditionally, raising `AttributeError: 'NoneType' object has no attribute 'clone'` on RecurrentGemma / Jamba / Qwen3-Next where recurrent layer slots have `.keys = None`. Extracted to a shared `_clone_cache` helper that skips empty slots and preserves the sparse cache shape produced by `retrieve_and_inject`.
- **`softmax_layer_count` / `layer_kind_labels`**: architecture classifier now recognises the full vocabulary of HF config layer-type names. Pre-fix, only literal `"attention"` and `"full_attention"` counted as softmax — `"sliding_attention"` (Gemma 2, Mistral, Phi-3), `"local_attention"` and `"global_attention"` (Longformer family) were silently misclassified, causing `is_supported` to report these uniform-softmax models as hybrid. Recurrent vocabulary likewise expanded to recognise `"mamba"`, `"delta_net"`, `"gated_delta_net"` (Jamba, Qwen3-Next, Zamba) alongside the existing `"recurrent"` / `"linear_attention"`. Runtime behavior was already correct (the engine looks at the actual cache shape, not the config label) — this fix makes the diagnostic surface accurate.

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
