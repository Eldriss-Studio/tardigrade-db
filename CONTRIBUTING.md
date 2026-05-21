# Contributing to TardigradeDB

Thanks for thinking about contributing. This document covers the full contributor workflow — environment setup, the ATDD discipline, CI gates, and the project conventions that aren't obvious from reading the code.

Before you start, please skim:

- The [Code of Conduct](CODE_OF_CONDUCT.md) — applies to all project spaces.
- The [Security Policy](SECURITY.md) — do **not** file security issues publicly; use the private security-advisory flow.
- [`docs/architecture.md`](docs/architecture.md) — the four-layer Aeon model. Understand which layer a change belongs in before writing code.
- [`CLAUDE.md`](CLAUDE.md) — the project's AI-collaboration conventions and standing rules. Even if you don't use AI tools, this is the most up-to-date capture of what "good work in this repo" looks like.

---

## Requirements

- **Rust** ≥ 1.95 (edition 2024) — MSRV enforced in CI. Local toolchain tracks `stable` via [`rust-toolchain.toml`](rust-toolchain.toml).
- **[just](https://github.com/casey/just)** — task runner (`cargo install just`).
- **[lefthook](https://github.com/evilmartians/lefthook)** — git hooks (`brew install lefthook` or `go install github.com/evilmartians/lefthook@latest`).
- **Python** 3.13+ — for `tdb-python` PyO3 bindings and the Python consumer surfaces.
- **Nightly Rust** (optional, for fuzzing: `rustup toolchain install nightly`).
- **PyTorch + Transformers** (optional, for end-to-end LLM demos: `pip install torch transformers`).

---

## Development Model

The `main` branch is **protected**. External contributors must open a pull request and receive at least **one approving review** before merging. Force pushes and branch deletion are blocked.

### For contributors

1. **Fork** the repository and create a feature branch.
2. Make your changes following the ATDD workflow below.
3. Run `just ci` locally — all checks must pass.
4. Open a pull request against `main`. Stale approvals are automatically dismissed when new commits are pushed, so keep your PR up to date.
5. Address review feedback, then wait for a maintainer to approve and merge.

### For maintainers

Maintainers with admin access may push directly to `main` when appropriate. The branch protection rules do not apply to admins. The bar for each direct commit is the same: `just ci` must pass completely before you push.

---

## Environment Setup

```bash
# Install dev tools (one-time)
just setup

# Verify everything works
just ci
```

`just setup` installs: `cargo-nextest`, `cargo-deny`, `cargo-llvm-cov`, `typos-cli`, and configures `lefthook` git hooks.

### Python bindings (tdb-python)

```bash
# Install maturin
pip install maturin

# Build and install into the active virtualenv
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop
```

The `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` flag is required for Python 3.14+ compatibility and is included in all `just` recipes that touch `tdb-python`.

---

## ATDD Workflow (Mandatory)

Every fix or feature follows Acceptance Test-Driven Development. No exceptions.

1. **Write the acceptance test first.** The test defines "done" before any implementation exists. It must fail on `main` before you start.
2. **Write the minimum implementation** to make the test pass. No speculative abstractions.
3. **Run `just ci`** — all 6 checks must pass: `fmt lint typos test deny doc`.
4. **Refactor pass** — naming, structure, duplication, SOLID adherence. Re-run `just ci`.
5. **Gap review** — missing edge cases, untested paths, incomplete error handling, architectural blind spots. Add tests for any gaps found.

Only after all 5 steps is the work considered complete.

---

## Design Principles

- **Tensor-native:** The stored unit is a KV-cache tensor pair, not text or embeddings. No tokenization round-trips.
- **No external DB dependencies:** Custom storage, custom indices. No Postgres, Neo4j, or vector DBs.
- **Latent-space retrieval:** Relevance is `q · k / √d_k`, the same formula used inside a transformer — not cosine over text embeddings.
- **Self-curating:** The AKL (Adaptive Knowledge Lifecycle) manages cell lifecycle autonomously. Application code does not evict or promote cells.
- **SOLID & Clean Code:** Small, focused files and functions. Single responsibility. Break complexity into smaller modules — prefer many small files over few large ones.
- **Design patterns:** Use named, well-known patterns. Document the pattern in the module-level doc comment.

---

## Crate Dependency Rules

The workspace has a strict layered dependency order. Never create upward dependencies:

```
tdb-core  (no internal deps)
  └─► tdb-storage
  └─► tdb-governance
  └─► tdb-index        (→ tdb-storage)
  └─► tdb-retrieval    (→ tdb-storage)
        └─► tdb-engine (→ all above)
              └─► tdb-python (PyO3 bindings)
```

If you need a type in multiple crates, define it in `tdb-core`.

---

## What Not To Do

- **Do not publish crates.** All crates have `publish = false`. `cargo publish` will fail.
- **Do not add `#[allow(...)]` annotations.** Fix the root cause of every warning. If Clippy flags something, address it.
- **Do not use `println!` / `eprintln!` in library code.** Clippy denies `print_stdout` and `print_stderr` workspace-wide.
- **Do not add `unsafe` blocks without a `// SAFETY:` comment.** Clippy denies `undocumented_unsafe_blocks`.
- **Do not add `dbg!` macros.** Clippy denies `dbg_macro`.
- **Do not write backwards-compatibility shims** for removed code. If something is deleted, delete it completely.
- **Do not add speculative features.** Build exactly what the current task requires. Three similar lines of code is better than a premature abstraction.

---

## Running Tests

### Rust

```bash
just test                                              # all workspace tests
just test-crate tdb-engine                             # single crate
cargo nextest run --workspace --exclude tdb-python    # equivalent without `just`
cargo test --doc --workspace --exclude tdb-python     # all doctests
cargo clippy --workspace --all-targets -- -D warnings # pedantic lint
cargo fmt --all -- --check                            # format check
just ci                                                # full CI (what the pre-push hook runs)
```

`tdb-python` is excluded from `cargo test` because it requires `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` and a built wheel.

### Python

```bash
source .venv/bin/activate
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop

pytest tests/python/ -v -m "not gpu"                    # CPU-only (safe everywhere)
pytest tests/python/test_vllm_integration.py -v -m gpu  # vLLM round-trip (Linux + NVIDIA GPU + vLLM ≥ 0.19)
```

## Running Benchmarks

### Criterion (per-crate)

```bash
just bench                                            # all criterion benchmarks
just bench-crate tdb-retrieval                        # SLB + SIMD dot product
just bench-crate tdb-index                            # Vamana build + WAL throughput
just bench-crate tdb-engine                           # end-to-end write/read
```

### Bench V1 harness (comparable smoke + full runs)

```bash
# Smoke matrix (Tardigrade + Mem0 + Letta)
PYTHONPATH=python python -m tdb_bench run \
  --mode smoke \
  --repeat 3 \
  --config python/tdb_bench/config/default.json \
  --output target/bench-v1/smoke-run.json

# Full LoCoMo + LongMemEval run (requires pinned dataset paths)
export LOCOMO_DATA_PATH=/abs/path/to/locomo_phase1.jsonl
export LONGMEMEVAL_DATA_PATH=/abs/path/to/longmemeval_phase1.jsonl
PYTHONPATH=python python -m tdb_bench run \
  --mode full \
  --repeat 3 \
  --config python/tdb_bench/config/default.json \
  --output target/bench-v1/full-run.json

# Markdown reports
PYTHONPATH=python python -m tdb_bench report \
  --input target/bench-v1/smoke-run.json \
  --format md \
  --output target/bench-v1/smoke-report.md

# Compare two runs
PYTHONPATH=python python -m tdb_bench compare \
  --baseline target/bench-v1/smoke-run.json \
  --candidate target/bench-v1/full-run.json \
  --format md \
  --output target/bench-v1/compare.md
```

## Coverage

```bash
just coverage        # HTML report → target/llvm-cov/html/index.html
just coverage-lcov   # lcov output for CI
```

---

## CI Gates

Six jobs run on every push and PR. All must pass before merge.

| Job | What it checks |
|-----|----------------|
| **Check & Lint** | `cargo fmt`, `clippy --pedantic`, `typos`, `cargo-deny` |
| **Test** | `cargo nextest` on Ubuntu + macOS |
| **Coverage** | `cargo-llvm-cov` with Codecov upload |
| **MSRV** | Verifies build on Rust 1.95 |
| **Documentation** | Rustdoc build with `-D warnings`, deployed to GitHub Pages on `main` |
| **Bench V1 Smoke Gate** | Runs the comparable-benchmark smoke matrix (Tardigrade + Letta) and enforces a non-OK-ratio quality gate. Imports `tardigrade_hooks` without building the PyO3 wheel — any module reachable at import time must avoid touching the native extension. The `scripts/lint_lazy_imports.py` pre-commit hook catches this regression class locally. |

### Pre-commit hooks

Lefthook runs automatically on commit (fmt + clippy + typos) and push (full CI: fmt + lint + typos + test + deny + doc). Install with `lefthook install` or `just setup`.

### API documentation

Rustdoc is built and deployed to GitHub Pages on every push to `main`:

**[eldriss-studio.github.io/tardigrade-db](https://eldriss-studio.github.io/tardigrade-db)**

Benchmark result pages:
- Criterion dashboard: [.../dev/bench/index.html](https://eldriss-studio.github.io/tardigrade-db/dev/bench/index.html)
- Bench v1 narrative + latest links: [.../dev/bench-v1/index.html](https://eldriss-studio.github.io/tardigrade-db/dev/bench-v1/index.html)
- Observed completed runs (sample / smoke + caveats): [.../dev/bench-v1/results.html](https://eldriss-studio.github.io/tardigrade-db/dev/bench-v1/results.html)

To build locally: `just doc` (output in `target/doc/`).

---

## Reliability & Consistency Contracts

These rules are mandatory for all storage / retrieval / index / engine changes. Every PR touching a write path must explicitly address them.

- **Durability contract required.** Every design / PR touching write paths must define the durability boundary and how `durable_offset` advances.
- **Consistency mode declaration required.** Any API or externally visible read / update behaviour must explicitly declare `confirmed` vs `unconfirmed` semantics.
- **Recovery contract required.** Any change to WAL / snapshot / replay must document crash boundaries and recovery behaviour.
- **Derived-state rebuildability required.** Indexes, caches, and materialized / derived state must be reconstructable from durable history + snapshots.
- **Fail-fast replay required.** Replay inconsistencies are hard errors; do not serve reconstructed-but-untrusted state.

Minimum acceptance tests for durability / recovery changes:

- Crash during append / write path.
- Crash between WAL commit and snapshot capture.
- Snapshot restore + WAL suffix replay correctness.
- Confirmed-read visibility (`durable_offset >= tx_offset`) when enabled.

Mandatory metrics for durability / recovery changes:

- Replay total time.
- Snapshot read / restore time.
- WAL replay count / time.
- Durability queue depth / backlog.

---

## Commit Message Format

Follow gitmoji conventions:

```
[emoji] type(scope): description
```

Examples:
- `✨ feat(retrieval): add INT8 SLB hot cache`
- `🐛 fix(storage): correct Q4 dequantization scale factor`
- `♻️ refactor(engine): extract governance hooks to separate module`
- `📝 docs(tdb-core): add MemoryCell builder example`
- `🧪 test(index): add WAL crash recovery acceptance test`

See [gitmoji.dev](https://gitmoji.dev) for the full emoji reference.

---

## Working with Claude (or other AI assistants)

This project uses AI-assisted development as a first-class workflow. [`CLAUDE.md`](CLAUDE.md) is the source of truth for the project's AI-collaboration conventions: ATDD discipline, design-pattern naming honesty, the "fix what you find" principal-engineer standard, brief-by-default communication, and end-of-phase product verification.

If you're contributing via Claude Code (or an equivalent assistant), read `CLAUDE.md` first — many of this project's standing rules (no `#[allow(...)]` annotations without a cited acceptance test, no plan-phase identifiers in source, no overwriting plans, etc.) are captured there rather than in CI.

Human contributors should still skim `CLAUDE.md` — most of it is project conventions that apply regardless of who's writing the code.
