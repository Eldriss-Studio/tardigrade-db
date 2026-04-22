# Contributing to TardigradeDB

## Development Model

TardigradeDB uses **push-to-main**. There are no long-lived feature branches and no pull request reviews. Every commit lands directly on `main` after passing the full local CI check.

This means the bar for each individual commit is high: `just ci` must pass completely before you push.

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

```bash
just test                    # all workspace tests
just test-crate tdb-engine   # single crate
just ci                      # full CI (what the pre-push hook runs)
```

## Running Benchmarks

```bash
just bench                   # all benchmarks, native CPU optimizations
just bench-crate tdb-index   # single crate
```

## Coverage

```bash
just coverage        # HTML report → target/llvm-cov/html/index.html
just coverage-lcov   # lcov output for CI
```

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
