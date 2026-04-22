# TardigradeDB development tasks

default:
    @just --list

# === Quality ===

# Check formatting
fmt:
    cargo fmt --all -- --check

# Fix formatting
fmt-fix:
    cargo fmt --all

# Run clippy lints
lint:
    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo clippy --workspace --all-targets -- -D warnings

# Run cargo-deny checks (license, advisory, bans)
deny:
    cargo deny check

# Spell-check code and docs
typos:
    typos

# === Testing ===

# Run all tests
test:
    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo nextest run --workspace

# Run tests with CI profile (retries, longer timeouts)
test-ci:
    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo nextest run --workspace --profile ci

# Run tests for a specific crate
test-crate crate:
    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo nextest run -p {{crate}}

# === Evals ===

# Run all release-mode evals (spec + aspirational)
eval:
    RUSTFLAGS="-C target-cpu=native" PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test --release --workspace --exclude tdb-python -- --ignored eval_ 2>&1

# Run only must-pass spec evals (Category A)
eval-spec:
    RUSTFLAGS="-C target-cpu=native" PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test --release --workspace --exclude tdb-python -- --ignored eval_spec_ 2>&1

# Run only aspirational evals (Category B+C, may print warnings)
eval-aspirational:
    RUSTFLAGS="-C target-cpu=native" PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test --release --workspace --exclude tdb-python -- --ignored eval_aspir_ 2>&1

# === Benchmarks ===

# Run all benchmarks with native CPU optimizations
bench:
    RUSTFLAGS="-C target-cpu=native" PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo bench --workspace --exclude tdb-python

# Run benchmarks for a specific crate
bench-crate crate:
    RUSTFLAGS="-C target-cpu=native" PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo bench -p {{crate}}

# Run benchmark harness smoke profile (<=10m target)
bench-v1-smoke:
    PYTHONPATH=python .venv/bin/python -m tdb_bench run --mode smoke --config python/tdb_bench/config/default.json --output target/bench-v1/smoke-run.json
    PYTHONPATH=python .venv/bin/python -m tdb_bench report --input target/bench-v1/smoke-run.json --format md --output target/bench-v1/smoke-report.md
    PYTHONPATH=python .venv/bin/python -m tdb_bench report --input target/bench-v1/smoke-run.json --format json --output target/bench-v1/smoke-report.json

# Run benchmark harness smoke profile with 3 replicates for confidence stats
bench-v1-smoke-r3:
    PYTHONPATH=python .venv/bin/python -m tdb_bench run --mode smoke --repeat 3 --config python/tdb_bench/config/default.json --output target/bench-v1/smoke-run-r3.json
    PYTHONPATH=python .venv/bin/python -m tdb_bench report --input target/bench-v1/smoke-run-r3.json --format md --output target/bench-v1/smoke-report-r3.md
    PYTHONPATH=python .venv/bin/python -m tdb_bench report --input target/bench-v1/smoke-run-r3.json --format json --output target/bench-v1/smoke-report-r3.json

# Run benchmark harness full profile
bench-v1-full:
    PYTHONPATH=python .venv/bin/python -m tdb_bench run --mode full --config python/tdb_bench/config/default.json --output target/bench-v1/full-run.json
    PYTHONPATH=python .venv/bin/python -m tdb_bench report --input target/bench-v1/full-run.json --format md --output target/bench-v1/full-report.md
    PYTHONPATH=python .venv/bin/python -m tdb_bench report --input target/bench-v1/full-run.json --format json --output target/bench-v1/full-report.json

# Run benchmark harness full profile with 3 replicates for confidence stats
bench-v1-full-r3:
    PYTHONPATH=python .venv/bin/python -m tdb_bench run --mode full --repeat 3 --config python/tdb_bench/config/default.json --output target/bench-v1/full-run-r3.json
    PYTHONPATH=python .venv/bin/python -m tdb_bench report --input target/bench-v1/full-run-r3.json --format md --output target/bench-v1/full-report-r3.md
    PYTHONPATH=python .venv/bin/python -m tdb_bench report --input target/bench-v1/full-run-r3.json --format json --output target/bench-v1/full-report-r3.json

# === Coverage ===

# Generate HTML coverage report
coverage:
    cargo llvm-cov --workspace --html
    @echo "Report: target/llvm-cov/html/index.html"

# Generate lcov output (for CI)
coverage-lcov:
    cargo llvm-cov --workspace --lcov --output-path target/llvm-cov/lcov.info

# === Build ===

# Debug build
build:
    cargo build --workspace

# Release build
release:
    cargo build --workspace --release

# Build documentation
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items --exclude tdb-python

# === Fuzz ===

# Run a fuzz target (requires nightly)
fuzz target:
    cd crates/tdb-storage && cargo +nightly fuzz run {{target}} -- -max_total_time=300

# === Setup ===

# Install dev tools and git hooks
setup:
    cargo install cargo-nextest cargo-deny cargo-llvm-cov typos-cli
    lefthook install
    @echo "Dev environment ready. Pre-commit and pre-push hooks installed."

# === CI-local ===

# Run the full CI check locally (same checks as GitHub CI)
ci: fmt lint typos test deny doc
    @echo "All CI checks passed."
