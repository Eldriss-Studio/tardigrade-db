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

# === Benchmarks ===

# Run all benchmarks with native CPU optimizations
bench:
    RUSTFLAGS="-C target-cpu=native" cargo bench --workspace

# Run benchmarks for a specific crate
bench-crate crate:
    RUSTFLAGS="-C target-cpu=native" cargo bench -p {{crate}}

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
