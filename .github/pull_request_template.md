# Self-Review Checklist

> TardigradeDB uses push-to-main. This checklist is a self-review before every commit.

## ATDD

- [ ] Acceptance test written **before** the implementation (test was failing on `main` first)
- [ ] Test names describe behavior, not implementation details

## CI

- [ ] `just ci` passes end-to-end (`fmt lint typos test deny doc`)
- [ ] No `#[allow(...)]` annotations added — root causes addressed instead
- [ ] No `println!` / `eprintln!` / `dbg!` in library code

## Code Quality

- [ ] Refactor pass complete — naming, structure, duplication, SOLID adherence
- [ ] Gap review complete — edge cases, untested paths, error handling, architectural blind spots
- [ ] No speculative abstractions added beyond what the task requires

## Documentation

- [ ] All public types and functions have `///` doc comments
- [ ] Any new `lib.rs` entry point has `//!` crate-level docs
- [ ] `cargo doc --workspace` generates no warnings (`just doc` passes)

## Durability (if touching write paths)

- [ ] Durability contract defined — where the boundary is and how `durable_offset` advances
- [ ] Crash scenarios covered — during append, between WAL commit and snapshot
- [ ] Recovery behavior documented in the module-level doc comment
