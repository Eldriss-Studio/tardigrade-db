---
name: Feature Request
about: Propose a new capability or architectural change
labels: enhancement
---

## Problem Statement

<!-- What problem does this solve? Be specific about the scenario that is currently impossible or painful. -->

## Proposed Solution

<!-- How should it work? Keep this at the design level — implementation details come later. -->

## Acceptance Criteria

<!-- These become the ATDD tests written before any implementation. Be concrete and testable. -->

- [ ] Given `<precondition>`, when `<action>`, then `<observable outcome>`
- [ ] Given ...

## Crates Affected

<!-- Which layers of the Aeon architecture does this touch? -->
- [ ] tdb-core (shared types)
- [ ] tdb-storage (block pool, quantization)
- [ ] tdb-retrieval (attention scoring, SLB)
- [ ] tdb-index (Vamana, Trace, WAL)
- [ ] tdb-governance (AKL, importance scoring)
- [ ] tdb-engine (facade / orchestration)
- [ ] tdb-python (Python bindings)

## Durability Contract

<!-- If this touches a write path, define the durability boundary: what is considered durable after this change, and how does `durable_offset` advance? -->

## Out of Scope

<!-- What related things are explicitly NOT part of this request? -->
