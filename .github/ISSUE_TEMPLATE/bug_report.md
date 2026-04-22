---
name: Bug Report
about: Something is broken or behaving incorrectly
labels: bug
---

## Crate

<!-- Which crate(s) does this affect? -->
- [ ] tdb-core
- [ ] tdb-storage
- [ ] tdb-retrieval
- [ ] tdb-index
- [ ] tdb-governance
- [ ] tdb-engine
- [ ] tdb-python
- [ ] CI / tooling

## Transformer Layer / Context

<!-- If relevant: model dimension, layer index, cell count at time of failure -->

## Observed Behavior

<!-- What actually happened? Include panic message, wrong output, or incorrect score. -->

## Expected Behavior

<!-- What should have happened? -->

## Reproduction Steps

```bash
# Minimal reproducer
```

## Durability Impact

<!-- Does this affect data already written to disk? Could cells be lost, corrupted, or misread? -->
- [ ] Yes — cells on disk may be affected
- [ ] No — in-memory state only
- [ ] Unknown

## Environment

- OS:
- Rust version (`rustc --version`):
- Python version (if tdb-python):
- Commit hash (`git rev-parse HEAD`):
