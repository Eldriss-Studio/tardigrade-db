# Security Policy

## Supported Versions

TardigradeDB follows a 0.x.y release cadence pre-1.0. Only the most recent minor series receives security fixes.

| Version | Supported |
|---------|-----------|
| 0.7.x   | ✅ Yes    |
| < 0.7   | ❌ No     |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security reports. The repository's public issue tracker is not a safe channel for vulnerability disclosure.

Use GitHub's private security advisory flow instead:

1. Navigate to the repository on GitHub.
2. Click the **Security** tab.
3. Click **Report a vulnerability**.
4. Fill in the form with a description, reproduction steps, and any proof-of-concept material.

You should receive an acknowledgement within 7 days. We will work with you to confirm the issue, ship a fix in the next minor release, and credit you in the advisory once the fix is public (unless you prefer to remain anonymous).

## Scope

In scope for this policy:

- Memory-safety bugs in the Rust crates (`tdb-core`, `tdb-storage`, `tdb-retrieval`, `tdb-index`, `tdb-governance`, `tdb-engine`).
- Persistence corruption that survives a crash or restart (segment / WAL / snapshot bugs that violate the durability boundary).
- Owner-isolation breaches — one owner's reads or writes leaking into another owner's namespace.
- PyO3 / GIL invariants — unsafe boundary code that can lead to use-after-free, data races, or crashes from Python.
- Privilege boundaries in the HTTP / REST bridge (`python/tardigrade_http/`) and the MCP server bridge.
- Authentication or authorization bypasses in any official integration.

Out of scope:

- Model-output quality, hallucination behavior, or factual accuracy — those are properties of the consuming LLM, not TardigradeDB.
- Benchmark methodology. Reproducibility concerns about benchmark numbers (LoCoMo, LongMemEval, etc.) are addressed via the audit trail in `docs/experiments/`, not the security advisory process.
- Third-party dependencies. Report those upstream; we'll bump versions when the fix is released.
- Self-DoS via legitimate API misuse (e.g. unbounded write loops). Use governance / quota at the consumer layer.

## Disclosure Timeline

- **Day 0:** Report received via GitHub private security advisory.
- **Day 1-7:** Acknowledgement + initial triage.
- **Day 7-30:** Fix developed, tested, and prepared for release.
- **Day 30+:** Fix shipped in the next minor release; advisory made public; reporter credited.

Timelines may extend for complex issues. We will keep you informed throughout.

## Hall of Fame

No advisories have been issued yet. Reporters who help us secure TardigradeDB will be listed here (with their permission) after their fix ships.
