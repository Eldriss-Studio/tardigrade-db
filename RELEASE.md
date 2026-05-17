# Release process

Pre-1.0 release procedure for `tardigrade-db`. Single-artifact release: the Python wheel published to PyPI. Rust crates are not currently published to crates.io — the workspace version exists for internal consistency only.

## When to release

- **After a feature batch ships green.** Don't release per-commit; group multiple related commits behind one tag. A milestone group (e.g. the HTTP bridge + Node.js example + consumer guide) is a good unit. A single internal refactor is not.
- **Out-of-cycle for security or critical bug fixes.** Tag the fix on its own, bump patch, ship.
- **Avoid sleepy releases.** If the working tree has been dormant for a sprint and nothing meaningful changed, don't release just for the heartbeat.
- **Avoid frenetic releases too.** Default cadence is weeks, not hours. Multiple releases in a single working session is a smell — bump consumers' update cost without giving them new behaviour proportional to the noise. When a feature lands, ask: "could this wait for the next planned release window?" If yes, hold it and batch with the next set of work. The exception is a security or data-loss fix, which ships immediately on its own.

Trigger check before kicking off:

1. Is there anything user-facing since the last tag? `git log v<prev>..HEAD --oneline` — look for ✨ feat or 🐛 fix that actually changes consumer behaviour.
2. Are all tests green? `just test-ci`.
3. Are clippy + fmt + docs clean? `just lint` + `cargo doc --workspace --no-deps --document-private-items --exclude tdb-python`.
4. Is the lint gate clean? `python3 scripts/lint_lazy_imports.py`.

If any of those fail, fix before releasing — never tag broken state.

## Version policy

Pre-1.0, semver-ish:

- **Minor bump (`0.X.0`)**: new user-facing surface, breaking internal changes that consumers may notice, batch of features.
- **Patch bump (`0.X.Y`)**: bug fixes, documentation, internal refactors with no consumer impact.
- **No major bumps until 1.0**: 1.0 happens when the public API is stable enough to commit to. Not yet.

The workspace version (`Cargo.toml`'s `[workspace.package].version`) and the Python distribution version (`pyproject.toml`'s `[project].version`) must match. The PyO3 binding inherits from the workspace.

## Pre-release checklist

```text
[ ] git switch main && git pull
[ ] git status is clean (working tree + index)
[ ] `just test-ci` is green
[ ] `just lint` is green (cargo clippy --workspace --all-targets -- -D warnings)
[ ] `cargo doc --workspace --no-deps --document-private-items --exclude tdb-python` is clean
[ ] `python3 scripts/lint_lazy_imports.py` returns 0
[ ] CHANGELOG.md: `[Unreleased]` content rolled into `[X.Y.Z] — YYYY-MM-DD`
[ ] CHANGELOG.md: fresh empty `[Unreleased]` section added back
[ ] Cargo.toml workspace version bumped to X.Y.Z
[ ] pyproject.toml version bumped to X.Y.Z
[ ] `maturin develop` succeeds with the new version
[ ] Commit the version bump: `git commit -m "🔖 release: vX.Y.Z"`
[ ] Tag: `git tag -a vX.Y.Z -m "vX.Y.Z"`
[ ] Push: `git push origin main && git push origin vX.Y.Z`
```

## Build and publish

The wheel is produced by `maturin`:

```bash
# Clean build, release profile, all platforms the runner supports.
maturin build --release --strip
# Inspect what was built:
ls -lh target/wheels/
```

Publish to PyPI:

```bash
# Use a scoped PyPI API token (not username/password).
# Token lives in ~/.pypirc or MATURIN_PYPI_TOKEN env var.
maturin publish --release --strip
```

For a release candidate before public publish, use TestPyPI:

```bash
maturin publish --repository testpypi
pip install --index-url https://test.pypi.org/simple/ tardigrade-db
```

## Publish the GitHub release

**Always create a GitHub release for every tag.** The PyPI page shows package metadata; the GitHub release page is what most people see first and what RSS / dependabot / "what's new" consumers pick up.

**Fact-check first.** Release notes go in front of users and stay there. Every factual claim ("Linux-only wheel", "X is fixed", "depends on Y") must be verified against the *published* artifact, not the local build. The local `target/wheels/` directory is one specific platform; the CI matrix produces many more. After the publish workflow finishes, pull the actual file list before writing the notes:

```bash
curl -s https://pypi.org/pypi/tardigrade-db/<version>/json \
  | python3 -c "import sys,json; [print(f['filename']) for f in json.load(sys.stdin)['urls']]"
```

If you publish wrong information, fix it the same session via `gh release edit --notes-file ...` — don't "leave it as a follow-up."

Use the `CHANGELOG.md` entry as the source of truth, but expand it into consumer-shaped sections (TL;DR, what's new grouped by theme, known limitations, full-changelog link). Attach the wheel(s) as release assets so people can download a known-good artifact without re-resolving from PyPI.

**Format the body as flowing markdown** — one paragraph per line, no hard wrap at fixed columns. The release page renders in the same fixed-width container as `.md` files on github.com; hard-wrapped prose looks awkward and word-counts oddly there. Same convention as the rest of the docs.

```bash
# Draft and publish in one shot. --notes-file lets you write
# the body in your editor and pass it in; --notes is fine for
# short releases written inline.
gh release create vX.Y.Z \
  --title "vX.Y.Z — <short tagline>" \
  --notes-file release_notes.md \
  target/wheels/*.whl
```

Things every release notes block should cover:

- One-line summary at the top.
- `pip install tardigrade-db==X.Y.Z` snippet for copy-paste.
- **What's new** — grouped by theme, not by commit hash order. Mirror the `Added` / `Changed` / `Fixed` structure from `CHANGELOG.md`.
- **Known limitations** — single-platform wheel, deprecated surface, anything that might bite a user. Honesty is cheaper than support tickets.
- Link to the relevant `CHANGELOG.md` anchor on the tag (use `https://github.com/.../blob/vX.Y.Z/CHANGELOG.md#xyz`, not `main`, so the link is stable across future edits).

## Post-release

1. Open a fresh `[Unreleased]` section at the top of CHANGELOG.md.
2. Announce in whatever channel is current (Slack/Discord/issue thread).
3. If the release fixed a tracked bug, link the tag from the bug tracker.

## What goes in CHANGELOG entries

Group by category. Use the headings consistently so consumers diffing two releases get a stable shape:

- **Added** — new public surface.
- **Changed** — behaviour change visible to consumers.
- **Deprecated** — still works but on the way out; name the replacement.
- **Removed** — gone.
- **Fixed** — bug fix.
- **Security** — vulnerability fix.

One bullet per user-facing item. Include the commit hash in parentheses if the diff is the canonical reference. Don't write "Phase 1A.2 lookback bug" — write "boundary-aware chunker no longer truncates mid-word at chunk edge (`abc1234`)" so the entry survives plan rewrites.

## Notes

- Cargo.toml `version.workspace = true` propagates to every crate; only edit the workspace version.
- `maturin develop` builds locally without publishing — use before tagging to confirm the new version installs cleanly.
- Never re-publish a yanked version under the same number — bump patch and ship the fix.
