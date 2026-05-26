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

The convention used by this project pre-1.0 — match it because every shipped tag (`v0.1.0` through current) has followed it:

- **Minor bump (`0.X.0`)**: a release that crosses a milestone consumers should pay attention to. New user-facing surface (new methods, new endpoints, new APIs that change what the engine *can do*); architectural primitives that reshape how consumers reason about the engine (the durability boundary, the metrics layer); coherent batches of work that ship together and want one anchor. Breaking changes also belong here — but in practice, pre-1.0 we have not shipped breaking changes, and minor bumps have been the "milestone marker" not the "incompatibility flag."
- **Patch bump (`0.X.Y`)**: incremental additions and bug fixes within the current track. Example: `0.7.7` closed a `CalibrationResult.best_score()` foot-gun; `0.7.6` fixed two vLLM connector save-path crashes; both shipped as patches because they were tactical fixes inside the v0.7 track, not new milestones.
- **No major bumps until 1.0**. 1.0 happens when the API is stable enough to commit to long-term. Not yet.

Why this isn't the by-the-book Cargo SemVer convention: Cargo treats `^0.2` as `>=0.2.0, <0.3.0`, which assigns *breakingness* to the minor bump and would push every additive feature batch to a patch. That works for libraries with external consumers who pin tight version ranges. Pre-1.0, with the sole maintainer and no external consumers pinning to specific minors, the milestone-marker convention is more useful to the project than the strict-incompatibility convention. **The expectation is to migrate to strict Cargo SemVer at 1.0**, when external consumers start pinning and the cost of misreading "minor = additive milestone" exceeds the cost of all releases being patches.

CHANGELOG.md's preamble names the same convention in one sentence ("minor bumps mark new user-facing surface, patch bumps mark fixes and internal changes"). Keep the two in sync — if the rule changes, change both.

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
[ ] pyproject.toml version bumped to X.Y.Z (drives the wheel filename — maturin reads this, not Cargo.toml's workspace version)
[ ] CITATION.cff `version:` field bumped, `date-released:` updated
[ ] README.md status banner and "Current version" line bumped to vX.Y.Z
[ ] SECURITY.md supported-versions table reflects new minor (only if the minor changed; patch bumps don't move support windows)
[ ] `maturin develop` succeeds with the new version
[ ] Commit the version bump: `git commit -m "🔖 release: vX.Y.Z"`
[ ] Tag the release commit explicitly (NOT main): `git tag vX.Y.Z <release-commit-sha>`
[ ] Verify the tagged tree carries the new version: `git show vX.Y.Z:Cargo.toml | grep '^version'`
[ ] Push: `git push origin main && git push origin vX.Y.Z`
[ ] `gh release create vX.Y.Z --notes-file <changelog-section>` (triggers the publish workflow via on: release: published)
[ ] Watch `gh run list --workflow=publish.yml --limit 1` until status = completed, conclusion = success
[ ] Verify the artifact actually landed: `curl -s https://pypi.org/pypi/tardigrade-db/X.Y.Z/json | jq -r .info.version` returns `X.Y.Z`
```

Two checklist entries to be aware of:

- **`pyproject.toml` AND `Cargo.toml` both need bumping.** Maturin reads `pyproject.toml`'s `[project].version` for the wheel filename, not the Cargo.toml workspace version. The publish workflow has guards against version-vs-tag drift, but the right fix is to keep both manifests synced.
- **Tagging main vs tagging the release commit.** GitHub Actions workflows triggered by `on: release: published` check out the *release's target commit*, not the tag's commit. If main has moved past the release commit and the release was created against main (the UI default), the workflow builds the wrong tree and PyPI rejects the wheel as a duplicate of the previous version's filename. `gh release create` from CLI uses the tag's target commit; the UI defaults to main. Verify with `git show vX.Y.Z:Cargo.toml | grep '^version'` before pushing.

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

The reader is the consumer of the package, not its author. They want: "what can I do now that I couldn't before, and how does my code need to change?" — not a reverse-chronological list of commits. Section headers are Title Case, themed to *what the release delivers*, not the Keep-A-Changelog template (Added / Changed / Deprecated / Removed / Fixed). The themed style was adopted starting around `v0.7.x` and the project has stayed with it; new entries should match.

Sections to draw from, used as needed:

- **General** — top-level capabilities, theme of the release, breaking changes
- **Public API** — new exports, signature changes, deprecations
- **HTTP API** — new endpoints, new query params, new error shapes
- **Behaviour** — things that work differently without an API change
- **Observability** — new metrics, new logging, new diagnostic surfaces
- **Performance** — measurable wins the user will notice
- **Test Infrastructure** — new test classes, drift-guards, parity gates (consumers care because their own CI inherits these)
- **CI** — workflow changes consumers will notice via passing/failing checks
- **Bug Fixes** — terminal section, purely factual
- **Community Standards** — `SECURITY.md`, `CITATION.cff`, etc. when bumped

Skip any section that has nothing in it. Per-entry shape: bold anchor (the symbol the user calls, in backticks if it's code) + colon + the delta. Use `→` for numerical or behavioural deltas (`"Cooldown 7s → 18s"`, `"Coverage threshold 80% → 75%"`). Tense: declarative, present, sentence fragments. "Sessions now expose …" not "Added event emission …".

What goes in: concrete user-callable changes (new exports, behaviour shifts, error-shape changes), numerical deltas with `→`, breaking changes with a one-line migration hint when non-obvious, bug fixes the user might have hit.

What stays out: internal refactors that don't change behaviour, build/lockfile/lint config changes, test framework migrations, PR numbers, SHAs, branch names, the *why* unless it's a security fix or a breaking-change rationale the user needs, internal-doc section references (no "§5.1", "punch list", "phase NN" — those rot fast and confuse future readers).

Include the commit hash in parentheses if the diff is the canonical reference. Don't write "Phase 1A.2 lookback bug" — write "boundary-aware chunker no longer truncates mid-word at chunk edge (`abc1234`)" so the entry survives plan rewrites.

Reference style: Valve's Dota 2 patch notes (<https://www.dota2.com/news/updates>) — the canonical example of changelog writing that the audience actually reads.

## Notes

- Cargo.toml `version.workspace = true` propagates to every crate; only edit the workspace version.
- `maturin develop` builds locally without publishing — use before tagging to confirm the new version installs cleanly.
- Never re-publish a yanked version under the same number — bump patch and ship the fix.
