# Design: qk-ci-referee

## Discarded alternatives

1. **One fat job.** Rejected: masked failure classes — rust and python
   failures look identical in one log; separate jobs give separate
   signals.
2. **`uv sync` without extras as "the" suite.** Rejected: reproduces the
   silent-skip disease at scale (the 62/16 green lie).
3. **`on: push` + `pull_request_target`.** Rejected: PR-content
   execution surfaces; the workflow needs zero privileged contexts.
4. **Tag-referenced actions (`uses: x/y@v4`).** Rejected: mutable
   supply chain on a public repo — tj-actions-class takeover makes the
   referee the attacker.
5. **Skip-count asserted by grep of colored output.** Rejected: parse
   the summary line deterministically (`== <line>` shape), no ANSI,
   no `tail | grep` without pipefail.
6. **npx-downloaded openspec in CI.** Rejected where avoidable: pin the
   openspec CLI version in the job and fail closed if it can't install —
   the spec being validated must be validated by the version the spec
   targets (config.yaml was authored against 1.13/1.14 semantics).

## Informed assumptions

1. **Public runners suffice.** The full suite is lean (sub-minute
   Python; small Rust crate); no GPU needed for torch-tier tests or the
   tiny smoke. Wrong-surfaces-as: runner OOM/timeout → job failure, which
   is loud, not silent.
2. **The driver-smoke tiny `Qwen3Config` is representative.** FALSIFIED
   by measurement (coder + sec-eng, 2026-10-20): a random-init model's
   next-token distribution is already uniform (PPL ~= vocab = 151,949),
   so quantization damage has no structure to destroy — rank-1
   near-total quantization read ratio 1.0000006 and a zeroed `o_proj`
   moved PPL 3 ppm. The tiny smoke proves plumbing only (report
   contract, exit-code agreement, provenance, one-pedestal invariant);
   detection and abort-path coverage come from the weights-cached abort
   leg, never from this fixture.
3. **No secrets ever.** Today's workflow needs none. Adding one later
   is a spec-amending change, not a workflow edit — the PR-run attack
   surface stays closed by default.

## Lean-leg recipe (proven 2026-10-20, numbers re-measured when the workflow was authored)

- `uv venv <dir>` + `uv pip install -e . pytest` — **no extras** — builds
  a true lean interpreter; `python -c "import torch, transformers,
  tokenizers"` then exits 1. In that interpreter the numbers stand at
  `tests/test_qk_report.py` 53 passed/0 skipped and the whole suite
  **122 passed/16 skipped** (16 = kernels Rust leg only; the 122 includes this
  change's own refusals-pins). The full-extras suite is **202 passed/0 skipped**.
  The lean tier's expected skip count is a documented constant, not a tolerance:
  the job asserts exactly 16, so a new silently-skipped test fails the job.
- Lesson burned twice already (this change's history contains a "47
  passed lean" that ran with torch synced): absence is proven BEFORE any
  skip count is parsed, and no count from an interpreter whose absence
  wasn't demonstrated is ever trusted — the absence assertion is
  first-class, not a comment. The workflow enforces the order structurally (the
  absence step precedes the pytest step in the same job), and a test pins that
  ordering plus the exact `--expect-skipped` value per tier.

## Weights cache mechanism (abort leg)

- **Seed job**: **never skippable** — it runs on all three triggers, so the abort
  leg's `needs: seed-weights` can never inherit a condition-skip and turn a real
  run into a green no-run. It restore-looks-up first: a hit is a pure no-op; a
  miss re-fetches the pinned Qwen3-0.6B snapshot + wikitext-2 parquet (network
  allowed ONLY in this job's fetch/save steps) and re-saves them under a key
  naming the checkpoint repo + revision. The no-PR-content property therefore
  lives at STEP level: the fetch and save steps carry
  `cache-hit != 'true' && github.event_name != 'pull_request'`, while the
  lookup-only restore is unconditional — restore reads ACROSS scope, save does
  not, so a PR run can read main's cache and can never write one.
- **Abort leg**: strictly offline — restores by that key; on cache miss the job
  FAILS (fail-closed) rather than downloading silently. **Recovery is the next
  push to main**, which self-heals the cache; a manual `workflow_dispatch` of
  the same job remains available as the out-of-band lever.
- **Threat model that makes this proportionate**: the key names the repo +
  revision, and GitHub's cache allowance is a per-repo 10 GB cap with a 7-day
  least-recently-used sweep — so there is no cross-repo collision to defend
  against, eviction on an idle repo is the sweep doing its job, and the
  per-push touch this job performs keeps its own cache the most-recently-used
  one in the repo. Eviction stops being a failure mode and becomes a
  self-healing hiccup that costs one download.
- **Documented residual**: if the cache is evicted and a PR runs before the next
  push to main, the PR's seed cannot fetch (the PR guard blocks network,
  correctly), its lookup misses, and the abort leg reds fail-closed until that
  push re-seeds — the design working, not a defect: a PR never fabricates a
  weights cache.
- Rationale: the leg's whole point is independent repeatable proof of
  the abort path; letting it download on miss would make the same run
  network-dependent and, worse, let a cache miss go unnoticed as a
  silent skip — the disease this repo just burned.

## Open Questions

**OQ-1: self-hosted runner for cache-warm runs?** Owned: user. Not
needed for the first landable revision; revisit only with measured
runner-limit evidence.

**OQ-2: the openspec job is red on `qk-flash-attention`, a change committed on
main before this one.** Owned: @planner (spec authoring, not this change's
code). Measured 2026-10-20 with the job's own body — one id per invocation,
`@fission-ai/openspec@1.14.3`, `--strict`, every active dir:
`qk-ci-referee` 0, `qk-divergence-guards` 0, `qk-e2e-quality-gate` 0,
`qk-fail-fast-loop` 0, `qk-nq-recreation-bar` 0, `qk-sage-attention` 0,
**`qk-flash-attention` rc=1**: `⚠ WARNING qk-kernels/spec.md: ADDED "CubeCL
tiled attention GPU face (exact, f32)": Requirement text is very long (>500
characters)` (measured 1,204 characters; committed in `8a07556`).
Consequence: the first CI run on this change's own push is red on the openspec
job. Remedy is a spec edit in that change (split the requirement so each states
one behavior) or archiving it — @planner's call. Weakening the job (excluding
dirs, dropping `--strict`) is rejected: surfacing exactly this class of defect
is the job working, and a first-run red that is a true signal is not a reason to
soften the referee.

## Trust model: main's run is the authoritative green

On `pull_request`, GitHub runs the workflow file **from the PR's merge ref** — a PR
can weaken or gut the referee and still print green checks under identical job
names. The mitigation is structural rather than procedural: the workflow also
triggers on `push: [main]`, so the landed referee re-runs at main's head on every
merge, and **that** run is the one that certifies the tree. PR green is advisory;
main green is authoritative. This is inert while landings are direct pushes, and
it becomes load-bearing the day this public repo takes an outside PR.
(Recorded from @sec-eng's task-5 review; wording @planner's to change.)

## Accepted residuals (task-5 review)

- `npx --yes @fission-ai/openspec@1.14.3` (every run) and
  `uvx --from huggingface-hub==2.2.0` (seed job only) are exact-version pins
  without integrity hashes: a registry compromise is code execution with
  `contents: read` and no reachable secrets. Proportionate at this exposure;
  vendoring under a lockfile with integrity is the upgrade path if it grows.
- Cache poisoning is not reachable: PR-facing jobs only RESTORE
  (`fail-on-cache-miss: true`), and the seed job's fetch and save steps are
  gated on `github.event_name != 'pull_request'`, so no PR run can write a cache
  entry — a PR's saves would land on the merge ref and be restorable only by
  that PR's own re-runs. The content is verified at the point it is written
  (pinned-revision fetch, snapshot assert, revision↔key pair pinned by test).

## Implementation crystallization

- Action SHAs SHALL be resolved at implementation time against the
  pinned repos' current heads (astral-sh/setup-uv, actions/checkout,
  etc.), recorded with their repo + tag provenance in the workflow
  comment — verified fresh at landing, never copied from memory.
- **Every command body in a workflow SHALL be executed locally before the
  workflow is committed.** The rehearsal of this file's fourteen bodies caught two
  real defects that no amount of reading would have: the smoke job as first
  written died at scale calibration because `QUANTIBBLER_KD_TMPDIR` must name an
  *existing* directory (`tempfile.TemporaryDirectory` does not create parents —
  and the driver's own default is the sandbox path `/opt/data/tmp`, which exists
  on no runner), and the openspec job is red on a change committed before this
  one (OQ-2). A job body that has never been run is a hypothesis, and CI is a
  terrible place to first test one: the referee's own failures must not be the
  thing that first teaches us CI works.
- **A pinned fetch needs its branch ref restored (task-5 SF-1, measured).** The
  seed job now fetches with `--revision <sha>` for both repos, so the fetch
  contract equals the cache-key contract and a re-warm no longer depends on Qwen's
  or Salesforce's default branch. But a revision-pinned fetch writes
  `snapshots/<sha>/` and **no `refs/`**, while the driver resolves
  `--model Qwen/Qwen3-0.6B` offline *through* `refs/main` — so the seed writes
  `refs/main` (bare 40-char sha, no newline, byte-identical to a branch fetch) and
  the abort leg's cache check keeps requiring it. Pinning one half without the
  other produces a cache that looks complete and cannot resolve a repo id.
- **Committed lockfiles mean locked resolution (task-5 SF-2).** Every
  `uv sync`, `cargo test`, `cargo clippy` and the maturin build now carry
  `--locked`, so a PR editing `pyproject.toml`/`Cargo.toml` cannot silently
  re-resolve and test different dependencies than main under the same job name.
  The lean leg is the one deliberate exception — it installs off the dependency
  floor unlocked because that float is its signal — and the exception is stated
  in the YAML so nobody "fixes" it. Each half is pinned by a test, including the
  absence of `--locked` in the lean job.
- **A routine re-warm must not print red.** The seed job's first step is a
  `lookup-only` cache restore whose `cache-hit` output gates both the fetch and
  the save, so dispatching an already-seeded key is a loud green no-op
  ("nothing to fetch, nothing to save") instead of a save-step "cache already
  exists" that a human has to learn to read as noise. Inputs and outputs of the
  pinned `actions/cache/restore` were read from its `action.yml` at the pinned SHA
  before this was written: `lookup-only` and the `cache-hit` output both exist in
  v6.1.0.
- **A pin asserts the write's shape and its binding; a later overwrite of the
  same path is the runtime leg's problem.** The ref-restore pin requires an
  actual `printf '%s' "$<var>" > "…refs/main"` write and that the variable be the
  one bound from the pinned pair — a substring (`"refs/main" in seed`) is not a
  pin, because an adjacent `echo` that mentions the path keeps it alive while the
  write is deleted (@bug-hunter's demonstrated bypass, now a parametrized test).
  What the pin deliberately does *not* chase: a second, later write to the same
  path (printf-then-`echo x > refs/main`). That is the nested-lambda /
  callee-mutates-the-dict tier — not a natural refactor shape, and a bad cache is
  caught on the first real runner run by the abort leg, which fails closed on a
  miss rather than downloading. One-clause hardening exists (collect every
  `refs/main` redirect, require exactly one and that it be the printf) and belongs
  in a follow-up commit, not a pre-landing rider: executing it now would move
  `tests/test_ci_referee.py` off the hash the sign-off transfer table records,
  which is a higher cost than the residual.

## Implementation crystallization — self-seeding weights job (2026-10-27)

**The problem the delta fixes, measured in the first real runs:** the seed job
was `workflow_dispatch`-only. That bought the no-PR-content property with a
job-level `if`, and the price was that the job never ran on its own — so on
every push and every PR the abort leg's fail-closed restore hit a cold cache and
redded, and the only way out was a human dispatching a seed. A referee that is
red until someone performs a manual chore is a referee whose reds stop being
read.

**The binding shape (frozen before implementation, per @planner):** the seed job
is never skippable — no job-level `if` on any trigger — and the no-PR-content
property moves to STEP level: the fetch and save steps carry
`steps.seeded.outputs.cache-hit != 'true' && github.event_name != 'pull_request'`,
the lookup-only restore runs unconditionally, and the abort leg declares
`needs: seed-weights`. The `needs:` is the load-bearing half: a job-level
condition on the seed would propagate a skip into the leg and turn a real run
into a green no-run, which is exactly the class of silence this referee exists
to kill.

**Why the PR path works (GitHub's own semantics, checked not assumed):** restore
is read-across-scope while save is not. A PR run's seed restores from the
default branch's cache scope (a hit), its fetch/save steps skip on the event
guard, and the abort leg's own fail-closed restore reads the same cache and
executes for real. So every PR run executes the leg without ever being able to
write a weights cache.

**The residual, recorded verbatim so a future red is read correctly.** If the
cache is evicted (7-day sweep on an idle repo) and a PR runs before the next
push to main, the PR's seed cannot fetch — the event guard blocks network,
correctly — its lookup misses, and the abort leg reds fail-closed until the next
push re-seeds. That is the design working, not a defect: **a PR never fabricates
a weights cache.** Same doctrine as the first-run red.

**Pins, and the battery that keeps them honest.** `_seed_shape_findings()` reads
the uncommented workflow and asserts the four properties directly: no `^    if:`
in the seed job, no `if:` on the lookup step, the exact gate literal on both
network steps, and `needs: seed-weights` on the abort leg. Four parametrized
bypass rows — job-level `if` reinstated, PR guard dropped from the fetch,
`needs` dropped, lookup restore made conditional — are each applied to the REAL
workflow text, each asserted non-no-op first, and each required to produce a
finding.

**Honesty boundary.** The `uses:` steps and the runner's trigger/skip semantics
still execute only on a runner. The four-row battery proves the pins reject the
mutations in the file; the in-the-wild proof — that a PR run's abort leg
*executes* rather than skips — is the post-landing `pr-proof-abort-leg` run the
planner pre-committed, and it is the delta's evidence, not a local rehearsal.
