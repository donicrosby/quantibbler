# Tasks: qk-ci-referee

- [x] (coder) Author `.github/workflows/ci.yml` per the final spec: SIX jobs (lean-python, full-python, rust-kernels, openspec, driver-smoke, abort-leg) plus a `workflow_dispatch`-only seed job; `bash -euo pipefail` defaults; SHA-pinned actions resolved fresh at landing (never from memory); `permissions: contents: read`; triggers `pull_request` + `push` to `main` (never `pull_request_target`); zero secrets. Lean job = proven recipe (fresh `uv venv` + `pip install -e . pytest`, no extras) with the absence-family assert (`import torch, transformers, tokenizers` must exit 1) BEFORE any skip-count parse; smoke labeled plumbing-only (no detection claim); abort leg restores the seeded cache strictly offline and fails closed on cache miss. Validate the YAML locally (yamllint) before first push.
- [x] (coder) Skip-count parsing as a helper script shared by both python jobs (parse the pytest summary line) — workflow shell stays thin, parsing testable in-repo.
- [x] (coder) Rehearse every job's command body locally in the sandbox and record each leg's real output (the lean/full/rust numbers, the smoke's offline run, the abort leg's 26–70 s abort with the recorded command) — CI YAML must not contain a command that has never been executed.
- [x] (bug-hunter) Review: every `uses:` line is a full SHA; no `pull_request_target`; no unrestricted `permissions`; full-python extras exactly `torch hf kernels`; openspec loop one-id-per-invocation with `set -e` semantics; lean job's absence-family assert precedes the skip-count parse; smoke asserts `exit_code == verdict_exit_code(status)` on the real report offline and claims nothing beyond plumbing; abort leg is offline-only with fail-closed cache miss; seed job is never-skippable with its network gated at step level (delta 2026-10-20 — see the section below, which supersedes the earlier `workflow_dispatch`-only shape). **DONE (bug-hunter, 2026-10-25): all nine clauses PASS on the landed bytes** — ci.yml sha256 `d2a33700…`, working tree == HEAD `6423998`. Evidence (real RCs): 17/17 `uses:` refs are 40-hex; all 4 unique pins resolve 200 at api.github.com, and checkout `v7.0.1`/cache `v6.1.0` compare `identical` (ahead_by 0) to their pins; `pull_request_target` only in the forbidding comment (ci.yml:5); sole `permissions:` is top-level `contents: read` (:28-29), zero job-level overrides/`if`s; full-python sync exactly `--extra torch --extra hf --extra kernels --locked` (:104); openspec loop one-id-per-invocation (:171-184) under `bash -euo pipefail {0}` (:33), rehearsed verbatim on current bytes: 24 active changes valid, 0 failures, yamllint rc=0; absence step (:75-81) precedes the parse (:85-90) — lean recipe re-measured: absence rc=1, 412 passed/16 skipped (all 16 = `test_kernels_parity.py:45` Rust leg), `pytest_summary --expect-skipped 16` rc=0 (cross-check: kernels-present run 428 passed/0 skipped = 412+16 exactly, so the landed 16-constant survived lane-4's post-record additions); smoke asserts `rc == exit_code(status)` on the real offline report (driver_smoke.py:235 → report_checks.py:80-81, strict parse :46-54, plumbing-only label ci.yml:187/:202-206); abort leg offline + `fail-on-cache-miss: true` (:220-221/:240) with fail-closed `cache_findings` before any driver run (abort_leg.py:60-85/:167-169); seed has no job-level `if`, unconditional `lookup-only` (:272-278), exact gate literal on fetch/save (:280/:319), abort-leg `needs: seed-weights` (:217) — independent PyYAML parse all-pass. Gates: `uv sync --all-groups --extra kernels` rc=0; `uv run pytest` 428 passed/0 skipped rc=0 (lean+dev+kernels env, torch/hf extras pruned by the sync); `tests/test_ci_referee.py` 69 passed rc=0. Findings: 0 blockers, 1 nit (concurrent-seed save race — no `concurrency` group; recorded in the review comment for a follow-up, not a spec violation); tasks.md:7 (sec-eng threat pass) left open — not mine.
- [x] (sec-eng) Threat pass: no secret access from PR runs (`secrets: none`), action SHAs match declared upstream heads, smoke/abort jobs cannot become executors (no `pull_request_target`, contents read-only), the seed job's live-network allowance cannot be triggered by PR content. **DONE 2026-10-25 — all four clauses PASS, zero merge-blocking findings; evidence below.**
- [x] (orchestrator) `npx --yes @fission-ai/openspec validate qk-ci-referee --strict`; fail-fast already landed (b5712d5), so sequencing is clear; scoped stage + commit (workflow + change dir, one commit); fenced push; watch the first real CI run — the push to main itself triggers the workflow, which is this change's own acceptance test. **DONE:** commits `a91684d` (tune/eval split + `_ABORT_IDENTITY` derivation + OOM advisory) and `b8815ce` (CI auto-seed + referee pins), pushed `f438177..b8815ce` with `PUSH_RC=0`, fence restored, key cleaned.
- [x] (orchestrator + team) Record the first CI run's verdict per job in this tasks file — the repo's first independent referee result. **DONE — run `39514036948` on `b8815ce` (event `push`), conclusion `success`: 7/7 jobs green.** Per job, **read back from the GitHub API rather than relayed**: `seed the weights cache` ✅ · `lean python` ✅ · `full python` ✅ · `rust kernels` ✅ · `openspec` ✅ · `driver smoke` ✅ · `abort leg` ✅. Step-level proof the delta works in the wild: the seed job's **lookup ran unconditionally** (step 4 success on a cold cache), the **fetch self-ran with no manual dispatch** (step 5), the "already seeded" no-op branch correctly **skipped** (step 6), the save succeeded (step 7); then the abort leg's fail-closed restore **hit** that freshly seeded cache (step 5) and executed the recorded abort command for real (step 6). No job skipped, no seed chore, no red mail.

## Evidence: workflow authorship and the command rehearsals (coder, 2026-10-20)

Every job body below was executed locally in this sandbox before the YAML was
written; the outputs are the real ones. `UV_PROJECT_ENVIRONMENT` was pointed at
fresh env dirs (`/opt/data/tmp/ci-full-venv`, `/opt/data/tmp/ci-rust-venv`) and
the lean interpreter at `/opt/data/tmp/ci-lean-venv`, because the sandbox has an
active `VIRTUAL_ENV` of its own that must not be pruned; in CI those are the
default `$PWD/.venv` / `.venv-lean` and nothing else changes.

### Task 1 — `.github/workflows/ci.yml` (six jobs + a seed job; the seed was `workflow_dispatch`-only as authored — superseded by the self-seeding delta at the end of this file)

- Actions are pinned by full commit SHA, **resolved live against the GitHub API
  while authoring** (never from memory), each line carrying repo + release
  provenance: `actions/checkout` v7.0.1 → `3d3c42e5aac5ba805825da76410c181273ba90b1`,
  `astral-sh/setup-uv` v10.3.0 → `1c37ad07a6a961277cf70c0d37d6f313000f5884`,
  `actions/cache` v6.1.0 → `55cc8345863c7cc4c66a329aec7e433d2d1c52a9` (the
  `restore` and `save` sub-actions resolve to it), `dtolnay/rust-toolchain` v1 →
  `683b588c487a69d01f6c6ddf40f036ff682cc2c8`. Input names were read from each
  action's `action.yml` at that SHA before use.
- `permissions: contents: read`; triggers `pull_request` + `push: [main]` +
  `workflow_dispatch`; never `pull_request_target`; no secrets anywhere. Every
  shell step runs under `bash -euo pipefail {0}` via `defaults.run.shell`, so no
  step can forget it.
- lean job builds the interpreter with the proven recipe and asserts the absence
  family first; full job syncs `torch hf kernels` explicitly and requires exactly
  zero skips; rust job pins the toolchain (1.98.1 + clippy/rustfmt), runs
  test/clippy `-D warnings`/fmt `--check` and `uv run maturin build`; openspec job
  validates one id per invocation with the CLI pinned to `1.14.3` and fails closed
  on an empty glob; smoke job is labeled plumbing-only and offline; abort leg
  restores the weights cache with `fail-on-cache-miss: true`; the seed job is the
  only place network is allowed — **never skippable, its fetch/save steps gated at
  step level** (as authored it ran only on `workflow_dispatch`; superseded by the
  self-seeding delta at the end of this file).
- YAML validated: `uvx --from yamllint yamllint .github/workflows/ci.yml` →
  **rc 0, no findings** (config committed as `.yamllint.yaml`: `relaxed` plus
  `line-length: max 200`, because the deliberate provenance prose would otherwise
  bury real findings in warnings). pre-commit's `check-yaml` hook also passes.

### Task 2 — the shared skip-count helper

- `tools/ci/pytest_summary.py` (used by both python jobs; the workflow shell only
  captures the log and passes pytest's own rc through) plus the referee's report
  assertions in `tools/ci/report_checks.py`. That logic lives in-repo so it is
  unit-tested: `tests/test_ci_referee.py` covers the parser's real summary shapes,
  the CLI's exit codes (0 green / 1 findings / 2 unparseable), every way each
  job's report contract can be broken, the abort leg's fail-closed cache check,
  and the workflow's own structural pins (full-SHA `uses:`, no privileged
  trigger, no `|| true`/`continue-on-error`/`set +e`, absence-before-count
  ordering, per-tier `--expect-skipped`, and the cache key naming the revisions
  the abort tooling asserts).
- Measured: `tests/test_ci_referee.py` → **42 passed, 0 skipped** (part of the
  lean tier's 122), rc 0.

### Task 3 — every command body rehearsed, with real output

| job body | command (as in the YAML) | output |
| --- | --- | --- |
| lean-python | `uv venv --python 3.12 …` + `VIRTUAL_ENV=… uv pip install -e . pytest` | fresh interpreter built from the dependency floor; `import torch, transformers, tokenizers` → **exit 1** (absence proven) |
| lean-python | `… -m pytest --color=no -q -rs`, then `pytest_summary.py --pytest-rc "$rc" --expect-skipped 16` | **122 passed, 16 skipped**; `skip discipline holds` (16 = `tests/test_kernels_parity.py` Rust leg, asserted exactly) |
| full-python | `uv sync --extra torch --extra hf --extra kernels --locked` | synced (torch 2.14.1, transformers, pyarrow, the PyO3 kernels wheel) |
| full-python | `uv run --no-sync pytest --color=no -q -rs`, then `--expect-skipped 0` | **202 passed, 0 skipped**; `skip discipline holds` |
| rust-kernels | `cargo test --manifest-path crates/quantibbler-kernels/Cargo.toml --all-targets` | **24 passed; 0 failed** |
| rust-kernels | `cargo clippy … --all-targets -- -D warnings` | clean (`Finished dev profile`) |
| rust-kernels | `cargo fmt --manifest-path … --all -- --check` | clean |
| rust-kernels | `uv run maturin build --manifest-path crates/quantibbler-kernels/Cargo.toml` | `📦 Built wheel for abi3 Python ≥ 3.11 to target/wheels/quantibbler_kernels-0.1.0-cp311-abi3-manylinux_2_35_x86_64.whl` |
| openspec | the one-id-per-invocation loop, `@fission-ai/openspec@1.14.3 --strict` | `qk-ci-referee`, `qk-divergence-guards`, `qk-e2e-quality-gate`, `qk-fail-fast-loop`, `qk-nq-recreation-bar`, `qk-sage-attention` all `is valid`; **`qk-flash-attention` rc=1** — see OQ-2 in design.md (pre-existing, committed in `8a07556`) |
| driver-smoke | `uv run --no-sync python tools/ci/driver_smoke.py --out …` | fixture built offline (3 blocks, rank 4, seq 32); driver **exit 0**; `status ok`, `artifact_exported true`, provenance = invoked args; `plumbing evidence: probe histories at depths [2, 3] against one pedestal 511.6183073091093`; contract held |
| abort-leg (seed) | `uvx --from huggingface-hub==2.2.0 hf download … --cache-dir "$PWD/hf-cache"` ×2 + the revision assert | **1.5G** cache; both snapshots landed at the pinned revisions (`models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`, `datasets--Salesforce--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3`) — upstream has not moved |
| abort-leg | `uv run --no-sync python tools/ci/abort_leg.py --cache … --out …` | driver **exit 1**; aborted at **depth 2 (`layers.1`)**, `prefix_ppl` 2,906,604.68 vs fp 83.08 → **ratio 34,983.698** against `line` 40.0; history depths `[2]`; the four end-of-run numerics `null`; no artifact exported; **wall 29.3 s**; contract held |

### Two defects the rehearsal caught (why task 3 is not optional)

1. **`QUANTIBBLER_KD_TMPDIR` must be an existing directory.** The first smoke
   rehearsal died at scale calibration: `scale_calib.py` hands that path to
   `tempfile.TemporaryDirectory(dir=…)`, which does not create parents
   (`FileNotFoundError: …/kd-tmp/quantibbler-kd-…`). The driver's own default is
   the sandbox path `/opt/data/tmp`, which exists on no CI runner, so a job that
   left the variable unset would die the same way. Both runner scripts now create
   the directory they point at.
2. **The openspec job is red on a change committed before this one.**
   `qk-flash-attention` fails `--strict` on a >500-character requirement text
   (1,204 chars, `qk-kernels/spec.md`). Recorded as OQ-2 in design.md, owned by
   @planner: the first CI run on this change's push cannot be green until that
   spec text is split or the change is archived. Softening the job (excluding
   dirs, dropping `--strict`) was rejected — surfacing this class of defect is
   the job working.

### Task-4/5 review fixes (SF-1, SF-2) — implemented and rehearsed (2026-10-20)

- **SF-1 (@sec-eng): the fetch is now pinned, and the rehearsal found its second
  half.** Both `hf download` calls carry `--revision <sha>`, so the fetch contract
  equals the cache-key contract. Measured while doing it: a revision-pinned fetch
  writes `snapshots/<sha>/` and **no `refs/`**, while the driver resolves
  `--model Qwen/Qwen3-0.6B` offline *through* `refs/main` — so the verify loop now
  both asserts the snapshot exists and restores the branch ref (`printf '%s'
  "$revision" > …/refs/main`, byte-identical to a branch fetch: bare 40-char sha,
  no newline). **Proof of the whole chain:** the abort leg runs green against a
  cache built by the new seed body —
  `seeded hf-cache/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca (refs/main -> c1899de2…)`
  then `runtime evidence: aborted at depth 2 (layers.1) with ratio 34983.69806211141
  against line 40.0; history depths [2]; wall 30.4s`.
- **Idempotent re-warm (closes @bug-hunter's operational flag with a third
  option).** The seed's first step is an `actions/cache/restore` with
  `lookup-only: true`, and its `cache-hit` output gates both the fetch and the
  save: dispatching an already-seeded key is a loud green no-op ("already seeded
  at … — nothing to fetch, nothing to save") instead of a red save-step that
  someone has to learn to read as noise. `lookup-only` and the `cache-hit` output
  were read from the pinned action's own `action.yml` (v6.1.0, SHA `55cc8345…`)
  before writing this.
- **SF-2 (@sec-eng): locked resolution everywhere it matters.** `--locked` added to
  all three `uv sync` invocations, to `cargo test` and `cargo clippy`, and to the
  maturin build (`uv run --locked`); the lean leg is explicitly documented in the
  YAML as floating by design so nobody "fixes" it into determinism. Re-rehearsed
  after the change: `uv sync --extra torch --extra hf --extra kernels --locked` +
  suite → **202 passed, 0 skipped**; `cargo test --locked` → **24 passed**;
  `cargo clippy --locked -- -D warnings` clean; `cargo fmt --check` clean;
  `uv run --locked maturin build` → wheel built; driver smoke → contract held.
- **Pins added so neither fix can silently regress:**
  `test_workflow_fetches_weights_by_pinned_revision_and_restores_the_branch_ref`
  (two `--revision` args equal to `PINNED_REVISIONS`, `refs/main` restored,
  `lookup-only` plus both `cache-hit` gates) and
  `test_workflow_never_resolves_dependencies_unlocked` (three `uv sync` runs and
  two cargo gates all `--locked`, maturin `--locked`, and `--locked` **absent**
  from the lean leg).
- **Shape pin for the ref restore (task-4 gap, fixed rather than asterisked).**
  The first version of this pin was a substring — `assert "refs/main" in seed` —
  and @bug-hunter demonstrated the bypass: delete the
  `printf '%s' "$revision" > …/refs/main` write, keep the adjacent
  `echo "seeded … (refs/main -> …)"`, and the substring stays alive while the seed
  ships a cache that cannot resolve a repo id offline — the exact failure the
  test's own docstring claims to prevent. Replaced with a shape assertion:
  `_ref_restore_writes()` requires an actual `printf '%s' "$<var>" > "…refs/main"`
  write line, and the written variable must be the one bound from the pinned pair
  (`revision="${pair##*:}"`), so a literal sha cannot drift away from the fetched
  revision. Four parametrized bypass mutations must each be rejected —
  delete-the-printf-keep-the-echo, echo-instead-of-printf (a newline, where a
  branch fetch writes none), write-a-different-ref, and literal-sha — and each
  test also asserts its mutation is *not* a no-op, so the battery fails loudly if
  the YAML ever drifts out from under it instead of silently going green.
- **Re-measured on a poisoned tree, not just argued.** With the real `ci.yml`
  carrying bug-hunter's bypass (backup sha `b637a0601e3b…`, restored byte-identical,
  `diff -q` clean, hash unchanged): the new pin test **FAILS**, the old substring
  form still evaluates True on the same text (printed as evidence), and the
  healthy tree is **46 passed**.
- **Tree at this record:** `tests/test_ci_referee.py` **46 passed** (incl. the four
  bypass mutations); lean tier **126 passed/16 skipped** *(skip discipline holds)*;
  full suite **206 passed/0 skipped**; pre-commit RC=0; yamllint RC=0;
  `validate qk-ci-referee --strict` RC=0. `ci.yml` sha256 `b637a0601e3b…` is
  **unchanged** by this repair — the only file that moved is
  `tests/test_ci_referee.py` (`59d8bd7fbc47…` → `3dffeb522be0…`), which is the
  value the sign-off transfer check needs for that path.
- **Honest boundary:** `uses:` steps (the cache lookup/restore/save and the action
  pins) execute only on a runner — their inputs/outputs were verified against each
  action's `action.yml` at the pinned SHA, and every shell body they gate is
  executed locally. That is the one thing in this file that cannot be rehearsed
  here, and it is stated rather than implied.

### First run on main (run `39444761866`): one real red, root-caused and fixed

- **Verdict shape:** openspec ✅, lean-python ✅ (torch proven absent), driver-smoke
  ✅, rust-kernels ✅, **full-python ❌**, abort-leg ❌ *by design* (cache miss —
  fail-closed doing its job; cleared by one `workflow_dispatch` seed run).
- **Root cause, measured rather than guessed:** the driver defaulted
  `QUANTIBBLER_KD_TMPDIR` to the **sandbox path** `/opt/data/tmp` (deliberate here:
  real disk, where `/tmp` is a small tmpfs). `scale_calib` hands that value to
  `tempfile.TemporaryDirectory`, which creates its leaf but **not its parents** —
  so on a runner, where `/opt/data` does not exist, the run dies. Reproduced by
  making the sandbox environment match a runner:
  `QUANTIBBLER_KD_TMPDIR=/nonexistent/ci-runner/dir` →
  `FileNotFoundError: [Errno 2] No such file or directory:
  '/nonexistent/ci-runner/dir/quantibbler-kd-7gtqpwsm'` raised from
  `scale_calib.py:139`, taking the ptq driver test down with it (pre-fix: **1
  failed** in that file, **15 failed** across the ptq tier; post-fix: 66 pass in
  both that shape and with the variable unset). This is precisely why every
  sandbox rehearsal was green: `/opt/data/tmp` exists here and on no runner.
- **Why the smoke leg stayed green on the same run:** `tools/ci/driver_smoke.py`
  sets `QUANTIBBLER_KD_TMPDIR` explicitly — its docstring documents this exact
  trap. The leg that went red is the one that runs the driver through pytest,
  which inherited the default.
- **Fix, three layers, minimal:**
  1. **Driver:** the sandbox scratch preference is now existence-probed with a
     platform-scratch fallback (`str(_SANDBOX_SCRATCH) if
     _SANDBOX_SCRATCH.is_dir() else tempfile.gettempdir()`), and breadcrumbs
     follow the resolved directory instead of a second hardcoded sandbox path.
  2. **`scale_calib._kd_scratch_dir()`:** a nominated scratch directory is
     created (`mkdir(parents=True, exist_ok=True)`) so a set-but-missing override
     cannot reach tempfile as a missing parent — the same defect class, at the
     point of use rather than only at the default.
  3. **Test:** the ptq driver test owns its KD directory
     (`env["QUANTIBBLER_KD_TMPDIR"] = <tmp>/kd-tmp`) instead of inheriting the
     host's scratch location.
- **Pins so this class cannot return:**
  `test_driver_scratch_dir_default_is_not_a_sandbox_only_path` (shape: exactly one
  default, existence probe present, fallback present, sandbox preference named —
  plus 4 rejected shapes: bare literal, no probe, no fallback, two defaults) and
  `test_kd_scratch_dir_creates_a_nominated_path_that_does_not_exist` /
  `..._defers_to_the_platform_default_when_unset`.
- **Gates after the fix:** lean **131 passed/16 skipped** (`skip discipline
  holds`); full suite **213 passed/0 skipped**; ptq tier **66 passed** both with
  `QUANTIBBLER_KD_TMPDIR` unset and with a bogus value exported; pre-commit RC=0
  (one ruff-format rewrite, re-verified after); all active changes strict-valid.
- **Honesty boundary:** the runner branch is proven by evaluating the shipped
  default expression with the existence probe forced (sandbox → `/opt/data/tmp`,
  absent → `/tmp`; both existing directories) and by the environment-matched
  reproduction above. Hiding `/opt/data/tmp` from a real process in this sandbox
  was not possible — a tmpfs or bind mount leaves the mount point a directory, and
  a namespace that removes the path also removes the repo path. The next real run
  is the end-to-end proof.

### Task-4 drift fix: the scratch-dir pin is structural, not token-grepping (2026-10-20)

@bug-hunter found two one-token drifts that the first version of the pin accepted
(both reproduced here against the real driver before fixing):

- **A — probe polarity flip.** `str(_SANDBOX_SCRATCH) if not
  _SANDBOX_SCRATCH.is_dir() else tempfile.gettempdir()` keeps all three tokens
  (`is_dir()`, `tempfile.gettempdir()`, and the sandbox path via the constant) and
  inverts the behavior: fine in the sandbox, crash on a runner.
- **B — nested fallback.** `… else str(Path(tempfile.gettempdir()) /
  "quantibbler-kd")` keeps all three tokens and reintroduces the identical
  missing-parent `FileNotFoundError`, because `TemporaryDirectory` creates its leaf
  and not its parents and `/tmp/quantibbler-kd` exists on no fresh runner.
- **Reproduction:** with v1 of the pin, both mutations of the *real* driver were
  **accepted** (empty findings); with the structural pin both are **rejected**, for
  the right reason each time — "the probe must be the true condition of the branch"
  and "the fallback must be exactly the platform scratch dir".
- **Fix (test-only, `tests/test_qk_report.py`, +112/−22):** the default is now
  pinned by AST shape — it must be an `ast.IfExp` whose `test` is a bare
  `.is_dir()` call (never wrapped in `UnaryOp(Not)`) and whose `orelse` unparses to
  exactly `tempfile.gettempdir()`. Path arithmetic on top of the platform scratch
  dir is rejected by construction: over-strict is the correct failure direction,
  since a deliberate fallback change must force a deliberate pin revisit.
- **Battery, 6 rows, all applied to the REAL driver source** (a synthetic snippet
  would be rejected by the undefined-constant check rather than by the shape under
  test — that is how A and B survived v1): polarity flip, nested fallback, bare
  literal, no probe, no fallback, two defaults. Each row also asserts the mutation
  is not a no-op, and the two token-grep escapes assert the three old tokens are
  all still present — the battery documents *why* the greps were insufficient.
- **Gates:** lean **133 passed/16 skipped** (`skip discipline holds`), full suite
  **215 passed/0 skipped**, pre-commit RC=0 (no rewrites), all active changes
  strict-valid. Nothing outside `tests/test_qk_report.py` changed, so the seven-file
  sign-off transfer table is untouched.

### Bug-hunter sign-off — final re-verify of the SF-1/SF-2 tree (2026-10-20)

- **Clean tree:** `tests/test_ci_referee.py` 46 passed / 0 skipped; lean suite
  126 passed / 16 skipped rc=0; pre-commit RC=0; `validate qk-ci-referee
  --strict` RC=0 — all re-measured in a torch-absent interpreter.
- **Independent mutation battery, 7 shapes, hash-guarded restore (byte-identical
  after each):** delete-printf-keep-echo (the demonstrated bypass), echo-instead-of-printf,
  write-different-ref, literal-sha, drop-`%s` (newline guarantee), swap the
  binding to the repo name — **all caught** by `_ref_restore_writes` == `["revision"]`
  plus the binding assert. Tree hashes at sign-off: `ci.yml` `b637a0601e3b…`
  (unchanged through the repair), `tests/test_ci_referee.py` `3dffeb522be0…`.
- **Named residual, accepted (adversarial-only, non-blocking):** a *later
  overwrite* of `refs/main` after the pinned printf (printf-then-`echo x >`
  refs/main) passes the shape pin — the pin asserts the write exists, not that
  it is the last write. Same tier as the nested-lambda and callee-mutates
  residuals: not a natural refactor shape, and the runtime abort leg catches a
  bad cache on the first real runner run. One-clause hardening available
  (collect every `refs/main` redirect; require exactly one and it must be the
  printf) — rides a future edit if wanted; doctrine one-liner suggested.
- **Transfer condition:** sign-off applies iff
  `git diff <landing-commit> -- .github/workflows/ci.yml .yamllint.yaml
  tools/ci/pytest_summary.py tools/ci/report_checks.py tools/ci/driver_smoke.py
  tools/ci/abort_leg.py tests/test_ci_referee.py` is empty against the hashes
  above.

### Rust-job scope fix: the purity guards now have a job that runs them (coder, 2026-10-20)

- **Finding (my own defect, surfaced while landing the cubek purity pair):** the
  rust job was `--manifest-path`-scoped to the kernels crate, so
  `crates/quantibbler-server`'s test targets never executed in CI —
  `binary_purity.rs` had been local-only since the server landed, and the new
  `cubek_purity.rs` pair would have shipped unenforced. Measured rather than
  argued: a deliberately failing test appended to `cubek_purity.rs` was
  **invisible** to the old form (0 mentions in its output) and red
  (`test result: FAILED`, rc=101) under the new one.
- **Fix:** `cargo test` and `cargo clippy` now select explicitly — `--locked -p
  quantibbler-kernels -p quantibbler-server --all-targets` — instead of
  `--manifest-path "$MANIFEST"`. `cargo fmt` keeps `--all`: measured that the
  manifest-path form *already* covered every member (a deliberately poisoned
  server file was flagged by it), so fmt's scope was never the gap; what the pin
  asserts there is workspace-wide `--all --check`. `MANIFEST` remains for the
  maturin build, where it is the correct argument.
- **Spec delta:** new `qk-ci` requirement "CI SHALL execute every crate that
  carries a purity guard" with two scenarios — the job executes the server's
  guards, and the referee pins the selection.
- **Pins (`tests/test_ci_referee.py`, +126):**
  `test_rust_job_executes_the_crates_carrying_the_purity_guards` reads the rust
  job's gate lines (comments dropped first, so the job's own explanatory comment
  cannot satisfy a check) and requires both `-p` packages in test and clippy,
  rejects `--manifest-path`/`--exclude` on those gates, and requires fmt to be
  workspace-wide. Five-row bypass battery, each rejected and each asserted
  non-no-op: revert-to-manifest-path, drop-server-from-test,
  drop-server-from-clippy, `--workspace --exclude quantibbler-server` (the
  lexical dodge that keeps the name alive), fmt-no-longer-workspace-wide.
- **Demonstrated on the real workflow, not only in the battery:** with `ci.yml`
  reverted in place to the pre-fix gate lines (the name still appears twice in
  the file, so a substring check passes), the pin **fails** naming both narrowed
  gates while `'quantibbler-server' in text` still evaluates True; then restored
  byte-identical — `8cfb8511f5e3` before and after — with the pin green again.
- **Rehearsed job bodies (offline, i.e. the cache-present case):**
  `cargo test --locked -p quantibbler-kernels -p quantibbler-server --all-targets`
  → **150 passed / 0 failed**, with `binary_purity.rs` (12 passed) and
  `cubek_purity.rs` (2 passed) both executing; `cargo clippy … -D warnings` rc=0;
  `cargo fmt --all --check` rc=0.
- **Existing pins checked, not assumed:** `test_workflow_never_resolves_dependencies_unlocked`
  still finds exactly two `cargo (test|clippy)` run lines with `--locked` in
  both; no pin asserts `--manifest-path` presence; the `rust-kernels` job key is
  unchanged, so `_job_block` still resolves it.
- **Gates at this record:** `tests/test_ci_referee.py` **52 passed** (46 + the
  positive pin + 5 battery rows); full suite **434 passed**; lean tier
  **352 passed / 16 skipped** (`skip discipline holds`); pre-commit RC=0;
  yamllint RC=0; `validate qk-ci-referee --strict` RC=0.
- **Fresh transfer hashes for this cycle** (the earlier seven-file transfer
  verified against `5e7e4e2` is closed and undisturbed; these are its
  successors, not a violation of it): `.github/workflows/ci.yml`
  `8cfb8511f5e3`, `tests/test_ci_referee.py` `718d1709327c`,
  `openspec/changes/qk-ci-referee/specs/qk-ci/spec.md` `88361731fee7`.
  `tools/ci/*` and `.yamllint.yaml` are **untouched** by this cycle.
- **Honesty boundary:** unchanged — the `uses:` steps (cache lookup/restore/save)
  still execute only on a runner; every shell body they gate is rehearsed here,
  the new gate bodies included.

### Self-seeding weights job — the delta (coder, 2026-10-20)

Binding shape frozen by @planner before implementation; @bug-hunter's four-row
battery specified up front and run against the REAL workflow text.

- [x] (coder) `ci.yml`: the seed job is never skippable (its job-level
  `if: github.event_name == 'workflow_dispatch'` is gone); the fetch and save
  steps are gated on the literal `steps.seeded.outputs.cache-hit != 'true' &&
  github.event_name != 'pull_request'`; the lookup-only restore stays
  unconditional; `abort-leg` gains `needs: seed-weights` so a condition-skip on
  the seed can never propagate into the leg.
- [x] (coder) Pins (`tests/test_ci_referee.py`, +148 lines): the new
  `_seed_shape_findings()` reads the *uncommented* workflow and asserts the four
  properties directly — no `^    if:` in the seed job, no `if:` on the lookup
  step, the exact gate literal on both network steps, and `needs: seed-weights`
  on the abort leg. The old `assert "workflow_dispatch" in seed` was **moved**,
  not deleted: the trigger set is now asserted positively
  (`pull_request` + `workflow_dispatch` + `branches: [main]`) in the new pin, and
  the negative is the absent job-level condition.
- [x] (coder) Battery, four rows, each applied to the real workflow text, each
  asserted non-no-op first, each required to produce a finding: job-level `if`
  reinstated, PR guard dropped from the fetch, `needs` dropped, lookup restore
  made conditional. **All four rejected.** `tests/test_ci_referee.py` →
  **57 passed** (was 52: +1 positive pin, +4 battery rows).
- [x] (coder) Spec delta: new `qk-ci` requirement "The weights seed SHALL be
  self-seeding and never skippable" with five scenarios —
  `seed_job_never_skippable`, `network_gated_at_step_level`,
  `cold_cache_self_heals_on_push`, `pr_runs_execute_the_abort_leg`,
  `pr_never_fabricates_a_cache`. The residual is recorded verbatim in design.md
  ("Implementation crystallization — self-seeding weights job (2026-10-20)").
- [x] (coder) Local gates at this record: yamllint **RC=0**;
  `tests/test_ci_referee.py` **57 passed**; full suite **444 passed** (was 439);
  lean tier **359 passed / 16 skipped**, `skip discipline holds`; pre-commit
  **RC=0**; `openspec validate qk-ci-referee --strict` valid.
- **The residual, verbatim (per the freeze):** if the cache is evicted (7-day
  sweep on an idle repo) and a PR runs before the next push to main, the PR's
  seed cannot fetch (the event guard blocks network, correctly), its lookup
  misses, and the abort leg **reds fail-closed** until the next push re-seeds —
  the design working, not a defect: **a PR never fabricates a weights cache.**
- **Honesty boundary for this delta:** the `uses:` steps and the runner's
  trigger/skip semantics execute only on a runner. The four-row battery proves
  the pins reject the mutations in the file; the in-the-wild proof that a PR
  run's abort leg *executes* rather than skips is the post-landing
  `pr-proof-abort-leg` run (planner pre-committed), and that run id is this
  delta's evidence — not a local rehearsal.

### Referee-only hardening cycle (coder, 2026-10-21) — after the first 7/7

@planner's contract for this commit, implemented as written: the
parse-don't-enumerate route for the seed-shape pin, `pyyaml` as an explicit
test-harness dependency, an extended battery, and the stale *current-mechanism*
prose corrected. The frozen decision was **no regex enumeration** — a text
matcher for a step's condition must enumerate YAML's key spellings *and* its
block boundaries by hand, and both enumerations leaked (measured last cycle: three
poison forms returned `rc=0`, i.e. the pin missed them).

- [x] (coder) `_step_condition` (hand-rolled text scan) **removed**.
  `_seed_shape_findings()` now `yaml.safe_load`s the workflow and inspects the
  parsed structure: no `if` key on the seed job; no `if` on the lookup step;
  `with["lookup-only"] is True`; both network steps' `if` VALUE exactly equal to
  `_SEED_GATE`; `jobs["abort-leg"]["needs"] == "seed-weights"`. A workflow that
  does not parse is itself a finding, never a traceback. `_triggers()` reads
  `on:` under both `True` (PyYAML resolves the bare key `on` to the boolean) and
  `"on"`, and the positive pin now asserts the trigger set as an **exact key
  set** — so an added `pull_request_target` reds instead of riding along as an
  unnoticed extra.
- [x] (coder) `pyyaml>=6,<7` added to the dev group with the regenerated
  `uv.lock` (**additive only**: two dev-group edges, no version churn), and
  installed **explicitly** in the lean CI recipe
  (`uv pip install -e . pytest "pyyaml>=6,<7"`) because PyYAML is not in the
  product floor — measured: the lean interpreter raised
  `ModuleNotFoundError: No module named 'yaml'` before this commit. The lean spec
  scenario now names the harness dependency, and a pin asserts `pyyaml` is in the
  lean recipe. The torch/transformers/tokenizers absence proof is unchanged.
- [x] (coder) Battery extended from four rows to **nine**, each applied to the
  REAL workflow text, each asserting non-no-op first, **and each now also
  asserting the poison still PARSES** (a syntax error would red the parser rather
  than the pin, proving nothing about the checks). Rows: job-level `if`
  reinstated · PR guard dropped from the fetch · `needs` dropped · lookup
  conditional placed first in the step · the same condition placed last · a blank
  line inside the step then the condition · a quoted `'if':` key · `if :` with a
  spaced colon · `lookup-only` flipped to `false`. `tests/test_ci_referee.py` →
  **62 passed**.
- [x] (coder) **Poisoned-tree proof on the real file's bytes** (the proof this pin
  owes, and the one the previous pin had), run on a mirror at `/tmp/poison-demo`
  so the shared tree was never written — verified by hash, `ci.yml` identical
  before and after. Healthy mirror control green (10 passed); **all seven poison
  forms CAUGHT**: `blank-line-inside-step`, `quoted-'if'-key`,
  `spaced-'if :'-colon`, `lookup-only->false`, `cond-before-uses`,
  `cond-after-lookup-only`, `job-level-if-back`. The first three are exactly the
  spellings the removed text matcher escaped; the parser closes them by
  construction.
- [x] (coder) Stale *current-mechanism* prose corrected, historical "was
  `workflow_dispatch`-only" passages preserved: `proposal.md` abort-leg bullet;
  `design.md` "Weights cache mechanism" (self-seeding on push, unconditional
  lookup, PR-gated fetch/save, the per-repo 10 GB / 7-day-LRU threat model,
  next-push recovery) and the cache-poisoning residual (the reason is now the
  step-level PR gate, not "dispatch-only"); `tasks.md` Task-1 heading/prose carry
  a supersede pointer instead of describing the old shape as current. `tools/ci/*`
  left **untouched**.
- [x] (coder) First main run recorded above: **`39514036948`, 7/7 green**, read
  back per job from the GitHub API — the run's own job list, not a relay.
- [x] (orchestrator + @user) **PR-shaped proof — WAIVED by @user (recorded
  date corrected — see the correction immediately below).**
  > **Date correction (coder, 2026-10-22).** The recorded date was five to six days
  > *after* every commit that records it: this line was authored 2026-10-22
  > 03:46 UTC and the string is impossible relative to the repository's own
  > clock. It was **not** a lone typo here — it was one of **33** occurrences across
  > 20+ records in this repo (`AGENTS.md:69`,
  > `crates/quantibbler-kernels/tests/cubek_cpu_reference_parity.rs:63,:74,:468`,
  > and the `qk-nq-recreation-bar`, `qk-ci-referee`, `qk-flash-attention` and
  > archived change dirs), a per-lane convention error. Whether the
  > clock or the string was wrong was @planner/@user's call, and the true date was
  > **not recoverable from the repository**. **@user ruled 2026-10-25
  > (`qk-pertarget-cycle` §18): FIX THE STRINGS** — each occurrence replaced
  > with the author date of the commit that records it (t_6f5dcf96). Census
  > and the shortest proof (commit `f438177`, authored 2026-10-20 20:33,
  > recording the phantom-dated decision): see `qk-pertarget-cycle/design.md` §9.
  The branch was cut and pushed (`pr-proof-abort-leg`, one empty commit), then
  deleted again: the user ruled CI sufficient ("Can we move on to fixing the
  actual root cause") before the click landed. Run 9 (`39519807660`) settled
  **7/7 success** on `08c2a29` — second straight all-green. The
  runner-semantics premise for `pull_request` events remains documented-but-
  unmeasured: the in-file pins hold, `push` semantics are proven by two 7/7
  runs, and the PR-trigger negatives are established only by the amended
  shape and the docs, not by a live run. Re-opening is one branch + one click.
- **Honesty boundary:** the battery and the mirror proof show the PINS reject
  mutations *in the file*. Runner trigger/skip semantics stay a runner fact: the
  7/7 establishes them for `push`, and the no-op PR is what establishes them for
  `pull_request`.

## Evidence: sec-eng threat pass (2026-10-25)

Scope: line 7. Audited bytes: `.github/workflows/ci.yml` as committed (working
tree clean for `.github/` and for this change dir; `git status` confirms). Repo
verified PUBLIC via the GitHub API (`"private": false, "visibility": "public"`,
default branch `main`) — the untrusted-fork-PR threat model is live, which is
what makes these clauses load-bearing. Every claim below is a command run in
this pass, not a read-back of the design's own prose.

### C1 — no secret access from PR runs (`secrets: none`): PASS

- `grep -nE "secrets|GITHUB_TOKEN|persist-credentials|inherit" ci.yml` → hits
  ONLY at `:9` and `:213`, both prose comments. Zero functional references: no
  `secrets.*`, no `secrets: inherit`, no explicit `GITHUB_TOKEN` use.
- Token ceiling: `permissions: contents: read` at ci.yml:28-29; job-level
  override scan (`awk 'NR>45 && /permissions/'`) → none anywhere.
- The only workflow file in the repo is `ci.yml` (`ls .github/workflows/`).
- Fork PRs on a public repo receive no repo secrets by platform rule; the only
  credential a PR run holds is the read-only `GITHUB_TOKEN`.
- Executable tripwire: tests/test_ci_referee.py:512 (`assert "secrets." not in
  text`) inside `test_workflow_is_supply_chain_hardened` (:503-519).

### C2 — action SHAs match declared upstream heads: PASS (verified live)

- All 17 `uses:` references are 40-hex full SHAs (non-full-SHA grep → empty,
  rc 1). `git ls-remote` against each upstream with tag peeling, run
  2026-10-25 — every declared tag resolves to exactly the pinned SHA today:
  - `actions/checkout` v7.0.1 → `3d3c42e5…` **MATCH** (ci.yml:51, 97, 121,
    158, 194, 224, 262)
  - `astral-sh/setup-uv` v10.3.0 → `1c37ad07…` **MATCH** (ci.yml:53, 99, 147,
    196, 226, 264)
  - `actions/cache` v6.1.0 → `55cc8345…` **MATCH** for both `cache/restore`
    (ci.yml:236, 274) and `cache/save` (ci.yml:320)
  - `dtolnay/rust-toolchain` v1 → `683b588c…` **MATCH** (ci.yml:127; the
    moving `v1` tag still points at the pinned commit)
- Tripwire: test_ci_referee.py:505-510 fullmatch-pins every `uses:`.
- Registry-installed tools stay exact-version pins without hashes
  (`npx @fission-ai/openspec@1.14.3` ci.yml:177; `uvx --from
  huggingface-hub==2.2.0` ci.yml:282, 284) — the design's recorded accepted
  residual; re-adjudicated proportionate at `contents: read` + zero secrets.
  Not merge-blocking.

### C3 — smoke/abort jobs cannot become executors: PASS

- Trigger set is exactly `{pull_request, push:[main], workflow_dispatch}`
  (ci.yml:22-26). `pull_request_target` appears only in the line-5 comment;
  the uncommented-text pins (test_ci_referee.py:511, :689) and the parsed
  exact-trigger-set pin (:687) make adding a privileged trigger a red.
- No `workflow_run` / `schedule` / `issue_comment` triggers (grep → rc 1).
- Top-level `contents: read`, no job-level escalation (C1 evidence). PR runs
  execute from the merge ref with a read-only token even when the PR rewrites
  the workflow — matches the recorded trust model (PR green advisory, main
  green authoritative).
- driver-smoke and abort-leg (ci.yml:186-247) do run PR-controlled code — that
  is inherent to CI that tests PR content — but cannot become PRIVILEGED
  executors: `ubuntu-latest` ephemeral runners, zero secrets, read-only token,
  and their own tooling is network-free by code
  (`grep -nE "requests|urllib|urlopen|socket|httpx|hf_hub_download|snapshot_download|from_pretrained" tools/ci/*.py`
  → zero hits) with `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` forced at
  ci.yml:190-191 and 220-221.
- Residual (inherent-to-class, non-blocking, adjacent to the recorded
  npx/uvx residual): a PR's own `pyproject.toml` names packages that
  ci.yml:71 (`uv pip install -e .`) and ci.yml:104 (`uv sync`) install and
  execute on the runner. No escalation path exists: ephemeral runner, no
  secrets, read-only token, PR-scope cache writes invisible to main.

### C4 — the seed job's live-network allowance cannot be triggered by PR content: PASS

- The only live-network steps in the file are the seed fetch (ci.yml:279-312)
  and save (ci.yml:318-323); both carry exactly
  `if: steps.seeded.outputs.cache-hit != 'true' && github.event_name != 'pull_request'`
  (ci.yml:280, 319). `github.event_name` is runner context, not
  PR-controllable: on every `pull_request` event it is `'pull_request'` and
  both steps skip.
- Only three `if:` conditions exist in the whole file (:280, :314, :319) — no
  other conditionality to subvert; the seed job carries no job-level `if`
  (ci.yml:249-259; pinned at test_ci_referee.py:694), and abort-leg
  `needs: seed-weights` (ci.yml:217) cannot inherit a skip.
- Every `${{ }}` in the file interpolates `env.*` only (9 occurrences:
  ci.yml:55, 101, 149, 198, 228, 239, 266, 277, 323) — no PR-controlled value
  reaches a `run:` or `if:` body, so there is no expression-injection route
  into the gate.
- A PR can gut the guard in its own workflow copy — but the fetch then runs on
  the PR's own ephemeral secret-free runner, and its `cache/save` writes the
  PR's merge-ref scope, which main-scope restores never read (GitHub cache
  branch isolation; recorded in design.md). The abort leg restores only,
  fail-closed, under the workflow-fixed `env.HF_CACHE_KEY` (ci.yml:43,
  235-240): no PR-reachable cache-poisoning path to main's weights.
- Boundary (not a finding): `workflow_dispatch` of an attacker's branch would
  pass the event guard, but dispatch requires write access — an actor with
  write can already push to main.

### Findings

- **Merge-blocking: none.** Zero proven vulnerabilities, zero plausible
  weaknesses beyond the design's already-recorded accepted residuals.
- Hardening opportunities (informational, non-blocking; do NOT hold the merge
  for either):
  1. `actions/checkout` defaults `persist-credentials: true`, leaving the
     read-only token in the runner's git config for the rest of the job. On a
     public repo with `contents: read` that token has zero exfiltration value;
     `persist-credentials: false` would be tidy if the file is ever touched
     again.
  2. The PR-controlled dependency-install residual (C3) is distinct from the
     recorded npx/uvx registry-compromise residual; one sentence naming it in
     design.md's accepted residuals would make the adjudication explicit.

### Gates (this pass)

- `uv sync --all-groups --extra kernels` → rc 0.
- `uv run pytest -q -rs` → **428 passed, 0 failed, 0 skipped, rc 0**
  (re-run confirmed rc 0). Tier honesty: this sandbox venv has
  torch/transformers/tokenizers ABSENT (proven via `importlib.util.find_spec`
  → three `False`) with the kernels extension present; `tests/ptq` collects
  228/231 lean (3 integration-marker deselections). The CI full-python job
  (torch extras, zero-skip assertion) is where the torch tier is enforced —
  that is not what ran here, and this count is not a substitute for it.

### Unticked items (not this task's scope)

- Line 6 (bug-hunter review) remains `[ ]` — no evidence of it having run was
  observed in this pass; it is not mine to tick.
