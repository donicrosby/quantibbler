# Tasks: qk-ci-referee

- [x] (coder) Author `.github/workflows/ci.yml` per the final spec: SIX jobs (lean-python, full-python, rust-kernels, openspec, driver-smoke, abort-leg) plus a `workflow_dispatch`-only seed job; `bash -euo pipefail` defaults; SHA-pinned actions resolved fresh at landing (never from memory); `permissions: contents: read`; triggers `pull_request` + `push` to `main` (never `pull_request_target`); zero secrets. Lean job = proven recipe (fresh `uv venv` + `pip install -e . pytest`, no extras) with the absence-family assert (`import torch, transformers, tokenizers` must exit 1) BEFORE any skip-count parse; smoke labeled plumbing-only (no detection claim); abort leg restores the seeded cache strictly offline and fails closed on cache miss. Validate the YAML locally (yamllint) before first push.
- [x] (coder) Skip-count parsing as a helper script shared by both python jobs (parse the pytest summary line) — workflow shell stays thin, parsing testable in-repo.
- [x] (coder) Rehearse every job's command body locally in the sandbox and record each leg's real output (the lean/full/rust numbers, the smoke's offline run, the abort leg's 26–70 s abort with the recorded command) — CI YAML must not contain a command that has never been executed.
- [ ] (bug-hunter) Review: every `uses:` line is a full SHA; no `pull_request_target`; no unrestricted `permissions`; full-python extras exactly `torch hf kernels`; openspec loop one-id-per-invocation with `set -e` semantics; lean job's absence-family assert precedes the skip-count parse; smoke asserts `exit_code == verdict_exit_code(status)` on the real report offline and claims nothing beyond plumbing; abort leg is offline-only with fail-closed cache miss; seed job is `workflow_dispatch`-only.
- [ ] (sec-eng) Threat pass: no secret access from PR runs (`secrets: none`), action SHAs match declared upstream heads, smoke/abort jobs cannot become executors (no `pull_request_target`, contents read-only), the seed job's live-network allowance cannot be triggered by PR content.
- [ ] (orchestrator) `npx --yes @fission-ai/openspec validate qk-ci-referee --strict`; fail-fast already landed (b5712d5), so sequencing is clear; scoped stage + commit (workflow + change dir, one commit); fenced push; watch the first real CI run — the push to main itself triggers the workflow, which is this change's own acceptance test.
- [ ] (orchestrator + team) Record the first CI run's verdict per job in this tasks file — the repo's first independent referee result — and re-run the seed job if the abort leg's cache needs warming before the first green.

## Evidence: workflow authorship and the command rehearsals (coder, 2026-10-20)

Every job body below was executed locally in this sandbox before the YAML was
written; the outputs are the real ones. `UV_PROJECT_ENVIRONMENT` was pointed at
fresh env dirs (`/opt/data/tmp/ci-full-venv`, `/opt/data/tmp/ci-rust-venv`) and
the lean interpreter at `/opt/data/tmp/ci-lean-venv`, because the sandbox has an
active `VIRTUAL_ENV` of its own that must not be pruned; in CI those are the
default `$PWD/.venv` / `.venv-lean` and nothing else changes.

### Task 1 — `.github/workflows/ci.yml` (six jobs + a workflow_dispatch-only seed job)

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
  only place network is allowed and runs only on `workflow_dispatch`.
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
