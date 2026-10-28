# Tasks: qk-spec-audit

The audit's execution ledger. The verdicts and their evidence live in
`design.md` (the reconciled verdict document); this file records what actually
ran. One row per action from the card's scope and definition of done.

## Scaffolding

- [x] Scaffold this change shell (proposal with Non-goals, design = the
  reconciled verdict document, tasks). Evidence: `openspec validate
  qk-spec-audit --strict` → valid, RC=0 (`skip_specs: true` in
  `.openspec.yaml`: the change declares no spec-level behavior change).
- [x] Scaffold `openspec/changes/qk-checkpoint-integrity/` from the orphan patch
  (ruling R1): proposal, design, `specs/qk-ptq` delta and an authored
  `tasks.md`. The patch's code hunks are NOT applied — the follow-up cards
  `t_1b582b7d` / `t_ed0efb97` own the code landing and carry superseding
  revisions. Evidence: `openspec validate qk-checkpoint-integrity --strict`
  → valid, RC=0.

## Verdict execution

- [x] Rewrite `tasks.md` for the alive changes from the reconciled verdict
  document, so each open row is the real backlog. Done for 12 changes
  (`qk-divergence-guards`, `qk-fix-ste-sign-adaptation`,
  `qk-fix-tunefp-teacher`, `qk-flash-attention`, `qk-launch-attribution`,
  `qk-nanoquant-alignment-lane1..lane4`, `qk-pertarget-cycle`,
  `qk-progress-instrumentation`, `qk-ptq-speedups`). **`qk-sage-attention` was
  deliberately NOT rewritten**: a co-tenant lane held uncommitted edits to that
  change's `design.md`/`tasks.md` recording the real-activation quality-gate
  measurement, and overwriting them with the audit's older stub would have
  destroyed that record.
- [x] Archive the fully-landed changes (deltas fold into `openspec/specs/`):
  `qk-ci-referee`, `qk-fp-model-identity`, `qk-geometry-conformance`,
  `qk-nq-recreation-bar`, `qk-solver-backend`. Evidence:
  `openspec/changes/archive/2026-10-25-*`.
- [x] Fold the spec-only deltas into `openspec/specs/` and archive them:
  `qk-fp-resume-shape-guard` (→ `specs/qk-ptq` +1), `qk-wrapper-contract`
  (→ `specs/qk-wrapper`, new capability, +7).
- [x] Reconcile the `rho_end` SHALL in `qk-fix-admm-solver` to `[0.0, 1.0]`
  with a dated crystallization entry BEFORE folding that delta (R4); then
  archive it into `specs/qk-ptq` (~3 modified). Evidence: the amended
  requirement text in `openspec/specs/qk-ptq/spec.md` plus
  `openspec validate qk-fix-admm-solver --strict` RC=0.
- [x] Apply the template-rot path fixes to affected `tasks.md` (R6):
  `tests/ci/` → `tests/test_ci_referee.py`, `tests/qk_report/` →
  `tests/test_qk_report.py`, bare `uv run maturin build` → the
  `--manifest-path` form. Carried in the rewritten files above.
- [x] Run the deletion two-level gate for every delete verdict. **The single
  delete verdict (`qk-test-discipline`) was NOT executed — its gate does not
  reproduce and the verdict is escalated to the planner card `t_b262f90d`
  instead of improvised.** L1 (symbol grep) holds: 13 delta-unique identifiers
  return 0 hits tree-wide outside the change dir. L2 (REQUIREMENT → mechanism)
  **fails**: `prose_fence_stays_hot` is implemented by the tracked
  `tests/test_prose_discipline.py` (landed `f4e5d37`) and
  `integration_marker_admission` by `pyproject.toml:57` `addopts` plus the
  marker description at `:59`. The dir is live — 15 commits in its history,
  most recent `b2864ce`, and lane1's `e5c6939` added 16 lines to its delta — so
  deleting it would discard the only requirement-of-record for mechanisms that
  ship. It is left byte-identical to HEAD pending a re-verdict; the shape the
  evidence supports is fold + archive, which is the planner's call.

## Store hygiene (required by the fold)

- [x] Real `## Purpose` written for the two capabilities the archive created
  (`specs/qk-ci`, `specs/qk-wrapper`); the tool's "TBD - created by archiving"
  placeholder is gone.
- [x] Two requirement bodies the fold carried over the 500-char `--strict` limit
  into `specs/qk-ptq` (699 and 757 chars, from `qk-fix-admm-solver`'s MODIFIED
  delta) split by behavior, content-preserving. No SHALL added, removed or
  changed; no requirement name that a live delta MODIFIES was renamed.
  `openspec validate --specs --strict` → 7 passed / 0 failed.

## Gates

- [x] `npx --yes @fission-ai/openspec validate <id> --strict` — ONE id per
  invocation — for every touched and every remaining change: all 16 remaining
  change dirs RC=0; `validate --specs --strict` 7 passed / 0 failed.
- [x] `set -o pipefail; uv run --no-sync pytest` →
  **1020 passed, 0 failed, 7 deselected, RC=0** in 257.52 s, on a HEAD stable
  before and after the run (`f47e26d`; interpreter `/opt/venv/bin/python3`,
  torch `2.14.1+cpu`, `torch.cuda.is_available()=False`). `--no-sync` is
  deliberate: a literal `uv run pytest` re-syncs the shared `/opt/venv` to the
  floor, drops the torch/transformers/pyarrow extras, and makes
  `tests/ptq/conftest.py`'s collect hook skip the whole ptq tier — a false
  green, not a lighter one. An earlier run of the same bytes at the pre-`f47e26d`
  revision produced `1 failed, 1014 passed, 3 skipped, 4 deselected`; that one
  failure was a concurrent-commit race in
  `test_scn_launch_identity_line_inside_a_checkout` (the test read
  `rev=52b98ea…`, the driver spawned moments later recorded the peer's
  `f47e26d`), it passes standalone, and no code or test was changed by this
  audit.
