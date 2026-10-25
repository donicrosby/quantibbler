# Proposal: qk-spec-audit

## Why

`openspec/changes/` had accumulated 22 change dirs whose checkbox state — not
the working tree — was the only progress record. A read-only audit (planner
card `qk-spec-audit: verdicts`, `t_b262f90d`, reconciled against HEAD `76c322c`
and re-verified at `52b98ea`) produced exactly one verdict per dir: archive,
fold, delete, or alive. This change is that audit's own change shell — it
carries the verdict document as its design, records the executions, and gives
the audit a lifecycle instead of leaving it as a card comment.

The audit found rot *classes*, not one-off mistakes:

- **Shipped-but-unarchived** — six changes whose code had landed and whose
  tasks were complete still sat in `changes/`, inflating every later
  `openspec list` and letting their deltas double-apply on the next fold.
- **Spec-only deltas never folded** — three changes describing behavior the
  code already ships, still unmerged into `openspec/specs/`, so the store read
  as older than the code it governs.
- **Stale task-template paths** — rows naming `tests/ci/` and `tests/qk_report/`
  directories that never existed (the files are `tests/test_ci_referee.py` and
  `tests/test_qk_report.py`), and a bare `uv run maturin build` that cannot work
  in this uv workspace without `--manifest-path`.
- **One proposal-only dir** whose delta must be re-homed rather than deleted,
  and **one orphaned implementation** with no change dir at all.

## What Changes

- Land `openspec/changes/qk-spec-audit/` — this change shell. Its `design.md`
  is the reconciled verdict document (one verdict per dir, with evidence and
  execution notes); `tasks.md` is the execution ledger.
- Execute the verdicts: archive the fully-landed changes (their deltas fold
  into `openspec/specs/` on archive), fold the spec-only deltas and archive
  them, and rewrite the alive changes' `tasks.md` so the open work is the
  honest backlog rather than a stale checklist.
- Scaffold the orphaned implementation as a new alive change
  (`qk-checkpoint-integrity`) so its patch has a home (ruling R1).
- Reconcile the `rho_end` SHALL in `qk-fix-admm-solver` to the lane-3 accepted
  regime before that delta is folded (ruling R4).
- Apply the template-rot path fixes where the verdict document routes them (R6).

## Impact

- **Affected specs**: `openspec/specs/` gains the capabilities the archived
  deltas carry. No requirement is weakened. Deletions carry two-level gate
  evidence (L1 symbol grep, L2 mechanism grep) in the commit body.
- **Affected code**: none. This change is store bookkeeping; it lands no
  production code and no test.
- **Not invariant-touching**: no qpk1 container format, no kernels, no server
  contract is touched. Compatibility impact: none.

## Non-goals

- Not a spec rewrite: no requirement's meaning changes except the single
  `rho_end` reconciliation (R4) the orchestrator ruled.
- Not an implementation lane: a change the audit keeps alive stays alive with
  its own open tasks; this audit does not absorb or advance them.
- Not a cleanup excuse: nothing is deleted without a reproduced two-level gate,
  and a verdict whose gate does not reproduce is escalated to the planner
  rather than executed.
