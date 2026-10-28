# Tasks: qk-nanoquant-alignment-lane1 (rewritten 2026-10-25)

> **Audit landing note (2026-10-25, `t_039035de`, ruling: verdicts from
> `qk-spec-audit`).** This `tasks.md` is a REWRITE, not the historical ledger:
> it replaces the prior task list so the open rows are the change's real
> backlog. Completed work is summarized above the open rows; the detailed
> row-by-row history is in git. Any numeric "suite state" reading below is the
> audit's, taken at the then-current HEAD — re-measure before quoting it.

## Completed (phases 1–5)

Reconciled 2026-10-25 (t_41af07ba). Code landed: `e5c6939`, `1484e55`. 5 tasks checked — core implementation complete.

## Open (phase 6)

- [ ] Missing fixture deliverables (3 tasks): the fixtures named in the acceptance criteria were never written. Document the gap and decide: write them or close with a disposition note.
- [ ] Unshipped q95 retirement: the q95 default retirement path was spec'd but not shipped. Document disposition.
- [ ] Partial shipment (1 task): one deliverable was partially shipped. Verify what's missing and close or complete.
- [ ] Audit (1 task): verify that every surviving REQUIREMENT from this change's delta is either in `openspec/specs/` or explicitly retired.

## Notes (2026-10-25)

The unticked items are test/helper debt, not spec gaps. The delta's SHALLs are in code. This change stays alive until the open items are resolved — then fold surviving REQUIREMENTS into specs and archive.

Template rot: `tests/ci/` → `tests/test_ci_referee.py`, `tests/qk_report/` → `tests/test_qk_report.py`, bare `maturin build` → `--manifest-path` form.
