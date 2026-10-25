# Tasks: qk-nanoquant-alignment-lane2 (rewritten 2026-10-25)

> **Audit landing note (2026-10-25, `t_039035de`, ruling: verdicts from
> `qk-spec-audit`).** This `tasks.md` is a REWRITE, not the historical ledger:
> it replaces the prior task list so the open rows are the change's real
> backlog. Completed work is summarized above the open rows; the detailed
> row-by-row history is in git. Any numeric "suite state" reading below is the
> audit's, taken at the then-current HEAD — re-measure before quoting it.

## Completed

Reconciled 2026-10-25 (t_75564534). Most tasks checked. Code landed: `a90f7e8`, `ec906a7`. Core implementation complete.

## Open

- [ ] Partial shipment (2 tasks): stream provenance fields and stage-map rewording were partially shipped. Verify what's missing against the delta and close or complete.
- [ ] Measurement tasks (Phase 3): two measurement tasks unrun. These are operational (gate on production data), not spec gaps.
- [ ] Fold surviving REQUIREMENTS into `openspec/specs/`, then archive.

## Notes (2026-10-25)

Template rot: `tests/ci/` → `tests/test_ci_referee.py`, `tests/qk_report/` → `tests/test_qk_report.py`, bare `maturin build` → `--manifest-path` form.
