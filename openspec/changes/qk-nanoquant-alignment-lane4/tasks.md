# Tasks: qk-nanoquant-alignment-lane4 (rewritten 2026-10-25)

> **Audit landing note (2026-10-25, `t_039035de`, ruling: verdicts from
> `qk-spec-audit`).** This `tasks.md` is a REWRITE, not the historical ledger:
> it replaces the prior task list so the open rows are the change's real
> backlog. Completed work is summarized above the open rows; the detailed
> row-by-row history is in git. Any numeric "suite state" reading below is the
> audit's, taken at the then-current HEAD — re-measure before quoting it.

## Completed

Most tasks checked. Code landed: `d8da177`, `45417d4`, `5c73fe0`, `afb220e`. Phases 1-4 mostly complete. Phase 5 (full gate suite) [x]. Orchestrator validate VALID.

## Open

- [ ] Phase 4, box 4: driver-stage-map pin — test/record-surface pin not yet written. Write the pin: a run carrying a fallback event surfaces it in the run record (stage_map/event counts).
- [ ] Fold surviving REQUIREMENTS into `openspec/specs/`, then archive.

## Notes (2026-10-25)

Template rot: `tests/ci/` → `tests/test_ci_referee.py`, `tests/qk_report/` → `tests/test_qk_report.py`, bare `maturin build` → `--manifest-path` form.
