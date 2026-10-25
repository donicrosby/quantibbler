# Tasks: qk-nanoquant-alignment-lane3 (rewritten 2026-10-25)

> **Audit landing note (2026-10-25, `t_039035de`, ruling: verdicts from
> `qk-spec-audit`).** This `tasks.md` is a REWRITE, not the historical ledger:
> it replaces the prior task list so the open rows are the change's real
> backlog. Completed work is summarized above the open rows; the detailed
> row-by-row history is in git. Any numeric "suite state" reading below is the
> audit's, taken at the then-current HEAD — re-measure before quoting it.

## Completed

GREEN implementation landed at `97a4725` + `dba35b8`. Config defaults: `admm_reg=3e-2`, `rho_start=0.0`, `rho_end=1.0`. Scale-relative ridge in continuous update with map-back. Solver probe rows carry new fields + sources. Orchestrator validate VALID RC=0. Gate suite recorded (fire-lane: 652 passed/4 skipped/3 deselected).

## Open

- [ ] Lane-order precheck: confirm Lane-0 baseline measurements (solver probe rows on production layers) and Lane-1 preconditioner fix have landed. Record Lane-0 post-ADMM pre-polish cos distribution and baseline `trace.min_eigs` floor in design.md. No Lane-3 number quoted before both land.
  - **Dependency**: gates on Lane-0/Lane-1 evidence. The acceptance-pin box (task 4, "set cos-improvement threshold FROM Lane-0 measurements") cannot be completed until this precheck is done.
- [ ] Measurement: run new regime on production layers with `QK_SOLVER_PROBE=1`. Record per-layer post-ADMM pre-polish cos (`cos_raw`) and `trace.min_eigs` floor. A/B entry Frobenius normalization (with/without) — record keep-or-remove decision in design.md.
- [ ] Acceptance pin: set cos-improvement threshold FROM Lane-0 measurements (never invented). Assert production-layer improvement and `min_eigs` floor rise as suite tests carrying measured constants.
- [ ] Bug-hunter review: (a) new fields join checkpoint fingerprint, (b) `QK_ADMM_REG` fail-closed, (c) solver probe rows carry new fields + sources, (d) Lane order honored, (e) Frobenius keep/remove decision recorded.
- [ ] SVD ablation: re-run `solver="svd"` under the new regime (`QK_SOLVER_PROBE=1`). Record regime-labelled comparison. Until this row exists, the "ADMM ≈ warm start" claim is barred.
