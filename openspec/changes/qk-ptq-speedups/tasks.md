# Tasks: qk-ptq-speedups (rewritten 2026-10-25)

> **Audit landing note (2026-10-25, `t_039035de`, ruling: verdicts from
> `qk-spec-audit`).** This `tasks.md` is a REWRITE, not the historical ledger:
> it replaces the prior task list so the open rows are the change's real
> backlog. Completed work is summarized above the open rows; the detailed
> row-by-row history is in git. Any numeric "suite state" reading below is the
> audit's, taken at the then-current HEAD — re-measure before quoting it.

## Completed

- **Task 1**: min_eigs gate — landed 2026-10-25 (t_2522c684). VERIFIED. `tests/ptq/test_min_eigs_gating.py` = 4 passed (one per scenario id). Gate = `SOLVER_PROBE_ENV` resolved once at the solve site. RED observed twice, GREEN after.

- **Task 4**: STE/TuneFP accelerator port — landed 2026-10-25 (t_c3fb5ce8). VERIFIED. `tests/ptq/test_stage_device_parity.py` = 4 passed. Stage records now NAME their device. Report half landed: stage records carry `elapsed_s` + `device`, `qk_report.stage_map` carries rows/elapsed_s/row_time_s per stage. Measured at production SHAPES (not layers): TuneFP 3.63×/ fire, STE 3.39×/linear (load 43-61 on 32 cores). Second run 1.77× / 4.07× — factor is ~3× order of magnitude with contention band.

- **Task 5**: Remaining accelerable sweep — enumerated in design.md with per-piece disposition.

## Open

- [ ] Task 2: Flag-on/off timing measurement on real production layers. Currently blocked on: (a) unmeasurable at 16 threads under co-tenant load (loadavg 51-66 on 32 cores, 176× inflation over 1-thread reading), (b) @user GO required for a run-bearing measurement. Evidence collected at 1 BLAS thread: 2×eigvalsh(512) = 585 ms vs 2×(matmul+Cholesky) = 212 ms/iteration (ratio 2.8). Script and exact launch line on ticket. Needs a quiet window or the ROCm image.
- [ ] Task 3: Full gate suite: `uv run pytest` (quote env + count), `npx --yes @fission-ai/openspec validate qk-ptq-speedups --strict`, scoped commit. Note: `tests/test_ci_referee.py` and `tests/test_qk_report.py` are the real files (not `tests/ci/` or `tests/qk_report/` directories).
- [ ] Task 4 (measurement half): Report per-stage row-time on PRODUCTION layers (not just production shapes). The row-time column now makes the next fire's own stage numbers that measurement — needs a fire (@user GO) to collect.
