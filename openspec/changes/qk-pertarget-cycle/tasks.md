# Tasks: qk-pertarget-cycle (rewritten 2026-10-25)

> **Audit landing note (2026-10-25, `t_039035de`, ruling: verdicts from
> `qk-spec-audit`).** This `tasks.md` is a REWRITE, not the historical ledger:
> it replaces the prior task list so the open rows are the change's real
> backlog. Completed work is summarized above the open rows; the detailed
> row-by-row history is in git. Any numeric "suite state" reading below is the
> audit's, taken at the then-current HEAD — re-measure before quoting it.

## Completed

All code landed in `d78f648` (branch `qk-pertarget-cycle`, = `7db1ab9` + this + `299f9b8`). All acceptance conditions except "Re-fire the faithful run" met:

- TuneFP hoisted into factorization loop ahead of each target's `_set_linear`; tunable set = still-FP remainder.
- Pre-group/remainder `split` machinery deleted.
- No-fire fallback names two causes separately.
- `TuneLatentSTE` records `residual_before` / `residual_after` over full capture + first-step `gradient_norm`.
- `qk_report.py` — `STEPS_EFFECT_EPS = 1e-3` beside `COLLAPSE_LINE`; stage-map cells re-pointed to per-target cycle.
- `tools/ci/report_checks.py` — `stage_evidence_findings()` + poison-shape unit tests.
- Test pins updated (N-1/N count discriminator + dominance assert).
- Seed-lottery flake recorded (seed 44 → 47).
- Watchdog: `tools/qkb-stall-watchdog.sh` + `run-ptq-container.sh` integration.
- Bug-hunter review: RED-frist executed, 3 failed + 1 ImportError vs predicted-green.
- Slop-remover: 5/6 spec points verified.
- Sec-eng: watchog reviewed, stage-map rows confirmed.
- Stale verdict claims annotated.
- Phantom-date census recorded (design.md §9).
- Artifact gate LANDED and wired (shape (a): finite trio presence + per-cycle net-reduction).
- Smke fixture rebuilt (init scale 0.02 → 0.4, generator-explicit).
- Divergence-gate flake closed (2/24 band, seeds 16 and 18).
- Rebased onto `3b6faf4` (8-commit replay surface).

## Open

- [x] Re-fire the faithful run with the per-target cycle live. This is a ~21.8 h run (918 s × 28 blocks, plus TuneFP/STE overhead). The change stays alive as the tracking vehicle for this fire.
  - Fired 2026-10-25 (v5, attributed launch `qk-pertarget-v5-t_83fc6e93-20261025T1905Z`) and **decided negative at 46 minutes**: probe abort at `layers.1`/depth 2, ratio 8,035.19 vs the original fire's 64,151.52 (identical site); the per-target cycle was live and measured **inert** (12 TuneFP + 14 STE cycles, net 0.0 across all). Evidence: `artifacts/qwen3-0.6B-r512-admm-paperfaithful-v5-pertarget/quantize_report.json` (sha256 `7612ce32…`), design.md §20–§21, card t_83fc6e93. Ticked per orchestrator ruling: the row is the fire, and the fire occurred with a decided verdict.
  - **Planner decisions recorded (design.md §18)**: (1) Shape (c) counterfactual deferred until after re-fire, with trigger condition and +7.1 h pricing on record. (2) Phantom-date ruling: @user confirm.
- [ ] After the re-fire records the per-target cycle numbers, fold delta + archive.
