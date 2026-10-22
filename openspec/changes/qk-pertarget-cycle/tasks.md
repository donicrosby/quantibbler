# qk-pertarget-cycle — tasks

## Landed in `d78f648` (branch `qk-pertarget-cycle`, = `7db1ab9` + this + `299f9b8`)

- [x] (coder) Ruling (a) implemented: TuneFP hoisted into the factorization
      loop ahead of each target's own `_set_linear`; tunable set = still-FP
      remainder `quantizable_linears[idx:]`; skipped at `idx == 0`; comment
      `:1128-1129` states "target 1 is factorized first, WITHOUT a preceding
      TuneFP"; TuneFP in-loop `:1160`, STE per target `:1273`.
- [x] (coder) Pre-group/remainder `split` machinery and `_is_attention_projection`
      **deleted** — they served only the overruled reading.
- [x] (coder) No-fire fallback names its two causes separately: fewer than two
      quantizable linears (legitimate) vs a truncated statistics book.
- [x] (coder) `TuneLatentSTE` records `residual_before` / `residual_after` over
      the **full capture** plus first-step `gradient_norm`, per fire.
- [x] (coder) `qk_report.py` — `STEPS_EFFECT_EPS = 1e-3` beside
      `COLLAPSE_LINE`; `tune_fp`/`ste` stage-map cells re-pointed to the
      per-target cycle.
- [x] (coder) `tools/ci/report_checks.py` — `stage_evidence_findings()` +
      four poison-shape unit tests + STE-row and skip cases.
- [x] (coder) `tests/ptq/test_driver_stage_map.py` — **pin 3** replaced:
      `tune_fp.invocations == factorized_linears - blocks` (derived, 18 on the
      tiny model), `deviation` asserted *for* `PER-TARGET` and *against*
      `once per block`.
- [x] (coder) `tests/ptq/test_tunefp_teacher.py` — N-1/N count discriminator
      (`:238-257`) + the dominance call-site assert that replaces the pin that
      could never go red.
- [x] (coder) Seed-lottery flake recorded: `test_full_budget_is_per_linear_not_a_pooled_split`
      moved seed 44 -> 47 with the 4/24 measurement written into the test.
- [x] (coder) Watchdog: `tools/qkb-stall-watchdog.sh` + `run-ptq-container.sh`
      integration, alert-only by default (`WATCHDOG_KILL_AFTER_MIN=0`).
- [x] (bug-hunter) Red-first executed on a scratch clone against `7db1ab9`:
      3 failed + 1 collection ImportError; the four predicted-green stayed
      green. Detector proven to detect.
- [x] (slop-remover) Five of six spec points verified against the diff.
- [x] (sec-eng) Watchdog reviewed as delivered; corrected stage-map rows
      confirmed live.
- [x] (coder) Stale verdict claims: one dated annotation per location —
      `qk-fix-ste-sign-adaptation/tasks.md:153`, `:165`, `design.md:193`,
      `qk-fix-tunefp-teacher/design.md:149`.
- [x] (coder) Phantom-date census recorded (design.md §9), including the
      one-instance correction to my own `299f9b8` waiver line.

## OPEN — acceptance conditions before re-fire

- [ ] (coder) **Wire the artifact gate — BLOCKED on a spec decision, measured.**
      The ruled test shape (`residual_after <= residual_before * (1 - EPS)` per
      fire) fails on the very fixture it would hang on: the last TuneFP fire
      reduces 5.86e-04 against a 1e-3 floor, and the decay is monotonic in
      remainder size (design.md §13). Threshold right, shape wrong — three
      candidate shapes listed there for @planner. `stage_evidence_findings()` is
      complete, unit-tested, shares the one constant — and is called by
      nothing. It cannot be wired today: the tiny fixture's TuneFP moved its
      residual **up** (`3.96e-06 -> 4.38e-06`), every effect measure below
      `STEPS_EFFECT_EPS`. Until this lands, **an inert STE record in a real run
      fails nothing.** The fixture gets a live objective; the threshold does
      not move. (design.md §5)
- [ ] (planner / coder) **Give the smoke fixture a live objective** so its
      `tune_fp` and `ste` records clear `STEPS_EFFECT_EPS` honestly. This is a
      fixture-shape decision, hence planner.
- [ ] (planner) **Pin the divergence-gate fixture seeds** (or fix the lottery)
      in the CI-referee config. A full-suite gate carries a **~1-in-6
      false-red rate** today; "suite green" is not yet a checkable fact.
      (design.md §8)
- [ ] (planner / user) **Rule on the phantom date.** 33 records date their own
      events after the commits that record them, relative to the repo's own
      clock. Clock wrong, or string wrong? Mechanical once decided; two of the
      four locations @sec-eng named are outside this change's blast radius.
      (design.md §9)
- [x] (coder) **Rebased onto `3b6faf4`** — replayed onto the Oct-21 leak-rewrite
      tip fetched from `/opt/data/qkb-rewrite/qkb-mirror` (no ssh, no creds);
      zero conflicts. The replay surface is **8 commits**, not one, two or
      three: `700bb9d 43da5df b6dd2be 2bb04fb 7db1ab9 d78f648 299f9b8 3c2400c`
      (was `1b1b5b3 d7caea0 8eef993 666b57c e7f69a3 284a296 d533c04 ba1a201`).
      Guardrail (a) closed by the replay itself — see design.md §12.
- [ ] (coder) Re-fire the faithful run on the per-target cycle. Budget
      ≈ 21.8 h per the measured constants (design.md §11).

## Blast radius — what this change does NOT touch

`lb_admm`, `config.py`, the driver's stage ordering, the fidelity-verdict
machinery, `zero_crossings`' diagnostic job, the TuneFP vacuity guard
(`factorized_in_block <= 0` stays as a safety net; production-unreachable under
(a)), and the fire tree's artifacts and checkpoints.
