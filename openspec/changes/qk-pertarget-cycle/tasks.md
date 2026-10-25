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

- [x] (coder) **Artifact gate LANDED and wired.** Shape (a), @planner's ruling:
      presence of the finite trio everywhere, plus a per-cycle net-reduction test
      against `STEPS_EFFECT_EPS` **on runs claiming paper fidelity** — STE grouped
      per block, not per linear. Non-faithful runs are shape-checked only, because
      the driver's non-faithful `stage_pre` is `epochs=0` and leaves TuneFP one
      step of the paper's 256: asserting the floor there would red-light correct
      behaviour. Proven on a real artifact, not a synthetic poison: the smoke
      report is clean as-is, and 3 findings when relabelled `paper_faithful: true`.
      Stated limit, in the gate's docstring: no-op detector, not a counterfactual.
- [x] (coder) **Smoke fixture rebuilt** (`tools/ci/driver_smoke.py`): init scale
      0.02 -> 0.4 (the unit fixture's), all draws generator-explicit. Measured:
      TuneFP capture residual 3.77e-06 -> 2.2-8.9, STE net 45x the floor. Banner
      and docstring rewritten with the measured numbers.
- [x] (planner) **[DROPPED — superseded] Give the smoke fixture a live
      objective.** Done as the rebuild above rather than as a perturbation; the
      perturbation reading would have needed a product hook, which the no-knob
      spec point forbids.
- [ ] (planner) **Decide shape (c)** — the same-solver counterfactual, priced at
      +7.1 h on the faithful run (918 s x 28: 21.8 -> 28.9 h). The only shape that
      measures TuneFP's marginal effect on the admm geometry; a plan-level
      decision, not a gate.
- [x] (coder) **Divergence-gate flake closed by cause.** `torch.set_num_threads(1)`
      pinned in `tests/ptq/conftest.py`; the fixture's global `nn.Linear` draw made
      generator-explicit; the band is **2/24, seeds 16 and 18**, byte-identical
      across two separate invocations; a band-owning test asserts the guard fires
      on seed 16. (design.md §8)
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
