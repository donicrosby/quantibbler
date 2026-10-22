# qk-pertarget-cycle — design

## 1. The ruling, and what it replaces

**Ruling (a)** (quoted verbatim in proposal.md): factorize target 1 first with
no preceding TuneFP; then per target TuneFP(still-FP remainder) -> factorize ->
STE. TuneFP fires N-1, STE N.

The overruled reading held that Figure 3 showed **one TuneFP pass per block,
after Q/K/V were factorized**. Three detectors were pinned on that reading.
This is the record of all three, because the incident's shape was *three
instruments agreeing on the same mistake* — and the reason nobody caught it is
that the pins made the reading unfalsifiable rather than confirmed.

| # | pin | where | detector that would catch it | catches it now? |
|---|-----|-------|------------------------------|-----------------|
| 1 | TuneFP call site sits *after* the factorization loop | `tests/ptq/test_tunefp_teacher.py` (the "never fires again" conjunction) | `test_tunefp_call_site_precedes_the_targets_own_factorization` — a **dominance** assert, source-order independent | yes |
| 2 | the stage-map text says "once per block" | `src/quantibbler/qk_report.py` | the `deviation` cells are now asserted *for* `PER-TARGET` and *against* `once per block` | yes |
| 3 | **`tune_fp.invocations == capture.invocations`** — the driver-level end-to-end pin | `tests/ptq/test_driver_stage_map.py` | replaced by a **derived** count: `factorized_linears - blocks` (21 - 3 = 18 on the tiny model) | yes |

Pin 3 was found during this work, not before it: it is the one that would have
kept the *end-to-end* stage map green through the whole rework, and no earlier
review had named it.

## 2. The spec (six points, @slop-remover — verified against the diff)

1. **STE `per` rows carry `residual_before` / `residual_after` / `gradient_norm`,
   measured over the full capture, not the first minibatch**
   — `block_recon.py:663`, `:676-678`, `:761-763`.
2. **TuneFP's trio is untouched** — no new field, no renamed key.
3. **The unit inequality asserts against the shared EPS, and the failure
   message names the inert-stage semantics** — `test_ste_sign_adaptation.py:128-130`.
4. **`STEPS_EFFECT_EPS = 1e-3` is a plain module constant beside
   `COLLAPSE_LINE`** — `qk_report.py:88`. No knob, no env var, no per-stage override.
5. **YAGNI fence held** — no per-step residual history, no monitor abstraction,
   verdict machinery untouched.
6. **The artifact gate must become a gate, not a function** — see §5. **OPEN.**

## 3. Red-first evidence — both states, executed

Detector-proves-it-detects, by execution on a scratch clone (not taken from the
commit message; executed by @bug-hunter against `e7f69a3`'s `block_recon.py` in
the ptq image):

**RED on `e7f69a3`:**

- `tests/ptq/test_tunefp_teacher.py` — **2 failed**:
  `test_scn_tunefp_fires_per_target_cycle` (AssertionError at `:309`, the N-1
  count) and `test_tunefp_call_site_precedes_the_targets_own_factorization`
  (the dominance upgrade that replaced pin 1 — the replacement goes red, which
  the pin it replaced never could).
  The four tests that must stay green under (a) **stayed green**, as predicted.
- `tests/ptq/test_driver_stage_map.py` — **1 failed**:
  `test_paper_faithful_run_records_the_algorithm1_stage_map`.
- `tests/ptq/test_ste_sign_adaptation.py` — dies at **collection**:
  `STEPS_EFFECT_EPS` cannot be imported from `e7f69a3`'s `qk_report`. (The
  commit body recorded this as `KeyError: 'gradient_norm'`; it is the same
  detector tripping one frame earlier, at import. Red either way — the commit
  body is corrected here rather than rewritten.)

**GREEN on `284a296`:** 545 passed, 0 failed — see §7 for the fingerprint.

## 4. Effect evidence: `STEPS_EFFECT_EPS`

One constant, `1e-3`, on the same footing as `COLLAPSE_LINE`. It is the floor
below which a stage record cannot distinguish "did work" from "did nothing",
and it is what the stage-evidence gate compares against. The justification is
dimensional, not fitted: a stage whose own effect measure moves less than a
tenth of a percent has not demonstrated an effect at the precision the report
quotes its other numbers to.

## 5. The artifact gate is built and **unwired** — acceptance condition

`stage_evidence_findings()` is complete, unit-tested (four poison shapes, the
STE-row case, the skip case) and shares the one constant. It is **deliberately
not wired** into the smoke or faithful legs, and the reason is measured:

> On the tiny 3-block fixture, a **full 256-step TuneFP budget moved the capture
> residual UP**: `3.96e-06 -> 4.38e-06`. `weight_delta_rel` ranges from
> `5.25e-05` (first fire) down to exactly `0.0` (last fire, `gradient_norm`
> `4.76e-13`). Every one of those is below `1e-3`.

Wiring the gate against that fixture would make it fire on plumbing — a
false-alarm gate trains the re-run-until-green habit that makes gates theater.
Per the fixture-honesty clause already written at the call site: **the fixture
gets a live objective; the threshold does not move.**

**Acceptance condition (must land before re-fire):** give the smoke fixture a
live objective so its `tune_fp` and `ste` records clear `STEPS_EFFECT_EPS`
honestly, then wire `stage_evidence_findings` into the smoke leg. Until then
**an inert STE record in a real run fails nothing** — the records are auditable
post-hoc, but the gate is not yet a gate.

## 6. Stale verdict claims riding this branch — annotations landed

The rework touches zero pre-existing openspec paths, so verdict claims emitted
under the pre-re-rule reading stay on the branch untouched. @sec-eng flagged
four; **one dated annotation per location, annotate-don't-rewrite** — the
entries honestly record what was believed then, the annotation records what we
know now.

| location | stale claim |
|---|---|
| `qk-fix-ste-sign-adaptation/tasks.md:153` | `paper_faithful: true` with **all eight verdict conditions true** |
| `qk-fix-ste-sign-adaptation/tasks.md:165` | a shortened-ADMM run "still computes `paper_faithful: true`" |
| `qk-fix-ste-sign-adaptation/design.md:193` | the label-rule / closed-mechanism-list definition |
| `qk-fix-tunefp-teacher/design.md:149` | the amendment's verdict definition (mechanism list) |

## 7. Environment fingerprint for the green suites

The `545 passed` figure describes **the host interpreter, not the ptq image**:

- host `/opt/venv`, CPython 3.12, **torch 2.14.1+cu130**, **pytest 9.1.1**
- run as two bounded invocations (the sandbox backend repeatedly dies on one
  monolithic `pytest tests/ -q`):
  - `tests/ptq` + `test_ci_referee` + `test_qk_report` + `test_run_ptq_wrapper` → **293 passed** (4m30s)
  - `test_attention_tiled` + `test_kernels_dispatch` + `test_kernels_parity` + `test_package` + `test_representation` → **252 passed** (1.5s)
- **caveat (@bug-hunter):** `quantibbler-ptq:latest`'s venv ships **without pip
  or pytest** (torch only) — reproducing the red-first state inside that image
  needed `ensurepip`. The 545 number is not reproducible in that container as
  shipped.

## 8. Known flake: the divergence-gate seed lottery — acceptance condition

At rank 3, the `_SevenLinearBlock` fixture family trips the post-balance
divergence gate on **4 of 24 seeds, in BOTH arms** (4/24 under this rework,
4/24 on `e7f69a3`, different seed sets). The rate belongs to the fixture, not
to the placement — but `test_full_budget_is_per_linear_not_a_pooled_split` sat
on the gate at seed 44 (fine at baseline, tripped under the cycle) and moved to
**47** with the measurement recorded in the test.

**Consequence:** any full-suite gate carries a **~1-in-6 false-red rate**.
Pin the fixture seeds in the CI-referee config, or fix the lottery, **before**
the rebase/push protocol treats "suite green" as a checkable fact. A gate with
a known 1-in-6 false-red trains exactly the habit that makes gates theater.

## 9. Phantom date: `2026-10-27` — census, and what is *not* established

@sec-eng found the string in three records. **The census is 33 occurrences
across 20+ files**, and the shortest proof is this one:

```
$ git log --format='%h %ai %s' -S "decision 2026-10-27" \
      -- crates/quantibbler-kernels/tests/cubek_cpu_reference_parity.rs
f438177 2026-10-20 20:33:49 +0000  Land 2.3: cubek cpu-reference third parity face, budget adjudicated
```

A commit **authored 2026-10-20 20:33** records a decision dated **2026-10-27**
— and attributes it to @sec-eng. Every *other* date in the corpus
(`2026-10-17/18/19/20/21`, 33/24/20/17/33 occurrences) is consistent with the
commit timeline, and the archive directories use real dates
(`2026-10-18-*`, `2026-10-19-*`, `2026-10-21-*`). `2026-10-27` is the only
impossible one, and it is systematic across lanes, not a lone typo in a coder
record.

**Honest boundary — what this does and does not establish.** Git author dates
and `date -u` are the *same clock*. So the finding, stated without overclaiming,
is: **relative to the only clock the repository has, 33 records date their own
events after the commits that record them.** That is *either* the box clock
running 5-6 days behind the real world, *or* the string being wrong. I cannot
break that tie from inside the repository, and I will not pick a corrected date
by inventing one. **@planner / @user:** the verdict (fix the clock vs. fix 33
strings) is yours; either way the correction is mechanical once decided.

Per-location corrections to the four records @sec-eng named are **not** applied
here — two of the four (`AGENTS.md`, the Rust file) are outside this change's
blast radius and their owners should land them. This change corrects only the
one instance I authored: the `qk-ci-referee` waiver line in `d533c04`.

## 10. Watchdog (host-side stall detection)

Both recorded incidents silenced the **in-process** breadcrumb writer, so
anything sharing the container's fate could not report it. The watchdog runs on
the host: `tools/qkb-stall-watchdog.sh`, launched by `tools/run-ptq-container.sh`
(`WATCHDOG=1` default, `WATCHDOG_STALL_MIN=15`, `WATCHDOG_KILL_AFTER_MIN=0` =
**alert-only** by default). It watches total line-silence in the breadcrumb log
— heartbeats gap ~5 s, legitimate stage markers ~21 min — and alerts into the
breadcrumb tail and `$OUT/watchdog.log`. `DRY_RUN` exits before launch.

## 11. Measured cost of (a)

Constants derived from the fatal session's stage clock: **TuneFP ≈ 133.1 s/fire**
(256 steps @ bs=4), **STE ≈ 154.2 s/linear** (1024 steps @ bs=1, seq 2048),
LB-ADMM ≈ 80.4 s (1024×1024) / 198.9 s (3072×1024).

| reading | TuneFP fires | TuneFP cost | faithful total |
|---|---|---|---|
| overruled (once/block) | 28 | ≈ 1.04 h | ≈ 16.6 h |
| **this change (a)** | **168** | **≈ 6.21 h** | **≈ 21.8 h** |
| alt (b) | 196 | ≈ 7.25 h | ≈ 22.8 h |

Levers, measured and fidelity-hostile: `TUNE_SEQ_LEN=512` saves ≈ 11 h but
LB-ADMM drops 918 s → ~10 s/block and the fidelity claim is not honest.
