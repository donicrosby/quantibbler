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
commit message; executed by @bug-hunter against `7db1ab9`'s `block_recon.py` in
the ptq image):

**RED on `7db1ab9`:**

- `tests/ptq/test_tunefp_teacher.py` — **2 failed**:
  `test_scn_tunefp_fires_per_target_cycle` (AssertionError at `:309`, the N-1
  count) and `test_tunefp_call_site_precedes_the_targets_own_factorization`
  (the dominance upgrade that replaced pin 1 — the replacement goes red, which
  the pin it replaced never could).
  The four tests that must stay green under (a) **stayed green**, as predicted.
- `tests/ptq/test_driver_stage_map.py` — **1 failed**:
  `test_paper_faithful_run_records_the_algorithm1_stage_map`.
- `tests/ptq/test_ste_sign_adaptation.py` — dies at **collection**:
  `STEPS_EFFECT_EPS` cannot be imported from `7db1ab9`'s `qk_report`. (The
  commit body recorded this as `KeyError: 'gradient_norm'`; it is the same
  detector tripping one frame earlier, at import. Red either way — the commit
  body is corrected here rather than rewritten.)

**GREEN on `d78f648`:** 545 passed, 0 failed — see §7 for the fingerprint.

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

**Update, after measurement: the ruled test shape does not survive contact with
the fixture — see §13.** The gate's *threshold* is settled; its *shape* is not,
and the smoke fixture's degeneracy has a cause other than the one assumed.

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

## 8. The divergence-gate seed lottery — cause found, band pinned and gated

At rank 3 the `_SevenLinearBlock` fixture family trips the post-balance
divergence gate on a minority of seeds, in **both** arms (4/24 under this rework
and 4/24 on the previous placement — the rate is the fixture's, not the
placement's; the two seed sets differ). `test_full_budget_is_per_linear_not_a_pooled_split`
sat on that band at seed 44 and moved to **47**, off it.

@slop-remover refused a seed-shopping fix, and was right to. The flake was then
chased to a cause and closed:

- **Cause: reduction order.** BLAS picks its reduction order from the thread
  count. `tests/ptq/conftest.py` now pins `torch.set_num_threads(1)`.
- **Cause: a global draw.** The fixture's `nn.Linear(...)` consumed the GLOBAL
  stream for a default init the next line overwrote — deterministic values,
  non-deterministic global stream. The fixture now builds through
  `_explicit_linear()`, which forks the construction so no global draw escapes.
  `torch.manual_seed` no longer appears in that file at all.
- **Proof, not assumption.** The same 24-seed sweep, run in **two separate
  invocations**, is **byte-identical**: 24 rows, gate trips at **seeds 16 and 18**
  (2/24), both runs, stderr clean.
- **The band is owned, not avoided.**
  `test_the_divergence_gate_fires_on_a_known_band_seed` runs seed 16 and asserts
  `QuantizationDivergenceError` is raised, so the guard's *firing* is tested — a
  regression that widens or narrows the band now changes a result. Verified
  seed-sensitive: seed 47 does not raise.

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
one instance I authored: the `qk-ci-referee` waiver line in `299f9b8`.

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


## 12. Post-rebase record (`3b6faf4`, 8 commits)

| pre-rebase | replayed |
|---|---|
| `1b1b5b3` | `700bb9d` |
| `d7caea0` | `43da5df` |
| `8eef993` | `b6dd2be` |
| `666b57c` | `2bb04fb` |
| `e7f69a3` | `7db1ab9` |
| `284a296` | `d78f648` |
| `d533c04` | `299f9b8` |
| `ba1a201` | `3c2400c` |

Every citation in this change dir names the replayed SHA (@sec-eng's item).

**Guardrail (a) — the redaction conflict — was closed by the replay itself, and
it was not hypothetical.** The whole content delta of the rebase is one file,
`openspec/changes/qk-e2e-quality-gate/tasks.md`, one line: the pre-rebase side
carried a **live vault reference and its key fingerprint**; the replay resolved
both to the base commit's already-redacted placeholders. The payload is
deliberately **not reproduced in this file** — the base commit and the archive
carry the authoritative redacted text, and a narration that quotes a secret
re-creates the leak it documents. That is not a hypothetical either: this change
has now produced that exact mistake three times (§14).

So the pre-rebase branch **carried a live vault reference in a reachable
commit**, and the rebase is what removed it. The leak audit is **clean** — trees
*and* commit messages, over the full range, including the commit that records
the audit (§14).

**The standing object sweep does not clear the leaky objects. Measured, not
assumed.** `git reflog expire --expire=now --all && git gc --prune=now` was run
in the tree; afterwards both targets are still present:

```
4229e8a -> commit
2a92995 -> commit
4229e8a: pinned by refs/heads/main, refs/remotes/origin/HEAD, refs/remotes/origin/main
2a92995: pinned by refs/heads/main, refs/remotes/origin/HEAD, refs/remotes/origin/main
```

Both are **ref-pinned**, so gc cannot reach them. `refs/heads/main` is still the
pre-rewrite tip `e7f69a3`, and the two `refs/remotes/origin/*` refs still point
at the stale pre-force-push `2a92995`. **Order matters: move the refs first
(fetch origin's real `main` = `3b6faf4`, re-point local `main`, drop the stale
remote-tracking refs), then gc, then sweep — otherwise the "post-rebase gate" is
a no-op for exactly the two objects it exists to remove.** Publication-card item,
not coder-lane.

## 13. The gate's reduction test does not survive measurement — flagged, not guessed

@planner ruled: perturb the smoke fixture so its stages have something to
reduce, then wire `assert residual_after <= residual_before * (1 - STEPS_EFFECT_EPS)`.
I measured both halves before implementing, and the second half fails on the
fixture it is meant to hang on.

Measured on the existing unit fixture — `_SevenLinearBlock`, seed 47, rank 3,
the same fixture `test_full_budget_is_per_linear_not_a_pooled_split` uses:

| fire (`factorized_in_block`) | `residual_before` | `residual_after` | reduction | vs `STEPS_EFFECT_EPS` = 1e-3 |
|---|---|---|---|---|
| 1 | 0.25901 | 0.24874 | 3.97e-02 | pass |
| 2 | 0.50557 | 0.49041 | 3.00e-02 | pass |
| 3 | 0.62073 | 0.60878 | 1.92e-02 | pass |
| 4 | 1.16799 | 1.15487 | 1.12e-02 | pass |
| 5 | 1.47361 | 1.47088 | 1.86e-03 | pass |
| 6 | 1.51489 | 1.51400 | **5.86e-04** | **FAIL** |

STE rows: `self_attn.q_proj` 1.37e-03, `self_attn.k_proj` 2.20e-03 — pass, but by
under 2.2x.

**The reduction decays monotonically as the remainder shrinks, and the last fire
falls below the floor.** The driver fixture shows the same structure
independently (gradient collapsing 1.4e-09 -> 4.76e-13 across its six fires). It
is a property of ruling (a), not of any one fixture: the final fire tunes a
**single** still-FP linear whose optimum the previous fire already approached.
**A uniform per-fire 1e-3 floor red-lights the legitimate per-target cycle.** The
threshold is right; the test *shape* is not. For @planner to rule:

1. apply the reduction test only to fires whose remainder has >= 2 linears;
2. test the **cumulative** cycle effect (first fire's `residual_before` -> last
   fire's `residual_after`) instead of each fire;
3. scale the floor by the remainder size.

Not chosen here — that is a spec decision, and picking one without you would be
redesign.

**The smoke fixture is degenerate for a different reason than assumed.** Its
TuneFP residual is `3.77e-06` against the unit fixture's `0.259` — five orders
apart — and the cause is construction, not tiny-model-ness: `driver_smoke.py`
builds random-init weights at a small scale, so the block output is nearly
input-independent already (its own banner measures "zeroing a whole block's
`o_proj` moves its PPL by 3 ppm"). A noise injection cannot fix that, and
scaling the same fixture is actively dangerous: measured, x10 scaling drives
`||Wq||_F/||W||_F` to `6.35e+18` and trips the **divergence gate**
(`QuantizationDivergenceError: layer 'self_attn.k_proj' ... outside band
[1/10000, 10000]`).

So "give the smoke fixture a live objective" most plausibly means **rebuild the
smoke fixture to resemble the unit fixture's construction** (an init scale at
which rank-r factorization is genuinely lossy) rather than inject a perturbation
into it — and @slop-remover's `randn_like` catch then becomes moot, which is the
better outcome than fixing the RNG on a perturbation nobody should add. Either
reading is implementable; the choice between them is the fixture-shape decision
and it is @planner's. I am not going to guess which fixture the smoke leg is
supposed to be, and I am not going to wire a gate I have measured red-lighting
the fixture it hangs on.

## 14. The leak-sweep rule, and three reproductions of one mapping

The audit that verified the rebase was itself reported clean while the branch it
reported on was not. The failure shape is worth naming once, precisely: **a
narration about a redaction reproduces the redacted payload.**

Three reproductions of one mapping, all inside 24 h:

1. `4229e8a` — the original: a commit whose "provenance" note carried the live
   vault reference *and* the key fingerprint in its message body, on a commit
   still reachable from `refs/heads/main`.
2. `7daea90` — this change's own sweep record, which quoted the pre-state diff
   verbatim inside the paragraph documenting its removal. The audit had ranged
   over the eight replayed commits and the recording commit **postdated its own
   range**.
3. A room message from the same author, quoting the same pre-state diff.

**Rules this earns:**

- **A leak sweep's range must include the commit that records the sweep** —
  otherwise the record is the one unexamined artifact, and it is written by the
  person holding the payload. Mechanically: `git rev-list <base>..<branch>`,
  **trees and messages** (`git log --format=%B`), never a hand-picked SHA list.
- **Quote the post-state, never the payload.** `[REDACTED …]` or a placeholder
  (`<live vault ref>`, `<live fingerprint prefix>`) is the only form a durable
  record may carry. The authoritative redacted text already exists in the base
  commit; there is nothing a quotation adds except a second copy.
- **The rule is not repo-scoped.** Reproduction 3 is a chat message: the doctrine
  governs anything durable, and chat transcripts are durable.
- **Refs before gc.** An amended commit's *pre-amend* object stays reachable
  through the branch reflog until `reflog expire` runs, and its tree carries the
  payload, so gc cannot prune the blob while the commit survives. Order is
  amend -> move refs -> expire -> gc -> verify absent.
- **Two fence faces.** The original leak lived in a tracked file *and* in a
  commit message body; a tree-only `gitleaks detect` sees only the first face.
  The outgoing-range leg must be the history form (`--log-opts` over
  `remote..local`, which scans messages and diff text) **plus** the plain tree
  scan for the working state.

**Live-state addendum from @sec-eng's check at the time of writing:** the
must-be-absent set is three objects, not two — `4229e8a`, `2a92995`, and the
**pre-amend** `7daea90`. The box-holds-no-token verdict is false while that
third one lives as a loose object, which is why the amend has to slot *before*
the ref work rather than after it.

## 15. Two corrections to my own previous report, from measurement

1. **The band is 2/24 with the seeds named, not "~1 in 6".** The earlier figure
   came from non-reproducible multi-threaded runs. After the thread pin the
   measurement reproduces byte-identically, which is what turns it from a rumour
   into a number.
2. **The `randn_like` catch was aimed at the wrong artifact, but the instinct was
   right.** @planner's recipe did draw the global RNG (correct), and the fixture
   it would have been injected into is the wrong fix (§13) — but the surviving
   requirement, *every draw generator-explicit*, turned out to be real. It was
   just located in the **existing** fixture builder rather than in a new
   injection.

**And the cumulative-cycle criterion (§13) cannot be computed from the two
residual fields as they stand — measured, not argued.** Per-fire
`residual_before` is measured at that fire's own student state, and the prefix
accumulates quantization damage as targets are factorized, so it *rises*
monotonically across the cycle: 0.25901, 0.50557, 0.62073, 1.16799, 1.47361,
1.51489. Comparing the first fire's `before` to the last fire's `after` is
0.25901 -> 1.51400 — a **5.8x increase** — so the criterion as stated would
red-light every block. The per-fire pair is a valid measurement (tune vs no-tune
at one student state); the **cross-fire comparison is the invalid step**. A
genuine cumulative criterion needs a same-prefix counterfactual, which means
recomputing the factorizations from the untuned weights: SVD-only cost, no
optimizer steps, but a second pass over the block.
