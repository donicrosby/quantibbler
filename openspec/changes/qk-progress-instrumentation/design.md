# qk-progress-instrumentation — design

## Sink routing

| sink | consumer | format | when |
|---|---|---|---|
| stdout | user (live monitoring), witness (`container.log`) | one newline-terminated JSON line per update | always — every pipeline stage, every mode |
| telemetry CSV | user, planner (artifact read) | one column `progress_json` per row, 2-second cadence matches existing `MemTelemetry` | when the telemetry subsystem is active (`MemTelemetry` thread running) |
| breadcrumbs | watchdog (`qkb-stall`), user | `{epoch} qkb:progress <phase> iter=<N>/<total> rate=<r> eta=<s> eta_sane=<b>` | when the breadcrumbs subsystem is active (watchdog armed) |

All three sinks are fed from a single `ProgressReporter` object. The calling
site never checks which sinks are live — the reporter handles routing
internally, and each long-running loop (capture, statistics, solver, block
recon, TuneFP, STE) gets one identical call.

## Heartbeat

The `MemTelemetry` thread already samples every 2 seconds and knows the current
phase. Progress output piggybacks on that cadence: every sample writes a progress
line with the phase and cumulative counts. Critically, a **progress line IS a
heartbeat** — two seconds of silence is a real stall, not a quiet run, because
the telemetry thread writes on its own timer regardless of what the main loop is
doing. The heartbeat writes to breadcrumbs so the existing watchdog reads
liveness without modification.

## Sanity-check constants

Measured per-phase rates (seq_len=2048, batch=1, torch fp32, gfx1151):

| phase | rate (s / unit) | unit |
|---|---|---|
| capture | — | per-block forward, negligible |
| stats_shipped (Arm A) | — | one-shot, negligible |
| stats_ce (Arm B) | — | one-shot, negligible |
| ADMM solve | ~89 | per linear (400-iter solve) |
| TuneFP | ~133 | per target factorize |
| STE | ~154 | per linear sign-optimize |

The ETA for a fire is computed as `Σ(phase_rate × count)` using these constants
plus the harness's known geometry. Before rendering, the computed ETA is checked
against a sanity fence: if the predicted ETA exceeds `MAX_SANE_ETA` (default 4
hours for a 20-solve fire), the bar reports `eta=unsafe` instead of a number —
the user sees "not rendering" rather than a lie.

### Implementation crystallization (2026-10-25) — planner rulings

**Cadence count (Decision 1).** The requirement's "~80 lines" for a 400-step
ADMM loop at 0.2s/iter disagrees with the cadence rule it states (every 5 s or
every 10 iterations, whichever comes first). At 0.2 s/iter the 10-iteration arm
fires first, producing ~40 progress lines (plus the `complete()` marker for 41
total). The code faithfully implements the stated rule. Ruling: the spec number
is wrong; amend it to "~40 lines." The rule is unchanged.

**ETA unit model (Decision 2).** The shipped wiring is loop-level:
`(total - done) * rate_for(phase)`, where `total` counts loop steps while the
constants are per-linear (admm 89 s, ste 154 s) or per-fire (tunefp 133 s).
The units are incoherent: an admm solve of 1 linear with 400 iterations renders
`eta=unsafe` for its whole length. This file's fire-level `Σ(phase_rate × count)`
model is the aspirational architecture; the implemented model is per-loop with a
physical-work-unit count. Ruling: adopt per-loop ETA with a `work_units: int = 1`
parameter on `ProgressReporter`. The ETA becomes
`fraction_remaining × work_units × phase_rate` where `fraction_remaining =
(total - done) / total`. With `work_units=1` (the default — each reporter is
local to one logical work unit — one linear for admm/ste, one fire for tunefp),
all three loops produce coherent ETAs that pass the sanity fence
(admm at iter 100/400 → ~67s; tunefp at iter 1/40 → ~130s; ste at iter 1/200 →
~153s). The three call sites (`lb_admm.py:483`, `block_recon.py:806,1059`) do
not need to change (default is correct); the change is `progress.py:_record`
adding the `work_units` field and `fraction_remaining` factor.

**Re-scope the five open rows (Decision 3).** Rows 1.5, 3.1, 4.1, 4.2, 5.1
reference symbols that no longer exist at HEAD (`MemTelemetry`, `run_arms`,
`progress_json`). Their implementations were deleted with the lane0 harness by
e5c6939. Rulings:

- **Row 1.5** (heartbeat in `MemTelemetry`): **deferred** — the host was
  retired; `ProgressReporter.heartbeat()` exists but has zero production
  callers. Needs a timer thread that exists.
- **Row 3.1** (wire into `run_arms` statistics phase): **moot** — `run_arms`
  retired; the replacement `calibration.collect_stats` is one-shot and does
  not need a progress reporter.
- **Rows4.1/4.2** (`progress_json` column in telemetry CSV): **deferred** —
  `jsonl_sink` and `attach_telemetry` exist in `progress.py` but have zero
  production callers. Wire-up needs a telemetry collector that registers the
  sink.
- **Row 5.1** (`qkb:progress` breadcrumbs in ptq mode): **deferred** —
  `breadcrumb_sink` and `attach_breadcrumbs` exist in `progress.py` but have
  zero production callers. The existing `_note_block_progress` writes stage
  markers directly (not through ProgressReporter). Wire-up: call
  `attach_breadcrumbs` at reporter construction time in the three existing
  loops (admm, tunefp, ste).

**Spec wording.** The change's `spec.md` named `MemTelemetry` in two
requirements and their scenarios. Those references are reworded to describe the
general mechanism ("a telemetry collector that registers a sink," "a timer-driven
heartbeat") without naming dead classes. The core progress capability ships in
production (stdout for admm, tunefp, ste); the multi-sink routing, heartbeat,
and telemetry CSV integration are designed but not yet wired to a host loop at
HEAD.

**Non-blocking recorded findings.** Row 2.2 and `constants_out_of_date_detected`
name a *date* among the compared values, while `constants_stale` compares commit,
ROCm, and PyTorch only — each constant carries its `measured_on` date,
uncompared. Row 6.4's name still says `to_two_lines` while the shipped 10-step
loop writes 1 line (the docstring says 1). Both are cosmetic and do not block
archival.

### Earlier

The five rows left open by the reconcile are not lagging checkboxes. Rows 1.5,
4.1, 4.2 and 5.1 landed in `21663d5` (`MemTelemetry` gained the `progress_json`
column and its sampler thread called `reporter.heartbeat()`) and row 3.1 in
`3e87a59` (lane0's shipped statistics loop); those implementations were then
deleted with their host files by `e5c6939` ("Lane 1: preconditioner rewrite"),
which removed `tools/lane0_precond_probe.py` and `tests/ptq/test_lane0_probe.py`.
At HEAD `MemTelemetry`, `run_arms` and `progress_json` occur only in this change
directory, and `attach_telemetry`, `attach_breadcrumbs` and
`ProgressReporter.heartbeat` have no production caller — the tests are their only
callers. So the telemetry-CSV and breadcrumb arms of `progress_sink_routing` and
all of `progress_heartbeat` are unimplemented in the shipped tree. The rows stay
open: closing them needs a host loop that exists, which is not something a
checkbox pass can supply.

Row 1.2 was amended in place to the rule the code implements — a step delta since
the last emit, not `iter % PROGRESS_INTERVAL_STEPS`. The two are not equivalent,
which is why the text mattered: a time-cadence emission re-arms a modulo window
and can emit a few steps later, the per-iteration verbosity the cadence exists to
bound.

Two measured divergences are recorded here for the planner; neither was resolved
by this reconcile, because resolving either changes a number a requirement
states:

- `progress_cadence_bounded` says a 400-step ADMM loop produces "~80 lines".
  Measured with the shipped cadence at 0.2 s/iteration: **40** progress lines,
  **41** with the completion marker. The requirement's own rule — every 5 s or
  every 10 iterations, whichever comes sooner — cannot produce ~80 at that pace
  (10-iteration rule: 40; 5-second rule: 16). The number and the rule disagree;
  the code implements the rule.
- The ETA is computed per loop as `(total - done) * rate_for(phase)`, where
  `total` counts loop steps while the constant's unit is per-linear (admm 89 s,
  ste 154 s) or per-fire (tunefp 133 s). Measured: a 400-iteration admm solve
  renders `eta=unsafe` for its whole length (399 × 89 s = 9.86 h), and a 40-step
  tunefp loop renders a numeric `eta=5187s` that clears the 4 h fence. This
  file's model of a fire-level `Σ(phase_rate × count)` is not what the loop-level
  wiring implements.

Row 2.2 and `constants_out_of_date_detected` name the recorded measurement date
among the compared values; the shipped `constants_stale` compares commit, ROCm and
PyTorch only (each constant carries its `measured_on` date, uncompared). Recorded
rather than amended, because that row quotes the requirement text.

### Implementation crystallization (2026-10-25) — Decision 2 follow-through

Landed by the wave-disposition card (t_6ed481cc) from t_993f1df5's code half, which
had blocked on the test question rather than guess.

**What shipped.** `progress.py:_record` computes
`fraction_remaining = (total - done) / total; eta_s = fraction_remaining *
self._work_units * phase_rate`, with `work_units: int = 1` on
`ProgressReporter.__init__` (keyword-only). The three call sites are unchanged —
each reporter is local to one logical unit. `tasks.md` row 1.3 was amended in
place: it still stated the superseded `(total - done) * rate_for(...)` formula.

**The stale pins (four tests, not one).** `tests/ptq/test_progress.py` pinned the
OLD formula in four places. Three are mechanical re-pins (`6*10 -> 0.6*10` and
`eta=60s -> eta=6s`; `19*89 -> 0.95*89`; `eta=801s -> eta=80s`). They are not
weakened assertions: the property each pins — "the prediction follows the measured
table, not the observed pace" — is unchanged and still asserted against the same
constants.

**Re-anchoring the fence test (the design call t_993f1df5 asked for).** Under the
per-loop model a single loop's ETA is bounded by one rate constant, so
`test_eta_renders_unsafe_when_it_exceeds_max_sane_eta`'s `20 steps @ 4000 s/unit`
no longer clears the 4 h fence. Chosen: express the requirement's scenario as
`work_units=20` (a 20-solve fire) rather than inflating the injected constant,
because it reaches the same `0.95 × 20 × 4000 ≈ 21 h` the requirement names
(`eta_suppressed_when_unreasonable`: "an ETA of 21 hours for a 20-solve fire"),
leaves the injected table as the thing that makes it unreasonable, and gives
`work_units` a reader in the suite — otherwise the new parameter would ship with
zero consumers.

**Recorded finding, not resolved (routed to planner).** With `work_units=1` and the
shipped constants (admm 89 s, ste 154 s, tunefp 133 s), the maximum renderable ETA
is the phase constant itself, so `MAX_SANE_ETA` (4 h) can no longer fire on a real
run: the fence is now reachable only from a fire-level reporter (`work_units > 1`)
or a pathological constant. That is a direct consequence of the per-loop model and
is not a defect in it — but it means the requirement's fire-level scenario has no
wired producer, and the shipped loops can no longer demonstrate "unsafe". If the
fence is meant to guard a fire rather than a loop, a fire-level reporter must
exist; that is a design decision, so it is recorded here rather than improvised.
