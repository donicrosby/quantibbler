# qk-ptq delta: qk-pertarget-cycle

## MODIFIED Requirements

### Requirement: TuneFP SHALL fire per-target within the factorization loop, not once per block before factorization

The TuneFP stage SHALL be hoisted into the factorization loop, firing ahead of each target's own `_set_linear`. Its tunable set SHALL be the still-FP remainder (`quantizable_linears[idx:]`), with `factorized_in_block = idx`. The first target (`idx == 0`) SHALL be factorized with NO preceding TuneFP — a pass before it is bitwise vacuous (student == pristine teacher on the same `x_b`, residual and gradient identically zero). TuneFP therefore fires N−1 times per block and STE fires N times, where N is the number of quantizable linears in the block.

#### Scenario: per_target_cycle_structure

- **WHEN** the block reconstruction loop iterates over quantizable linears in a block with N ≥ 2
- **THEN** TuneFP fires N−1 times, each fire tuning the still-FP remainder
- **AND** the first target (`idx == 0`) is factorized with zero preceding TuneFP invocations
- **AND** STE fires N times, once per target after its factorization

#### Scenario: single_linear_block_skips_tunefp

- **WHEN** a block contains exactly one quantizable linear
- **THEN** TuneFP fires zero times (N−1 = 0)
- **AND** the block is still eligible for factorization of its sole linear — a block with fewer than two quantizable linears is a legitimate skip, not an error

#### Scenario: still_fp_remainder_is_accurate

- **WHEN** TuneFP fires at position idx
- **THEN** its tunable set is `quantizable_linears[idx:]` (the still-FP remainder)
- **AND** `factorized_in_block` equals `idx`, so the vacuity guard `factorized_in_block <= 0` remains a safety net but is unreachable in production (first call always has idx ≥ 1)

### Requirement: The pre-group/remainder split machinery SHALL be deleted

The `split` logic that divided quantizable linears into a pre-group and remainder, and the `_is_attention_projection` helper, SHALL be deleted — they existed only to serve the overruled "one TuneFP pass per block, fired after Q/K/V were factorized" reading.

#### Scenario: split_machinery_absent

- **WHEN** the codebase is searched for the pre-group/remainder split and `_is_attention_projection`
- **THEN** neither exists — both are deleted
- **AND** no caller references either deleted function

### Requirement: TuneLatentSTE SHALL record objective per-fire evidence

`TuneLatentSTE` SHALL record, per fire, `residual_before` and `residual_after` over the FULL capture (not a sampled subset), plus the first-step gradient norm. This replaces the overruled `zero_crossings` metric as the primary evidence of whether the stage did anything.

#### Scenario: objective_evidence_per_fire

- **WHEN** TuneLatentSTE completes a per-target fire
- **THEN** `residual_before` and `residual_after` are recorded over the full capture block's teacher/student outputs
- **AND** the first-step gradient norm is recorded
- **AND** the evidence is persisted in the artifact rows for downstream verification

### Requirement: The driver-level stage-map invariant SHALL be derived from the ruling, not asserted from the overruled shape

The test `test_driver_stage_map.py` SHALL replace the overruled pin (`tune_fp.invocations == capture.invocations`) with a derived invariant: `tune_fp.invocations == factorized_linears - blocks` (18 on the tiny model), asserting FOR the per-target ruling and AGAINST "once per block."

#### Scenario: derived_invariant_asserts_per_target

- **WHEN** the driver stage-map test runs on the tiny model (14 attn + 14 mlp = 28 quantizable linears across 14 blocks)
- **THEN** `tune_fp.invocations` equals `factorized_linears − blocks` (28 − 14 = 14, N−1 per block)
- **AND** the test explicitly asserts that the observed invocations are inconsistent with "once per block" (which would predict 14 TuneFP fires, not 14 — the degeneracy at 14 blocks / 14 linears means `once_per_block == 14` and `per_target == 14`, so the test uses the tiny model's 2-linears-per-block guarantee at block 0 for a strict distinction)

### Requirement: A CI report gate SHALL verify stage evidence presence

`tools/ci/report_checks.py` SHALL include a `stage_evidence_findings()` gate that inspects report.json for the presence of objective stage evidence. On runs claiming paper fidelity, the gate SHALL assert a per-cycle net-reduction test against `STEPS_EFFECT_EPS=1e-3`.

#### Scenario: stage_evidence_gate_on_faithful

- **WHEN** a report claims `paper_faithful: true`
- **THEN** the gate verifies that STE's per-cycle net reduction exceeds `STEPS_EFFECT_EPS` for at least one linear — a no-op stage on a faithful run is a finding

#### Scenario: stage_evidence_gate_on_non_faithful

- **WHEN** a report does NOT claim `paper_faithful: true` (e.g. epochs=0 non-faithful runs)
- **THEN** the gate performs shape checks only — the non-faithful driver's `stage_pre` sets `epochs=0`, leaving TuneFP at one step of the paper's 256, and asserting the floor there would red-light correct behavior

### Requirement: A host-side stall watchdog SHALL detect frozen pipeline runs

`tools/qkb-stall-watchdog.sh` SHALL monitor the container's stdout for stall conditions. It SHALL be alert-only by default (`WATCHDOG_KILL_AFTER_MIN=0`) and SHALL integrate with `run-ptq-container.sh` so that every container fire carries the watchdog.

#### Scenario: watchdog_alert_only_by_default

- **WHEN** the watchdog is started with default configuration
- **THEN** it emits alerts on stall detection but does NOT kill the container
- **AND** setting `WATCHDOG_KILL_AFTER_MIN > 0` enables auto-kill after the specified minutes of stall
