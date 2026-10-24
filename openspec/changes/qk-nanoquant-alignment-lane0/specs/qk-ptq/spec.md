# qk-ptq delta: qk-nanoquant-alignment-lane0

## ADDED Requirements

### Requirement: Lane 0 SHALL be a measurement-only decisive experiment that lands nothing in the artifact path

The lane SHALL be a decisive preconditioner-divergence experiment built on the existing `lb_admm` + `magnitude_balance` solver pair and the repo's ablation/probe conventions, adding measurement code only. It SHALL NOT modify any shipped pipeline module's algorithm, SHALL NOT reach the qpk1 export path, and SHALL NOT emit any OK/DEGRADED/FAILED verdict band. Any non-default measurement knob SHALL be recorded in provenance.

#### Scenario: measurement_only_no_export

- **WHEN** the lane-0 probe completes on its pinned probe set
- **THEN** no code path on the lane reaches `extract_and_pack`, qpk1 serialization, or any export site, the report records `artifact_exported: false` with no effective-BPW field, and no verdict band appears: "No verdict bands, no export."

#### Scenario: shipped_pipeline_modules_untouched

- **WHEN** the lane executes against the repo
- **THEN** `calibration.py`, `lb_admm.py`, `block_recon.py`, and `config.py` carry no algorithm changes (empty diff on their solver logic), and all new code lives in a new tool module plus its test file
- **AND** progress instrumentation imports in `lb_admm.py` and `block_recon.py` (the `ProgressReporter` belt code from `qk-progress-instrumentation`) are the sole allowed diff — they are additive instrumentation that changes no solver behavior, and their presence is expected state, not a drift

### Requirement: The decisive probe SHALL run a fixed set of real production layers with full identity recorded

The probe set SHALL be a fixed constant pinned before any measurement: real production layers, one entry per (model, block index, layer name). Every report row SHALL record those three identity fields. Both arms SHALL run the identical set in the identical order; per-run ad-hoc set edits SHALL be impossible by construction.

#### Scenario: fixed_probe_set_identity_recorded

- **WHEN** the lane-0 report is written
- **THEN** every probed-layer row records `model`, `block_idx`, and `layer_name` populated from the pinned fixed set, and the set itself is a named constant in the harness with no CLI override

#### Scenario: both_arms_identical_probe_set

- **WHEN** the two arms execute
- **THEN** arm A and arm B consume the same pinned probe set in the same order, so every comparison is within-layer

### Requirement: The intended preconditioner SHALL be constructed the paper-intended way

The intended arm SHALL build `W~ = D~_out W D~_in` ([NQ] B.1 Eq. 26) with diagonal entries folded once per side as square roots of robust second-moment statistics (sqrt fold), the gradient side from a real cross-entropy backward through the head ([NQ] App. C :3011), per-token norm clipping at the robust 99.9th percentile, and the paper's clip-and-shrinkage robustness ([NQ] Eq. 3) applied on top.

#### Scenario: sqrt_fold_once_per_side

- **WHEN** the intended preconditioner is constructed for a probed layer
- **THEN** `W~_intended = D~_out W D~_in` holds exactly, each preconditioner diagonal is applied once per side, the diagonal entries equal the square root of the robust clipped-and-shrunk second-moment statistics, and the magnitude-balance unfold uses the same diagonals

#### Scenario: gradient_side_real_ce_backward

- **WHEN** the intended arm collects gradient statistics
- **THEN** they come from a backward of the cross-entropy loss of the fp teacher's logits against the calibration targets through the head, never a uniform-ones cotangent, and any layer with no reachable gradient records the fallback it used

#### Scenario: per_token_norm_clip_999

- **WHEN** calibration batches contain outlier tokens
- **THEN** per-token vector norms are clipped at the robust 99.9th percentile of the batch's per-token norms before statistics accumulate, and the applied percentile is recorded in the report

### Requirement: Both arms SHALL run the existing lb_admm and magnitude_balance unchanged

Each arm SHALL feed its `W~` variant into the shipped `lb_admm(w_tilde, cfg)` and `magnitude_balance(p_u, p_v, d_out_tilde, d_in_tilde, target=W)` entry points under a production-default `PTQConfig`. The only difference between arms SHALL be the preconditioner construction. Neither arm SHALL invoke polish, STE refinement, or any altered solver parameter.

#### Scenario: same_solver_pair_both_arms

- **WHEN** the two arms run a probed layer
- **THEN** both call the shipped solver entry points with the same production-default `PTQConfig`, and the only per-arm variation is `(D~_in, D~_out, W~)`

#### Scenario: pre_polish_stage_boundary

- **WHEN** a probed layer's measurements are taken
- **THEN** `greedy_polish` and TuneLatentSTE are never invoked on the lane, so every quantity reflects the post-ADMM pre-polish state

### Requirement: Per probed layer each arm SHALL record the four decisive quantities

For every probed layer and both arms the report SHALL record: the post-ADMM pre-polish reconstruction cosine against the original fp weight; the full consensus residual trajectory; the ADMM iteration count `n_iter`; and the `min_eigs` floor over the solve's system matrices. A missing or truncated quantity SHALL fail the run.

#### Scenario: four_quantities_per_layer_per_arm

- **WHEN** a probed layer completes both arms
- **THEN** its rows each carry the post-ADMM pre-polish recon cos, the complete consensus residual trajectory, `n_iter`, and the min_eigs floor; any missing quantity fails the run closed instead of writing a partial row

#### Scenario: recon_cos_against_original_weight

- **WHEN** the recon cos is computed
- **THEN** it is `cos(W, U_bal V_bal^T)` against the ORIGINAL un-preconditioned fp weight at the post-balance pre-polish boundary, matching the solver-probe `cos_raw` convention, and the intended arm's band comparison uses that same quantity

### Requirement: The lane SHALL end in a quantize_report-style JSON and a winner-or-falsified verdict paragraph

The decisive run SHALL write a quantize_report-style JSON (strict-parseable, provenance-complete) and a verdict paragraph that names, per probed layer, the arm owning the strictly higher post-ADMM pre-polish recon cos, or declares the hypothesis falsified. The domination rule and the Lanes 1-3 gate are defined in the design. "No verdict bands, no export."

#### Scenario: winner_named_per_probed_layer

- **WHEN** the decisive run completes
- **THEN** the report is a quantize_report-style JSON plus a verdict paragraph naming the winner per probed layer, and a layer where the intended arm does not strictly win is recorded as such rather than omitted

#### Scenario: falsified_hypothesis_gates_lanes

- **WHEN** the intended preconditioner does not dominate (per the design's domination rule)
- **THEN** the verdict paragraph declares the hypothesis falsified and the report carries the lane gate for Lanes 1-3: STOP and re-analyze before landing anything, with Lane 0 gating Lanes 1-3 because lane order is load-bearing

### Requirement: The CE head SHALL be chunked and statistic-exact

The CE-gradient pass SHALL chunk the head per 256-token slice off a full-context trunk forward under `torch.utils.checkpoint`, sum the chunk losses, apply the mean gauge to the total, and take ONE backward, bitwise identical to the whole-batch path. Per-slice `empty_cache` SHALL bound the allocator.

#### Scenario: chunked_ce_preserves_statistic

- **WHEN** the equivalence fixture runs the legacy whole-batch path and the checkpointed-head path on identical weights and inputs (tiny geometry, one sequence, 128 tokens, CPU)
- **THEN** the accumulated gradient-square vectors match to machine precision — identity, not tolerance
- **AND** the fixture goes red against the naive trunk-rechunk variant AND against per-chunk backward (attention couples rows, so squared partials do not sum)

### Requirement: The trunk forward SHALL gradient-checkpoint every production block

The trunk forward SHALL checkpoint every production block — backward recomputes per block, so the retained activation graph is bounded to one block, the statistic is unchanged bitwise, and the ~30% stats-phase slowdown is accepted. The fixture SHALL run under the attention implementation the fenced rehearsal will use.

#### Scenario: trunk_checkpointing_preserves_statistic

- **WHEN** the equivalence fixture runs the retained-graph path against the checkpointed-trunk path on identical weights and inputs (tiny geometry, CPU, the container's attention implementation)
- **THEN** the gradient-square vectors match bitwise — identity, not tolerance
- **AND** the fixture goes red against any build where a production block is left uncheckpointed

### Requirement: The stats accumulator SHALL count each row exactly once

The accumulation hooks SHALL gate on `torch.is_grad_enabled()` so each row is counted exactly once on the recompute pass, and the trunk checkpoint SHALL be reentrant mode — the discriminator's premise. The equivalence fixture SHALL assert RAW-sum identity against the uncheckpointed path, not just the mean.

#### Scenario: raw_sums_are_the_correct_version

- **WHEN** the equivalence fixture runs the checkpointed-trunk path against the uncheckpointed path
- **THEN** `d_in_raw` and `d_out_raw` match bitwise — raw, not mean — each row counted exactly once, so the raw sums are the naive-magnitude values and no future reader can be surprised by a ×2

#### Scenario: doubling_trap_is_a_red_case

- **WHEN** the fixture exercises an ungated accumulator or a non-reentrant checkpoint (both passes grad-enabled)
- **THEN** the raw sums double relative to the uncheckpointed path and the fixture goes red, so the trap fails loudly instead of surviving as folklore

#### Scenario: whole_head_checkpointed_trunk_fenced

- **WHEN** the harness is asked to run the whole-batch head with checkpointed blocks — the combination whose `d_out_raw` moved ~0.8% while raw sums and rows stayed identical, executed by no shipped path
- **THEN** it fails closed before any statistics pass, the error names the finding and its commit, and the configuration is not launchable until the divergence is root-caused

#### Scenario: rehearsal_red_retired

- **WHEN** any future Lane 0 rehearsal executes
- **THEN** the parent-bytes path is never re-executed — the incident record is the artifact: one packed batch on parent bytes with the fence present killed the host twice, so the per-batch peak of that path is larger than the host and cannot be measured on it; any launch attempting the unchunked or uncheckpointed stats path SHALL be refused

#### Scenario: rehearsal_green1_then_16

- **WHEN** the checkpointed build's rehearsal runs
- **THEN** it starts at ONE batch measuring the true per-batch peak from `torch.cuda.max_memory_reserved`; the full 16-batch endurance run proceeds only if that peak lands at or under 15G; and the refire happens only after the 16-batch peak leaves solve-compatible headroom under 32G, the red artifacts, and both fixtures are green
- **AND** the rehearsal constructs its batches through the same packed loader as the protocol run — one batch is one packed batch of BATCH×seq_len tokens, never a degenerate corpus-sized blob — with explicit rehearsal sequences recorded as provenance (`rehearsal=true, sequences=N, protocol_sequences=128`) so the artifact cannot be misread as a protocol statistic
- **AND** the rehearsal SHALL release as it runs: caching allocator disabled, explicit reference release at every chunk boundary, per-chunk empty_cache, 2-second memory telemetry flushed line-by-line to the artifact, and the attention implementation pinned — so a death leaves a trace naming the allocation, and the box survives via the host memory-floor net
- **AND** no fire SHALL execute a batch count the CPU suite has never run: a rehearsal with N batches requires the e2e fixture to have run N batches red-first, so green-16's batch-2 Arm B death class — first execution of multi-batch Arm B in campaign history — is impossible by construction
- **AND** the rehearsal SHALL exercise the target-load path against the same cache shape the fire uses — a fixture-cache and the mounted `HF_HOME=/hf` are different shapes (the mount is the *parent* of `hub/`; a fixture laid out under a bare root misses the offset), and the fire's death — `load_layer_weight` never matched — lived precisely in that difference; the rehearsal fixture SHALL stage a safetensors file one level deep under `hub/models--…/snapshots/…/` matching `_hf_cache_roots`'s normalization rule, so `real_target` is exercised at the gate, not discovered at the fire

#### Scenario: rehearsal_geometry_guard_exempts_explicitly

- **WHEN** the rehearsal passes an explicit sequences value
- **THEN** the geometry validation checks the resolved count against THAT explicit value, the decisive fire's 128-sequence guard stays untouched, and any rehearsal without an explicit sequences value is refused rather than silently defaulted

#### Scenario: rehearsal_peak_reads_device_memory

- **WHEN** the rehearsal writes its peak
- **THEN** it reads device memory (`torch.cuda.mem_get_info` / the telemetry CSV maximum) — never `max_memory_reserved`, which is meaningless under the disabled caching allocator — and both arms' per-phase peaks are recorded separately so the gate judges the actual measurement path, not a reference arm dressed as it

### Requirement: Harness changes touching a production import SHALL pass the surface-table gate

Any harness change that touches a production import SHALL be accompanied by the production-surface diff table — one row per touchpoint (collect_stats, robust_diag, lb_admm, magnitude_balance), each citing the production call-site line and reading `matches` or `DEFECT`. No fire SHALL launch while any row reads DEFECT.

#### Scenario: surface_table_gate_before_fire

- **WHEN** a Lane 0 fire is staged
- **THEN** the harness change's surface table exists with every production touchpoint row citing its call-site line, all rows read `matches`, and any DEFECT row blocks the launch — instance six gets caught by a grep, not a floor-switch

### Requirement: Every Lane 0 launch SHALL ride the wrapper

Every Lane 0 launch SHALL go through `tools/run-ptq-container.sh` under `MODE=lane0-rehearsal` or `MODE=lane0-fire` — hand-rolled docker launches SHALL NOT exist and SHALL NOT be certifiable: launch.env carries the wrapper's provenance block plus the mode, and an artifact without it is not a Lane 0 result. Both lane0 modes SHALL fail closed without GPU=1, SHALL carry a wall-clock stop net, and SHALL record the pre-launch free-RAM baseline beside the artifact.

#### Scenario: wrapper_only_provenance

- **WHEN** a Lane 0 artifact is read
- **THEN** its launch.env carries the wrapper provenance block with mode, commit, tree state, image digest, and the pre-launch free-RAM baseline — an artifact without them is refused as evidence

#### Scenario: wall_clock_stop_net_armed

- **WHEN** a lane0 rehearsal or fire launches
- **THEN** a host-side stop net is armed for the container's wall-clock deadline, so a run that neither completes nor OOMs cannot hang the box, and the deadline is recorded in launch.env

### Requirement: container_log_witness

The wrapper SHALL capture the container's stdout/stderr to `$OUT/container.log` — wrapper-level `tee`, no in-container changes, no image rebuild — and SHALL record the container exit status in `launch.env` via POSIX `PIPESTATUS`. A fire of ANY mode SHALL fail closed on completion when `container.log` is absent or empty.

#### Scenario: any_mode_death_leaves_testimony

- **WHEN** any fire — lane0 or ptq production — dies
- **THEN** its traceback SHALL exist in `container.log`: a death without a trace is a launch defect, not an acceptable outcome — green-16's batch-2 Arm B death and v2's fifth blind death both left none, and the CPU suite cannot be the witness for device-only faults

#### Scenario: witness_mechanics_verify_without_box_time

- **WHEN** the wrapper is invoked with an unpullable image
- **THEN** the redirect and status-capture mechanics prove themselves in seconds — the pull fails before any container exists — without spending box time or CPU hours on redirect theater
- **AND** the mechanical check records container.log presence and the propagated exit status in the artifact

#### Scenario: refire_requires_witness

- **WHEN** a rehearsal or decisive fire refires after a death
- **THEN** the refire SHALL complete with `container.log` on disk; any subsequent death leaves its traceback named in the artifact, and the first read of that traceback is the suspect from the prior death's boundary analysis — for green-16's class, `release_device_memory`'s grad-enabled `empty_cache()` between batches

### Requirement: fire_monitoring_cadence

Every fire SHALL announce its poll cadence at launch with a named owner, alongside its announced wall time — a fire nobody is scheduled to look at is unmonitored by construction. When the completion channel is known dead at launch, the launcher SHALL arm the poll at launch, not remember it later.

#### Scenario: v2_silence_class_shall_not_repeat

- **WHEN** a fire's wall time exceeds one hour
- **THEN** a poll cadence with a named owner SHALL exist from launch, and the launch announcement names both
- **AND** a fire launching with a dead completion channel SHALL have its first poll scheduled at launch — v2, a 21.8-hour fire with a known-dead channel and no poll order, died at 03:43:42Z and went unread for 21 hours; that class SHALL NOT exist again

### Requirement: guards_shall_be_tri_state

A guard whose verification step fails SHALL refuse, never proceed. Three states: held SHALL refuse; free SHALL proceed; unverifiable SHALL refuse with the captured failure in the message. `2>/dev/null || true` reading as free is the named defect class: the pre-flight name guard fell open exactly this way — an inspect that failed, or a payload on stderr, both read as free and the wrapper exited 0 through to `container: exit=0`.

#### Scenario: unverifiable_refuses_with_captured_stderr

- **WHEN** a guard's verification call returns rc≠0 or the payload lands on stderr
- **THEN** the guard refuses with the captured stderr in the refusal message — both collapse modes produce exit 2, not 0
- **AND** the fix lands red-first against two stub fixtures: one answering rc≠0, one emitting the payload on stderr; both currently yield rc=0, both SHALL yield exit 2 after the fix
- **AND** the stub logs three lines per invocation — argv, emitted stdout, rc — so form-selective dispatch cannot hide inside a pass

#### Scenario: seam_oracle_is_dry_run_rc

- **WHEN** a wrapper gate test runs
- **THEN** the wrapper SHALL execute under `DRY_RUN=1` — DRY_RUN walks every gate in order (digest probe, rocm-tag, name guard) and exits before anything can launch, so the run's rc IS the seam oracle at zero launch risk
- **AND** the stub SHALL be form-selective: the `{{.Id}}` probe answers with the pinned digest via the `LANE0_IMAGE_DIGEST_ALLOW` door (a stub must satisfy the pin to reach the guard), and the name-inspect answers with a held payload
- **AND** one run proves the seam and the guard together: rc=0 means the seam is dead and the wrapper talked to the real daemon; rc=2 means the seam is alive AND the guard refused through the full chain — the 2026-10-24 dead-seam run read as a guard pass precisely because nothing stubby ever answered

#### Scenario: fail_direction_audit_required

- **WHEN** a guard chain is audited
- **THEN** every verification call SHALL be classified by fail direction, and a call that fails toward launch is a defect — the 2026-10-24 audit found the nets' label reads and the watchdog's `docker ps` fail safe, and the pre-flight guard the only fail-open on the launch path

### Requirement: wrapper_failclosed_branches_hermetic_tests

Every fail-closed branch of the wrapper SHALL be exercised by a hermetic test — a stub `docker` answering `inspect` with a fixture payload, the interception proven in-band before the wrapper runs — never by "verified by construction".

#### Scenario: refusal_branch_proven_before_fire

- **WHEN** a lane0 refire or decisive fire is staged
- **THEN** the staging proof includes the hermetic refusal test green — a fire SHALL NOT ride a guard that cannot be demonstrated
- **AND** the branch list SHALL cover: pre-flight held-name refusal, digest pin, rocm-tag gate, GPU gate, witness fail-closed, the nets' label-decline, and the watchdog's kill-path label check
- **AND** the named class is the 2026-10-24 pre-flight miss: claimed verified by construction, a hermetic stub measured it NOT firing — wrapper exited 0 through to `container: exit=0`
- **AND** a guard whose red test is reverted to keep the suite green is a staging blocker, not a silent gap
- **AND** the one legal exception to a red fixture in the lane is `xfail(strict=True)` when the red is a broken fixture harness, not a broken guard: the marker's reason names the harness defect, the docstring names the un-xfail condition, and `strict=True` makes the day the harness gets fixed and the test passes an XPASS failure — the suite itself demands the un-xfail; reverting the test out of the lane is never the exception

### Requirement: launch_lines_derived_at_firing

A staged launch line SHALL be re-derived against the tree at firing time — the wrapper's preflight is the derivation, and its printed launch record is what gets fired, not a remembered line. A knob value recorded under an old semantic is a defect to catch at the gate, not a launch to replay: v3 fast-failed because the staged line carried `EVAL_SEQS=18` in the old batches unit, which the re-expressed sequence-counting knob correctly refused.

#### Scenario: replayed_line_refused_at_gate

- **WHEN** a launch line recorded before a knob re-expression is replayed verbatim
- **THEN** the fire SHALL fail fast at the gate with the traceback in the witness — the correct response is re-deriving the value against HEAD (18 batches → 144 sequences), never bypassing the gate

### Requirement: launch_env_single_writer

`launch.env` SHALL have exactly one writer at a time: concurrent appends are a provenance defect, not a race to tolerate — v2's launch.env ended in a corrupted bare-`?` append line. Writers SHALL serialize (one append path) or write atomically (write-then-replace); a malformed launch.env invalidates the artifact's provenance block.

#### Scenario: corrupted_provenance_invalidates_artifact

- **WHEN** launch.env carries a corrupted line (v2's bare-`?` append)
- **THEN** the artifact's provenance block is refused as evidence until the writer defect is fixed — provenance that can be interleaved by two writers is not provenance

### Requirement: infrastructure_interception_proof

A test that overrides `PATH` to intercept a command which launches real infrastructure SHALL prove the interception took effect in-band — assert the resolved binary is the shim — before invoking the code under test. A silent fall-through to the real binary is not a red test; it is real infrastructure with the test's own timeout as the only alarm.

#### Scenario: shim_proven_before_code_runs

- **WHEN** a test shims a command that launches real infrastructure (docker, container runtimes)
- **THEN** the test SHALL assert the resolved binary is the shim in the same environment handed to the code under test, before that code runs — a standalone probe that passes while the real test falls through is the exact failure the rule catches
