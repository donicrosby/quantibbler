# qk-ptq delta: qk-nanoquant-alignment-lane0

## ADDED Requirements

### Requirement: Lane 0 SHALL be a measurement-only decisive experiment that lands nothing in the artifact path

The lane SHALL be a decisive preconditioner-divergence experiment built on the existing `lb_admm` + `magnitude_balance` solver pair and the repo's ablation/probe conventions, adding measurement code only. It SHALL NOT modify any shipped pipeline module, SHALL NOT reach the qpk1 export path, and SHALL NOT emit any OK/DEGRADED/FAILED verdict band. Any non-default measurement knob SHALL be recorded in provenance.

#### Scenario: measurement_only_no_export

- **WHEN** the lane-0 probe completes on its pinned probe set
- **THEN** no code path on the lane reaches `extract_and_pack`, qpk1 serialization, or any export site, the report records `artifact_exported: false` with no effective-BPW field, and no verdict band appears: "No verdict bands, no export."

#### Scenario: shipped_pipeline_modules_untouched

- **WHEN** the lane executes against the repo
- **THEN** `calibration.py`, `lb_admm.py`, `block_recon.py`, and `config.py` are byte-identical to their pre-lane state (empty diff), and all new code lives in a new tool module plus its test file

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

### Requirement: Every Lane 0 launch SHALL ride the wrapper

Every Lane 0 launch SHALL go through `tools/run-ptq-container.sh` under `MODE=lane0-rehearsal` or `MODE=lane0-fire` — hand-rolled docker launches SHALL NOT exist and SHALL NOT be certifiable: launch.env carries the wrapper's provenance block plus the mode, and an artifact without it is not a Lane 0 result. Both lane0 modes SHALL fail closed without GPU=1, SHALL carry a wall-clock stop net, and SHALL record the pre-launch free-RAM baseline beside the artifact.

#### Scenario: wrapper_only_provenance

- **WHEN** a Lane 0 artifact is read
- **THEN** its launch.env carries the wrapper provenance block with mode, commit, tree state, image digest, and the pre-launch free-RAM baseline — an artifact without them is refused as evidence

#### Scenario: wall_clock_stop_net_armed

- **WHEN** a lane0 rehearsal or fire launches
- **THEN** a host-side stop net is armed for the container's wall-clock deadline, so a run that neither completes nor OOMs cannot hang the box, and the deadline is recorded in launch.env
