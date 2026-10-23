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

### Requirement: The real-statistics pass SHALL be memory-fenced and statistic-exact

The CE-gradient pass SHALL chunk the head (logits/CE/backward) per 256-token slice off a full-context trunk forward — never re-running the trunk per slice. Per-chunk backward SHALL use `reduction="sum"` with `1/N_c^2` post-scaling, preserving both the per-row statistic and the mean-loss gauge. Per-slice `del` + `empty_cache` SHALL bound the allocator, and the refire SHALL be preceded by a fenced rehearsal whose peak is recorded in launch.env.

#### Scenario: chunked_ce_preserves_statistic

- **WHEN** the equivalence fixture runs the old whole-batch CE path and the new head-chunked path on identical weights and inputs (tiny geometry, one sequence, 128 tokens, CPU)
- **THEN** the accumulated gradient-square vectors match to machine precision — identity, not tolerance — and the fixture goes red against the naive trunk-rechunking variant

#### Scenario: rehearsal_red_first

- **WHEN** the bounded rehearsal runs on pre-chunk bytes inside the fenced container
- **THEN** it reproduces the death as an in-container OOM kill — inside the fence, never a box death

#### Scenario: rehearsal_peak_recorded_before_refire

- **WHEN** the chunked rehearsal completes all 16 stats batches under the fence
- **THEN** the measured memory peak is recorded in launch.env with enough headroom for the 20 solves and model weights, and the refire happens only after the peak, the red-first artifact, and the equivalence fixture are all green
