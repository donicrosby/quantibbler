# qk-ptq delta: qk-nanoquant-alignment-lane1

## ADDED Requirements

### Requirement: Phase-1 statistics SHALL derive the input-side statistic from activation energy and the output-side statistic from the task-loss gradient

`collect_stats` SHALL record, per linear layer, d_in = E[x²] (per-feature mean square of input activations) and d_out = E[g²] (per-feature mean square of the task-loss gradient with respect to the layer's output), both float64. The gradient SHALL come from a real loss: cross-entropy backward through the model head, or a documented caller-supplied head/cotangent for bare-block use. The uniform-cotangent gradient term SHALL be absent from d_in.

#### Scenario: gradient_statistic_on_output_side

- **WHEN** `collect_stats` runs a fixture whose loss has an analytic output-side gradient (e.g. `loss = 0.5·‖z − t‖²` on the layer output `z` with a fixed target `t`)
- **THEN** `d_out_raw` equals the analytic per-feature mean of `(z − t)²` — the task-loss gradient statistic, measured on the OUTPUT side
- **AND** `d_in_raw` equals the plain per-feature mean of `x²` of the input activations, with no gradient term added

#### Scenario: real_loss_through_model_head

- **WHEN** the calibration target is a whole model carrying a head
- **THEN** the gradient statistics come from a single cross-entropy backward through the model head ([NQ] App. C :3011 — memory-efficient CE), with per-batch next-token targets derived from the calibration sequences themselves
- **AND** the uniform-`ones_like` cotangent backward is no longer the statistics source for whole-model collection

#### Scenario: caller_supplied_head_for_bare_block

- **WHEN** `collect_stats` targets a bare block that produces no scalar loss and the caller supplies neither a head nor a cotangent factory
- **THEN** collection fails closed with an error naming the missing loss source
- **AND** when the caller does supply a documented head/cotangent (a callable mapping model output to a scalar loss), that object is the gradient source and the collected statistics reflect it

#### Scenario: uniform_cotangent_retired_from_d_in

- **WHEN** statistics are collected for any linear layer
- **THEN** `d_in_raw` carries no mean-square-gradient term: the OQ-2 informed choice (gradient statistic mirrored onto the input side) is retired, and the retirement is recorded in design.md

### Requirement: Per-token norm clipping SHALL precede squaring on both statistics paths

Before squaring, each token's activation vector and each token's output-gradient vector SHALL be clipped by L2 norm to a robust ceiling: the 99.9th percentile of the per-token norms (a robust max, not the true max). Clipping SHALL apply identically on the reference path and the chunk_rows streamed path, from the same per-batch threshold; online accumulation is permitted. Clipped vectors are then squared and accumulated in float64.

#### Scenario: per_token_clip_before_squaring

- **WHEN** a calibration batch contains outlier tokens whose vector norms dominate the population
- **THEN** the accumulated statistic equals the analytic mean of squares of the CLIPPED token vectors
- **AND** every token whose norm exceeds the cap contributes exactly the cap squared to the sum of squares, so the outlier energy is bounded before it is squared

#### Scenario: clip_quantile_is_robust_max

- **WHEN** a batch contains one extreme token whose norm is many times the 99.9th percentile of per-token norms
- **THEN** the ceiling equals the 99.9th percentile (a robust max), never the true maximum, so no single token can dominate the statistic

#### Scenario: clip_applies_to_gradient_side_too

- **WHEN** output-gradient vectors for a layer are accumulated
- **THEN** each token's gradient vector receives the same per-token norm clip, at the same 99.9th percentile rule, before it is squared and accumulated in float64

### Requirement: The preconditioned target SHALL fold as sqrt(d_out) ⊙ W ⊙ sqrt(d_in), and magnitude balancing SHALL recover against the same sqrt'd vectors

W̃ formation SHALL compute W̃ = sqrt(d_out) ⊙ W ⊙ sqrt(d_in): the sqrt fold happens at target formation, exactly once, on the robust_diag outputs. The same sqrt'd vectors SHALL be handed to magnitude balancing for the Eq. 9 recovery, one shared definition. robust_diag SHALL NOT apply sqrt; it applies clipping and Eq. 3 shrinkage to the raw statistics.

#### Scenario: sqrt_fold_at_formation

- **WHEN** the preconditioned target is formed for a linear layer
- **THEN** W̃ equals `sqrt(d_out)[:, None] * W * sqrt(d_in)[None, :]` elementwise (fixture asserts against an independent reference)
- **AND** magnitude balancing receives those same sqrt'd vectors as D̃_out / D̃_in for the Eq. 9 recovery `Û = D̃_out⁻¹ P_U` — one shared definition, no second sqrt site

#### Scenario: eq3_shrinkage_precedes_fold

- **WHEN** shrinkage coefficient gamma > 0
- **THEN** the statistic that gets sqrt'd at formation is the Eq. 3 shrunk statistic `(1−γ)·clip(D, τ) + γ·mean(clip(D, τ))`, never the raw one ([NQ] Step 2-1, Eq. 3)

### Requirement: robust_diag SHALL retire the per-feature tail-gated q95 auto-cap and keep explicit tau as the paper's Lemma 1 ceiling

The auto-derived tail-gated q95 ceiling (tau unset, _TAU_TAIL_GATE) SHALL be retired: with tau unset no per-feature cap is applied — outlier robustness is carried by the per-token clip. An explicit numeric tau SHALL remain the verbatim absolute cap of [NQ] Lemma 1 (assumption A-1). Zero-variance statistics SHALL still yield finite positive entries via the floor for any finite calibration data, including block-0-style heavy tails.

#### Scenario: tau_unset_means_no_per_feature_cap

- **WHEN** the configuration leaves tau unset
- **THEN** no per-feature cap is derived or applied: the raw (per-token-clipped) statistics pass through to Eq. 3 shrinkage untouched
- **AND** the retired q95 tail-gate policy is gone from `robust_diag` and from `config.py`'s tau documentation, with the retirement decision and rationale recorded in design.md

#### Scenario: explicit_tau_stays_verbatim

- **WHEN** the configuration pins an explicit numeric tau
- **THEN** that value caps the statistics entries exactly, before shrinkage, with entries provably in [0, tau] after — unchanged from the landed A-1 semantics

#### Scenario: heavy_tail_still_finite_positive

- **WHEN** block-0-style heavy-tailed statistics (r137 class: embedding-scale activations with rare extreme tokens) flow through `collect_stats` with the per-token clip and `robust_diag` with tau unset
- **THEN** the pipeline terminates, every preconditioner entry is finite and positive, and the preconditioned target W̃ is finite — the r137 block-0 heavy-tail case stays protected by the token-level clip that replaced the retired per-feature cap

### Requirement: The reference/stream parity scenario SHALL stay green under the new statistics

The existing reference/stream parity pin (chunk_rows streamed vs chunk_rows=None reference) SHALL remain green at its 1e-12 relative tolerance under the new statistics (real-loss gradients, per-token clip before squaring). Both paths SHALL compute the clip threshold from the same per-batch token norms so the clipped statistics agree by construction; the streamed path SHALL keep state O(feature width) plus bounded per-batch norm state and never materialize full activations.

#### Scenario: parity_green_under_new_statistics

- **WHEN** `collect_stats` runs the reference path and the `chunk_rows` streamed path over the same calibration data under the new statistics
- **THEN** `d_in_raw` and `d_out_raw` agree within the existing 1e-12 relative/absolute tolerance for every layer, on fixtures both with and without active clipping

#### Scenario: streamed_path_still_streams

- **WHEN** the streamed path runs
- **THEN** it retains no full float64 activation tensor (the D-7 structural pin holds: both the input and the output legs accumulate streamed per-feature sums), and its extra per-token norm state is bounded by the batch's row count, never by feature width × rows
