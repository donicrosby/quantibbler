# qk-ptq Specification

## Purpose

Define the offline Python post-training quantization pipeline that transforms caller-supplied transformer weights and calibration data into validated qpk1-ready factorized layers without network access.

## Requirements

### Requirement: The PTQ pipeline SHALL expose a single entry point that consumes a caller-supplied model and calibration data

`quantibbler.ptq.quantize_model(model, calibration_data, config)` SHALL accept a caller-constructed PyTorch `nn.Module` (the fp teacher), an iterable of caller-supplied calibration batches, and a `PTQConfig`, and SHALL return a `QuantizedModel` carrying packed binaries and float scales for the quantized layers. The pipeline SHALL perform no network access: the model and calibration data are caller-supplied, per the no-network rule of `openspec/config.yaml`.

#### Scenario: caller_supplied_model_and_data

- **WHEN** a caller constructs a tiny transformer `nn.Module` locally and passes it with an iterable of synthetic token-id batches and a default `PTQConfig` to `quantize_model`
- **THEN** the call returns a `QuantizedModel` without any network access, and no download, fetch, or hub-resolution code path exists in the `quantibbler.ptq` package

#### Scenario: config_defaults_paper_values

- **WHEN** a `PTQConfig` is constructed with no arguments
- **THEN** it exposes rank, robust-diagonal (tau, gamma), ADMM (K, penalty schedule, lam, epsilon), stage step counts with learning rates and batch sizes, seed, and temperature fields, with defaults drawn from [NQ] App. C and the informed assumptions in the qk-ptq design

### Requirement: Rust kernel dispatch SHALL match declared-dtype NumPy parity fixtures on probe inputs

When the `quantibbler-kernels` Rust extension is present, its output on probe inputs at a declared dtype SHALL match same-declared-dtype NumPy parity fixtures within the per-dtype tolerances declared by the qk-kernels capability (bf16 backward 1e-2 and f32 backward 1e-6 of the row term scale; float64 rel 1e-12), with packed bits identical (sign-agreement invariant), per the verification gates of `openspec/config.yaml`.

#### Scenario: rust_matches_parity_fixtures

- **WHEN** a kernel probe (pack, unpack, or a solve helper) executes with the Rust extension installed at any declared dtype
- **THEN** the Rust output matches the same-dtype NumPy parity fixture within the declared per-dtype tolerance, and packed bits are identical between the two paths

### Requirement: Global calibration SHALL produce robust diagonal preconditioners from activation and gradient statistics

Phase 1 SHALL run the calibration data through the fp teacher once, collecting per-linear-layer input/output activation and gradient statistics, and SHALL construct diagonal preconditioners `D~_in` and `D~_out` per layer by clipping diagonal entries to a ceiling tau and applying shrinkage `(1−γ)·entry + γ·mean(D)` (Eq. 3, [NQ] Step 2-1). Entries SHALL be nonnegative and finite. Tau SHALL be an explicit pin or the per-vector 95th percentile.

#### Scenario: preconditioner_shape_robustness

- **WHEN** calibration statistics are collected for a linear layer with weight shape (d_out, d_in)
- **THEN** `D~_in` has d_in diagonal entries, `D~_out` has d_out entries, all entries are clipped to tau before shrinkage and lie in [0, tau] after, and zero-variance activations yield finite positive entries rather than NaN or zero

#### Scenario: stats_cover_all_layers

- **WHEN** calibration batches flow through a multi-block model
- **THEN** every linear layer slated for quantization has statistics recorded after Phase 1, in module declaration order

#### Scenario: preconditioner_tau_default_q95

- **WHEN** the configuration leaves tau unset and a raw statistic vector is heavy-tailed (maximum more than 10x its 95th percentile)
- **THEN** the derived ceiling equals that vector's 95th percentile, entries above it are clipped before shrinkage, and the resulting preconditioner entries are strictly bounded below the raw maximum

#### Scenario: preconditioner_tau_tail_gate_inert

- **WHEN** the configuration leaves tau unset and a raw statistic vector is moderately tailed (maximum at most 10x its 95th percentile)
- **THEN** no clipping is applied, and the preconditioner equals the no-clip shrinkage path exactly

#### Scenario: preconditioner_tau_explicit_override

- **WHEN** the configuration pins an explicit numeric tau
- **THEN** that value is the cap exactly as before, independent of the per-vector statistics

### Requirement: Each linear layer SHALL be initialized via LB-ADMM with Hessian-aware coupling and stabilized Cholesky solves

For each linear layer, the pipeline SHALL form the preconditioned target at `compute_dtype` and SHALL solve the ridge-regularized ADMM problem of Eq. 4 ([NQ] Step 2-2). Normal equations and Cholesky SHALL execute at the accumulation dtype; latents, scales, and bias SHALL be stored at `compute_dtype`. The solver SHALL execute on the module's device with no host round-trip.

#### Scenario: admm_spd_solves

- **WHEN** LB-ADMM runs on a random preconditioned target at any rank r and penalty endpoints, at any declared compute_dtype
- **THEN** every system matrix `VᵀV + (ρ+λ)I` (and its transpose-side counterpart) is symmetric positive definite, the solve is performed by Cholesky at the accumulation dtype, and the stabilized fallback engages only when positive definiteness is numerically violated

#### Scenario: admm_dtype_generic

- **WHEN** LB-ADMM completes at bf16 compute
- **THEN** the returned latents U, V, scales s1, s2, and bias are bf16 tensors on the module's device

#### Scenario: admm_on_device

- **WHEN** LB-ADMM factorizes a layer on an accelerator device
- **THEN** the solve executes entirely on-device with no `.cpu().numpy()` round-trip, using `torch.linalg.cholesky` and `torch.linalg.solve_triangular` on device tensors

#### Scenario: admm_parity_per_dtype

- **WHEN** LB-ADMM factorizes a fixed target at each compute_dtype against a NumPy parity fixture
- **THEN** the relative difference of the FIRST ADMM update residual `‖U₁ − U₁_ref‖ / ‖U₁_ref‖` against the NumPy fixture satisfies: ≤1e-2 for bf16, ≤1e-4 for float32, and ≤1e-12 for float64 — measured on the continuous-update iterate before the first sign projection

### Requirement: LB-ADMM SHALL iterate continuous updates, SVID proxies, and scaled duals to a stated stopping rule

The continuous update SHALL solve `(VᵀV + (ρ+λ)I)Uᵀ = VᵀW~ᵀ + ρ(Z_U − Λ_U)ᵀ` (Eq. 5) via stabilized Cholesky decomposition, symmetrically for V ([NQ] App. B.4, Lemma 2). The proxy update SHALL apply the two-sided optimal rank-1 SVID operator to the consensus variables (Eq. 6): `Z = sign(P) ⊙ (g hᵀ)` where (g, h) is the top rank-1 SVD of `sign(P)∘P` ([NQ] :592, "optimal rank-1 approximation that preserves the sign structure"). The dual update SHALL accumulate `Λ ← Λ + U − Z` ([NQ] App. B.3).

#### Scenario: admm_consensus_convergence

- **WHEN** LB-ADMM factorizes a matrix with an exact rank-r structure
- **THEN** the consensus residual `max(||U−Z_U||_F, ||V−Z_V||_F)` decreases across iterations and terminates below epsilon or at the K-iteration cap
- **AND** the SVID proxy preserves the sign structure of its input at every proxy update

#### Scenario: svid_is_two_sided_optimum

- **WHEN** the SVID proxy processes a consensus variable `P = U + Λ` with sign structure `S = sign(P)`
- **THEN** the returned Z is `S ⊙ (g hᵀ)` where (g, h) are the top rank-1 SVD factors of `S∘P`
- **AND** for a column-constant input the two-sided optimum is exactly the column-only solution — the fix is a strict superset, never a regression on the existing test fixtures

#### Scenario: svid_sign_preserved

- **WHEN** SVID processes any input matrix
- **THEN** `sign(Z) == sign(input)` entrywise, with sign(0) = +1 per the A-11 convention
- **AND** dead columns (all-zero input) receive the positive `DEFAULT_SCALE_FLOOR`

### Requirement: The ADMM penalty SHALL follow a linear schedule over K iterations with epsilon as an early stop

The penalty parameter SHALL follow a linear schedule across K outer iterations ([NQ] App. C :3008), and ADMM SHALL stop early when the consensus residual falls below the tolerance epsilon; otherwise it SHALL run to the K-iteration cap.

#### Scenario: penalty_schedule_stopping

- **WHEN** LB-ADMM runs with default ADMM parameters on a fixed target
- **THEN** the penalty value increases linearly per iteration up to K, and the solver halts at whichever comes first: residual below epsilon or the K-th iteration

### Requirement: The rho endpoint SHALL lie in the stated interval in the normalized regime

The rho endpoint SHALL stay within the stated endpoint interval so the consensus term does not overwhelm the unit-scale data term: in the normalized regime (target divided by its Frobenius norm at entry per [NQ] Step 2), `rho_end` SHALL lie in the closed interval `[0.0, 1.0]`, with the schema refusing `rho_end < rho_start`. The paper prescribes the linear shape; the endpoints are an implementation choice, and an endpoint that prevents convergence is a defect.

#### Scenario: rho_end_bounded_for_convergence

- **WHEN** LB-ADMM factorizes a planted rank-32 target at rank-64 factorization (`gamma=0.2`; warm-start cos ≥0.99 — the fixture the convergence gate exercises) with the paper's `tau`/`gamma` preconditioning; real-layer evidence: all 20 recorded `layers.0` probe rows in `artifacts/lane0-fire-r3-20261024/rows.jsonl` are non-monotone (2–66 rises each; one row — arm B `self_attn.o_proj` — does not net-decrease over the first 200), confirming the shape is structural, not a fixture artifact
- **THEN** the consensus residual makes net progress across the first 200 iterations (net decrease from start of the second half, with the first half dominated by the consensus-term pull onto the ±1 proxy before the data term reasserts itself) and the solver terminates at the K-iteration cap — `converged=False` is expected with a ±1-structured proxy and does not indicate solver failure
- **AND** the post-iteration continuous reconstruction satisfies `cos(UVᵀ, W) > 0.5`, proving the ADMM produces a usable initializer (cos loss from the unconstrained truncated-SVD warm start is expected; the SVID proxy pushes latents into the ±1-signed rank-1 subspace, and the paper's pipeline compensates in STE refinement per [NQ] Step 3)
- **AND** `cos(UVᵀ, W) > cos(sign(zu) @ sign(zv)ᵀ, W)` — the continuous reconstruction beats a naive sign-only assignment, proving the ADMM iteration does meaningful work beyond trivial binarization

### Requirement: LB-ADMM SHALL conclude with latent magnitude balancing that yields Ub, Vb, s1, s2

Upon ADMM convergence, the pipeline SHALL recover unscaled proxies `Û = D~_out⁻¹ P_U`, `V̂ = D~_in⁻¹ P_V`, compute the equilibrium factor `η = sqrt(||V̂||_F / ||Û||_F)` (Eq. 7), extract scales `[s1]_i = mean(|η û_i|)`, `[s2]_j = mean(|η⁻¹ v̂_j|)` (Eq. 8), and set final latents `U = η Û`, `V = η⁻¹ V̂` (Eq. 9) ([NQ] Step 2-3). Initialization of the layer is complete only when all of (Ub, Vb, s1, s2) are produced.

#### Scenario: magnitude_balancing_scales

- **WHEN** ADMM converges and magnitude balancing executes on the recovered proxies
- **THEN** the balanced factor norms satisfy the η relation of Eq. 7, the extracted scales equal per-row mean absolute values per Eq. 8, and no scale entry is NaN; an all-zero row receives the clamped scale floor rather than an undefined value

### Requirement: The ADMM exit SHALL rescale to preserve scale equivariance

The ADMM exit SHALL rescale factors from the normalized space back to the target's scale with `factor_scale = sqrt(target_scale)` applied to each of `(U, V, P_U, P_V)` — a single factor of `target_scale` on the product `UVᵀ`, maintaining scale equivariance: scaling the entry target by `c` SHALL scale the returned product by `c`, not `c²`.

#### Scenario: exit_rescale_is_equivariant

- **WHEN** the ADMM entry target is scaled by a nonzero factor c
- **THEN** the returned product `UVᵀ` scales by exactly c — not c² — verified within relative tolerance 1e-5
- **AND** the iteration trajectory in the normalized space is identical regardless of the entry scale

### Requirement: Block-wise reconstruction SHALL refine factors against quantized-prefix inputs and teacher targets

Phase 2 SHALL process transformer blocks sequentially. For each block it SHALL capture inputs `X_b` by pushing calibration data through the already-quantized prefix and teacher outputs `Y_b` from the fp block ([NQ] Algorithm 1 lines 9–10), and SHALL tune the block's fp weights against (X_b, Y_b) for T_pre steps before factorization ([NQ] Step 1, error propagation mitigation).

#### Scenario: quantized_prefix_inputs

- **WHEN** block b > 0 is reconstructed in a multi-block model
- **THEN** its captured inputs `X_b` are the outputs of the quantized blocks before b, not of the fp prefix, and `Y_b` is the fp block's output on those inputs

### Requirement: Factorized components SHALL be refined per block with the Straight-Through Estimator

After initialization, each block's continuous latents and scales SHALL be jointly refined with the STE against Eq. 10 ([NQ] Step 3) for T_post steps at `compute_dtype`. The STE passthrough SHALL operate at `compute_dtype`. Binaries SHALL be extracted as `sign(U)`, `sign(V)` at `compute_dtype` and packed.

#### Scenario: ste_refinement_improves_mse

- **WHEN** factorized-component refinement runs on a fixed toy block with straight-through sign gradients at any supported compute_dtype
- **THEN** gradients reach U, V, s1, and s2 with the sign function treated as passthrough, and the block output MSE decreases over the T_post steps

#### Scenario: sign_extraction_packing_parity

- **WHEN** refinement converges and binaries are extracted as `sign(U)`, `sign(V)` and packed via the qk-kernels pack path (mapping −1→0, +1→1 per [NQ] Fig. 2 stage c)
- **THEN** unpacking the packed bits reproduces the extracted sign matrices exactly, and the same bits are produced by the NumPy reference and Rust dispatch paths
- **AND** the sign convention at zero is **+1** (`sign(0) = +1`, implemented as `x >= 0` not `torch.sign`) — exact-zero latents (common under bf16 underflow) map to bit-0, preserving the encoding

#### Scenario: sign_zero_plus_one_convention

- **WHEN** a latent tensor under bf16 compute contains exact zeros (from underflow or initialization)
- **THEN** the packed bits encode those positions as +1 (bit-0), and the unpacked sign matrix is identical to the input under the +1-at-zero convention

### Requirement: Global scale calibration SHALL tune only the scale vectors under a KL objective with binaries frozen

Phase 3 SHALL optimize the per-layer scale vectors `{s1, s2}` of all quantized layers to minimize the KL divergence between teacher and quantized logit distributions under temperature T (Eq. 11, [NQ] Sec 3.3), with teacher logits cached and the packed binary weights bit-frozen throughout. No other parameter SHALL receive gradients during this phase.

#### Scenario: kl_scale_calibration_frozen_bits

- **WHEN** global scale calibration runs on a tiny quantized model with cached teacher logits
- **THEN** the KL objective decreases over T_glob steps, the packed bits are bit-identical before and after the phase, and only scale parameters carry gradients

### Requirement: The pipeline SHALL round-trip through the qpk1 container with byte-exact fidelity

A `QuantizedModel` SHALL serialize to the qpk1 container defined by the qk-representation capability and SHALL reload from it such that packed binaries are byte-exact, float scales are preserved bit-for-bit in float64, and a forward pass after reload matches the forward pass before serialization on the same probe inputs. Byte accounting for the artifact SHALL come from the serialized container, per the qk-representation capability.

#### Scenario: qpk1_round_trip_byte_exact

- **WHEN** a quantized tiny model is serialized to qpk1 and reloaded into a fresh object
- **THEN** the packed bits compare byte-exact, the scales compare bit-for-bit in float64, and probe-input forwards before and after the round trip agree to float64 round-off

### Requirement: The end-to-end tiny-model gate SHALL bound logit delta and require effective BPW below 1.0 measured from bytes

The verification gate SHALL build a seeded tiny random transformer block, run quantize → serialize qpk1 → reload → forward on probe inputs, and assert that the maximum absolute logit delta against the fp tiny model is bounded by the threshold recorded in the gate test. This gate mirrors the end-to-end smoke gate of `openspec/config.yaml` and runs as part of the full gate suite.

#### Scenario: e2e_tiny_gate_passes

- **WHEN** the seeded tiny random transformer block is quantized, serialized, reloaded, and forwarded on the gate's probe inputs
- **THEN** the max logit delta versus the fp model is within the recorded bound

### Requirement: The gate SHALL assert byte-derived BPW below 1.0 and round-trip bit parity

The gate SHALL compute effective bits-per-weight as serialized bytes from the qpk1 container via the qk-representation accounting — never from parameter counts — and SHALL require it strictly below 1.0; it SHALL also assert pack/unpack parity (bit identity) across the round trip.

#### Scenario: e2e_bpw_and_parity

- **WHEN** the gate measures the serialized qpk1 artifact of the tiny model
- **THEN** effective BPW computed from serialized bytes is below 1.0 and packed bits are identical across the serialize → reload round trip

#### Scenario: e2e_gate_detects_corruption

- **WHEN** a single packed bit in the serialized qpk1 artifact is deliberately flipped and the gate reruns
- **THEN** the gate fails on the logit-delta or parity assertion, demonstrating the gate can detect serialization corruption

### Requirement: factor-mode fp removes the binary constraint

The PTQ pipeline SHALL support a `factor_mode` of `binary` (default) or `fp`.
In `fp` mode the factorized layers compute with continuous latents — no sign,
no straight-through estimator, no bit-flip polish — while capture, TuneFP,
solver, balance, scale initialization, latent refinement, the progress probe,
and the report run unchanged.

#### Scenario: fp_mode_forward_has_no_sign_anywhere

- **WHEN** a run carries `factor_mode=fp`
- **THEN** the factorized forward computes `diag(s1) @ U @ V^T @ diag(s2)` with
  no `sign` operation in any of the three latent-term sites, in either the
  grad-enabled or the `no_grad` regime, and all three sites produce their terms
  through one shared helper

#### Scenario: helper_bitwise_contract_in_binary_mode

- **WHEN** a binary run computes latent terms through the shared helper
- **THEN** the grad-enabled branch returns the straight-through passthrough
  `sign(u) + u - u.detach()` (so latent gradients are nonzero), the `no_grad`
  branch returns `torch.sign` bitwise (`torch.equal` holds on the f64 fixture
  tier), the regime is selected by an explicit `torch.is_grad_enabled()` test,
  and a helper returning literal sign in the grad-enabled branch is rejected
  by the battery

#### Scenario: helper_reachability_is_observed_twice

- **WHEN** the helper pins run
- **THEN** spies observe BOTH the live STE path and `_set_linear` consulting the
  helper, AND the weight `_set_linear` writes is `torch.equal` to the helper's
  `no_grad` output — a helper satisfied by only one site does not pass

#### Scenario: runtime_gradient_contract_catches_dead_ste

- **WHEN** the binary grad-enabled helper is exercised by a real backward pass
- **THEN** its latent gradient equals the analytic `d/dU = s1` contract, and a
  detached-alias implementation that returns the right values but zeroes that
  gradient fails even if the structural AST pin accepts its shape

#### Scenario: default_binary_behavior_is_byte_identical

- **WHEN** `factor_mode` is unset
- **THEN** sign, STE, polish, and packing run exactly as in the pre-knob
  pipeline

#### Scenario: fp_mode_skips_only_the_discrete_stage

- **WHEN** a run carries `factor_mode=fp`
- **THEN** the bit-flip polish stage is not invoked, balance-stage latents flow
  unchanged into refinement, and every other stage's recorder events still fire
  in the same order

### Requirement: untuned-signs ablation exports sign of the balance latents

The pipeline SHALL support an `ablation` mode `untuned-signs` for binary runs:
the exported bits are `sign()` of the balance-stage latents with zero
refinement — no polish rounds and no TuneLatentSTE steps.

#### Scenario: untuned_signs_runs_zero_refinement

- **WHEN** a binary run carries `ablation=untuned-signs`
- **THEN** the exported bits equal `sign()` of the balance latents exactly, and
  the run log records zero polish invocations and zero STE events

#### Scenario: default_refinement_unchanged

- **WHEN** `ablation` is unset
- **THEN** polish and TuneLatentSTE run with their configured budgets exactly
  as before

### Requirement: layer scope factorizes only the named component

The pipeline SHALL support a `layer_scope` of `all` (default), `self_attn`, or
`mlp`: only linears whose module path contains the scope component are
factorized; every other linear stays full precision in prefix capture,
refinement, and export.

#### Scenario: attention_scope_quantizes_only_attention

- **WHEN** a run carries `layer_scope=self_attn`
- **THEN** only `self_attn` linears are factorized, and every MLP weight is
  byte-identical to its pre-run value

#### Scenario: mlp_scope_quantizes_only_mlp

- **WHEN** a run carries `layer_scope=mlp`
- **THEN** only `mlp` linears are factorized, and every attention weight is
  byte-identical to its pre-run value

#### Scenario: all_scope_is_the_default

- **WHEN** `layer_scope` is unset
- **THEN** the quantizable set is unchanged from the pre-knob pipeline

### Requirement: ablation runs are measurements, never artifacts

A run carrying any non-default knob (`factor_mode=fp`, `ablation=untuned-signs`,
or `layer_scope != all`) SHALL be a measurement: the full gate, probe, and
report run with the knob values recorded in provenance, the verdict derives
from the held-out PPL ratio with the unchanged band, and a `fp`-mode run never
reaches the qpk1 export.

#### Scenario: fp_run_reports_but_never_exports

- **WHEN** a run carries `factor_mode=fp`
- **THEN** `extract_and_pack` — the only site that converts latents to bits —
  raises on fp latents rather than sign-converting them, the run completes with
  host write-back via the helper's fp branch, the report is written with a
  verdict, provenance records the knobs, `artifact_exported` is false,
  `effective_bpw` is null, scale calibration is skipped as the ablation
  boundary, and the driver's export refusal remains as a backstop

#### Scenario: binary_ablation_runs_may_export

- **WHEN** a binary run carries `ablation=untuned-signs` or a scoped
  `layer_scope`
- **THEN** the qpk1 export path is unchanged and provenance records the knobs

#### Scenario: verdict_band_is_mode_independent

- **WHEN** any knob is set
- **THEN** status is derived from the held-out PPL ratio against the same
  `OK_LINE` / `COLLAPSE_LINE` band as default runs

### Requirement: chunked statistics stream feature moments

When `collect_stats` receives `chunk_rows`, its forward hooks SHALL retain only
float64 per-feature sum-of-squares and token counts; retained state SHALL be
independent of the number of rows/chunks. It SHALL NOT retain or concatenate
full activation tensors. Gradient accumulation remains unchanged. The
`chunk_rows=None` path SHALL remain byte-identical to its committed fixture records at the declared dtype.

#### Scenario: streamed_stats_match_reference

- **WHEN** the same f64 fixture is collected once unchunked and once through
  the streamed chunk path
- **THEN** every input, output, and gradient-derived statistic matches at
  `rtol=1e-12, atol=1e-12`

#### Scenario: streamed_state_is_bounded

- **WHEN** the streamed path processes arbitrarily many row chunks
- **THEN** each layer side retains one feature-width f64 sum vector and one
  token count, never a list of full activation tensors; a poison that restores
  tensor appends is rejected by the structural pin

#### Scenario: default_reference_path_is_unchanged

- **WHEN** `chunk_rows` is unset
- **THEN** the existing list/concatenate reference computation and its
  byte-identical fixture behavior remain unchanged

### Requirement: The pipeline SHALL offer a per-block progress probe against which prefix perplexity can be gated

`PTQConfig` SHALL expose a `progress_probe` callable field, default `None`, invoked by the sequential block reconstruction after each block is written back into the host model, with the live model, the 1-based prefix depth, and the block name; the probe may raise to abort the pipeline, and an unset probe SHALL change nothing.

#### Scenario: probe_invoked_each_block

- **WHEN** a `progress_probe` is configured and a multi-block model is quantized
- **THEN** the probe observes the live host model once per block in construction order, with depth counting quantized blocks 1-based, after that block's write-back and before the next block's capture begins

#### Scenario: probe_absent_no_change

- **WHEN** the config leaves `progress_probe` unset
- **THEN** the pipeline path is byte-identical to a no-probe run (no invocation overhead in control flow, no checkpoint fingerprint change from a callable being absent), and existing configs and tests keep working untouched

#### Scenario: probe_abort_propagates

- **WHEN** the probe raises its abort exception
- **THEN** the pipeline halts immediately with the block and depth named, no later block is captured or reconstructed, and the error surfaces from `quantize_model` unchanged

### Requirement: The driver probe SHALL fail the run at the first prefix whose ratio crosses the paper-derived collapse line

The driver-side probe SHALL precompute fp-prefix perplexities at every depth on the head slice of the held-out eval, measure each quantized prefix's perplexity on the same slice, compare through the `qk_report` verdict band, and SHALL abort the run when a prefix ratio exceeds `COLLAPSE_LINE` at any probed depth. The probe is kill-only: no prefix measurement SHALL certify `ok` or `degraded`, which come exclusively from the end-of-run full held-out gate.

#### Scenario: exploded_prefix_aborts_fast

- **WHEN** a reconstructed block pushes the quantized prefix's slice ratio above `COLLAPSE_LINE` (min 40.0, the [LB2]-derived line) at depth 2 or later
- **THEN** the run aborts at that block boundary — before any later block's solver work — naming the block, depth, and both prefix perplexities

#### Scenario: healthy_prefix_passes

- **WHEN** the prefix ratio at every probed depth stays at or below `COLLAPSE_LINE`
- **THEN** no probe abort fires, the pipeline completes normally, and the end-of-run verdict alone classifies the artifact

#### Scenario: probe_abort_leaves_report

- **WHEN** the probe aborts at a block boundary
- **THEN** the driver still writes `quantize_report.json` carrying `status: failed`, the aborting block, the depth, both prefix perplexities, and `artifact_exported: false`, and exits nonzero — a probe abort never leaves the run without durable evidence, mirroring the e2e gate's forensics contract

#### Scenario: report_single_shape_both_paths

- **WHEN** any run completes, aborted or not
- **THEN** the report key set is identical on both paths: every field the completion report carries is present on an abort report (unmeasured end-of-run numerics written as `null`), and `artifact_exported` appears on both (`false` on abort, `true` on completion) — strict consumers never KeyError on either shape

#### Scenario: abort_report_construction_lean_tested

- **WHEN** the suite runs in a lean environment without torch
- **THEN** the abort-report construction — building the report dict from a `PrefixProbeAbort`'s fields plus history, routed through `dump_report` with the failed status — is covered by tests that execute without skips, so the catch→report→status logic has repeatable proof independent of weights and CI

#### Scenario: probe_history_recorded_on_completion

- **WHEN** a run completes without a probe abort
- **THEN** `quantize_report.json` carries the per-depth probe history — `(depth, prefix_ppl, fp_prefix_ppl, ratio)` for every probed block — written through the same `dump_report` path as every other report value, so the first surviving run becomes the probe's own calibration data instead of an unauditable near-miss

#### Scenario: probe_comparator_reuses_gate_band

- **WHEN** the probe's decision logic runs
- **THEN** it is the same torch-free `qk_report.verdict` comparator as the end-of-run gate — the missing/non-finite check runs first, and `COLLAPSE_LINE` is the shared derived constant, not a second copy

#### Scenario: probe_slice_same_eval

- **WHEN** the driver builds the probe slice
- **THEN** the slice is the head of the same held-out evaluation batches, in the same order, with the same seed discipline as the end-of-run measurement — never a separately generated dataset

#### Scenario: fp_prefix_baselines_same_depth

- **WHEN** the driver precomputes prefix baselines
- **THEN** each depth's baseline is the fp model's perplexity over the same slice at the same prefix depth, so the probe ratio isolates quantization damage and not depth effects

### Requirement: The probe SHALL be configurable in interval, bounded in cost, and fixed in threshold

The driver SHALL read probe configuration from environment variables `QK_PROBE_INTERVAL` (positive integer blocks, default 1) and `QK_PROBE_BATCHES` (positive integer eval batches, default 2). The abort threshold SHALL be the derived `COLLAPSE_LINE` with no per-run override, and probes SHALL begin no earlier than depth 2.

#### Scenario: probe_knob_defaults_and_validation

- **WHEN** the driver starts with no probe env vars set
- **THEN** the probe measures every block boundary from depth 2 on a 2-batch slice, and a non-positive `QK_PROBE_INTERVAL` or `QK_PROBE_BATCHES` is rejected at startup

#### Scenario: probe_env_var_override

- **WHEN** `QK_PROBE_INTERVAL=4` and `QK_PROBE_BATCHES=1` are set
- **THEN** the probe fires every fourth block on a 1-batch slice

#### Scenario: probe_threshold_not_overrideable

- **WHEN** any run executes
- **THEN** no flag or config value re-pins the probe's abort line above or below the shared `COLLAPSE_LINE` constant

#### Scenario: probe_math_lean_tested

- **WHEN** the suite runs in a lean environment without torch
- **THEN** the probe decision logic (slice-ratio verdict, boundary semantics, NaN first) is covered by tests that execute without skips, while model-eval glue tests live in the torch tier per the repo's tier conftest pattern

#### Scenario: probe_empty_env_var_treated_as_unset

- **WHEN** `QK_PROBE_INTERVAL=` (empty string) is set
- **THEN** the default interval 1 applies — an empty value is treated as unset, and the empty-string case is covered by a fixture test

#### Scenario: probe_non_integer_rejected

- **WHEN** `QK_PROBE_INTERVAL=abc` or `QK_PROBE_BATCHES=1.5` is set
- **THEN** the driver exits nonzero at startup naming the invalid input, with the same named refusal as a non-positive value

### Requirement: The real-model driver SHALL report an end-to-end perplexity ratio against the same run's bf16 baseline

The real-model driver SHALL compute `ppl_ratio = ppl_quant_native / ppl_baseline_bf16` over one held-out eval partition through one logits path (the host object quantize_model wrote the artifact back into), and SHALL record both perplexities, the ratio, and the byte-derived effective BPW in `quantize_report.json`.

#### Scenario: ratio_single_measurement_path

- **WHEN** the driver measures quantized perplexity after `quantize_model` writes the artifact back into the host
- **THEN** the reported ratio uses exactly that measurement — no re-import or second measurement path — and `ppl_baseline_bf16`, `ppl_quant_native`, and `ppl_ratio` all appear in `quantize_report.json`

#### Scenario: ratio_history_all_refused

- **WHEN** the verdict logic replays the recorded production shapes (ratios 1,156,577.1; 943.49; 5,517.8; 463,742.5; 524,894.8; and NaN)
- **THEN** every shape classifies `failed` — no historical artifact ever receives `ok` or `degraded`

### Requirement: Every driver run SHALL receive a three-value end-to-end quality verdict from a paper-calibrated ratio band

The driver SHALL classify each run `ok`, `degraded`, or `failed` from the end-to-end ratio against two derived band edges, expose the classification in the report `status` field, and recur the same classification into the process exit code.

#### Scenario: verdict_ok_within_functional_envelope

- **WHEN** a run's ratio is at or below `OK_LINE` = `ceil(23.74/6.10)` = 4.0 — the functional envelope at the extreme budget end of [LB2] Sec. 5.2 Table 1 (LittleBit-2 at 0.1 bpp, PPL 23.74, against the FP16 6.10)
- **THEN** the verdict is `ok` and the process exits 0

#### Scenario: verdict_degraded_between_lines

- **WHEN** a run's ratio exceeds `OK_LINE` and is at or below `COLLAPSE_LINE` = `ceil(241.96/6.10)` = 40.0 — the documented method collapse of [LB2] Sec. 5.2 Table 1 (STBLLM at 0.55 bpp, PPL 241.96, against the FP16 6.10)
- **THEN** the verdict is `degraded`, the process exits 0, and the report carries the degraded status rather than a silent pass

#### Scenario: verdict_failed_above_collapse_line

- **WHEN** a run's ratio exceeds `COLLAPSE_LINE`, or the ratio is missing or non-finite
- **THEN** the verdict is `failed` and the process exits nonzero; the missing/non-finite check always precedes any threshold comparison

#### Scenario: verdict_edge_arithmetic_pinned

- **WHEN** the suite runs
- **THEN** a test asserts both edges' derivations exactly — `ceil(241.96/6.10) == 40` and `ceil(23.74/6.10) == 4` — and asserts the boundary semantics (`ratio == 40.0` → `degraded`; `ratio == 40.0001` → `failed`) plus the trap that `nan > 40.0` is False so a threshold comparison alone can never catch NaN

#### Scenario: verdict_helpers_torch_free_tested

- **WHEN** the test suite runs in a lean environment without torch
- **THEN** the verdict and sanitizer tests execute (no skips, no collection errors) because the logic lives outside the torch-importing driver, in the compressor package

### Requirement: quantize_report.json SHALL be provenance-complete and strict-JSON-parseable

The report SHALL name the actual run — top-level `solver` and `scale_dtype` from the argument values, `status` from the verdict — and SHALL emit `regime.artifact` and `config_f64.dtype_note` from those same arguments rather than hardcoded literals, and SHALL write every non-finite float as `null` so the report parses under a NaN-rejecting strict parser.

#### Scenario: report_mirrors_run_args

- **WHEN** a run executes with `--solver svd --scale-dtype float32`
- **THEN** the report records `solver: svd` and `scale_dtype: float32` top-level, and `regime.artifact` names float32 scales — the float16 literal never appears for a float32 run

#### Scenario: report_fp16_opt_in_named

- **WHEN** a run executes with the `--scale-dtype float16` opt-in
- **THEN** `regime.artifact` and `config_f64.dtype_note` name float16 scales, recorded from the arguments rather than from a constant

#### Scenario: report_nonfinite_as_null

- **WHEN** any report value is NaN or infinite — including a numpy scalar float such as `np.float32(nan)` — or writing the full report fails for any reason
- **THEN** non-finite values are written as `null` with `status: failed` and the file parses under a `parse_constant`-rejecting strict JSON parser; if serializing or writing the report itself raises, the driver SHALL still write a minimal `{"status": "failed", "report_error": ...}` report, exit nonzero, and never leave the run without a durable status; a healthy report parses with `status: ok`

### Requirement: The driver process SHALL fail closed on a failed verdict while always exporting the artifact

`main()` SHALL return 0 only for `ok` or `degraded` verdicts and a nonzero exit code for `failed`, and SHALL write the qpk1 export and `quantize_report.json` regardless of verdict, so a failed run still leaves a forensically complete artifact.

#### Scenario: failed_run_exits_nonzero

- **WHEN** a run's verdict is `failed` (including the missing-or-non-finite ratio case)
- **THEN** the process exit code is nonzero, distinguishable from a pass by any CI or orchestrator chain

#### Scenario: healthy_run_exits_zero

- **WHEN** a run's verdict is `ok` or `degraded`
- **THEN** the process exits 0

#### Scenario: export_and_report_persist_on_failure

- **WHEN** the gate fails the run
- **THEN** the qpk1 model directory and `quantize_report.json` are still written, with the report carrying `status: failed` as the durable trace

### Requirement: Checkpoint resume SHALL fail closed on empty or incomplete block archives

Resume SHALL never advance a block with fewer reconstructed layers than the
block is owed. Guards SHALL be driven from the live model's expected
quantizable names, not from the archive's contents.

#### Scenario: empty_archive_is_never_written

- **WHEN** a block completes with zero reconstructed layers
- **THEN** `save_block_checkpoint` raises and no `block-*.npz` file is
  written for that block

#### Scenario: empty_archive_is_never_counted_done

- **WHEN** `completed_blocks` scans a checkpoint directory
- **THEN** archives with zero keys (or no readable quantized-layer names) are
  excluded from the done set

#### Scenario: restore_asserts_expected_equals_present

- **WHEN** a resumed run applies checkpointed layers to a block
- **THEN** the set of layer names reconstructed from the archive is asserted
  equal to the block's expected quantizable names from the live model, and
  any symmetric difference raises an error naming the block and the missing
  names before the prefix advances

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

### Requirement: Harness changes touching a production import SHALL pass the surface-table gate

Any harness change that touches a production import SHALL be accompanied by the production-surface diff table — one row per touchpoint (collect_stats, robust_diag, lb_admm, magnitude_balance), each citing the production call-site line and reading `matches` or `DEFECT`. No fire SHALL launch while any row reads DEFECT.

#### Scenario: surface_table_gate_before_fire

- **WHEN** a Lane 0 fire is staged
- **THEN** the harness change's surface table exists with every production touchpoint row citing its call-site line, all rows read `matches`, and any DEFECT row blocks the launch — instance six gets caught by a grep, not a floor-switch

### Requirement: Checkpoint resume SHALL bind the model identity

The resume fingerprint SHALL bind the identity of the model being quantized: the
quantized-parameter scope's module inventory, each module's `(dout, din)` shape, and
its weight dtype, all derived from the LIVE model at fingerprint-write time. A resume
SHALL refuse a diverged identity, naming the diverged component(s), and SHALL refuse a
directory with no recorded identity. A repository id or path SHALL NOT be the identity.

#### Scenario: model_identity_binds_the_resume

- **WHEN** a checkpoint directory written while quantizing model A is resumed by a run
  quantizing model B under an identical `PTQConfig`
- **THEN** `check_fingerprint` raises before any block archive is applied, and no
  block of B is restored from A's archives

#### Scenario: model_identity_mismatch_names_the_divergence

- **WHEN** the live model's quantized-parameter scope differs from the recorded
  identity in module membership, in a per-module `(dout, din)` shape, or in a weight
  dtype
- **THEN** the refusal names the diverged module(s) and what diverged about them, so
  the mismatch is diagnosable from the error alone

#### Scenario: legacy_fingerprint_without_model_identity_is_refused

- **WHEN** a checkpoint directory's `fingerprint.json` carries no model-identity
  component (it was written before this binding existed)
- **THEN** the resume refuses and says the directory predates the model-identity
  component, so its provenance with respect to the live model cannot be verified

#### Scenario: driver_default_checkpoint_dir_is_per_run

- **WHEN** `tools/quantize_hf_model.py` runs without an explicit `--checkpoint-dir`
- **THEN** it checkpoints into a FRESH per-run directory rather than a shared stable
  path, so two defaulted launches never share block archives; resumption is opt-in by
  naming the directory, and the chosen directory is named in the launch's output

### Requirement: Calibration and eval data knobs SHALL count sequences, not packed batches

The `--calibration-seqs` and `--eval-seqs` knobs SHALL denote **sequences**, matching the [NQ] protocol unit (128 calibration sequences, [NQ] Sec 3.4/App. C). The loader SHALL convert sequences to packed batches internally (`n_chunks = N // BATCH`, `BATCH = 8`), fail-closed when N is not divisible by BATCH, and the reported counter SHALL be computed at the same boundary as the chunking — before any tune-window slice that changes the chunk count.

#### Scenario: calibration_knob_counts_sequences

- **WHEN** the driver runs with `--calibration-seqs 128`
- **THEN** the loader produces chunks covering exactly 128 sequences (16 packed batches of 8)
- **AND** the reported `calibration_rows` equals 128, matching the [NQ] protocol, so the report's deviation flag does not fire
- **AND** the counter is computed before the `tune_seq_len != seq_len` slice at the loader/driver boundary (a fix placed after that slice computes the value 4× too large under `TUNE_SEQ_LEN=512`)

#### Scenario: calibration_knob_fails_closed_on_odd_count

- **WHEN** `--calibration-seqs` is not divisible by the packed batch size
- **THEN** the driver exits nonzero with an error naming the knob and the divisibility requirement

#### Scenario: eval_knob_counts_sequences

- **WHEN** the driver derives the maximal local eval window
- **THEN** `--eval-seqs` denotes sequences and the window derivation is expressed in sequences (18 batches of 8 re-expressed as 144 sequences), with the eval gate comparing like units at every comparison

#### Scenario: red_first_unit_test

- **WHEN** the unit tests run against the pre-fix driver
- **THEN** an end-to-end assertion `len(calib) * BATCH == CALIBRATION_SEQS` fails with the current value 1024 (red-first), and passes after the fix with 128

### Requirement: The fail-fast verdict machinery SHALL prove geometry-invariance or re-derive constants per geometry

Every gate constant used by the prefix fail-fast verdict SHALL either be geometry-invariant by construction or be re-derived per geometry. The collapse line, pedestals, and the prefix probe each follow the classification stated in the scenarios below.

#### Scenario: collapse_line_is_paper_anchored

- **WHEN** any run evaluates the depth-2 prefix ratio against the collapse line
- **THEN** the line is exactly the published-anchor derivation asserted in `tests/test_qk_report.py`, with no run-specific calibration and no code movement on geometry change

#### Scenario: pedestal_tracks_geometry

- **WHEN** a run changes calibration or eval geometry
- **THEN** its baseline and fp-prefix pedestals are measured at that run's own geometry and recorded in the run's provenance before any ratio verdict is computed

#### Scenario: prefix_probe_same_depth_same_object

- **WHEN** the prefix probe computes its ratio
- **THEN** the quantized-prefix PPL and the fp-prefix PPL are measured on the same object at the same depth, dividing depth-dependent floor effects out of the ratio

### Requirement: The pipeline SHALL be held to a paper-recreation bar pinned to [NQ] Table 2 rows for Qwen3-0.6B

An acceptance run SHALL demonstrate, on Qwen3-0.6B over [NQ]'s WikiText-2 sequence-length-2048 protocol, a perplexity at or below `1.15 * row` for one of the three NanoQuant rows of [NQ] Sec. 4.2 Table 2 (27.56 @1.00 bits, 33.79 @0.80 bits, 52.94 @0.55 bits, against the 12.66 baseline), with byte-derived `effective_bpw` at or below the targeted row's bit budget.

#### Scenario: bar_constants_asserted

- **WHEN** the suite runs
- **THEN** a test asserts the four row constants and the tolerance arithmetic (`1.15 * row` for each targeted row) are the values pinned in this delta, and that no runtime path recomputes them

#### Scenario: bar_undefined_below_lowest_row

- **WHEN** an artifact's bit budget is below 0.55 bits
- **THEN** no recreation row exists for it; it is beyond-paper research — never recorded under the word recreation, never granted ledger status on interpolation

#### Scenario: bar_tolerance_provisional_until_measured

- **WHEN** artifacts are still far from any row
- **THEN** the 1.15 tolerance stands as the pre-registered provisional, and the first artifact landing within 2x of a targeted row obligates re-deriving the tolerance from that artifact's measured run-to-run eval variance and re-pinning it as a spec change

#### Scenario: bar_never_recalibrates_the_band

- **WHEN** a band-edge recalibration is ever proposed
- **THEN** moving an edge so a paper-reference artifact (including [NQ]'s own rows, e.g. @0.55 ratio 4.18) reads `ok` is prohibited — the band is never tuned to flatter a reference; it recalibrates only from production healthy-extreme evidence

#### Scenario: bar_not_purchased_with_bits

- **WHEN** an artifact claims a bar row
- **THEN** its serialized-byte `effective_bpw` is at or below that row's bit budget — a lower PPL at a higher bit budget meets no row

#### Scenario: bar_requires_paper_protocol

- **WHEN** an acceptance run executes
- **THEN** it evaluates at sequence length 2048 over the WT2 test partition per [NQ] Sec. 4.2; seq-512 numbers alone are recorded alongside, never substituted for the bar

#### Scenario: bar_recorded_not_gated

- **WHEN** an acceptance run completes
- **THEN** a ledger entry records model id, commit hash, targeted row, eval protocol, measured PPL, ratio, effective_bpw, and the operational-gate verdict — the bar is a documented human-gated acceptance criterion, not a per-run gate verdict

#### Scenario: bar_distinct_from_landing_gate

- **WHEN** any artifact is classified
- **THEN** the `ok/degraded/failed` band remains the only per-run verdict; an `ok` artifact may still be below the bar, and no artifact above the bar receives `ok` on bar evidence alone

### Requirement: The PTQ driver SHALL support independent tune and evaluation sequence lengths

The driver's `--seq-len` SHALL remain the evaluation geometry, and a `--tune-seq-len` flag SHALL bound the per-block reconstruction windows independently. When `--tune-seq-len` is unset it SHALL default to `--seq-len`, preserving single-geometry behavior exactly.

#### Scenario: split_mode_preserves_eval_geometry

- **WHEN** the driver runs with `--seq-len 2048 --tune-seq-len 512`
- **THEN** evaluation PPL runs over 2048-token WT2 test sequences while block reconstruction tunes on 512-token windows, and the report records both lengths

#### Scenario: default_single_geometry_unchanged

- **WHEN** `--tune-seq-len` is unset
- **THEN** tuning windows use `--seq-len` exactly as before the flag existed — existing run behavior is unchanged and no silent geometry change occurs

### Requirement: The ADMM solver SHALL default to the torch backend with NumPy unit-test-only

The solver SHALL default to the torch backend, the only backend that produces artifacts, resolved fail-closed once per run with the `QK_RHO_END` provenance idiom. NumPy SHALL execute ONLY inside unit tests, as the parity reference — never in a fire, harness, or integration path — with NO fallback: a torch request without an accelerator SHALL fail rather than resolve to NumPy. `solver_is_admm` SHALL remain an algorithm condition.

#### Scenario: backend_knob_provenance

- **WHEN** a run selects `SOLVER_BACKEND=torch` (or resolves the default)
- **THEN** launch.env, the checkpoint fingerprint, and every solver-probe row record backend and dtype
- **AND** the default resolves to torch — the accelerated backend
- **AND** a torch backend without an available accelerator device fails closed at startup, on the same pattern as `GPU=1` requiring a rocm-tagged image — a torch backend running on CPU is a slowdown wearing an optimization's name

#### Scenario: numpy_is_unit_test_only

- **WHEN** the NumPy backend executes
- **THEN** it executes inside a unit test and nowhere else — no fire, no harness, no integration path

#### Scenario: admm_algorithm_unchanged

- **WHEN** either backend factorizes a matrix
- **THEN** it executes the same continuous-update, SVID-proxy, dual-update, and magnitude-balance steps, and `solver_is_admm` remains true on either backend

### Requirement: Backend parity SHALL be established by unit tests with torch-only production rows

The torch backend SHALL be accepted on its unit-test parity suite — the only place the NumPy reference executes — together with the campaign's measured tolerance record (fp32 GPU ≡ fp64 CPU within ±0.002 on production layers). Production and probe fires are torch-only, and every probe row SHALL carry row-time so the measured win replaces the estimate.

#### Scenario: torch_only_fire_rows

- **WHEN** the Lane 0 fire executes
- **THEN** all 20 solves run on torch, no NumPy process runs, and every row carries row-time alongside recon cos, residual trajectory, n_iter, and min_eigs

#### Scenario: unit_test_parity_is_the_gate

- **WHEN** the backend's parity suite runs as unit tests
- **THEN** tests compare reconstructed products (sign-ambiguity-safe) within the tolerance class, and a failing parity test blocks acceptance

#### Scenario: tolerance_breach_named

- **WHEN** any parity unit test moves outside the tolerance class
- **THEN** the failing test names the operation and its delta, and acceptance fails instead of passing silently

### Requirement: Checkpoint restore SHALL refuse a mis-shaped state before writing any layer

Checkpoint restore SHALL validate each stored state's arrays against the live
linear's weight and bias shapes before writing any layer. A state whose
Ub/Vb/s1/s2/bias cannot factorize to the live `(dout, din)` weight SHALL raise
naming the block, layer, and stored shapes, leaving the block untouched.

#### Scenario: restore_refuses_mis_shaped_state

- **WHEN** a resumed run applies a checkpointed state whose arrays cannot
  produce the live linear's weight `(dout, din)` — Ub not `(dout, r)`, Vb not
  `(din, r)` with a shared r, s1 not `(dout,)`, s2 not `(din,)`, or a stored
  bias not `(dout,)`
- **THEN** `apply_layer_states` raises a RuntimeError naming the block, the
  layer, the stored shapes, and the live weight shape, refusing to write the
  state

#### Scenario: mis_shaped_refusal_leaves_block_untouched

- **WHEN** a right-key/wrong-shape archive is applied to a multi-layer block
- **THEN** the raise fires before the first `copy_`, and every linear's weight
  in the block is bit-identical to its value before the restore

### Requirement: A resumed run SHALL declare its restart in the report as a record, never a gate

The real-model driver SHALL emit a top-level `resume` record in `quantize_report.json` naming whether the run resumed, which blocks were restored from a prior launch, the checkpoint directory, the container image and digest, and the fingerprint that admitted the restore. The record SHALL be informational: it SHALL NOT enter the verdict, the exit code, or any gate, and SHALL NOT be folded into the checkpoint fingerprint.

#### Scenario: resume_record_on_resumed_run

- **WHEN** a launch resumes — its `--checkpoint-dir` holds block archives a prior launch wrote, admitted by `check_fingerprint`
- **THEN** `quantize_report.json` carries a top-level `resume` record with `resumed: true`, `restored_blocks` equal to the completed blocks present at launch (the `completed_blocks` scan), `checkpoint_dir` naming the directory, and `image`/`image_digest` read from the launch environment

#### Scenario: resume_record_fresh_run_consistent_shape

- **WHEN** a run starts fresh (no completed block archives at launch)
- **THEN** the `resume` key is still present on the report, with `resumed: false`, an empty `restored_blocks`, `checkpoint_dir: null`, `image`/`image_digest` read from the launch environment (or `"unknown"` when unset), and `admitted_by_fingerprint: null`, so the completion, abort, and stage-failure key sets stay identical and a strict consumer never KeyErrors on either shape

#### Scenario: resume_true_only_when_admitted

- **WHEN** a fingerprint check refuses the restore (the digest or model identity diverges)
- **THEN** the refusal propagates a `RuntimeError` before any report is written, so no report carries `resumed: true` from a refused resume — `resumed: true` SHALL appear only on runs whose `check_fingerprint` admitted the restored blocks
- **AND** a future change that catches the refusal downstream (post-report-write) SHALL flip `resumed: true` to an attempt-based record with an explicit `admitted: false` field, never silently land a false restoration claim

#### Scenario: resume_record_is_a_record_not_a_gate

- **WHEN** any run carries a resume record
- **THEN** the `ok`/`degraded`/`failed` verdict, the process exit code, the band edges, and every fail-fast, probe, and stage gate behave byte-identically to an otherwise-equivalent non-resumed run — the record changes no decision and is never read by a gate

#### Scenario: resume_marker_never_folds_the_fingerprint

- **WHEN** the resume record is emitted
- **THEN** no resume marker enters `PTQConfig`, the `config_fingerprint` payload, or the `_FINGERPRINT_PROVENANCE` record, so a resumed run computes the same config digest as the pre-resume blocks and admits itself rather than refusing its own checkpoint

#### Scenario: resume_identity_is_evidence_not_an_assertion

- **WHEN** the resume record names the restored blocks' provenance
- **THEN** it records the restored checkpoint's `fingerprint` digest, its `model_identity` digest, and its provenance members verbatim — the `_FINGERPRINT_PROVENANCE` tuple (`solver_backend`, `compute_dtype`, `solver`, `rank`, `factor_mode`, `admm_reg`, `rho_start`, `rho_end`) plus the separately recorded `solver_device`
- **AND** it does not emit a computed `identical_bytes` boolean — cross-reboot source-byte identity is a host-side fact (a git diff across the two tips) recorded in the launch trail, which the in-container driver without `git` cannot compute

#### Scenario: resume_record_on_failure_paths

- **WHEN** a resumed run aborts via the prefix probe or fails a fail-closed stage gate
- **THEN** the `resume` record appears on the abort and stage-failure report shapes too, with `resumed` and `restored_blocks` populated from launch state, so the declaration survives a run that died before completion

### Requirement: The report's fidelity label SHALL be a computed verdict from declared configuration against the paper regime

The report's `paper_faithful` label SHALL be the conjunction of eight config-match conditions plus all existing stage-level conditions. The `paper_faithful_flag` condition SHALL NOT exist. Every condition SHALL be reported. No hardcoded dtype SHALL replace the cut flag.

#### Scenario: all_conditions_satisfied_yields_true

- **WHEN** a run executes with `solver=admm`, `compute_dtype=bf16`, `factor_mode=binary`, `ablation=none`, `layer_scope=all`, `admm_iter=400`, `rho_end` in [0.0, 1.0], `device` on accelerator (not cpu), no runtime deviations, no polish rounds, no vacuous teacher, capture from fp stream, no stage failure, and every required stage executed
- **THEN** the label is `true` and every condition in the set is `true`

#### Scenario: condition_set_names_every_config_field

- **WHEN** the fidelity conditions are computed for any run
- **THEN** the condition record includes `config_solver_is_admm`, `config_compute_dtype_is_bf16`, `config_factor_mode_is_binary`, `config_ablation_is_none`, `config_layer_scope_is_all`, `config_admm_iter_matches_regime`, `config_admm_rho_end_matches_regime`, and `config_device_is_accelerator` — one condition per paper-regime config field — plus all existing stage-level conditions (`no_runtime_deviations`, `polish_off`, `no_discrete_flip_pass`, `no_scale_only_rescope`, `no_vacuous_teacher`, `capture_is_fp_stream`, `no_stage_failure`, `solver_is_admm`, and `required_*_executed`)

#### Scenario: solver_mismatch_names_the_deviation

- **WHEN** a run executes with `--solver svd` and all other config matches the paper regime
- **THEN** `config_solver_is_admm` is `false`, the label is `false`, and a reader can identify the denying condition from the condition record alone

#### Scenario: compute_dtype_mismatch_names_the_deviation

- **WHEN** a run executes with `--compute-dtype float32` and all other config matches
- **THEN** `config_compute_dtype_is_bf16` is `false` and the label is `false`

#### Scenario: admm_iter_mismatch_names_the_deviation

- **WHEN** a run executes with `--max-admm-iter 200` and all other config matches
- **THEN** `config_admm_iter_matches_regime` is `false` and the label is `false`

#### Scenario: device_mismatch_names_the_deviation

- **WHEN** a run executes with `--device cpu` and all other config matches
- **THEN** `config_device_is_accelerator` is `false` and the label is `false`

#### Scenario: no_hardcoded_dtype_in_cut_surface

- **WHEN** the implementation is complete
- **THEN** the cut surface (driver, report, fidelity conditions) contains no hardcoded `torch.float64`, `torch.float32`, `torch.bfloat16`, `np.float64`, or `np.float32` type constants — every dtype reference is parametric on `compute_dtype` or a named accumulation dtype per pipeline-dtype's no-hardcoded-dtype requirement

#### Scenario: label_denied_by_stage_condition_still_reports_config_match

- **WHEN** a run's config matches the paper regime but a vacuous-teacher event fires
- **THEN** all config-match conditions are `true`, `no_vacuous_teacher` is `false`, the label is `false`, and both facts are visible in the condition record

#### Scenario: no_request_flag_anywhere

- **WHEN** any run executes
- **THEN** no CLI flag, config field, or report identity field carries a fidelity-request value; the `paper_faithful_flag` key is absent from the condition record; the label is always computed and always reported

#### Scenario: eval_window_guard_dies_with_the_flag

- **WHEN** a run launches with a non-maximal `--eval-seqs` window
- **THEN** startup does not refuse it — the `--paper-faithful` eval-window guard is deleted with the flag; the sliced window is recorded in the report's evaluation record (selected blocks, source-token coverage) and never denies the computed label, because evaluation comparability is outside the method verdict

### Requirement: The driver CLI SHALL present an operator-grade help surface

Every surviving driver flag SHALL carry help text naming the flag's unit, its default value, and its effect on the artifact. Flags SHALL be grouped by function (model/data, solver/quality, output, resources). Help text SHALL NOT cite paper sections, change ids, lane names, or bot vocabulary. The word "paper" SHALL NOT appear in any flag's help string.

#### Scenario: help_output_has_operator_sections

- **WHEN** the driver is invoked with `--help`
- **THEN** the output includes grouped sections ("Model and data:", "Solver and quality:", "Output:", "Resources:") and every surviving flag appears under its section

#### Scenario: help_naming_unit_and_default

- **WHEN** the driver is invoked with `--help`
- **THEN** every flag's help text states its unit (tokens, sequences, iterations, fraction) and its default value

#### Scenario: help_free_of_bot_vocabulary

- **WHEN** the driver is invoked with `--help`
- **THEN** the output does not contain "paper", "faithful", change ids, or lane names

#### Scenario: cut_flags_absent_from_help

- **WHEN** the driver is invoked with `--help`
- **THEN** the output does not mention `--paper-faithful`, `--ste-flip-pass-budget`, `--probe-interval`, `--probe-batches`, `--factor-mode`, `--ablation`, or `--layer-scope`

### Requirement: Non-user mechanisms SHALL use environment variables with the QK_ prefix

Lane-internal controls that survive the CLI cut SHALL be settable only via environment variables with the `QK_` prefix. The driver SHALL read them at startup and SHALL NOT accept them as CLI flags. The controls this change migrates to env vars are `QK_PROBE_INTERVAL`, `QK_PROBE_BATCHES`, `QK_FACTOR_MODE`, `QK_ABLATION`, and `QK_LAYER_SCOPE`. Pre-existing `QK_*` env vars (`QK_RHO_END`, `QK_ADMM_REG`, `QK_SOLVER_PROBE`) are unaffected.

#### Scenario: env_var_wires_to_config

- **WHEN** `QK_FACTOR_MODE=fp` is set in the environment
- **THEN** `PTQConfig.factor_mode` is `"fp"` — the env var feeds the config field through the driver, not through a second config path

#### Scenario: env_var_defaults_preserve_behavior

- **WHEN** no `QK_*` env vars are set
- **THEN** the pipeline behavior is byte-identical to the pre-cut default run: `binary` mode, `none` ablation, `all` scope, probe interval 1, probe batches 2

#### Scenario: env_var_not_a_cli_flag

- **WHEN** a caller passes `--factor-mode fp` on the command line
- **THEN** the parser rejects it as an unrecognized argument — `QK_FACTOR_MODE` is the only path

#### Scenario: env_var_validation_rejects_invalid

- **WHEN** `QK_FACTOR_MODE=invalid` is set
- **THEN** the driver exits nonzero at startup naming the invalid value and the accepted choices

#### Scenario: env_var_empty_treated_as_unset

- **WHEN** `QK_FACTOR_MODE=` (empty string) is set
- **THEN** the default `"binary"` applies — an empty value is treated as unset, and this case is covered by a fixture test

#### Scenario: env_var_non_integer_rejected

- **WHEN** `QK_PROBE_INTERVAL=abc` or `QK_PROBE_BATCHES=1.5` is set
- **THEN** the driver exits nonzero at startup naming the invalid input, with the same named refusal path as a non-positive value

### Requirement: The discrete sign-flip pass SHALL be removed from the pipeline

The non-paper discrete sign-flip mechanism SHALL be deleted at every site: the `_discrete_flip_pass` definition, import, re-export, and call path. STE event records SHALL NOT carry a `flip_pass` key. `PTQConfig` SHALL NOT carry a `ste_flip_pass_budget` field. `no_discrete_flip_pass` SHALL remain in the condition set and SHALL always be `true`. The four runtime gates that consumed `PTQConfig.paper_faithful` SHALL re-key on the config-match conjunction.

#### Scenario: flip_pass_code_path_gone

- **WHEN** the STE stage completes on any layer
- **THEN** no discrete sign-flip pass executes; the post-STE path goes directly to sign extraction and packing per [NQ] Algorithm 1 lines 20–23, and the symbol `_discrete_flip_pass` does not exist anywhere in `src/` — definition (`_ste.py`), import (`_block_stage.py`), and re-export (`block_recon.py`) are all deleted

#### Scenario: flip_pass_never_in_conditions

- **WHEN** the fidelity conditions are computed for any run
- **THEN** `no_discrete_flip_pass` is `true` — it is always satisfied and can never be the denying condition

#### Scenario: ste_flip_pass_budget_field_absent

- **WHEN** `PTQConfig` is constructed
- **THEN** the object has no `ste_flip_pass_budget` attribute; accessing it raises `AttributeError`

#### Scenario: paper_ste_declared_from_config_match

- **WHEN** the STE stage completes on a paper-regime run (config matches all eight regime fields)
- **THEN** the per-linear `declared` field is `"paper_ste"` — the diagnostic path (discrete flip pass, scale-only rescope) is never invoked, and `no_scale_only_rescope` is always `true` on paper-regime runs

#### Scenario: non_paper_polish_runs_diagnostic_path

- **WHEN** a run with `factor_mode=binary` and `ablation=none` that deviates from the regime elsewhere (e.g. `--solver svd` or `--max-admm-iter 200`) reaches the STE stage
- **THEN** polish still runs (the gate is `factor_mode=binary ∧ ablation≠untuned-signs ∧ not config_matches_paper_regime`); on the post-STE ladder a linear with sign crossings declares `"paper_ste"` and a zero-crossing linear declares `"scale_only"` — the `declared` vocabulary is `{"paper_ste", "scale_only"}` and `discrete_flip_pass` no longer exists

#### Scenario: vacuous_teacher_abort_on_paper_regime_only

- **WHEN** a TuneFP pass encounters a zero first-step gradient
- **THEN** on a paper-regime run the driver raises `VacuousTeacherError` (fail-closed); on a non-paper run the event is recorded and the stage continues with pristine weights — the diagnostic, not the denial

#### Scenario: paper_regime_stage_budgets_are_app_c

- **WHEN** a paper-regime run (config matches all eight regime fields) constructs its stage budgets
- **THEN** the budgets are the [NQ] App. C triple — `stage_pre=StageHyper(1e-4, 4, 8)`, `stage_latent=StageHyper(1e-5, 1, 8)`, `stage_global=StageHyper(1e-6, 1, 8)` (lr, batch_size, epochs) — and any run failing the conjunction keeps the gate-shaped triple `StageHyper(1e-4, 4, 0)` / `StageHyper(2e-3, 4, 3)` / `StageHyper(1e-3, 4, 4)`

#### Scenario: stage_map_discrete_flip_pass_always_false_on_paper_regime

- **WHEN** the report stages summary is computed for a paper-regime run
- **THEN** the `ste.discrete_flip_pass` key is `false` — it stays in the schema for backward compatibility but is never `true` on a paper-regime run

### Requirement: The exporter SHALL write the remainder at the declared dtype

The real-model driver SHALL expose `--remainder-dtype {fp32, bf16}` as an explicit override; when unset, the default follows the pipeline compute-dtype mapping (`bf16`→`"bf16"`, `float32`→`"fp32"`, `float64`→`"fp32"`). The writer SHALL apply ONE shared round-to-nearest-even conversion helper with a post-cast finiteness gate on the bf16 path, and SHALL declare the dtype in the manifest and the `weight_tying` suffix so the declarations agree (qk-server RD-1).

#### Scenario: RDW-1.1 — The fp32 path is byte-identical
- **WHEN** the driver exports with remainder dtype fp32, whether explicit or resolved from the float32 compute default
- **THEN** the remainder files' bytes equal the pre-change writer's
  pinned bytes, and the manifest carries `dtype = "fp32"`

#### Scenario: RDW-1.2 — A bf16 export is a loadable declared artifact
- **WHEN** the driver exports with `--remainder-dtype bf16`
- **THEN** the output directory carries `[sidecar] dtype = "bf16"`, a
  matching `@bf16` weight-tying suffix, and bf16 remainder files, and
  the Rust loader accepts it (qk-server RD-2.1)

#### Scenario: RDW-1.3 — Non-finite values refuse on the bf16 path
- **WHEN** a bf16 export encounters a non-finite remainder value
- **THEN** the cast stage refuses with a named error; the fp32 export
  path is unchanged

#### Scenario: RDW-1.4 — The export self-check uses the declared itemsize
- **WHEN** the exporter self-checks the sidecar byte length
- **THEN** the check validates against the float count times the
  declared dtype's itemsize, so a miscount fails at export, not at
  server load

### Requirement: The report SHALL name the remainder dtype

The report's fidelity fields SHALL name the remainder storage dtype
from the run's arguments: a top-level `remainder_dtype` field, and
`regime.artifact` and the dtype note strings extended to name it —
emitted from the argument values, never from hardcoded literals.

#### Scenario: RDW-2.1 — A bf16 run records bf16
- **WHEN** a run executes with `--remainder-dtype bf16`
- **THEN** the report records `remainder_dtype: bf16` and the artifact
  and dtype-note strings name bf16, recorded from the argument

#### Scenario: RDW-2.2 — The default records the dtype resolved from the arguments
- **WHEN** a run executes without `--remainder-dtype` at any compute dtype
- **THEN** the report records the resolved remainder dtype (the compute-dtype mapping, never a hardcoded literal) — `"fp32"` unless the compute dtype maps to bf16, and the bf16 literal never appears for a float32-compute run

### Requirement: The repack utility SHALL convert the remainder without re-training

`tools/repack_remainder.py` SHALL convert an artifact's remainder
from fp32 to bf16 into a NEW directory, never mutating the source: layer
streams and tokenizer files SHALL be copied byte-identically, the
remainder re-encoded through the shared conversion helper, and the
manifests rewritten with the declared dtype. The output manifest SHALL
record the conversion provenance and SHALL carry no wall-clock
timestamps, so identical sources produce identical output bytes.

#### Scenario: RDP-1.1 — The repacked fixture loads
- **WHEN** the committed fp32 fixture is repacked to bf16
- **THEN** the output directory loads in the Rust loader with the
  pinned bf16 values, and its manifest carries `dtype = "bf16"` plus a
  `[repack]` record naming the source model id, the conversion
  (`remainder fp32 -> bf16`), and the sha256 of the source remainder
  files

#### Scenario: RDP-1.2 — An already-bf16 source refuses
- **WHEN** the source artifact's remainder is already declared bf16
- **THEN** the utility refuses with a named error naming the artifact

#### Scenario: RDP-1.3 — A non-finite source refuses
- **WHEN** any source remainder value is non-finite
- **THEN** the utility refuses with a named error; no output directory
  is written

#### Scenario: RDP-1.4 — Non-remainder files are byte-identical
- **WHEN** the repack copies the layer streams and tokenizer files
- **THEN** every non-remainder file in the output is byte-identical to
  its source counterpart

#### Scenario: RDP-1.5 — Repack is deterministic
- **WHEN** the same source is repacked twice
- **THEN** the two output directories are byte-identical

### Requirement: bf16 remainder conversion SHALL bound elementwise error at half an ulp

The fp32-to-bf16 conversion SHALL round to nearest even, so every
converted finite element satisfies |bf16(x) - x| <= 2^-8 |x| + 2^-134 —
and this bound, not the float64 1e-12 parity tolerance, SHALL govern
bf16 round-trip comparisons.

#### Scenario: RDP-2.1 — Elementwise round-trip within the half-ulp bound
- **WHEN** a deterministic fp32 matrix is converted to bf16 and
  compared elementwise
- **THEN** every element satisfies the 2^-8|x| + 2^-134 bound exactly

### Requirement: The exporter SHALL write a single-file blob

The real-model driver SHALL expose `--container-format {blob,dir}`
and SHALL emit a `.qpk` blob when `blob` is named (default) and a
directory when `dir` is named. The blob path SHALL render the header
JSON (B2), section table (B3), and aligned payloads (B4) per the blob
specification, with the remainder at the declared `--remainder-dtype`.
The report SHALL record the emitted blob's sha256 (B6 external
fingerprint). The directory path SHALL preserve the pre-change export
behaviour.

#### Scenario: BW-1.1 — A blob export loads in the Rust loader

- **WHEN** the driver runs with the default `--container-format blob` on
  a tiny model
- **THEN** the output `.qpk` file passes every loader validation gate
  (qk-server BL-1.1 and BL-2.1).

#### Scenario: BW-1.2 — Directory export is still accepted

- **WHEN** the driver runs with `--container-format dir`
- **THEN** the output directory loads in the legacy loader path
  (qk-server BL-1.2) and its content is byte-identical to a pre-change
  directory export.

#### Scenario: BW-1.3 — The report records the emitted blob's sha256

- **WHEN** the driver exports a blob
- **THEN** `quantize_report.json` carries the blob's sha256 — the hash
  is recorded outside the container, never as a header field (B6).

### Requirement: The same model exported twice SHALL produce identical blobs

The same model packed twice to blob SHALL produce byte-identical files.
The writer SHALL use the fixed canonical section order, SHALL zero all
padding bytes, and SHALL emit no wall-clock timestamp in the header.
This SHALL be pinned by packing a model twice and asserting
`sha256(blob1) == sha256(blob2)`.

#### Scenario: BW-2.1 — Pack-twice sha256 is equal

- **WHEN** a tiny model is exported to blob, then exported again from
  the same state
- **THEN** `sha256(file1) == sha256(file2)`.

### Requirement: The repack utility SHALL convert a directory to a blob

`tools/repack_remainder.py` SHALL gain a dir→blob leg: given a
directory artifact, it SHALL produce a single `.qpk` file carrying every
layer stream verbatim (SF-2.1 byte-identical round-trip bytes), the
remainder at the declared or converted dtype, the tokenizer files as
raw sections, and a complete header + section table. Non-remainder
content SHALL be byte-identical to the source directory.

#### Scenario: BW-3.1 — A dir→blob repack loads

- **WHEN** the committed server-model directory fixture is repacked to
  blob with `--container-format blob`
- **THEN** the output `.qpk` loads in the Rust loader and produces a
  `ModelDirectory` identical field-for-field to the one loaded from the
  source directory (qk-server BL-1.1 + BL-2.1).

#### Scenario: BW-3.2 — Layer streams are byte-identical

- **WHEN** a directory is repacked to blob
- **THEN** the bytes of every layer stream section in the blob equal the
  bytes of the corresponding `.qpk1` file in the source directory —
  verified by extracting each section and comparing sha256.

### Requirement: The repack utility SHALL convert a blob's remainder dtype

The repack SHALL accept a `.qpk` blob as input and produce a new `.qpk`
blob with the remainder converted to a different declared dtype (fp32
→ bf16, through the shared RNE cast helper from the parent change). The
header's `remainder_dtype` and `weight_tying` suffix SHALL be rewritten
to agree. Section offsets and lengths SHALL be recomputed. Layer
streams and tokenizer sections SHALL be carried over byte-identically.

#### Scenario: BW-4.1 — A blob→blob fp32→bf16 conversion loads

- **WHEN** the committed blob fixture (fp32 remainder) is repacked with
  `--remainder-dtype bf16`
- **THEN** the output blob loads (qk-server BL-1.1), its
  `remainder_dtype` reads `"bf16"`, and the loaded remainder values
  satisfy the parent change's RDP-2.1 half-ulp bound.

#### Scenario: BW-4.2 — Layer streams survive blob→blob unchanged

- **WHEN** a blob is repacked to a new blob with a different remainder
  dtype
- **THEN** the byte ranges of every layer stream section in the output
  blob are identical to the input blob's corresponding ranges (except
  at different offsets), and the tokenizer raw sections are also
  byte-identical.

### Requirement: TuneFP SHALL persist the tuned bias

The TuneFP stage's tunable set SHALL include each remainder linear's bias, and
the factorized replacement built after that pass SHALL take the bias the pass
produced. Snapshotting and restoring a pre-tuning bias at factorization SHALL
NOT occur: the compensation the stage computed is the artifact's bias, and the
packed layer state SHALL carry it into the container.

#### Scenario: tunefp_bias_lands_in_the_artifact

- **WHEN** a block with bias-bearing linears is reconstructed
- **THEN** each factorized linear's packed state carries the bias the module
  held at its factorization — the value the preceding TuneFP pass wrote — and
  the host module's bias after reconstruction equals it, so the host forward
  is the artifact's forward.

### Requirement: The TuneFP record SHALL declare the bias movement

The TuneFP stage record SHALL report `bias_delta_rel` — the tuned linears'
relative bias movement — beside `weight_delta_rel`, which spans all tuned
parameters and cannot alone attest the bias compensation. The record SHALL name
which of three causes left the figure absent, never a confident `0.0`: no
bias-bearing tunables (`bias_tunables` 0); a stage that never ran (`steps` 0);
or an all-zero bias baseline (`steps` non-zero, division undefined).

#### Scenario: tunefp_record_declares_bias_movement

- **WHEN** the stage runs over linears that carry a bias
- **THEN** its record's `bias_delta_rel` equals the relative bias movement
  recomputed from a pre-stage snapshot, and when the tunable linears carry no
  bias the field carries no value.

#### Scenario: tunefp_record_bias_movement_undefined_over_all_zero_baseline

- **WHEN** the stage runs over bias-bearing linears whose selected biases are
  all exactly `0.0` at stage entry
- **THEN** its record's `bias_delta_rel` carries no value while `bias_tunables`
  and `steps` are both non-zero — the relative movement has no denominator, so
  it is left undefined rather than reported as a confident `0.0`.

### Requirement: The STE refinement SHALL run once per block over the block's whole factorized set

Per block, after every factorization completes and before the stream swap, TuneLatentSTE SHALL run exactly once over ALL of the block's factorized linears: one optimizer holding every latent and scale parameter set, and one loss on the block output against the frozen teacher target `y_b` ([NQ] Eq. 10). The paper's per-target-layer reading is a declared deviation (design.md).

#### Scenario: ste_refines_all_factorized_linears_in_one_block_pass

- **WHEN** a block carrying N factorized linears is reconstructed
- **THEN** TuneLatentSTE is invoked exactly once for that block, with all N linears as free parameters
- **AND** every per-linear row of that block carries the same block-objective measurement, because the pass had one objective

### Requirement: The STE stage SHALL emit one block-scope record carrying per-linear rows

Each block's refinement SHALL emit exactly one STE record. The record SHALL carry the stage's block-scope summary (steps, wall time, gradient floor, vacuity, inertness, and a scope marker) and one row per factorized linear; each row SHALL carry that linear's latent and scale movement, its sign-flip counts, and its dead-column count under the joint pass.

#### Scenario: ste_record_is_one_block_record_with_per_linear_rows

- **WHEN** a block's refinement completes
- **THEN** exactly one STE record is emitted, carrying a row for every factorized linear and a scope marker declaring the firing block-joint
- **AND** no per-linear invocation count is implied by the record

#### Scenario: dead_columns_are_recorded_per_linear

- **WHEN** a factorized linear carries a rank index whose outer-product column is zero (either factor's column identically zero)
- **THEN** that linear's row reports the number of such dead rank components

### Requirement: The joint STE pass SHALL weight its loss by the normalized terminal statistic and guard on the resolution floor

The joint pass SHALL weight the block-output loss by the terminal linear's Phase-1 output statistic normalized to unit mean, and SHALL test its first-step gradient norm against the floor derived from the optimizer eps and the parameter dtype's relative resolution, taking the existing vacuous/inert declaration path. Both quantities SHALL arrive resolved from the block's resolve site rather than re-derived inside the stage.

#### Scenario: joint_pass_uses_the_normalized_weight_and_the_floor

- **WHEN** the joint pass runs on a block whose terminal statistic resolved to a weight
- **THEN** the weight handed to the loss has unit mean, both pre- and post-normalization means are recorded, and the record carries the resolution-derived gradient floor

### Requirement: The joint STE pass SHALL consume the App. C factorized-stage regime once per block

The joint pass SHALL run `epochs x (rows // batch_size)` steps at the factorized stage's learning rate under the stage's cosine schedule ([NQ] App. C), consumed ONCE for the whole block. No per-linear share and no block-pooled allocation SHALL be applied on top of that regime.

#### Scenario: joint_budget_is_the_appendix_c_regime_once_per_block

- **WHEN** the joint pass runs on a block of N factorized linears
- **THEN** the recorded stage budget is `epochs x (rows // batch_size)` and every per-linear row records that same budget as its scheduled steps

### Requirement: The terminal-statistic tuning loss weight SHALL be normalized to unit mean before use

The TuneFP and TuneLatentSTE losses SHALL be weighted MSE by the terminal linear's robust-diagonal Phase-1 output statistic normalized to unit mean, preserving its relative per-feature structure while restoring its absolute scale to the residual's own scale ([NQ] App. C weighted MSE; the statistic's construction is Algorithm 1 lines 2-5, Step 2-1 Eq. 2). Both stage records SHALL carry the pre-normalization and post-normalization weight means.

#### Scenario: weight_is_normalized_before_the_loss

- **WHEN** the block's terminal statistic resolves to a weight whose mean is not 1
- **THEN** the weight handed to TuneFP and TuneLatentSTE has mean 1 while every pairwise ratio of its entries is unchanged
- **AND** both stage records carry the pre-normalization mean and the post-normalization mean

### Requirement: The vacuous-stage guard SHALL test a resolution-derived gradient floor, not exact zero

A tuning stage whose first-step gradient norm is below a floor derived from the optimizer's eps and the parameter dtype's relative resolution SHALL be recorded vacuous and SHALL take the existing fail-closed path. An identically-zero gradient SHALL remain a special case of that floor. The stage record SHALL carry the floor, the optimizer eps, and the dtype relative resolution the floor was derived from.

#### Scenario: below_floor_gradient_declared_vacuous

- **WHEN** a tuning stage's first-step gradient norm is nonzero but below the resolution-derived floor
- **THEN** the stage record reads `vacuous: true` with a reason naming the floor, and the stage takes the existing fail-closed path
- **AND** the record carries the floor, the optimizer eps, and the dtype relative resolution it used
- **AND** an identically-zero gradient reaches the same vacuous outcome as a special case of that floor

### Requirement: Tuning-stage records SHALL declare inertness, not imply it

A tuning stage that completes its full step budget with parameter movement below the parameter dtype's relative resolution SHALL record `inert: true`. A TuneLatentSTE row whose per-linear movement is below that resolution SHALL record inert on its own per-linear row. A stage record without the declaration SHALL be read as unknown, never as an effective stage.

A movement figure carries no value, never a confident `0.0`, when the stage never ran or an exactly-zero baseline leaves no denominator.

#### Scenario: full_budget_without_movement_is_declared_inert

- **WHEN** a tuning stage runs its full step budget and its parameter movement is below the parameter dtype's relative resolution
- **THEN** its stage record carries `inert: true`
- **AND** a stage whose movement exceeds that resolution carries `inert: false` rather than the key being absent

#### Scenario: movement_figure_undefined_over_all_zero_baseline

- **WHEN** a tuning stage ran over parameters whose entry baseline is exactly zero — an aggregate `weight_delta_rel` over an all-zero weight baseline, a `bias_delta_rel` over an all-zero bias baseline, or a TuneLatentSTE row figure over an all-zero entry parameter
- **THEN** the affected figure carries no value while `steps` attests the stage ran — the relative movement has no denominator, so it is left undefined rather than reported as a confident `0.0`
- **AND** the inertness declaration carries no value for the affected TuneLatentSTE row and for the stage — never a verdict aggregated over a figure that does not exist

### Requirement: The launch crumb line SHALL attribute the run to code, process, and output target

The PTQ driver's breadcrumb trail SHALL begin with a launch line
carrying the process id, the exact argv, the repository git revision
(literal `unknown` when unresolvable, e.g. running outside a
checkout), and the resolved output directory — so that a surviving
trail self-attributes its run even when the launch protocol is
violated, without recourse to kernel records or container metadata.

#### Scenario: launch identity line inside a checkout

- **WHEN** the driver starts inside a git checkout with a fake argv
  and a temp scratch dir,
- **THEN** the trail's first line contains the pid, the fake argv, the
  repo's `HEAD` revision, and the resolved output directory.

#### Scenario: launch identity line outside a checkout

- **WHEN** the driver starts with the repo directory absent (no `.git`
  reachable),
- **THEN** the trail's first line still contains pid, argv, and the
  resolved output directory, with the revision field literally
  `unknown` (never a crash, never a fabricated value).

### Requirement: Concurrent driver runs SHALL not interleave crumb trails

When `QUANTIBBLER_BREADCRUMBS` is unset, the trail file SHALL be
per-launch: `breadcrumbs-<pid>.log` inside the resolved KD scratch
directory, so two concurrent launches write two files. When the
variable is set, the configured path SHALL be honored exactly (the
single-run container contract; the wrapper's run headers and the
forensics read path are unchanged).

#### Scenario: default path disambiguates concurrent launches

- **WHEN** two driver processes run concurrently with the variable
  unset and a shared KD scratch dir,
- **THEN** each writes its own `breadcrumbs-<pid>.log` and no line
  interleaves.

#### Scenario: configured path honored exactly

- **WHEN** `QUANTIBBLER_BREADCRUMBS` points at a fixed path,
- **THEN** the driver appends only to that path (no pid suffix) and
  its launch identity line still appears first.

### Requirement: A driver run SHALL name its owner or refuse to start

Every PTQ driver run SHALL resolve a launch owner from `QKB_LAUNCH_OWNER`
(label spellings `QKB_LAUNCH_LABEL_OWNER` / `qkb_launch_owner` are the same
evidence) and SHALL refuse to start -- named error, nonzero exit, nothing
written to the trail -- when no owner is present. The launch line SHALL record
`owner=<owner>` and `card=<card|unset>` after `out`, the owning card being a
board fact that is recorded literally as `unset` when absent.

#### Scenario: driver_refuses_unattributed_launch

- **WHEN** the driver starts with none of the accepted owner spellings in its environment
- **THEN** it exits nonzero with an error naming `QKB_LAUNCH_OWNER`, and no crumb trail is written
- **AND** this is the 2026-10-25 `qk-pertarget-v4` incident's defect: an ownerless paper-faithful fire ran, and nothing in the launch path could say whose run it was

#### Scenario: label_spelled_owner_accepted

- **WHEN** only the label spelling `qkb_launch_owner` / `qkb_launch_card` is present
- **THEN** the run proceeds and its launch line records that owner and card — the incident ruling named the container labels, which a container can only receive as env

#### Scenario: owner_without_card_records_unset

- **WHEN** an owner is present and no owning card id is
- **THEN** the launch line records `card=unset` rather than omitting the field

### Requirement: Eigenvalue diagnostics SHALL be gated behind the probe flag

The per-iteration `min_eigs` eigenvalue computation SHALL run only when the solver-probe diagnostic flag is enabled. `trace.n_jitter` — the always-on record of non-SPD events from the Cholesky solve — SHALL remain the production-path SPD signal, so no safety property is lost by the gating. The gating SHALL NOT alter any solver state: the checkpoint fingerprint and the solver's arithmetic SHALL be bit-identical with the flag off.

#### Scenario: production_run_pays_no_diagnostic_tax

- **WHEN** a production run executes the ADMM loop without the probe flag
- **THEN** no per-iteration eigenvalue computation occurs, and the solver's measured wall time drops by the measured per-iteration diagnostic cost (benchmarked: 2× `eigvalsh(512×512)` ≈ 142 ms/iteration at 16 threads vs 62 ms for the 2× matmul + 2× Cholesky it guards)

#### Scenario: diagnostic_run_keeps_the_column

- **WHEN** a diagnostic run enables the probe flag (the Lane 0 harness does)
- **THEN** every probe row still carries `min_eigs`, satisfying `qk-nanoquant-alignment-lane0`'s harness requirement unchanged

#### Scenario: gating_is_behavior_neutral

- **WHEN** the same target factorizes with the flag on and off
- **THEN** post-ADMM cos, n_iter, residuals, and the checkpoint fingerprint are identical — the gating moves only wall time

#### Scenario: spd_signal_preserved

- **WHEN** a Cholesky solve jitters during a production run
- **THEN** `trace.n_jitter` still records the event, so the SPD proof carried by the solve itself is never left unrecorded

### Requirement: Torch-autograd stages SHALL run on the accelerator device

The STE, TuneFP, and TuneScalesKD stages SHALL execute on the accelerator with no algorithm change — same per-stage loss, optimizer, scheduler, and batch sizes — via activation-as-authority device resolution, never globals. TuneScalesKD's disk-backed teacher cache SHALL remain on the host.

#### Scenario: ste_tunefp_gpu_zero_algorithm_change

- **WHEN** a run selects an accelerator device
- **THEN** the STE and TuneFP stages execute on that device with the identical loss, optimizer, scheduler, and batch configuration as the CPU reference
- **AND** per-stage row-time is reported, translating the measured ~133 s/TuneFP-fire and ~154 s/STE-linear costs

#### Scenario: tunescaleskd_gpu_zero_algorithm_change

- **WHEN** a run selects an accelerator device
- **THEN** the TuneScalesKD stage executes its model forward/backward on that device with the identical KL loss, optimizer, scheduler, and batch configuration as the CPU reference
- **AND** the disk-backed teacher cache and post-stage scale readback stay on the host; only the per-step model forward/backward and paged teacher logits move to the device
- **AND** `ub`/`vb` are built on device at stage scope (hoisted above the per-step loop), not rebuilt from NumPy inside `scale_forward`; the device authority is `linear.weight.device`

#### Scenario: kd_gpu_device_parity

- **WHEN** a TuneScalesKD stage run on the accelerator is compared against the CPU reference
- **THEN** the per-step KL trace decreases monotonically on both legs
- **AND** the final scales are finite and stay within the stage's trust region on both legs
- **AND** the frozen-bits invariant holds (packed Ub/Vb unchanged; already tested by `test_scn_kl_scale_calibration`)

#### Scenario: ste_tunefp_fp32_gpu_within_tolerance_class

- **WHEN** an STE or TuneFP fp32 stage run is compared against the fp64 CPU reference on production layers
- **THEN** cosine movement stays within the campaign's ±0.002 tolerance class; a breach names the layer and fails that stage's acceptance
- **AND** TuneScalesKD (f64 scale parameters, native-dtype activations) is excluded from this class; its device equivalence is asserted through KL trace agreement (scenario `kd_gpu_device_parity`)

#### Scenario: device_authority_not_globals

- **WHEN** tensors move to the accelerator
- **THEN** movement routes through the existing activation-as-authority path (`flat.device`, `linear.weight.device`) — no torch global device state, no `torch.set_default_device`

### Requirement: The pipeline SHALL compute in bf16 by default with float32 and float64 as documented opt-in only

All solver and accumulation paths SHALL compute in bf16 unless the caller sets `compute_dtype` to `"float32"` or `"float64"`. No stage SHALL silently upcast from bf16; widening to accumulation dtype is an explicit per-site declaration.

#### Scenario: default_numerics_bf16

- **WHEN** the pipeline runs on any layer with a default `PTQConfig`
- **THEN** intermediate solver arrays are bf16, accumulation operations widen to float32 at named sites, and the artifact records `compute_dtype: bf16`

#### Scenario: float32_opt_in

- **WHEN** the caller sets `compute_dtype = "float32"`
- **THEN** solver entry points operate in float32 and accumulation equals float32

#### Scenario: float64_opt_in

- **WHEN** the caller sets `compute_dtype = "float64"` with `--solver-backend numpy`
- **THEN** the pipeline is bit-identical to the pre-change float64 path on the same inputs, and every existing NumPy-backend fixture at that dtype passes unchanged
- **WHEN** the caller sets `compute_dtype = "float64"` with the torch backend
- **THEN** the reconstruction cosine vs a same-declared-dtype NumPy fixture satisfies ≥1−1e-12 (cosine drop 1−cos ≤1e-12) — the declared float64 cell of `admm_parity_per_dtype`

### Requirement: Every pipeline stage SHALL be parameterized by the declared compute_dtype with no hardcoded numeric type

Every tensor creation, cast, and conversion in the PTQ pipeline SHALL use `compute_dtype` — never hardcoded `torch.float64`, `torch.float32`, or `np.float64`. The codebase SHALL be grep-evidenced with zero unjustified hits.

#### Scenario: grep_evidenced_no_hardcoded_dtype

- **WHEN** the implementation is complete
- **THEN** `grep -rnE '\.double\(\)|torch\.float32|torch\.float64|np\.float32|np\.float64|astype\(.*float' src/quantibbler/ptq/ tools/quantize_hf_model.py` returns zero hits that are not justified by one of three named resolution classes: **(a) parametric** — the site uses `compute_dtype` or `accum_dtype` from config; **(b) accumulation-site** — a named accumulation dtype with arithmetic justification in a comment; **(c) reference-path** — a deliberate dtype literal in the NumPy solver path (`solver_backend.py`, unit tests, reference fixtures) carrying an explicit `# reference-path: <dtype> by design` comment

#### Scenario: every_hardcoded_site_enumerated

- **WHEN** the pre-change grep runs against the full pattern set
- **THEN** every existing hardcoded-dtype hit is enumerated and each has a planned resolution class (a/b/c above) in the design.md crystallization section

### Requirement: Where bf16 mantissa width is insufficient, each accumulation site SHALL declare an explicit accumulation dtype

Every operation summing more values than bf16's 7-bit mantissa can accommodate SHALL declare a named accumulation dtype with arithmetic justification. The mapping SHALL be a pure function of compute_dtype: float32 under bf16 compute, identity at float32 and float64.

#### Scenario: accumulation_dtype_is_named_per_site

- **WHEN** any bf16 run executes
- **THEN** each accumulation site (normal equations, Cholesky, SVD, running statistics, Frobenius norms) names its accumulation dtype with a comment citing the term-count bound

#### Scenario: accumulation_identity_at_higher_dtypes

- **WHEN** compute_dtype is float32
- **THEN** the accumulation dtype equals compute_dtype
- **WHEN** compute_dtype is float64 with `--solver-backend numpy`
- **THEN** the accumulation dtype equals compute_dtype, and the pipeline is bit-identical to the pre-change float64 path on the same inputs
- **WHEN** compute_dtype is float64 with the torch backend
- **THEN** the accumulation dtype equals compute_dtype, and the first-update residual relative difference vs the NumPy parity fixture satisfies ≤1e-12 (per `admm_parity_per_dtype`)

### Requirement: The LB-ADMM and SVD solvers SHALL execute on the accelerator device when present

The torch solver backend SHALL execute on the module's device. On an accelerator, the solve SHALL use torch device tensors throughout with no host round-trip. The NumPy backend is a unit-test-only reference path at the declared dtype.

#### Scenario: solver_on_device

- **WHEN** a model's parameters are on an accelerator
- **THEN** the ADMM solve executes on that device, and the two per-layer host↔device crossings of the v6 pipeline are eliminated

#### Scenario: solver_cpu_explicit_only

- **WHEN** `--device cpu` is set
- **THEN** the torch solver backend executes on CPU tensors — no silent fallback from a failed accelerator probe

#### Scenario: numpy_backend_unit_test_only

- **WHEN** `--solver-backend numpy` is set
- **THEN** the NumPy backend path executes in a unit test only; the fire path never resolves to NumPy

### Requirement: The pipeline SHALL place every stage on the declared device without host round-trips

Every stage SHALL derive its device from the model's parameters and SHALL keep all tensors on that device: capture, TuneFP, solver, STE, TuneScalesKD, and eval. The host-side stages are sign/extract/pack (latent-to-bit conversion boundary) and qpk1 export (disk write).

#### Scenario: per_stage_device_from_module

- **WHEN** any pipeline stage executes
- **THEN** the device is read from the owning module's parameters — not from a global or environment variable

#### Scenario: device_auditor_catches_mismatch

- **WHEN** `--device-audit` is set and a module's `__call__` receives tensors on a different device
- **THEN** a named error identifies the module, expected device, and actual device

#### Scenario: per_stage_timing_recorded

- **WHEN** any pipeline fire completes
- **THEN** the report records elapsed-wall seconds per stage (capture, TuneFP, solver, STE, sign/extract/pack, TuneScalesKD, eval, export) as a per-layer JSONL row, appended as each stage produces its timing — enabling comparison against the ~8h v6-class baseline

### Requirement: The CPU path SHALL size thread pools to available cores with load-aware backoff

At startup, the pipeline SHALL read CPU affinity and 1-minute loadavg, compute a thread count, set `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` before any BLAS import, and apply the same count to torch's own pool via `torch.set_num_threads` after import (torch does not read `TORCH_NUM_THREADS`). A `--max-threads N` cap and `--no-good-guest` opt-out SHALL exist.

#### Scenario: thread_sizing_light_load

- **WHEN** the 1-minute loadavg is below half the available cores
- **THEN** the thread count is `max(2, n_cores - 2)`

#### Scenario: thread_sizing_heavy_load

- **WHEN** the 1-minute loadavg exceeds the available core count
- **THEN** the thread count is `max(1, n_cores // 4)`

#### Scenario: thread_sizing_moderate_load

- **WHEN** the 1-minute loadavg is at or above half the available cores but does not exceed the core count
- **THEN** the thread count is `max(2, n_cores - 4)`

#### Scenario: thread_sizing_small_affinity

- **WHEN** the affinity set contains fewer than 4 cores (e.g. a container with `--cpus 2`)
- **THEN** the light-load formula produces `max(2, n_cores - 2) = 2` and the heavy-load formula produces `max(1, n_cores // 4) = 1` — a computed value of 1 is acceptable when only 2 cores are visible

#### Scenario: max_threads_cap

- **WHEN** `--max-threads 8` is set
- **THEN** the thread count is `min(computed, 8)`

#### Scenario: no_good_guest_disables

- **WHEN** `--no-good-guest` is set
- **THEN** no thread-pool environment variables are set, and the GPU memory cap is skipped

### Requirement: GPU memory allocation SHALL be capped at a default fraction with a user override

When an accelerator is present, the pipeline SHALL call `torch.cuda.set_per_process_memory_fraction(0.85)` at startup. A `--gpu-memory-fraction F` flag SHALL override the cap.

#### Scenario: gpu_memory_cap_applied

- **WHEN** the pipeline starts on a GPU with default settings
- **THEN** at most 85% of GPU memory is available to this process

#### Scenario: gpu_memory_fraction_override

- **WHEN** `--gpu-memory-fraction 0.5` is set
- **THEN** the cap is 0.5

### Requirement: The artifact remainder dtype SHALL default to the pipeline compute dtype without silent upcast

The `remainder_dtype` for export SHALL default to the mapping of compute_dtype (`bf16`→`"bf16"`, `float32`→`"fp32"`, `float64`→`"fp32"`). The `--remainder-dtype` flag SHALL override. The export path SHALL NOT silently upcast to fp64.

#### Scenario: remainder_dtype_defaults_to_compute_dtype

- **WHEN** a bf16 compute run exports without `--remainder-dtype`
- **THEN** remainder sidecars are written in bf16 and the manifest declares `dtype = "bf16"`

#### Scenario: remainder_dtype_override_independent

- **WHEN** a bf16 compute run exports with `--remainder-dtype fp32`
- **THEN** remainder sidecars are fp32, compute and storage dtypes are separately declared in the manifest

#### Scenario: no_fp64_intermediate_in_export

- **WHEN** the export path writes remainder sidecars
- **THEN** no `.to(torch.float64)` or `.astype(np.float64)` intermediate is applied

### Requirement: FactorizedLinear SHALL store all parameters at the declared compute_dtype

`FactorizedLinear.__init__` SHALL store latents, scales, and bias as `nn.Parameter` at `compute_dtype`. The STE passthrough SHALL be algebraically identical at any float dtype.

#### Scenario: latents_at_compute_dtype

- **WHEN** a `FactorizedLinear` is constructed with `compute_dtype=torch.bfloat16`
- **THEN** all five parameter tensors (u_latent, v_latent, s1, s2, bias) have `dtype == torch.bfloat16`

#### Scenario: from_weight_casts_to_compute_dtype

- **WHEN** `FactorizedLinear.from_weight` is called with a bf16 weight tensor
- **THEN** the returned module's latents and scales are bf16 — the SVD init path casts to compute_dtype before storing

### Requirement: The bias path SHALL carry bias tensors at the declared compute_dtype through every stage

Bias tensors SHALL be stored at `compute_dtype`, carried through solver and STE at `compute_dtype`, and exported at the declared remainder dtype. No stage SHALL zero or drop the bias a module declares.

#### Scenario: bias_preserved_through_pipeline

- **WHEN** a quantized layer has a non-zero fp32 bias from the teacher
- **THEN** the exported bias is non-zero at the declared remainder dtype — never zeroed by a silent dtype conversion or skipped write-back

#### Scenario: bias_zero_assertion_gate

- **WHEN** the e2e gate quantizes a tiny model in which at least one quantized linear declares a bias (`linear.bias is not None` — attention_bias or MLP bias present) with a non-zero source value
- **THEN** every such linear's exported bias is non-zero, and an artifact whose declared-bias linears all carry exactly-zero (or absent) bias fails the gate even if the PPL ratio and every other gate condition pass

#### Scenario: bias_zero_assertion_gate_not_fired_on_legit_zero_biases

- **WHEN** the quantized model's architecture declares no tunable bias on any quantized linear (`bias=None` everywhere — the Qwen3-0.6B class with `attention_bias=false` and no MLP bias), or declares biases whose source values are all exactly zero
- **THEN** the gate performs no bias-non-zero assertion, an artifact whose bias fields are all exactly zero passes the gate, and the bias-free outcome is a present, exactly-zero field rather than an omitted one

#### Scenario: bias_dtype_tracks_compute_dtype

- **WHEN** a FactorizedLinear is constructed with bf16 compute_dtype
- **THEN** `bias.dtype == torch.bfloat16`

### Requirement: The pipeline SHALL gate each reconstructed layer against the fp weight's magnitude before any refinement or downstream use

Immediately after magnitude balancing yields `(Ub, Vb, s1, s2)` for a linear layer, the pipeline SHALL verify the binary reconstruction's Frobenius norm against the fp weight's within the band factor `PTQConfig.divergence_band_factor` (default 10000.0, a catastrophe bound calibrated to the measured honest range of balance-time ratios), and SHALL require every scale entry to be finite. A violation SHALL abort quantization with a dedicated error naming the layer, the measured ratio, and the band.

#### Scenario: divergence_gate_healthy_passes

- **WHEN** magnitude balancing produces scales whose binary reconstruction's Frobenius norm is within the configured factor of the fp weight's Frobenius norm
- **THEN** the gate passes without raising, and the pipeline proceeds to refinement with the balanced factors

#### Scenario: divergence_gate_diverged_aborts

- **WHEN** a layer's balanced scales reconstruct a binary weight whose Frobenius norm exceeds the fp weight's Frobenius norm by more than the band factor
- **THEN** quantization aborts with the layer's module path, the measured ratio, and the configured band in the error message, raised before Straight-Through refinement and before any later block's calibration inputs are captured, and no qpk1 bytes are exported for the model

#### Scenario: divergence_gate_nan_rejected

- **WHEN** any scale entry produced by balancing is NaN or infinite
- **THEN** the gate rejects the layer with the same abort contract as a magnitude violation

### Requirement: The latent refinement stage SHALL make sign adaptation effective or declare itself scale-only in the run record

The straight-through refinement stage ([NQ] Sec 3.2 Step 3) SHALL instrument
per-step sign-flip counts for both factors of every factorized linear in the
block — the seven scoped linears (attn q/k/v/o, mlp gate/up/down) — and SHALL
refine the paper's joint parameter set (U, V, s1, s2) per [NQ] Eq. 10, never
finishing a run as a silent sign no-op. Per the 2026-10-21 user directive, the
paper's own STE mechanism is attempted first; adopted deviations are named in
the run record.

#### Scenario: flip_counts_are_recorded

- **WHEN** the STE stage runs its full per-linear budget on a binary-factor
  block
- **THEN** the run artifact records per-linear, per-factor sign-flip counts,
  the latent relative movement, the scale (s1, s2) movement, and the
  scheduled step count for every factorized linear in the block — the seven
  scoped linears — so neither a zero-flip U/V nor a frozen scale can pass
  unnoticed

#### Scenario: sign_adaptation_is_effective_or_declared

- **WHEN** the stage completes with zero accepted flips on a scoped linear
  on the non-faithful (diagnostic) path
- **THEN** either the run record declares the stage scale-only for that
  linear with the re-scoping marker — the literal
  `ste_scale_only_declared` (`STE_RESCOPE_MARKER`,
  `src/quantibbler/ptq/block_recon.py`) — or the gate for this change
  fails
- **AND** the paper-faithful path is exempt from the re-scope branch: it
  records the measured zero crossings and completes as the paper's
  mechanism — [NQ] prescribes sign+pack on whatever signs STE produced
  (Algorithm 1 lines 20-23), never declaring a re-scope the paper does not
  have

#### Scenario: flip_acceptance_uses_block_reconstruction_objective

- **WHEN** a flip pass proposes sign changes to a factor entry of a
  factorized linear
- **THEN** each flip is accepted only if the block-reconstruction objective
  `‖block_tuned(x_b) − y_fp‖²` decreases, where `block_tuned(x_b)` is the
  block's forward with the proposed flip applied to the linear under
  refinement and every other block parameter held at its current value —
  so the objective typechecks for every factorized linear, including GQA
  k/v projections whose layer outputs are not d_model-shaped, and never
  scores a layer-local residual against the block-level target
- **AND** acceptance uses a deterministic sweep order with a recorded
  acceptance count per linear
- **AND** `(x_b, y_fp)` is the same corpus-pinned cached capture that feeds
  TuneFP ([NQ] Algorithm 1 lines 9–10, 19): `x_b` is the compressed-prefix
  stream and `y_fp` is the pristine FP block's output on it, captured once
  per block before any tuning, never recomputed from tuned or quantized
  weights
- **AND** the artifact records the same teacher provenance (source
  fingerprint + capture index) as the TuneFP stage

#### Scenario: refinement_covers_every_factorized_linear_with_paper_parameter_set

- **WHEN** the STE stage runs on a block
- **THEN** refinement covers every factorized linear in the block — the
  seven scoped linears — scheduled per-layer on the just-factorized linear
  per [NQ] Figure 3's right panel, never a partial set that refines only
  the O/G/U/D exemplar path
- **AND** each linear's refinement consumes the FULL per-linear budget —
  `epochs × (n // batch)` steps per [NQ] App. C (corpus line 3007: the
  factorized tuning stage consists of 8 epochs at batch 1, so 8 × 128 =
  1024 steps per linear at the protocol's calibration) — never a
  block-pooled split across linears, and the artifact records each
  linear's scheduled steps
- **AND** the refinement parameter set is the paper's joint set (U, V, s1,
  s2) per [NQ] Eq. 10 and the Figure 3 caption (corpus line 577:
  "fine-tunes the continuous latent matrices and scales"): flip counts
  instrument the U/V sign crossings and scale movement is recorded per
  linear alongside them — omitting scales from the parameter set is a named
  deviation, never silent

### Requirement: The paper-faithful path SHALL execute the paper's Stage 2 to 3 mechanism without extra-paper discrete stages

On the paper-faithful path the latent factorization SHALL initialize per
[NQ] Stage 2 — LB-ADMM, Eq. 7/8 balancing, Eq. 9 latents — with the greedy
bit-flip polish disabled, and the Eq. 10 STE refinement SHALL run on those
continuous latents and scales for the full per-linear budget before
sign+pack (Algorithm 1 lines 20-23).

#### Scenario: paper_faithful_disables_greedy_polish

- **WHEN** a run configures paper_faithful
- **THEN** the factorization event records polish_rounds 0 and the STE
  stage receives the continuous balanced latents (Eq. 9 initializers),
  never polish-binarized ±1 bits
- **AND** the production path (paper_faithful unset) keeps the greedy
  polish unchanged — this gate touches the faithful path only

#### Scenario: paper_faithful_uses_the_paper_solver

- **WHEN** a run configures paper_faithful
- **THEN** the factorization solver is the paper's LB-ADMM with the App. C
  linear penalty schedule over 400 factorization steps per weight matrix
  (the pipeline's `lb_admm`, default `max_admm_iter=400`)
- **AND** the SVD warm start — the measured production ablation — cannot
  carry the faithful label (the fidelity verdict flips the label on
  solver != admm)

### Requirement: TuneFP SHALL tune the still-FP remainder against the cached teacher at the pertarget-cycle placement

Per [NQ] Sec 3.2 Step 1, Algorithm 1, Figure 3: the stage SHALL target
`Y_b = B^FP_b(X_b)` — the pristine FP block's output, captured once
pre-tuning — and SHALL fire per-target within the factorization loop
(N-1 times per block for N quantizable linears), tuning the still-FP
remainder after earlier in-block layers are factorized. The TuneFP
internal loss formula is the informed assumption recorded in design.md.

#### Scenario: teacher_is_captured_before_tuning

- **WHEN** block reconstruction tunes fp weights for T_pre steps
- **THEN** the teacher tensor is the pristine block's output on the
  compressed-prefix stream, captured once before any tuning or in-block
  factorization, detached, with gradients disabled, and never recomputed
  from the tuned weights
- **AND** the teacher block's weights are the pristine FP weights loaded
  from the pinned model safetensors (or a frozen reference copy), never the
  live module's mutable state
- **AND** the artifact records the teacher source fingerprint (hash of the
  pristine weight source) and the capture index (which block the teacher
  belongs to), so provenance is provable from the artifact alone
- **AND** the same capture feeds TuneLatentSTE per [NQ] Algorithm 1 line 19,
  so Steps 1 and 3 share one teacher

#### Scenario: tunefp_fires_per_target_within_factorization_loop

- **WHEN** block reconstruction reaches a block's factorization loop
- **THEN** TuneFP fires N-1 times (for N quantizable linears), each fire
  tuning the still-FP remainder before the target is factorized — matching
  the pertarget-cycle ruling (a) where `idx >= 1` gates the TuneFP call
- **AND** the first target (`idx == 0`) is factorized with NO preceding
  TuneFP — a pass before it has nothing factorized in-block, so the
  student block IS the pristine teacher block on the same input and the
  objective is identically zero
- **AND** the teacher target `Y_b` is captured from the fp-stream reading
  (per lane 2), and the same capture feeds TuneLatentSTE per [NQ]
  Algorithm 1 line 19, so Steps 1 and 3 share one teacher

#### Scenario: vacuous_teacher_fails_closed

- **WHEN** the pre-tune objective gradient norm is zero at the first optimizer
  step — whether at the operative post-Q/K/V call or at any call emitted
  with no in-block layer factorized (the legacy pre-loop placement, which
  the operative placement SHALL never produce)
- **THEN** the stage aborts with recorder event `tune_fp_vacuous_teacher`
  naming the block, and the event carries BOTH the first-step gradient norm
  and the initial residual norm — gradient-zero with residual-zero is a
  broken teacher (target equals current output), gradient-zero with
  residual-nonzero is a detached student (no gradient path)
- **AND** in a diagnostic run the pipeline continues with pristine weights
  and the event is visible in the run record
- **AND** in a run that claims paper fidelity, any `tune_fp_vacuous_teacher`
  event invalidates the claim: the run SHALL fail or record
  `paper_faithful: false`, never silently continue

#### Scenario: tune_fp_effect_is_recorded_honestly

- **WHEN** a TuneFP stage runs to completion
- **THEN** the artifact records teacher provenance naming the source
  (pristine safetensors or the frozen reference copy, with its fingerprint),
  the informed-assumption marker, and the fp weight/output delta versus
  pristine, and the delta is nonzero on at least one scoped linear — the
  operative call always runs with factorized attention projections, so a
  real effect is required whenever in-block factorization error is nonzero

#### Scenario: teacher_target_is_persisted_and_recovered_on_resume

- **WHEN** a TuneFP run checkpoint includes the fp-stream teacher target
- **THEN** the target is persisted as the reserved checkpoint npz key `y_b`
  (per `checkpoints.py:_YB_KEY`) and recovered on resume via
  `load_block_yb` to reconstruct the fp-stream prefix exactly
- **AND** the persistence is fingerprint-bound to the checkpoint's
  config+code identity (the existing `config_fingerprint` mechanism that
  gates resume) so a config/code change refuses a stale checkpoint
- **AND** a legacy checkpoint predating lane 2 carries no `y_b` key, and
  the resume path falls back to recomputing the fp stream through the
  pristine copy — never silently starting from a wrong activation

### Requirement: The paper_faithful label SHALL be a computed verdict, never a forwarded flag

The reported `paper_faithful` field SHALL be computed from the run's own
record — the flag AND zero runtime deviations AND zero mechanism-class
deviations AND evidence that every required stage actually executed — so a
run can never carry the label while its protocol deviates, and a label
backed by an empty or incomplete stage record (which certifies nothing)
SHALL be false.

#### Scenario: label_flips_on_runtime_deviations

- **WHEN** the run identity records any runtime deviation (split tune
  geometry, non-protocol calibration row count, non-finite eval geometry,
  or a non-protocol eval sequence length — `non_protocol_eval_geometry`,
  eval length != the protocol's 2048)
- **THEN** the reported label is false regardless of the flag
- **AND** the report records the deviation list alongside the label

#### Scenario: label_flips_on_mechanism_deviations

- **WHEN** a paper-faithful-flagged run's record shows any of: lb_admm
  polish_rounds greater than 0, a STE discrete flip pass fired, a STE
  scale-only re-scope marker, a vacuous-teacher event, a stage failure, a
  solver other than admm, or a required stage (per the stage inventory's
  `required` column) that never executed
- **THEN** the reported label is false
- **AND** each condition is evaluated and recorded, so the label is
  auditable from the artifact alone

#### Scenario: required_stage_execution_reads_the_same_record

- **WHEN** the verdict evaluates required-stage execution
- **THEN** it derives run-level stage counts from the same normalized
  stage record every other condition reads, so an empty stage record
  denies the label on every required stage
- **AND** a stage event carrying a skip marker does not count as
  executed (`tune_fp` with `skipped` set, `tune_scales_kd` with
  `artifact` falsy) — executed means fired, not merely recorded
- **AND** the required set is materialized as a column of the stage
  inventory (`_STAGE_LINES`), never a second list beside the check, so
  the stage map and the verdict cannot drift
- **AND** required-stage conditions are recorded on every report path,
  naming the missing stage, but no non-requested run is denied by them —
  the flag condition already denies non-requested runs

#### Scenario: informed_assumptions_do_not_flip_the_label

- **WHEN** the record carries informed-assumption markers (the TuneFP loss
  formula)
- **THEN** the label is unaffected — those markers are documentation, not
  deviations

### Requirement: Phase-1 statistics SHALL be collected from the pristine model on the raw calibration set

RobustDiag statistics SHALL be collected once, in a Phase-1 pass over the
raw calibration set through the pristine FP model ([NQ] Algorithm 1 lines
2-5), before any block reconstruction; the block loop SHALL consume those
pre-collected per-linear statistics, and the per-block capture no longer
doubles as the statistics source.

#### Scenario: phase1_stats_from_pristine_model

- **WHEN** a run begins
- **THEN** one pre-loop pass runs the calibration rows through the
  pristine FP model, collecting per-linear statistics for every scoped
  linear (chunked/streamed collection at real scale, state O(feature
  width))
- **AND** block reconstruction consumes that pre-collected book — the
  compressed-prefix stream remains the tuning input ([NQ] line 9), never
  the statistics source

#### Scenario: stage_map_reflects_the_hoist

- **WHEN** the run records its stage map
- **THEN** the phase1_statistics entry carries no compressed-prefix-source
  deviation note
- **AND** the capture entry names only the line-9/10 teacher capture

### Requirement: Evaluation evidence SHALL state coverage and table comparability separately from method fidelity

The local PPL quality gate SHALL record its exact WT2 coverage and evaluator
semantics. Because [NQ] names WT2 perplexity and `lm_eval=0.4.9` but the pinned
corpus does not specify exact coverage/windowing, the local fixed-window metric
SHALL NOT claim paper-table comparability or redefine the method-fidelity
verdict.

#### Scenario: evaluation_coverage_is_artifact_provable

- **WHEN** the WT2 test shard is tokenized for baseline and quantized PPL
- **THEN** every report path records evaluator kind, source token count,
  complete blocks available, remainder tokens, selected blocks/tokens,
  scored predictions, complete-block and source-token coverage, selection
  seed, and a fingerprint of the ordered selected block indices
- **AND** the report records `paper_table_comparable: false` for the local
  fixed-window evaluator

#### Scenario: faithful_fire_uses_maximal_local_evaluation_window

- **WHEN** a paper-faithful-flagged run uses the local fixed-window evaluator
- **THEN** the requested eval batches equal the largest complete BATCH-sized
  group derivable from the tokenized test shard; smaller coverage fails
  preflight before the expensive solve
- **AND** for the pinned Qwen3-0.6B tokenizer/test shard measurement this is
  EVAL_SEQS=18: 144 of 145 complete 2,048-token blocks, with the 1,978-token
  remainder and one complete block explicitly recorded as uncovered

#### Scenario: evaluation_limit_does_not_masquerade_as_method_deviation

- **WHEN** the local evaluator uses its maximal complete-batch window
- **THEN** `paper_faithful` remains the computed verdict over quantization
  method and calibration protocol conditions
- **AND** evaluation comparability remains separately false, so neither field
  can be read as the other

### Requirement: A resumed run's stage record SHALL be recovered from the checkpoint directory that records it

Post-fire row: the stage journal SHALL be written as ``stage-events.jsonl``
beside the solve checkpoints, one JSONL line per stage event appended as it
fires, and SHALL be read back on resume only behind the fingerprint guard and
only for a directory with a completed block; every re-read line SHALL be
validated against the known payload shapes and finiteness — a name-only or
saturated record SHALL never reach the payload-blind required-stage verdict
(sec-eng t_5855d059).

#### Scenario: resume_journal_is_fingerprint_bound_and_never_synthesizes

- **WHEN** a launch resumes from a checkpoint directory
- **THEN** the journal is read only after ``check_fingerprint`` passes
  (config, code version, and model identity — the live guard), so a
  stale-config resume raises before any line is parsed
- **AND** a directory with no completed block inherits nothing (the
  launch re-runs every stage — fresh-start semantics)
- **AND** a missing journal yields the pre-journal record (empty list,
  never a synthesized event that would falsely attest an unexecuted stage)
- **AND** a fresh-start reset — ``reset_stage_log`` called where the
  fingerprint is written — drops the journal of the fingerprint it
  replaces, so a later resume cannot inherit another config's events

#### Scenario: resume_journal_line_is_shape_checked_and_finiteness_gated

- **WHEN** a journal line is re-read
- **THEN** the line must be a JSON object whose ``event`` names a stage the
  recorder can emit (registered in ``_STAGE_PAYLOAD_KEYS``) and whose
  ``info`` carries the exact keys that event always sends — a name-only
  line, a truncated payload, or an unregistered event name raises rather
  than counting as an execution
- **AND** the writer sanitizes every value through ``json_safe`` before
  writing and refuses non-finite tokens (``allow_nan=False``), so a
  saturated measurement is recorded as strict-JSON ``null`` and a re-read
  saturated line raises — the monotone-leverage corruption class must
  deny, never bless the payload-blind verdict
- **AND** a reader that detects a non-finite value after JSON parse also
  raises, so both the write-side and read-side gates cover the saturation
  path
- **AND** the shape check is not a signature — a line whose keys match the
  expected set but whose values are forged does not raise on shape; what
  keeps a journal honest is the fingerprint of the directory it sits in

#### Scenario: resume_record_matches_the_uninterrupted_record

- **WHEN** a run crashes in the pipeline tail (post block loop, pre
  completion) and resumes from its checkpoints
- **THEN** the resumed run's fidelity condition record equals the
  uninterrupted run's — every required stage's condition reads the same
  value — so the label (their conjunction) means the same thing on a
  resumed artifact as on a clean run's
- **AND** Phase-1 statistics re-fires on every launch and is not
  checkpointed, so the journal carries only the block-loop stages; the
  resumed launch's own Phase-1 line and the journaled block-loop lines
  together produce the equivalence

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

### Requirement: Block reconstruction SHALL advance dual activation streams with an fp-stream teacher capture

Block reconstruction SHALL advance TWO streams: an fp stream through pristine blocks that produces the teacher target `Y_b = B^FP_b(X_fp)`, and a compressed stream through quantized blocks that produces the student input `X_b` ([NQ] Algorithm 1 :730); the student sees `X_b`. Feeding the teacher the fp-stream activation deviates from :735's literal `Y_b ← B^FP_b(X_b)`; design.md records the deviation and its rationale ([NQ] :525, :577).

#### Scenario: dual_stream_advances_in_lockstep

- **WHEN** block reconstruction advances from block b to block b+1
- **THEN** the fp stream advances through the pristine frozen-reference block and the compressed stream advances through the quantized block, in lockstep
- **AND** the student consumes the compressed stream: student sees `X_b`

#### Scenario: teacher_target_reads_fp_stream

- **WHEN** the block-b teacher capture runs
- **THEN** `Y_b = B^FP_b(X_fp)` — the pristine block applied to the fp-stream activation
- **AND** `X_b` is `M̂_<b(X_cal)` per [NQ] Algorithm 1 line 9 (:730), and `Y_b` is not a function of the compressed stream

#### Scenario: stream_separation_fixture

- **WHEN** a fixture corrupts the compressed stream before the block-b capture while the fp stream is intact
- **THEN** fixtures prove stream separation: corrupt the compressed stream, assert `Y_b` unchanged
- **AND** the assertion holds for every block after the first, through both direct `reconstruct_block` calls and the `reconstruct_model` driver loop

#### Scenario: target1_tunefp_vacuity_gone_by_construction

- **WHEN** TuneFP fires for a block's first target linear and at least one prefix block is quantized
- **THEN** the target-1 TuneFP objective would be nonzero by construction — the streams have diverged — so under dual-stream the single-stream target-1 vacuity disappears
- **AND** the universal idx==0 skip (landed `if idx >= 1:` at block_recon.py:1595, per qk-pertarget-cycle ruling (a)) still applies: at `idx==0` no in-block layer is factorized, so TuneFP has nothing to compensate — the skip is universal across ALL blocks, not limited to the block-1 identity prefix
- **AND** the skip's justification is "nothing factorized in-block yet," not "teacher equals frozen deepcopy" (the stale single-stream rationale that must be rewritten — planner reconcile routes to coder)

#### Scenario: suite_green

- **WHEN** the lane-2 change runs the full gate suite
- **THEN** the suite is green: `pytest tests/ptq/ tests/ci/ tests/qk_report/`, `cargo test`, `cargo clippy --all-targets -D warnings`, `cargo fmt --check`, and `maturin build` all pass

### Requirement: The tune_fp_vacuous_teacher guard SHALL remain a plumbing detector with a re-pinned rationale

The guard SHALL stay a detector, behavior unchanged: zero first-step gradient with zero initial residual still means broken plumbing (stream cross-talk, detached student, capture reading live or compressed state), recorded as `tune_fp_vacuous_teacher` and fail-closed under `paper_faithful`. The rationale SHALL be rewritten: under dual-stream every fired TuneFP call has a nonzero objective by construction, so a zero event is a construction defect, never nothing-to-compensate.

#### Scenario: guard_zero_grad_zero_residual_means_broken_plumbing

- **WHEN** a fired TuneFP call records a zero first-step gradient and a zero initial residual
- **THEN** the `tune_fp_vacuous_teacher` event fires carrying both norms, because zero-grad + zero-residual still means broken plumbing
- **AND** the event carries the stream provenance of the capture that fed the call

#### Scenario: guard_rationale_rewritten

- **WHEN** the guard's documentation is read (docstrings, the `paper_faithful` config comment, design.md)
- **THEN** the rationale names dual-stream separation as the reason a zero objective is a defect, and no text claims target-1 vacuity is inherent to the placement
- **AND** the rationale records that a genuinely zero prefix divergence still trips the guard — fail-closed, never silent

#### Scenario: paper_faithful_fails_closed_on_vacuous_event

- **WHEN** a run claiming paper fidelity records any `tune_fp_vacuous_teacher` event
- **THEN** the claim is invalidated exactly as before: the run fails or records `paper_faithful: false`, never silently continues

### Requirement: The paper_faithful verdict SHALL be re-pinned to fp-stream teacher provenance

The computed `paper_faithful` verdict SHALL be re-pinned to dual-stream capture semantics: fp-stream teacher provenance becomes a label requirement, so a capture recorded with compressed-stream provenance (the legacy single-stream capture) or with missing provenance is a mechanism deviation that denies the label. Every existing verdict condition — the flag AND no runtime deviations AND no mechanism deviations AND required stages executed — is unchanged.

#### Scenario: label_requires_fp_stream_teacher_provenance

- **WHEN** the run record's teacher captures carry stream provenance
- **THEN** every capture names its stream, and a paper-faithful label requires all captures to be fp-stream: a compressed-stream capture or an unprovenanced capture sets the label false
- **AND** the recorded deviation list names the offending capture

#### Scenario: dual_stream_deviation_marker_is_documented_not_label_flipping

- **WHEN** the record carries the dual-stream deviation marker (fp-stream teacher input versus [NQ] :735's literal compressed-prefix feed)
- **THEN** the marker is documentation recorded in design.md and the run record, and the re-pinned verdict treats fp-stream provenance as the faithful form, so the marker does not flip the label

### Requirement: Recorder events SHALL carry stream provenance

Capture, `tune_fp`, `ste`, and `tune_fp_vacuous_teacher` recorder events SHALL record which stream produced each tensor they reference — the teacher target's stream (fp-stream) and the student input's stream (compressed-stream) — alongside the existing teacher fingerprint and capture index, so provenance is provable from the artifact alone.

#### Scenario: recorder_events_gain_stream_provenance

- **WHEN** any reconstruction recorder event references `x_b` or `y_b`
- **THEN** recorder events gain stream provenance: each event names the stream that produced each referenced tensor
- **AND** the capture event keeps the teacher fingerprint and capture index fields, so the artifact proves both the teacher source and the stream separation

### Requirement: Prefix-probe and checkpoint fingerprints SHALL be unchanged in shape

The dual-stream capture SHALL introduce no new solver state: the progress probe keeps its signature `probe(model, depth, name)` and per-block call site, and the checkpoint fingerprint keeps its shape — sha256 over `asdict(PTQConfig)` plus code identity with `progress_probe` the only exclusion — with no new joined config field.

#### Scenario: prefix_probe_and_checkpoint_fingerprints_unchanged_in_shape

- **WHEN** a dual-stream run exercises the prefix probe and checkpoint resume paths
- **THEN** prefix-probe and checkpoint fingerprints are unchanged in shape: the probe still fires as `probe(model, depth, name)` after each block write-back, and the fingerprint payload keys and the joined PTQConfig field set are exactly as before the lane
- **AND** resume semantics are unchanged: a mismatched config or code state is refused, and the lane boundary changes the code identity, so a pre-lane checkpoint is refused resume exactly like any other stale-state case

### Requirement: The continuous update SHALL regress against row/col-normalized regressors with a scale-relative ridge

The LB-ADMM continuous update SHALL build its design matrix from the row/col-normalized regressor X (the fixed partner factor of Eq. 5, [NQ] :2877) and SHALL replace the absolute ridge shift `(ρ+λ)I` of Eq. 37 ([NQ] App. B.4 :2916) with the scale-relative stabilizer `ρ·mean(diag(XᵀX)) + reg`, where `reg` is a config field (default 3e-2). The `QK_RHO_END` env-provenance pattern SHALL extend to the new fields, and the checkpoint fingerprint SHALL pick up every new solver-state field.

#### Scenario: normalized_regressor_design_matrix

- **WHEN** the continuous update solves for U given the fixed partner factor (symmetrically for V)
- **THEN** the design matrix is built from the row/col-normalized proxy (the fixed partner factor serving as the regressor of Eq. 5, [NQ] :2877), so the Gram matrix XᵀX is O(1)-scale with unit-scale diagonal
- **AND** the returned iterate is mapped back to the unnormalized factor basis, so the product UVᵀ remains a valid factorization of the target

#### Scenario: stabilizer_scale_relative_ridge

- **WHEN** the ridge shift is computed at schedule value ρ
- **THEN** the shift is exactly `stabilizer = ρ·mean(diag(XᵀX)) + reg` with `reg` read from a config field whose default is 3e-2
- **AND** the stabilizer depends only on the normalized regressor, never on raw layer scale, so one ρ schedule serves every layer

#### Scenario: reg_env_override_explicit_fail_closed

- **WHEN** `QK_ADMM_REG` is set in the environment
- **THEN** the effective `reg` is the env value with source `"env"`, recorded beside the value in the solver probe rows and the run report exactly like `rho_end_source` (the `rho_end_provenance` pattern, `lb_admm.py:325-341`)
- **AND** an unparseable `QK_ADMM_REG` raises an error naming the variable, never falling back to the default silently

#### Scenario: new_solver_state_joins_fingerprint

- **WHEN** a checkpoint is resumed
- **THEN** two configs differing only in a new solver-state field (`admm_reg`, recalibrated ρ endpoints) produce different checkpoint fingerprints
- **AND** the new fields are absent from `_FINGERPRINT_EXCLUDED`, asserted by a suite test so a future exclusion-list edit cannot silently drop solver state

### Requirement: The ADMM penalty SHALL follow the paper's linear schedule over K=400 with endpoints re-calibrated for the normalized regime

ρ SHALL follow the paper's linear schedule across K=400 factorization steps ([NQ] App. C :3008) with endpoints re-calibrated for the normalized-regressor regime: linear ~0→1. With the scale-relative ridge, a final ρ of ~1 no longer creates the consensus dominance qk-fix-admm-solver capped at `1e-1`; that cap's rationale is scoped to the absolute-ridge regime this change replaces. The tension between the two sibling deltas is named in this change's proposal.

#### Scenario: rho_endpoints_recalibrated_normalized_regime

- **WHEN** LB-ADMM runs with default ADMM parameters in the new regime
- **THEN** ρ follows the paper's linear schedule over K=400 ([NQ] App. C :3008) with endpoints ~0→1
- **AND** at schedule end the ridge `ρ·mean(diag(XᵀX)) + reg` sits at the data term's own scale, so the continuous fit is not crushed (the defect qk-fix-admm-solver measured and capped)
- **AND** well-posedness holds at ρ≈0: the `reg` floor keeps the continuous subproblem SPD per [NQ] App. B.4 Lemma 2

### Requirement: The entry Frobenius normalization SHALL be kept or removed deliberately under the new regime

The entry Frobenius normalization (target divided by its Frobenius norm, `lb_admm.py:376-380`) may become redundant once the regressor is normalized and the ridge is scale-relative. Its keep-or-remove decision SHALL be made from with/without measurements, deliberately, and the decision with its rationale SHALL be recorded in this change's design.md.

#### Scenario: frobenius_normalization_keep_or_remove_recorded

- **WHEN** the new regime is measured with and without the entry Frobenius normalization
- **THEN** the keep-or-remove decision is deliberate and recorded in this change's design.md with the measurements and rationale
- **AND** the default path SHALL NOT silently drop the normalization before that decision is recorded

### Requirement: The ADMM regime SHALL be accepted only on Lane-3 measurements against the Lane-0 baseline

The new regime SHALL be accepted only on Lane-3 measurements against the Lane-0 baseline. Acceptance requires: on production layers, post-ADMM pre-polish cos improves materially over the Lane-0 baseline (threshold set FROM Lane-0 measurements, not invented); the `trace.min_eigs` floor rises; the `solver="svd"` ablation is RE-RUN under the new regime before anyone quotes "ADMM ≈ warm start" again. Lane order is load-bearing: Lane 0 gates 1-3 and Lane 1 lands before Lane 3.

#### Scenario: post_admm_pre_polish_cos_over_lane0_baseline

- **WHEN** the new regime runs on production layers with the solver probe recording per-layer cos
- **THEN** on production layers, post-ADMM pre-polish cos improves materially over the Lane-0 baseline (threshold set FROM Lane-0 measurements, not invented)
- **AND** no threshold constant is authored in code or spec before the Lane-0 measurements land; the pinned threshold is recorded in this change's design.md from those measurements

#### Scenario: min_eigs_floor_rises

- **WHEN** the new regime runs on a fixed production target
- **THEN** the `trace.min_eigs` floor rises
- **AND** the floor is bounded below by `reg` at every iteration (the stabilizer at ρ≈0), and the measured floor exceeds the current regime's floor on the same target

#### Scenario: svd_ablation_rerun_under_new_regime

- **WHEN** the `solver="svd"` comparison is used to claim what ADMM buys over the warm start
- **THEN** the `solver="svd"` ablation is RE-RUN under the new regime before anyone quotes "ADMM ≈ warm start" again
- **AND** old-regime ablation rows are inadmissible for the claim — every comparison row records its regime (ρ endpoints, `reg`, normalization, sources)

#### Scenario: lane_order_gates_acceptance

- **WHEN** Lane-3 acceptance numbers are quoted as evidence
- **THEN** Lane 0 and Lane 1 have landed — solver regime numbers only mean something against the fixed preconditioner
- **AND** numbers quoted before both land are marked provisional, never acceptance evidence

### Requirement: Block tuning losses SHALL be weighted MSE with the terminal linear's Phase-1 output statistic

The TuneFP and TuneLatentSTE losses SHALL be weighted MSE, elementwise, by the robust-diagonalized Phase-1 output statistic of the block's terminal linear. The weight SHALL be detached, non-trainable, and applied in the stage's compute dtype. The canonical terminal linear SHALL be `down_proj`; arch-equivalent expansion is a future mechanism.

#### Scenario: weighting_reaches_the_loss_poisoned_weight_changes_loss

- **WHEN** fixtures run TuneFP and TuneLatentSTE on a toy block whose terminal linear's Phase-1 output statistic is poisoned at one entry (sentinel magnitude)
- **THEN** the poisoned weight vector changes the loss value relative to the run with the un-poisoned statistic
- **AND** the per-feature weight formula is `w = robust_diag(d_out_raw[terminal], tau=cfg.tau, gamma=cfg.gamma)` per [NQ] App. C :3012 and Algorithm 1 lines 2-5 Eq. 2
- **AND** this lane pins `CANONICAL_TERMINAL_NAMES = ("down_proj",)`; arch-equivalent names expand from this set via a future arch-to-role table

### Requirement: Uniform fallback when the canonical layer is absent SHALL be recorded, never silent

When the canonical terminal linear (`down_proj`) is absent from the block, or carries no Phase-1 statistic, the tuning stages SHALL fall back to uniform weights. A stage whose `loss_weight_source == "uniform_fallback"` SHALL carry the `tune_loss_uniform_fallback` event on record naming the block and the missing canonical layer. A stage whose `loss_weight_source == "uniform"` (explicit ablation) SHALL NOT emit the event.

#### Scenario: uniform_fallback_recorded_on_down_proj_less_block

- **WHEN** a down_proj-less block (no `down_proj` among its discovered linear names, e.g. a gpt2 `mlp.c_proj`) is tuned with terminal-gradient weighting requested
- **THEN** the uniform fallback fires
- **AND** the fallback is recorded as an event naming the block and the missing canonical layer — never silent

### Requirement: Test harnesses SHALL discover linear names from the loaded model, not hardcode Qwen3-specific names

All weighting and resolution fixtures SHALL discover linear layer names from the model's structure (`named_modules()`, `state_dict()` keys, or HF config). No fixture SHALL hardcode a Qwen3-specific layer name. Arch-to-role expansion is a future spec change; this lane's fixtures SHALL parameterize over known-good architectures and SHALL fail loudly on an unknown architecture.

#### Scenario: fixtures_discover_names_not_hardcode_qwen3

- **WHEN** the toy block fixture provides a synthetic block whose module tree carries non-Qwen3 names (`fc1`, `fc2`, `c_proj`)
- **THEN** the canonical-terminal resolution discovers the terminal by searching for `down_proj` among the block's actual names
- **AND** a test parameterized over `[qwen3, llama, gpt2]` passes for qwen3 and llama (weighting path) and passes for gpt2 (fallback path with recorded event)
- **AND** no test file contains a hardcoded Qwen3 layer name literal (`"q_proj"`, `"k_proj"`, `"v_proj"`, `"o_proj"`) outside of arch-map declaration rows

### Requirement: Stage records SHALL carry the weighting mode

The `tune_fp` and `ste` stage records SHALL carry the weighting mode actually used (`terminal_grad` or `uniform`) and the weight source (the terminal linear's name, or a reference to the recorded fallback event). The `block_loss_weight` config knob SHALL be solver state and SHALL join the checkpoint fingerprint.

#### Scenario: stage_records_carry_weighting_mode

- **WHEN** the `tune_fp` and `ste` stage records are emitted
- **THEN** each record carries the weighting mode
- **AND** a fallback run's records name the fallback event reference as the weight source

#### Scenario: loss_weight_knob_joins_checkpoint_fingerprint

- **WHEN** two configurations differ only in the `block_loss_weight` knob
- **THEN** their checkpoint fingerprints differ, so resume refuses across the knob change (solver state)

### Requirement: Block checkpoint archives SHALL carry and verify a content digest

Every `block-*.npz` this pipeline writes SHALL record a sha256 over its own
logical content, and a load SHALL refuse an archive whose content does not hash
to the recorded digest. The digest key itself SHALL NOT be part of the hashed
content. An archive carrying no digest SHALL load unverified, with a notice: an
absent digest is never a mismatch.

#### Scenario: block_archive_digest_written_at_save

- **WHEN** `save_block_checkpoint` writes an archive
- **THEN** the archive carries the sha256 of every array it stores — key name,
  dtype, shape and bytes, the reserved teacher-target key included — under a
  digest key that derives no layer name and is excluded from the hashed content

#### Scenario: stale_digest_refuses_rewritten_archive

- **WHEN** an archive's content is regenerated with different values — including
  a moved reserved teacher-target key — and the digest is carried over unchanged
- **THEN** `load_block_checkpoint` raises an error naming the archive and both
  digests, instead of rebuilding layer states from it

#### Scenario: unreadable_archive_is_refused_by_name

- **WHEN** an archive is bit-flipped or cut down in place so that its container
  no longer parses
- **THEN** `load_block_checkpoint` raises a named checkpoint refusal carrying
  the container's own exception class in its message, rather than letting that
  class escape to the caller

#### Scenario: legacy_archive_without_digest_loads_unverified

- **WHEN** an archive carries no digest
- **THEN** it loads, and the load prints a notice stating the archive is
  unverified

### Requirement: The checkpoint scan SHALL apply one contract to every name it cannot read

`completed_blocks` SHALL skip, with a printed notice naming the file, any
`block-*.npz` whose name carries no block index — the same notice-and-skip
contract it applies to an unreadable or empty archive. No file present in the
directory may abort the scan.

#### Scenario: malformed_block_stem_is_not_counted_done

- **WHEN** `completed_blocks` scans a directory holding an archive whose file
  name carries no block index
- **THEN** that archive is excluded from the done set with a printed notice
  naming it, and every well-formed archive in the directory is still reported

### Requirement: A fresh start SHALL preserve, never overwrite, prior checkpoint state

When a launch starts fresh in a checkpoint directory that already holds state
from a previous launch, the existing files SHALL be moved into a timestamped
subdirectory of that directory before the new fingerprint is written, leaving
the previous record (fingerprint, stage journal, archives) readable after the
relaunch. A directory with nothing to preserve SHALL NOT gain a snapshot
directory.

#### Scenario: fresh_start_preserves_prior_checkpoint_state

- **WHEN** a launch starts fresh in a checkpoint directory holding a previous
  launch's fingerprint, stage journal and an unfinished archive
- **THEN** those files are moved into a single `prior-<timestamp>` subdirectory
  with their bytes intact, and the directory's live fingerprint is the new
  launch's

#### Scenario: empty_directory_gains_no_snapshot

- **WHEN** a launch starts fresh in an empty checkpoint directory
- **THEN** no snapshot subdirectory is created

#### Scenario: prior_state_snapshots_are_additive

- **WHEN** a second fresh start happens in the same directory, including within
  the same second as the first
- **THEN** it archives the live state into its own `prior-<timestamp>`
  directory and leaves every earlier snapshot in place, un-nested
