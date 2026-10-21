# Spec Delta: qk-root-cause-ablation

## ADDED Requirements

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
`chunk_rows=None` float64 reference path SHALL remain byte-identical.

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
