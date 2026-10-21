# qk-ptq delta: qk-fix-ste-sign-adaptation

## ADDED Requirements

### Requirement: The latent refinement stage SHALL make sign adaptation effective or declare itself scale-only in the run record

The straight-through refinement stage ([NQ] Sec 3.2 Step 3) SHALL instrument
per-step sign-flip counts for both factors of every factorized linear in the
block — the seven scoped linears (attn q/k/v/o, mlp gate/up/down) — and SHALL
refine the paper's joint parameter set (U, V, s1, s2) per [NQ] Eq. 10, never
finishing a run as a silent sign no-op. Per the 2026-10-21 user directive, the
paper's own STE mechanism is attempted first; adopted deviations are named in
the run record.

#### Scenario: flip_counts_are_recorded

- **WHEN** the STE stage runs for T_post steps on a binary-factor block
- **THEN** the run artifact records per-linear, per-factor sign-flip counts,
  the latent relative movement, and the scale (s1, s2) movement for every
  factorized linear in the block — the seven scoped linears — so neither a
  zero-flip U/V nor a frozen scale can pass unnoticed

#### Scenario: sign_adaptation_is_effective_or_declared

- **WHEN** the stage completes with zero accepted flips on a scoped linear
- **THEN** either the run record declares the stage scale-only for that
  linear with the re-scoping marker, or the gate for this change fails

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
- **AND** the Algorithm-1 line-19 T_post budget is allocated
  deterministically across the block's factorized linears (equal split with
  the remainder to the earliest linears in fixed order) and the artifact
  records the per-layer step allocation
- **AND** the refinement parameter set is the paper's joint set (U, V, s1,
  s2) per [NQ] Eq. 10 and the Figure 3 caption (corpus line 577:
  "fine-tunes the continuous latent matrices and scales"): flip counts
  instrument the U/V sign crossings and scale movement is recorded per
  linear alongside them — omitting scales from the parameter set is a named
  deviation, never silent
