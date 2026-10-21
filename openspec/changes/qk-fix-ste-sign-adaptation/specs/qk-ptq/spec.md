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
  linear with the re-scoping marker, or the gate for this change fails
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
