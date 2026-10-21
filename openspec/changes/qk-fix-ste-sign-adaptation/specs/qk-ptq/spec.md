# qk-ptq delta: qk-fix-ste-sign-adaptation

## ADDED Requirements

### Requirement: The latent refinement stage SHALL make sign adaptation effective or declare itself scale-only in the run record

The straight-through refinement stage ([NQ] Sec 3.2 Step 3) SHALL instrument
per-step sign-flip counts for both factors of every scoped linear and SHALL
never finish a run as a silent sign no-op. Per the 2026-10-21 user
directive, the paper's own STE mechanism is attempted first; adopted
deviations are named in the run record.

#### Scenario: flip_counts_are_recorded

- **WHEN** the STE stage runs for T_post steps on a binary-factor block
- **THEN** the run artifact records per-linear, per-factor sign-flip counts
  and the latent relative movement for every scoped linear

#### Scenario: sign_adaptation_is_effective_or_declared

- **WHEN** the stage completes with zero accepted flips on a scoped linear
- **THEN** either the run record declares the stage scale-only for that
  linear with the re-scoping marker, or the gate for this change fails

#### Scenario: flip_acceptance_uses_block_local_objective

- **WHEN** a flip pass proposes sign changes to a factor entry
- **THEN** each flip is accepted only if the block-local objective
  `‖Wq@x_b − y_b‖` decreases, with a deterministic sweep order and a recorded
  acceptance count per linear
