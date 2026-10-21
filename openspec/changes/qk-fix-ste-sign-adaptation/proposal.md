# Proposal: qk-fix-ste-sign-adaptation

## Why

[NQ] Sec 3.2 Step 3 prescribes joint refinement of (U, V, s1, s2) with the
straight-through estimator against the block-output objective ([NQ] Eq. 10,
T_post steps). Measured on the stage trace: in 4,096 steps the stage flips
**zero** sign bits on all 7 scoped linears while moving latents 5–33%
(`latent_rel_change_u` 0.055–0.327). The binary factors are frozen by the
warm start, so the U/V half of the joint refinement is structurally inert —
only the scales (s1, s2) adapt. The stage's nominal target, sign adaptation,
is a silent no-op in every archived run.

This is the second "says it refines X, measurably never touches X" finding
this week (TuneFP is the other). The scales do real work — the T4 stage
improves the block output (0.322 → 0.254 attn, 0.508 → 0.276 mlp) — so this
is not a proposal to delete the stage; it is a proposal to make sign
adaptation effective or declare the stage honestly scale-only.

## What Changes

- The stage SHALL instrument per-step sign-flip counts (per linear, both
  factors) and report them in the run artifact.
- The stage SHALL either (a) achieve sign adaptation with decreasing
  block-output objective, or (b) be re-scoped as scale-only refinement in
  code, spec, and run record — **fail-honest, never fail-silent**.
- Mechanism ladder (design.md holds the full alternatives and the decision
  gate), ranked per the 2026-10-21 user directive — paper first: (1) the
  paper's own STE mechanism, instrumented and diagnosed so latents cross
  zero within [NQ] App. C parameter space; (2) magnitude decay toward zero
  with per-step re-binarization; (3) a bounded discrete sign-flip pass on
  `Ub`/`Vb` with block-local loss-decrease acceptance, recorded as an
  explicit deviation if adopted; (4) honest re-scope to scale-only.

## Impact

- **Affected specs**: `qk-ptq`, one ADDED requirement only (no MODIFIED;
  sibling changes carry live deltas under this capability).
- **Affected code**: `src/quantibbler/ptq/block_recon.py`
  (`_tune_latent_ste`, `_quantized_forward_block`), recorder/trace artifacts,
  tests.
- **Not invariant-touching**: no qpk1 format, kernels, or server change;
  binaries remain ±1 by construction.

## Non-goals

- No change to [NQ] Step 2 (greedy polish) or the scales-only global tuning
  requirement.
- No task-loss coupling: acceptance of any flip uses the block-local
  objective only.
- No stage budget/lr retuning in this change.
