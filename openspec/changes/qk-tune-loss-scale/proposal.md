# Proposal: qk-tune-loss-scale

## Why

Every tune stage in every recorded run has been numerically inert. The v5 report
(`quantize_report.json`, sha256 `7612ce32…`) carries TuneFP ×12 cycles and
TuneLatentSTE ×14 rows whose step-0 gradients sit at 7.9e-12…6.2e-11,
`weight_delta_rel` is exactly `0.0` on bf16 parameters, `u_rel_move` is ~1e-11 on
the f64 latents, and `residual_before == residual_after` on every row — while the
real unweighted block residual GROWS 0.0347 → 0.181 across block 0's seven
linears. `vacuous_events: 0`: the exact-zero guard cannot see this regime.

Root cause (derivation of record, `qk-pertarget-cycle` design.md §23): the App. C
weighted MSE is weighted by the terminal linear's `d_out` robust diagonal — a
mean-SQUARED-gradient statistic whose raw scale at hidden-state level is ~1e-10.
The weight's RELATIVE per-feature structure is its statistical content; its
ABSOLUTE scale is a free numerical degree of freedom. At raw scale the gradient
falls below Adam's absolute `eps` (1e-8), the update degrades to ~`lr·g/eps`
~1e-8 absolute, and on bf16 parameters (~8e-3 relative resolution) every step
rounds back to the identical value. The stages run full budgets and move nothing.

## What Changes

1. The terminal-statistic loss weight SHALL be normalized to unit mean before use
   in both tune stages; both stage records SHALL carry the pre/post-normalization
   mean so the applied scale is auditable.
2. The vacuous-stage guard SHALL test a **resolution-derived floor** instead of
   exact zero, so a gradient below the optimizer-eps × dtype-resolution product is
   declared vacuous and takes the existing fail-closed path; exact zero stays a
   special case.
3. Stage records SHALL **declare** inertness: a stage completing its full budget
   with weight movement below the parameter dtype's relative resolution is
   recorded `inert: true` rather than silently reading `steps: 256`.

## Impact

- **Affected specs**: `qk-ptq`, three ADDED requirements (no MODIFIED; sibling
  changes carry live deltas under this capability).
- **Affected code**: `src/quantibbler/ptq/block_recon.py`
  (`_resolve_loss_weight`'s caller, `_tune_fp`, `_tune_latent_ste`), stage
  records, tests.
- **Not invariant-touching**: no qpk1 format, kernels, server, or solver-backend
  change. Nothing here moves a bit or a scale; it restores the scale of a loss
  weight the method already prescribes.

## Paper anchor

[NQ] App. C (weighted MSE) + Eq. 2 / Algorithm 1 lines 2-5 (the statistic's
construction is gradient-based): normalizing the loss weight does not alter the
preconditioner's role, only the loss weight's numerical scale. No third-party
implementation references (clean-room law).

## Non-goals

- No re-fire. The fire question goes to @user after this lands.
- No change to the solver, the factorization, the polish pass, or any stage
  budget / learning rate.
- No new code-identity stamp or hash gate: version control plus the CI gates
  already answer "did the code change".
