# Design: qk-tune-loss-scale

## The defect, restated as numerics

The App. C weighted MSE is `mean(w ⊙ (B(X_b) − Y_b)²)` with `w` the terminal
linear's robust-diagonalized Phase-1 output statistic (`d_out_raw`). That
statistic is a mean-SQUARED-gradient quantity, so at hidden-state level its raw
scale is ~1e-10. The weight's relative per-feature structure is the statistical
content ([NQ] Step 2-1 Eq. 2); its absolute scale is a free numerical degree of
freedom that [NQ] never pins — the paper writes the loss form, not its scale.

At raw scale the consequences measured on the v5 report are:

- the weighted loss sits ~1e10x under the unweighted residual (`~1e-13` vs
  `~1e-3`), so the step-0 gradient is ~1e-12;
- Adam's absolute `eps` (1e-8) dominates the denominator, so the update degrades
  to `lr·g/eps` ~1e-8 absolute;
- on bf16 parameters (~8e-3 relative resolution) that update rounds back every
  step: `weight_delta_rel` is exactly `0.0`, not merely small;
- the exact-zero guard (`gradient_norm == 0.0`) is blind to a nonzero-but-dead
  gradient: `vacuous_events: 0` on a fully inert run.

## Decision 1: normalize the weight to unit mean

Restore the weight's absolute scale to the residual's own scale by dividing by
its own mean. This preserves every pairwise ratio (the statistical content) and
is exactly the paper's loss form with its unspecified scale fixed to the natural
one. The normalization happens ONCE, where the weight is resolved
(`_resolve_loss_weight`'s caller in `reconstruct_block`), so both stages receive
the same normalized vector — not two independent scalings that could drift.

`_resolve_loss_weight` itself keeps returning the raw robust diagonal: its
contract is "the robust diagonalization of the terminal statistic", it is pinned
by tests in `tests/ptq/test_tune_loss_weighting.py`, and the raw value is the
pre-normalization quantity the stage record must be able to name.

## Decision 2: the floor derivation of record

`floor = optimizer_eps × dtype_relative_resolution`, evaluated on the COARSEST
parameter dtype in the stage (a bf16 parameter is the one that rounds back).

Derivation: at step 0 the bias-corrected Adam update is
`Δ = lr · g / (|g| + eps)`. In the eps-dominated regime `|g| ≪ eps` this is
`Δ ≈ lr·g/eps`. The parameter dtype can only represent a relative change above
`finfo(dtype).eps`, so the asymptotically-smallest gradient whose eps-damped step
still clears the dtype floor for a unit-magnitude parameter is the product
`eps · finfo(dtype).eps`. Measured against the v5 rows: bf16 gives
`7.8125e-11`, and every recorded step-0 gradient (7.9e-12…6.2e-11) sits below
it; the post-fix (normalized) production gradient is the residual's own scale,
orders above the floor. Exact zero remains a special case of `< floor`.

The number is *not* a tuned threshold: both factors are read off the live
optimizer (`opt.defaults["eps"]`) and the live parameters
(`torch.finfo(p.dtype).eps`), and all three (`floor`, `floor_optimizer_eps`,
`floor_dtype_rel_resolution`, `floor_param_dtype`) travel in the stage record, so
a reader can re-derive it from the artifact.

## Decision 3: inertness is declared, not inferred

A stage that runs its full budget and moves its parameters by less than the
dtype's relative resolution is inert. The record says so (`inert: true`) instead
of leaving a reader to reconcile `steps: 256` with `weight_delta_rel: 0.0`. The
STE half declares it per-linear, from the recorded `u/v/s1/s2` relative moves.

## Named deltas (unclear statements implemented to the letter)

- **Requirement 2's "existing fail-closed path" for TuneLatentSTE.** TuneFP has
  one (restore the snapshot, emit `tune_fp_vacuous_teacher`, raise under
  `paper_faithful`). TuneLatentSTE has none. The letter is implemented as: record
  `vacuous: true` with the floor on the record, restore the entry snapshot, end
  the stage, and mark the per-linear rows inert — no new event name (a new event
  would change the stage sequence every existing consumer asserts) and no raise
  (the STE path has no fidelity-fail mechanism today). With f64 latents the
  derived floor is `1e-8 · 2.22e-16 ≈ 2.2e-24`, so the guard is constructible
  only by an extreme weight scale; the RED fixture drives it that way.
- **Requirement 3's "weight movement" for TuneFP** is the recorded
  `weight_delta_rel` (a relative Frobenius movement), compared against the
  relative dtype resolution; for TuneLatentSTE it is each per-linear relative
  movement.

## Crystallization (2026-10-25)

Implementing against the live tree crystallized three facts:

1. The normalization belongs at the resolve site, not inside the stages: a stage
   that normalized its own input could never observe the un-normalized regime the
   floor guard exists to catch.
2. The floor's dtype is the coarsest parameter dtype, so a mixed
   fp32/bf16 block is guarded by its bf16 parameters.
3. `_STAGE_PAYLOAD_KEYS` in `qk_report.py` is a required-keys shape check
   (extra keys pass), and its `_non_finite_path` refuses non-finite numbers in a
   payload — so every new stage field is recorded finite (`None` for "not
   measured", never NaN).

## RED → GREEN (measured 2026-10-25, `/opt/venv` torch 2.14.1+cpu, no accelerator)

RED, on the pre-change tree (`git worktree add --detach … HEAD` at `cfa3311`,
`PYTHONPATH=<worktree>/src`):

- `tests/ptq/test_tune_loss_scale.py` → collection `ImportError: cannot import
  name 'LOSS_WEIGHT_MEAN_NORMALIZATION' from 'quantibbler.ptq.block_recon'`
  (`_optimizer_gradient_floor` equally absent).
- behavioral probe on the v5-shape fixture (bf16 params, raw weight `1e-10`):
  `gradient_norm = 1.9717375279177577e-11`, `gradient_floor: <ABSENT>`,
  `vacuous: False`, `inert: <ABSENT>`, `steps: 2` — a dead stage booked its
  budget and the record said nothing. The weight `reconstruct_block` handed to
  the loss had mean `1.0605747542709931` (raw, un-normalized).

GREEN, on this change (`PYTHONPATH=/opt/data/quantibbler/src`):

- same probe: `gradient_floor = 7.8125e-11` (= `1e-8 × 0.0078125`),
  `vacuous: True`, `vacuous_reason: below_resolution_gradient_floor`,
  `inert: True`, `steps: 0` — the v5 signature is now named instead of silent;
  the weight handed to the loss has mean exactly `1.0`.
- `uv run --no-sync --extra torch --extra kernels --extra hf pytest
  tests/ptq/test_tune_loss_scale.py -q` → **7 passed**.
- measured inertness: bf16 params + unit-mean weight → `weight_delta_rel`
  ~1.7e-4 below the bf16 relative resolution 7.8125e-3 → `inert: true`; f64
  params + the same weight → ~3.8e-4 above f64's 2.22e-16 → `inert: false`.

The interpreter must be named with the number: the shared `/opt/venv` carries an
editable `quantibbler` whose `.pth` currently points at a sibling worktree, so
`PYTHONPATH=<repo>/src` is what makes the shared checkout's bytes win over
site-packages.

## Not done here

- No re-fire: this change makes the stages able to move; measuring them is the
  next decision, owned by @user.
- No hash/anchor table, no code-identity stamp: version control plus the CI gates
  answer "did the code change".
