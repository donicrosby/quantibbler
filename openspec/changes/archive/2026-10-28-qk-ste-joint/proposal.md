# Proposal: qk-ste-joint

## Why

The block-reconstruction stage's objective is BLOCK reconstruction error: `y_b`
against the quantized block's output. After factorization the free parameters of
that objective are the block's ENTIRE factorized set. The shipped per-target
structure hands TuneLatentSTE one linear at a time — each linear's latent set is
optimized against a context in which its neighbours are frozen at whatever value
they were committed to, so every linear is refined as if the others were exact
and no correction that requires two linears to move together can be expressed.
The measured shape of that mismatch is in the record: the seven per-linear rows
of one block carry seven DIFFERENT `residual_before` readings (0.2098, 0.2321,
0.5454, 0.5727, 1.0080, 1.4611, 1.5653 on the fixture), because each row measures
a block objective that moved between passes, not one objective measured once.

Ruling of record (@user, 2026-10-25): **"STE should tune all at once not per
linear."** Algorithm 1 line 19's "the current target linear layer" prose is
superseded by it; the deviation is declared in the spec delta and in design.md.

## What Changes

1. Per block, after every factorization completes and before the stream swap,
   TuneLatentSTE SHALL run ONCE, jointly over all of the block's factorized
   linears: one optimizer over every latent/scale parameter set, one loss on the
   block output against `y_b`.
2. The stage's run record SHALL be ONE block-scope record carrying one row per
   factorized linear, with that linear's movement, sign-flip and dead-column
   counts, and it SHALL declare its scope (block-joint, never implied
   per-linear).
3. The joint pass SHALL consume the normalized terminal loss weight and the
   resolution-derived gradient floor already shipped for both tune stages — it
   arrives resolved from the block's resolve site, and is not re-derived inside
   the stage.
4. The App. C factorized-stage regime (stage lr, 8 epochs, cosine) applies to the
   joint pass and is consumed ONCE for the whole block. No per-linear share and
   no block-pooled allocation is applied on top of it.

## Impact

- **Affected specs**: `qk-ptq`, four ADDED requirements (no MODIFIED — the
  existing "refined per block ... jointly" requirement already says the
  refinement is joint; this change makes the implementation match it and adds
  the record contract).
- **Affected code**: `src/quantibbler/ptq/block_recon.py` (`reconstruct_block`'s
  per-target loop and the STE call site + declaration ladder;
  `_tune_latent_ste`'s record), `src/quantibbler/qk_report.py` (the STE stage
  map row's deviation text and the stage row-time unit), tests.
- **Not invariant-touching**: no qpk1 format, kernels, server, or solver-backend
  change. TuneFP's placement, the factorization, the polish pass and the
  declaration ladder's semantics per linear are unchanged. The only pipeline
  behavior that moves is when and over what the refinement runs, which is the
  point of the change.

## Paper anchor

[NQ] Step 3 / Algorithm 1 line 19 and Eq. 10, with App. C's factorized-stage
regime. Line 19's per-target-layer prose is a DECLARED deviation (design.md).
No third-party implementation references (clean-room law).

## Non-goals

- No re-fire. The fire question returns to @user after the stage-fix set lands.
- No change to the solver, the factorization, the polish pass, the stage budget
  VALUES, or the declaration ladder's per-linear semantics.
- No new code-identity stamp or hash gate (version control plus the CI gates
  already answer "did the code change").
