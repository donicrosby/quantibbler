# Design: qk-ste-joint

## 1. Ruling of record

@user, 2026-10-25: **"STE should tune all at once not per linear."**

This card implements that ruling. It is a deliberate, declared deviation from
[NQ] Algorithm 1 line 19's prose, which names "the current target linear layer"
(singular) and whose Eq. 10 writes the optimization variables as one linear's
`(U, V, s1, s2)`.

## 2. Implementation crystallization (2026-10-25)

**The deviation, declared.** The stage is the refinement of a block's factorized
components against the BLOCK's reconstruction objective — `y_b` against the
quantized block's output. After factorization, the free parameters of that
objective are the block's ENTIRE factorized set. The paper's per-target reading
refines each linear against a context in which its neighbours are already
committed: each linear is optimized as if the others were exact, and a correction
that requires two linears to move together is not in the reachable set of any
single pass. The joint reading optimizes the block's quantized remainder as one
system, so one linear's residual can be partially cancelled by another. The joint
reading is the correct structure for a block-level objective, and it is what the
existing `qk-ptq` requirement of record already says ("each block's continuous
latents and scales SHALL be jointly refined ... against the block-output
objective") — this change makes the implementation match that requirement and
supersedes the per-target reading recorded by `qk-fix-ste-sign-adaptation`.

**Where the pass runs.** After the factorization loop, before the stream swap. In
`reconstruct_block` that is after `_set_linear` has frozen every factorized
linear into the module and before the extraction/packing block; the emitted
stage sequence is unchanged (`capture -> tune_fp -> lb_admm -> ste ->
sign_extract_pack`), so no consumer's ordering assertion moves.

**Budget.** The App. C factorized-stage regime (stage lr, 8 epochs, cosine) is
unchanged; `epochs x (rows // batch_size)` is consumed ONCE for the whole block
where the per-target cycle handed it to EACH linear. Dividing the regime per
linear would put every linear back on a fraction of the paper's budget; pooling
it per block would invent a scheme the paper does not describe. Consuming it once
is the reading that keeps the paper's regime verbatim while changing only the
scope it applies to. Every per-linear row records that single block budget as
`scheduled_steps`, so a row cannot be misread about the budget it ran under.

**Record contract.** ONE record per block, not one per linear. The record is the
stage's own summary (`steps`, `elapsed_s`, `gradient_floor`, `vacuous`, `inert`,
`scope`) beside its per-linear rows. `scope` is the new marker
(`STE_JOINT_SCOPE = "block_joint"`): the same `declared` values
(`paper_ste` / `discrete_flip_pass` / `scale_only`) mean something different when
one joint pass over N linears produced them than when N per-linear passes did,
and the rows alone cannot tell a reader which. A firing is therefore declared
block-joint rather than implied per-linear.

**Per-linear `elapsed_s` is null under the joint scope, and the stage's row time
moves to the record.** A joint call over several facs cannot split its wall time
by linear — the one measurement belongs to the whole set, and dividing it would
report a share no clock observed. The unchanged `_stamp` already reports null for
a multi-fac call; this change adds the record's own `elapsed_s`, and
`qk_report.stage_row_times` now counts one `ste` row per fire (one block) instead
of one per refined linear, so the measured cost is not nulled and not
multiplied by the linear count.

**Dead-column counts.** Each row records how many rank indices carry no sign,
i.e. how many outer-product columns are zero because either factor's column is
identically zero. That is the same notion `lb_admm` floors in place of a sign.
Without it a reader cannot tell an effective joint pass from one whose rank
budget is partly inert by construction.

**Consumed, not re-derived.** The normalized terminal loss weight and the
resolution-derived gradient floor already ship for both tune stages; the joint
pass receives the resolved (unit-mean) weight from the block's resolve site and
uses `_optimizer_gradient_floor` on its own optimizer/parameter set. Nothing in
the stage re-normalizes or re-derives either, so the floor guard still observes
the un-normalized regime it exists to catch.

## 3. What is deliberately NOT claimed

The per-linear budget tests that `qk-fix-ste-sign-adaptation` left behind
asserted that each scoped linear consumed the FULL per-linear budget. Those
assertions (every row's `steps` and `scheduled_steps` equal the recorded
`total_steps`) remain TRUE under the joint scope — every row books every step of
the one joint pass — but their prose is superseded, so the docstrings are amended
in place rather than left to assert a scheduling rule that no longer exists.

A residual ORDERING between the sequential composition and the joint pass is NOT
claimed and NOT pinned. Measured on a two-linear chained fixture across seeds,
budgets and learning rates, joint-vs-sequential residual is non-monotone (joint
wins 5/6 or 1/6 depending on the geometry), because the composition gets two
bounded passes while the joint pass gets one. The separation that IS provable is
structural and is what the fixture pins: a per-target pass's free parameters are
ONE linear's, so it cannot move a neighbour at all; the joint pass has every
linear free, so a pass that must move two linears to lower the block objective is
expressible only jointly.

## 3a. Requirement conflict, reported not laundered

`openspec/changes/qk-fix-ste-sign-adaptation/` (an UNARCHIVED change) carries an
ADDED requirement whose scenario `refinement_covers_every_factorized_linear_with_paper_parameter_set`
states that "each linear's refinement consumes the FULL per-linear budget —
`epochs × (n // batch)` steps per linear — never a block-pooled split". That is
the per-target reading this change supersedes.

It is NOT amended here. The requirement of record (`openspec/specs/qk-ptq/spec.md`,
"Factorized components SHALL be refined per block with the Straight-Through
Estimator") already reads "each block's continuous latents and scales SHALL be
jointly refined ... against the block-output objective", so the disagreement is
between that requirement and an unreleased proposal — and rewriting a SHALL
inside another change's delta is how a ruling launders a spec change nobody
ratified. Reported to the planner as a finding instead; the owning change's delta
needs the amendment when it lands or archives.

## 4. RED → GREEN

RED (pre-change tree, per-target structure): the block's seven per-linear rows
carried seven DISTINCT `residual_before` / `residual_after` / `gradient_norm`
readings — one block objective measured seven times at seven different contexts
(`0.209847 0.232117 0.545375 0.572653 1.008034 1.461143 1.565284`), and
`_tune_latent_ste` was invoked seven times per block, each with a one-linear
`facs` dict.

GREEN (post-change): exactly one invocation per block with the block's full
factorized set; the seven rows share ONE measurement
(`residual_before 1.517289`, `residual_after 1.507108`,
`gradient_norm 3.573410`); the record carries `scope: block_joint`,
`total_steps 16`, `steps 16`, `elapsed_s > 0`, and a `dead_columns` count per
row; the emitted stage sequence is unchanged.

Transcripts: `tasks.md` §4.
