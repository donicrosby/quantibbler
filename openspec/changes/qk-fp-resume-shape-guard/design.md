# Design: qk-fp-resume-shape-guard

## The hardening being ratified

`apply_layer_states` (in `src/quantibbler/ptq/checkpoints.py`) restores
checkpointed bits+scales into a block's linears by reproducing
`reconstruct_block`'s write-back: `W_q = (s1 ⊙ Ub) @ (s2 ⊙ Vb)ᵀ` copied into
the linear's dtype. The name-set guard asserts the archive's key set equals
the block's expected quantizable names, but is payload-blind — it says
nothing about the arrays behind those keys.

A right-key/wrong-shape archive therefore used to raise INSIDE the write
loop: `module.weight.copy_` on the second linear, after the first was already
overwritten. The F2 patch makes the apply two-phase — phase 1 matches every
archive layer to its module and validates per-array shapes (via
`_require_state_shapes`), phase 2 writes — so a mis-shaped state raises with
the live block untouched.

## The shape contract

`_require_state_shapes` reads the live linear's `(dout, din)` weight shape and
checks that the stored state can produce it: `Ub` is `(dout, r)`, `Vb` is
`(din, r)` with a shared `r`, `s1` is `(dout,)`, `s2` is `(din,)`, and a
stored bias is `(dout,)`. This is exactly the representation of record in
`openspec/config.yaml` (binary factors `Ub in {-1,+1}^(dout x r)`,
`Vb in {-1,+1}^(din x r)`, `diag(s1) @ (Ub @ Vb^T) @ diag(s2)`). No restorable
archive is turned away: a state the check rejects could only have raised or
produced a mis-shaped weight anyway.

## Archive wording correction (2026-10-25)

The archived `2026-10-25-qk-fix-fp-resume-guard/tasks.md` carries a ticked
sec-eng row that over-claims:

> "(sec-eng) Integrity pass: the new raise cannot corrupt in-flight state
> (raise happens before any write-back)"

The claim is TRUE for the name-set raise (which fires before any write) and
for the on-disk checkpoint (untouched), but was FALSE for the in-memory block
on HEAD 84ba3b4: `apply_layer_states` validated names before writing but
validated shapes only implicitly, inside the write loop, so a
right-key/wrong-shape archive raised from `module.weight.copy_` on the second
linear after the first was already overwritten (`self_attn.q_proj`
max|after-before| = 2.552724; torch "The size of tensor a (4) must match the
size of tensor b (3)").

This change SUPERSEDES that wording: the two-phase apply ratified here
(validate key set and per-array shapes in phase 1, write only in phase 2)
makes the claim true in full. The archive is left as history — not amended —
and this note is the dated correction, per the repo's pattern for retracted
wording. Cited by symbol (`apply_layer_states` phase ordering), not line
numbers.

## Discarded

- **Amending the archived delta or its frozen scenario.** Rejected: archived
  change dirs are history, and `openspec/specs/**` is the requirement of
  record. The correction lives here.
- **Folding the shape refusal into the existing name-set requirement via
  MODIFIED.** Rejected: MODIFIED replaces the whole block and collides with
  sibling live deltas under `qk-ptq`; an ADDED requirement is archive-order
  independent.
- **A single phase that validates lazily during the write.** Rejected: that
  is the pre-F2 behavior, and it is exactly what leaves the block
  half-quantized between the raise and teardown.
