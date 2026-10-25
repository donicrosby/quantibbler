# Proposal: qk-fp-resume-shape-guard

## Why

The two-phase-apply hardening (review finding F2, routed out of
`qk-fix-fp-resume-guard`) adds a fail-closed refusal in `apply_layer_states`
for a checkpointed state whose arrays cannot factorize to the live linear's
weight/bias — but no requirement of record covers it.
`restore_asserts_expected_equals_present` (folded into
`openspec/specs/qk-ptq/spec.md` at archive) is payload-blind: it asserts
name-set equality only, so a right-key/wrong-shape archive slips past it and
the raise used to land mid-write-loop, after earlier layers were already
overwritten.

The behavior is now observable (a raise naming the block, layer, and stored
shapes) and is a genuine fail-closed integrity guarantee — it belongs in the
spec of record, not only in code.

## What Changes

- Ratify the shape refusal as a SHALL requirement: `apply_layer_states`
  validates every stored state's array shapes against the live linear's
  weight/bias before any write-back, and raises on a state that cannot
  factorize to them, leaving the block untouched.
- Record a dated correction note for the archived sec-eng row that
  over-claims "raise happens before any write-back" (see design.md).

## Impact

- **Affected specs**: `qk-ptq`, one ADDED requirement (no MODIFIED; sibling
  changes carry live deltas under this capability).
- **Affected code**: none in this change — it documents behavior the F2 patch
  (`src/quantibbler/ptq/checkpoints.py` `apply_layer_states` +
  `_require_state_shapes`) already implements; the patch lands with the spec.
- **Not invariant-touching**: no qpk1 format, kernels, or server change; the
  shape contract is the representation of record already pinned in
  `openspec/config.yaml` (`Ub in {-1,+1}^(dout x r)`, `Vb in {-1,+1}^(din x r)`,
  `diag(s1) @ (Ub @ Vb^T) @ diag(s2)`).

## Non-goals

- No amendment to the archived `2026-10-25-qk-fix-fp-resume-guard` delta or
  its frozen scenario — archived change dirs are history.
- No change to the name-set guard (`restore_asserts_expected_equals_present`)
  or the existing "fail closed on empty or incomplete block archives"
  requirement.
- No code change: the F2 patch is the implementation; this change ratifies
  its observable behavior.
