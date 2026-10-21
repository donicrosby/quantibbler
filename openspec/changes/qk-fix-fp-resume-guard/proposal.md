# Proposal: qk-fix-fp-resume-guard

## Why

A resumed `factor_mode=fp` run is a **silent no-op on the prefix**, and every
gate stays green while it happens. The chain, verified on bytes
(`src/quantibbler/ptq/checkpoints.py`):

1. `extract_and_pack` refuses to produce layers in fp mode, so
   `result.layers == []`.
2. `save_block_checkpoint` writes that empty list anyway — a 22-byte,
   zero-key `block-N.npz` (measured: `rca-rung1-fp/checkpoints/block-1.npz`
   and `block-2.npz` are exactly 22 bytes, `np.load` → `[]`).
3. `completed_blocks` globs `block-*.npz` with no size or key check and
   counts those empty archives as done.
4. `load_block_checkpoint` returns `[]`; `apply_layer_states` applies nothing
   — and raises nothing, because its `missed` set is built **from the
   archive's own contents** (`by_rel`), so an empty archive yields an empty
   diff and the guard that exists to prevent unfrozen-prefix divergence is
   blind to the zero-names case.

A resumed fp run therefore advances a **pristine** prefix and reports a
normal completion with the fingerprint guard passing. Rung 1 itself is NOT
contaminated (verified: the OOM corpse holds only `fingerprint.json`; the
fresh run's block archives postdate the fresh run's start — those blocks were
solved in-run), but no fp resume can be trusted until this is fixed.

This is the **second fail-silent guard this week** (the stage-trace ratio
falsy-zero was the first) with the same shape: a guard built from *what
arrived* instead of *what was owed*.

## What Changes

- `completed_blocks` ignores empty (zero-key / undersized) archives.
- `save_block_checkpoint` refuses to write an empty layer list (raise — an
  empty archive must never exist on disk again).
- `apply_layer_states` drives its guard from the **live block's expected
  quantizable names**, asserts set-equality against the archive's contents,
  and raises on any missing or unexpected name.
- Class-fix sweep: audit the codebase for other guards built from
  arrived-content instead of owed-content; each gets an expected-vs-present
  assertion or a recorded waiver.

## Impact

- **Affected specs**: `qk-ptq`, one ADDED requirement only (no MODIFIED;
  sibling changes carry live deltas under this capability).
- **Affected code**: `src/quantibbler/ptq/checkpoints.py`, the resume path in
  `block_recon.py`, tests.
- **Not invariant-touching**: no qpk1 format, kernels, or server change;
  existing checkpoint archives are read-compatible (good archives still
  load; empty ones now raise).

## Non-goals

- No fix to fp-mode *extraction* itself in this change (the continuous-factor
  replay is a separate re-run, budgeted elsewhere).
- No migration of rung-1 artifacts — they are untouched and remain valid.
- No change to the divergence-guard change (`qk-divergence-guards`) — cited,
  not modified.
