# Proposal: qk-bias-carry

## Why

Every factorization discards a class of values the pipeline computed. TuneFP's
tuned set is `m.parameters()` over the still-FP remainder linears, so it tunes
the bias as well as the weight; `_factorize_one` then restores the PRE-TuneFP
bias before the factorized replacement is frozen, because the container had no
place to put it. The tuning stage therefore produced a bias compensation and
threw it away at the next step, at every target of every block.

[NQ] App. C makes the tuned values the product of the tuning stage. A stage
whose output is discarded is not "tuning for reconstruction"; it is a
measurement of a compensation that never ships.

## What Changes

1. **The container SHALL carry a per-linear bias vector.** Version 2 of the
   qpk1 layout adds the `(dout,)` bias between the scale vectors and the
   bitplanes. The writer emits version 2 only; version 1 stays readable as the
   pre-bias artifact (the zero map a biasless linear applies).
2. **Both faces SHALL carry it.** The Rust reader returns the vector, the
   Python reader returns it, the server's forward adds it, and a bias-carrying
   vector is pinned byte-identically on both faces.
3. **TuneFP SHALL persist the tuned bias.** The revert is deleted; the
   factorized layer takes the bias the tuning pass produced, and the stage
   record reports its movement as `bias_delta_rel` beside `weight_delta_rel`.

## Impact

- **Invariant-touching: yes — the storage format.** The qpk1 byte layout
  changes, so the compatibility impact is stated here and decided in
  `design.md` D11: a version-1 stream remains readable with an empty bias
  vector, and no existing artifact is invalidated. Every consumer of the
  container (Python compressor, Rust loader/engine) is updated in the same
  landing.
- **Affected specs**: `qk-representation` (modified layout/round-trip/
  accounting/planning requirements plus three added ones), `qk-ptq` (two
  added), `qk-server` (one modified forward requirement).
- **Affected code**: `src/quantibbler/qpk1.py`,
  `src/quantibbler/ptq/{block_recon,api}.py`,
  `crates/quantibbler-qpk1/src/lib.rs`,
  `crates/quantibbler-server/src/engine/mod.rs`,
  `tools/gen_server_fixtures.py`, `tools/quantize_hf_model.py`, tests,
  fixtures.
- **Bandwidth cost**: `dout` extra floats per layer at the stream's dtype —
  measured on the 4096x11008 r=100 shape, 227224 bytes against the pre-bias
  219032, i.e. BPW 0.0403 against 0.0389. Storage stays dominated by the
  packed binaries.

## Paper anchor

[NQ] App. C (the tuning stages' budgets and objective) and Algorithm 1 lines
12/19 (tuning precedes factorization): the tuned values are the stage's
product. No third-party implementation references (clean-room law).

## Non-goals

- No re-fire. The fire question returns to @user after this lands.
- No change to the solver, the rank planning *policy*, the polish pass, the
  stage budgets or the learning rates. The rank formula's inversion is
  updated only because the byte model it inverts gained a term.
- The continuous pre-sign latents stay unshipped: only the bias vector joins
  the stored fields.
- No frozen or hand-maintained byte tables beyond the existing pinned
  known-layer vectors.
