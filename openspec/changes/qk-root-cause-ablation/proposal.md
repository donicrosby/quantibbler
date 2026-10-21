# Proposal: qk-root-cause-ablation

## Why

Every archived production run collapses — ratios from 943x to 1.16e6x, never
anything in between. That is a cliff, not a quality spectrum. The fail-fast
probe data we already paid for localizes it further: the r512-d1 run aborted at
`layers.1`, depth 2, prefix ratio 21,161x. Per-block that is ~145x multiplicative
error at the second block, against the paper's implied ~1.03x per block
(2.18x total over 28 blocks). A sub-1-bit capacity problem degrades gradually
and composes; a 145x-per-block wreck at block 2 is not hardness — it is a bug in
the solver, the scale calibration, or the block write-back.

The user has directed the team to move on from CI and fix the actual root cause.
This change builds the measurement ladder that splits bug from method, cheapest
rung first (ladder authored by @sec-eng, 2026-10-27):

1. **fp factors** — rank-r512 reconstruction with the binary constraint off:
   pure fp factors through the existing pipeline. If prefix PPL still explodes,
   the defect is in factorization/scale/write-back and no quantization tuning
   matters yet.
2. **untuned signs** — binarization with `sign()` of the fp factors and zero
   refinement. Splits "binary representation cannot hold it" from "the
   optimizer is not finding it".
3. **per-component ablation** — attention-only and MLP-only scope, plus the
   per-depth probe history every run already records as a free collapse curve.

## What Changes

Three knobs, carried on `PTQConfig` and exposed as driver flags:

- `--factor-mode {binary,fp}` (default `binary`): `fp` removes the sign and the
  straight-through estimator from both factorized-forward sites and skips the
  discrete-only bit-flip polish. Every other stage runs unchanged.
- `--ablation {none,untuned-signs}` (default `none`): `untuned-signs` exports
  `sign(balance latents)` with zero refinement — no polish, no TuneLatentSTE.
- `--layer-scope {all,self_attn,mlp}` (default `all`): only linears whose module
  path contains the scope component are factorized; the rest stay full
  precision.

Ablation runs are measurements, never artifacts: the full gate, probe, and
report still run and record the knob values in provenance, but a `fp`-mode run
never reaches the qpk1 export (which hard-fails if asked), sets
`artifact_exported: false`, and records `effective_bpw: null`.

## Impact

- Affected capability: **qk-ptq** (ADDED requirements only).
- Affected code: `src/quantibbler/ptq/config.py`, `src/quantibbler/ptq/block_recon.py`,
  `tools/quantize_hf_model.py`, report provenance, plus tests.
- The qpk1 container and the server are untouched — no new container semantics.
- Default runs are byte-identical to today's pipeline (regression-guarded).
