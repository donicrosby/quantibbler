# Proposal: qk-artifact-dtype

## Why

The qpk1 container declares a dtype for its quantized sidecars (`ScaleDtype`
in the layer-stream header, tags 1/2) but the FP remainder is hardcoded fp32
in both faces: the Rust loader reads the embedding, lm_head, and norm sidecar
with `read_f32_bin_exact` (loader.rs:237/241/272) and the host writer casts to
float32 unconditionally (`tools/quantize_hf_model.py`). The `weight_tying`
grammar carries a literal `@fp32` suffix (loader.rs:554/664/689). The
remainder — embedding in particular, `vocab_size x d_model` floats — is the
largest byte contributor outside the packed bitplanes, and it is the last
undeclared dtype in the container.

bf16 is the natural first non-fp32 citizen: it halves those files' bytes with
a bounded, format-defined quality cost (8 significant bits — the same
representation the pipeline's model side already evaluates in), and its error
behavior is arithmetic-derivable from the IEEE bf16 format rather than an
empirical claim.

## What Changes

1. **Format — remainder dtype declaration.** The remainder tensors
   (embedding, lm_head, norm weights) gain a declared dtype: `fp32` (the
   current bytes, unchanged meaning) or `bf16`. Declaration surface: an
   optional `dtype` key in the manifest `[sidecar]` block plus an extended
   `weight_tying` suffix `@{fp32,bf16}` that SHALL agree with it. The loader
   resolves the dtype from the declaration only, never from file-size
   heuristics; a present-but-unknown token, or a suffix/key disagreement, is a
   named loud refusal. An absent key reads as `fp32` — the pre-declaration
   byte contract, so every pre-change artifact stays readable byte-identically
   (compatibility decided in design.md D1 with the byte-layout evidence).
2. **Writer.** The host pack path gains `--remainder-dtype {fp32,bf16}`
   (default fp32 — current behavior unchanged). bf16 is an opt-in whose cast
   is a checked stage (one shared round-to-nearest-even helper + a
   finiteness gate), and the report's fidelity fields name the remainder
   dtype from the run's arguments, never from hardcoded literals.
3. **Repack utility.** `tools/repack_remainder.py` converts an existing
   artifact's remainder fp32→bf16 without re-training: layer streams and
   tokenizer files are copied byte-identically, the remainder is re-encoded,
   the manifests are rewritten with the declared dtype, and the output
   manifest records the source identity and the conversion. Emits a NEW
   directory, never mutates in place; lossy-by-declaration; deterministic
   bytes (no wall-clock timestamps).
4. **Both faces + parity.** The Rust loader reads both dtypes at their
   declared itemsize; the host writer emits both; byte-exact cross-face
   fixtures and a loud-refusal battery pin the contract. A bf16 variant's
   decode-side logits sit within a declared tolerance derived from bf16's
   8-bit mantissa (half-ulp 2^-8 per element, accumulated — derivation in
   design.md D9), explicitly NOT the float64 1e-12 parity tolerance.

## Impact

- **Invariant-touching: yes — the storage format.** The compatibility impact
  is stated and decided in design.md D1: an extension, not a version bump;
  qpk1 v1 (the v6 artifact) and v2 (post-bias-carry) both remain readable
  after the change, and a pre-change loader fails closed on any bf16 artifact
  (grammar refusal + exact-length refusal — no silent misparse).
- **Affected specs**: `qk-server` (two MODIFIED loader requirements, three
  ADDED), `qk-ptq` (four ADDED: exporter, report fidelity, repack, cast
  bound). `qk-representation` is untouched — the per-layer stream layout does
  not change.
- **Affected code (implementation)**: `crates/quantibbler-qpk1/src/loader.rs`
  + `lib.rs` (new enum), `tools/quantize_hf_model.py`,
  `src/quantibbler/qk_report/reports.py`, `tools/repack_remainder.py` (new),
  `tools/gen_server_fixtures.py`, a committed bf16 twin fixture, and tests.
- **Co-touch note**: the active change `qk-serialization` owns pydantic
  adoption of the report surface; this change only demands a new report field
  and string content. The implementer reconciles against whatever
  `qk-serialization` has landed; this delta stays ADDED-only under `qk-ptq`.

## Paper anchor

No method change: no solver, stage, budget, or geometry surface is touched.
The bf16 error bounds derive from the IEEE bf16 format itself (1 sign + 8
exponent + 7 explicit mantissa bits → relative ulp 2^-7, round-to-nearest-even
half-ulp 2^-8) — arithmetic-derivable, no corpus claim needed. The remainder
storage dtype is a container-level decision, not a method element.

## Non-goals

- No bf16 for the qpk1 layer-stream scales: `ScaleDtype` stays fp16/fp32
  (the `qk-kernels-dtype-generic` lane owns kernel genericity; different
  surface, no overlap).
- No fp16/fp8/fp4 remainder dtypes; no bf16→fp32 conversion direction.
- No in-place repack; the source artifact is never mutated.
- No re-fire, and no touching the live fire's artifacts.
- No single-file blob container — `qk-artifact-blob` is chained after this
  change and will re-home this declaration surface into its header; this
  change deliberately keeps the directory format.
