# Design: qk-bias-carry

## D10 — Wire layout of the bias vector

Version 2 of the qpk1 stream inserts the `(dout,)` bias vector between the
scale vectors and the `Ub` bitplane:

```
header(24) | s1 (dout) | s2 (din) | bias (dout) | Ub bits | Vb bits
             \_______________ all in the stream's tagged dtype _______________/
```

Rationale for the position: the float section stays contiguous and the
bitplane offsets move by exactly `dout * itemsize`, so a reader computes one
extra offset rather than interleaving a new section type between bitplanes.
`dout` (not `din`) is the bias's width because the layer's map is
`x -> W x + b` with `b` indexed by output feature.

Byte model (D1's formula with the added term):

```
size(dout, din, r, w) = 24 + (2*dout + din)*w + ceil(dout*r/8) + ceil(din*r/8)
```

The rank-planning inversion (`rank_for_target_bpw`) is updated to this model,
because it inverts the byte size; the planning *policy* is unchanged. The
paper reduction ([NQ] App. F.5 Eq. 58-59) still holds for the weight storage
and gains exactly `16*dout` bits of bias overhead per layer.

## D11 — Legacy (pre-bias) reading: default-zero, not fail-closed

A version-1 stream carries no bias field. Reading one SHALL yield a layer
with no bias — the zero vector, which is exactly the map a biasless linear
applies — not an error and not a default for a *missing* field on a current
stream. The `deserialize` path (Python) returns `bias=None`; the Rust
`Layer` returns an empty bias vector; the server normalizes both to `dout`
zeros at load, so the forward needs no version branch.

Discarded alternatives:

- **Bump the version and reject version 1 (fail-closed).** This invalidates
  every committed pre-bias artifact — including the whole `server-model`
  fixture battery the Rust tests load — and would force a fixture
  regeneration whose bytes are not reproducible on this box (see D13). It is
  also the wrong semantics: the pre-bias artifact is well formed, and the
  zero map is what it described.
- **Keep the bias out of the container and ship it in a sidecar.** Rejected
  by the user's ruling of record ("the qpk1 container should include it") and
  by the container's own purpose: one self-describing per-layer stream. A
  sidecar re-introduces the cross-file drift the format exists to prevent.

## D12 — TuneFP's bias is the artifact's bias; STE still does not refine it

`reconstruct_block` no longer snapshots and restores `original_bias`. A
target's factorized replacement takes the LIVE bias of the module — i.e.
whatever the preceding TuneFP pass produced, or the pristine value for the
first target, which no pass precedes (`_factorize_one`'s per-target cycle).

TuneLatentSTE keeps its bias out of the refinement parameters: Eq. 10's
optimization variables are ONE linear's `(U, V, s1, s2)`, and a second,
unstated producer of the same artifact field would make the shipped bias
unreproducible from the record. The STE docstring is amended to state that
reason instead of "the container carries no bias field", which is no longer
true.

The stage record gains `bias_delta_rel` beside `weight_delta_rel`: the
aggregate figure is computed over ALL tuned parameters, so it cannot on its
own say whether the bias compensation is real. A `None` figure names WHICH of
three causes left it unmeasured, and is never a confident `0.0`:
`bias_tunables == 0` is a bias-free model, which has nothing to move;
`bias_tunables > 0` with `steps == 0` is a stage that never ran (empty tunable
set, or a vacuous-teacher early return); and `bias_tunables > 0` with `steps >
0` is a stage that ran over an all-zero bias baseline, where the relative
movement is undefined — never "the bias never moved".

## D13 — Which fixture carries the bias, and why the pipeline fixture does not

Two artifacts, deliberately:

- `tests/fixtures/server-model` stays a **version-1** artifact. Every Rust
  test that loads it is therefore exercising the declared legacy reading
  end to end (loader, engine construction, forward). It is the pinned pre-bias
  artifact, and regenerating it would also delete the separately authored
  `attention_gate.json`.
- `tests/fixtures/server-model-bias` is the **version-2 twin**: one
  transformer block at the same geometry, tokenizer and sidecars, with a
  deterministic non-zero bias on every projection, produced by the same
  stdlib generator (`tools/gen_server_fixtures.py`). A deterministic tool
  means this fixture regenerates byte-identically on any machine.
- `tests/fixtures/parity-tiny` (the Python-pipeline fixture) also stays
  version 1. Its generator is NOT byte-reproducible here: regenerating it on
  this box produces different weights and a different tokenizer file from the
  committed bytes (different `transformers`/`torch` init), so a regeneration
  would swamp the review with environment drift and bury the change. The
  cross-face battery therefore extends with a PINNED bias-carrying vector —
  the same literal asserted by `tests/test_representation.py` and by the Rust
  reader tests — plus the generated v2 twin above.

Discarded alternative: regenerate `parity-tiny` through the pipeline so the
end-to-end fixture carries bias. Discarded on measurement (D13 above): the
diff is dominated by unrelated bytes, and the generation run did not complete
reproducibly on this box.

## D14 — Unknown versions stay fail-closed

The reader accepts exactly 1 and 2. Any other version word, a version-2
stream whose body is short by the bias vector's bytes (truncation), and a
version-2 stream one byte past its extent (trailing garbage) are all hard
rejections — the bias field is never silently defaulted into existence.

## Open questions

None open. The two questions this change raised — whether the container
carries the bias (user ruling: yes) and what a pre-bias stream reads as
(D11: the zero map) — are both resolved above.

## Implementation crystallization (2026-10-25)

Recorded while implementing, so the choices cannot be re-litigated as
ambiguity:

- The version-2 number is `2`, exported as `REPRESENTATION_VERSION` in
  Python and `VERSION` in Rust; version 1 is
  `LEGACY_REPRESENTATION_VERSION` / `LEGACY_VERSION`.
- A version-2 stream ALWAYS carries the field. `serialize(..., bias=None)`
  writes zeros; there is no "field omitted" state on the current version, so
  a reader never has to infer presence from length.
- The server normalizes an absent (version-1) bias to `dout` zeros at
  `PackedLayer::from_layer`, and rejects a wrong-width vector at load rather
  than truncating it in the hot loop.
- Scope of the bias change: it is the layer's POST-tuning bias, i.e. the
  compensation the stage produced. Phase 3 (global scale calibration) still
  tunes `s1`/`s2` only; the bias is not a calibration parameter.
- Module placement follows the `block_recon` split that landed while this
  change was in flight: the stage edits live with their stages
  (`_tuning.py` for the TuneFP record and bias snapshot, `_ste.py` for the
  rationale, `_block_stage.py` for the two deleted reverts and the live-bias
  construction), and `block_recon.py` stays a pure façade. The test spy for
  `_tune_fp` is anchored on `_block_stage`, the module holding the call site.
- The gate fixture's `d`/`wide` are chosen against BOTH bounds the e2e gate
  asserts — byte-derived BPW < 1.0 and corruption delta < 1.0 — because the
  bias vector forces `d` up (at `d = 32` the float section alone exceeds one
  bit per weight for any `wide`; measured 1.037 even at `wide = 1024`) and the
  delta moves NON-monotonically with geometry. Measured at rank 8 / f16
  (`(d, wide) -> delta`): `(64,128) 1.188`, `(64,192) 0.518`, `(64,384)
  1.310`, `(96,128) 0.711`, `(112,128) 1.582`, `(128,128) 1.828`; BPW at
  `(64,192)` is 0.682. The first choice `(64,128)` passed pre-sibling (delta
  0.893) and stopped passing once `qk-ste-joint` landed its single joint
  TuneLatentSTE pass (delta 1.188) — the fixture had 0.107 of margin, and the
  joint pass consumed it. `(64,192)` restores ~0.48 of margin under the joint
  pass without relaxing either bound.
