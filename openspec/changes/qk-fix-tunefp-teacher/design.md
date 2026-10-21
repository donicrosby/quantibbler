# Design: qk-fix-tunefp-teacher

## Teacher construction (corpus-pinned, [NQ] Algorithm 1)

Verified 2026-10-21 on the pinned corpus bytes
(`docs/papers/nanoquant-2602.06694v3.md`, algorithm block): the figure text
IS carried by the conversion and pins the Step-1 construction:

- `X_b ← M̂_{<b}(X_cal)` — input activation after the already-compressed
  prefix.
- `Y_b ← B^FP_b(X_b)` — teacher output: the pristine FP block applied to
  `X_b`.
- `TuneFP(b, X_b, Y_b; T_pre)` — Step 1.

So `y_fp = block_fp(X_prefix_q)` where `X_prefix_q` is the
compressed-prefix stream, captured once per block BEFORE any tuning or
in-block factorization, under `torch.no_grad()`, `.detach()`ed, gradients
disabled, never recomputed from tuned weights. The tuned objective is
`‖block_tuned(X_prefix_q) − y_fp‖²`.

**Placement (figure-encoded, 2026-10-21).** The user pinned the method
image directly: [NQ] Figure 3, `block_recon.png` — vendored at
`docs/papers/assets/block_recon.png`, sha256
`3b4c5ed8cb12757f6a39cb366329418abcc00c372366762223ae9926ab798c61`,
source `https://arxiv.org/html/2602.06694v3/block_recon.png`. Panel
inventory (verified on the vendored bytes): left panel — Q/K/V GREEN
(already factorized), O/G/U/D YELLOW with flame icons (being tuned),
all four flames simultaneous; middle panel — scissors on O only; right
panel — O green with a flame on O only; sequence arrows left→right.
Official caption (pinned HTML): "(1) Error Propagation Mitigation to
adjust full-precision weights for accumulated errors". Together with
the Sec 3.2 Step 1 prose ("previously factorized layers in the current
block") the figure encodes the operative placement: **TuneFP fires ONCE
per block (Algorithm 1 line 12, honored), AFTER the attention
projections Q/K/V are factorized and BEFORE the remaining linears
O/G/U/D are, tuning the still-FP remainder against the cached
teacher.** At that placement the objective is never identically zero —
the compensation source always exists. The literal
line-12-before-everything placement (zero factorized siblings,
objective `‖B^FP_b(X_b) − Y_b‖²` ≡ 0 at init) is the measured defect:
the bit-identical T0 the stage trace recorded, ruled out by the
figure's own encoding. The same captured (X_b, Y_b) feeds TuneLatentSTE
per line 19 — pinned in the sibling change
`qk-fix-ste-sign-adaptation`; Figure 3's right panel (flame on O alone
after its factorization) confirms STE refinement is per-layer, on the
just-factorized layer.

## Why the objective is nonzero (prose is binding)

[NQ] Sec 3.2 Step 1 prose: TuneFP compensates "the error introduced by the
quantization of preceding blocks, as well as previously factorized layers
in the current block." With the Figure 3 placement, at TuneFP time the
attention projections are already factorized, so `block_tuned` (factorized
Q/K/V + tunable FP remainder O/G/U/D) deviates from the pristine teacher
by exactly the error those factorized layers introduced — nonzero
gradient, real compensation. The algorithm's line-12 placement is
schematic; at init there the objective is identically zero, which is
precisely the vacuity the guard below exists to expose (and the figure's
own encoding rules that placement out).

Cost: the teacher capture already exists for the `y_b = None` path
(`reconstruct_block` captures `y_b = block(x_b)` under `no_grad` before
tuning); the fix pins that capture as the teacher source, adds the single
block-level TuneFP pass between the attention projections and the output
projection inside the sequential factorization loop, and keeps one fp
forward per block for the teacher — budgeted, offline, same passes
as existing captures.

## Guard semantics (zero first-step gradient is always the defect)

Under the Figure 3 placement the TuneFP pass always runs with the
attention projections already factorized, so the first-step gradient MUST
be nonzero. A zero first-step gradient — whether at the legacy pre-loop
position (nothing factorized in-block, the shipped bit-identical
signature) or at the operative call (≥1 layer factorized but no
compensation path) — aborts with `tune_fp_vacuous_teacher` naming the
block and carrying BOTH the first-step gradient norm and the initial
residual norm: gradient-zero + residual-zero = broken teacher (target
equals current output — the shipped defect); gradient-zero +
residual-nonzero = detached student (no gradient path). Diagnostic runs
continue with pristine weights and the event on record. There is no
legitimate trivial-zero case under the figure-encoded placement, so the
guard has no carve-out.

## Remaining informed assumption (narrow)

The corpus names the TuneFP call but not its internal loss formula. This
design uses the block-output squared error against the cached teacher
(`‖block_tuned(x) − y_fp‖²`), the direct reading of the prose ("minimize
the error introduced by..."). Marker recorded in code and artifacts as
`tune_fp_loss: informed_assumption` citing the corpus block.
Wrong-assumption surfacing: the vacuous-teacher abort (case 2) plus the
nonzero-effect acceptance criterion.

## Reviewer amendments (2026-10-21, sec-eng integrity pass; folded by planner)

1. **A1 — teacher weights pinned to pristine, provenance recorded.** By
   block N the live model carries earlier blocks written back quantized, so
   the teacher block's weights SHALL load from the pinned pristine
   safetensors (or a frozen reference copy), never the live module's
   mutable state; the artifact records the source fingerprint and the
   capture index. Scope note (planner ruling): the input stream stays the
   compressed-prefix stream per the corpus (`X_b ← M̂_{<b}(X_cal)`); what is
   pristine-pinned is the teacher block's *weights*. The informed-assumption
   marker is only honest if the provenance it hangs on is recorded.
2. **A2 — teacher cache memory-only in v1.** The cached teacher is never
   persisted; on resume it is re-captured from the pinned pristine source.
   Any future persisted form is fingerprint-bound to the pristine source
   hash + capture index. (A stale restored teacher would be the third
   fail-silent surface of this class: ratio falsy-zero, empty-archive
   resume, stale-teacher cache.)
3. **A3 — two-norm event + fidelity invalidation.** `tune_fp_vacuous_teacher`
   carries the first-step gradient norm AND the initial residual norm:
   gradient-zero + residual-zero = broken teacher; gradient-zero +
   residual-nonzero = detached student (construction bug, no gradient path).
   Runs claiming paper fidelity treat any vacuous event as invalidating the
   claim (fail or `paper_faithful: false`); diagnostic runs keep the
   continue-with-pristine semantics of the guard section above.

## Discarded alternatives

- **Teacher = task loss / next-token logits.** Rejected for this change:
  couples the PTQ stage to an eval surface and reopens the probe-detection
  gate's territory; block-local objectives only (v1 scope).
- **Teacher = live module output recomputed per step (the shipped shape).**
  Rejected: self-referential — recomputing the teacher from the tuned
  module each step makes the difference identically zero; the measured
  defect.
- **Teacher on the unquantized FP-prefix stream (the former informed
  assumption).** Rejected: the corpus pins `X_b ← M̂_{<b}(X_cal)` (compressed
  prefix) and `Y_b ← B^FP_b(X_b)`; an unquantized-prefix teacher is a
  different mechanism the paper does not describe at this step.

## User directive (2026-10-21)

User: "Follow the paper TO THE LETTER; prove the paper's method is possible
first." Consequence: sequential in-block factorization + the Figure 3
TuneFP placement are in scope here (the former v1 non-goal is withdrawn);
the three fix dirs land before any further diagnostic runs; the ladder
fork fires only after a paper-faithful run exists.
