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

## Why the objective is nonzero (prose is binding)

[NQ] Sec 3.2 Step 1 prose: TuneFP compensates "the error introduced by the
quantization of preceding blocks, as well as previously factorized layers
in the current block," and Sec 3.2 opens with "We sequentially compress
each linear layer in each transformer block." Therefore TuneFP runs per
layer ℓ: layers <ℓ of the current block are already factorized, so
`block_tuned` (factorized in-block layers + tunable FP remainder) deviates
from the pristine teacher by exactly the error those factorized layers
introduced — nonzero gradient, real compensation. The algorithm's
once-per-block TuneFP line is schematic; at init there the objective is
identically zero, which is precisely the vacuity the guard below exists to
expose.

Cost: the teacher capture already exists for the `y_b = None` path
(`reconstruct_block` captures `y_b = block(x_b)` under `no_grad` before
tuning); the fix pins that capture as the teacher source, adds the
per-layer TuneFP call inside the sequential factorization loop, and keeps
one fp forward per block for the teacher — budgeted, offline, same passes
as existing captures.

## Guard semantics (trivial-zero vs vacuous-zero)

Two distinct zero cases, handled differently:

1. **First in-block layer, nothing factorized yet** — objective zero at
   init BY CONSTRUCTION (the block still equals the teacher). Expected:
   record `tune_fp_trivial_first_layer`, proceed. Not an abort.
2. **≥1 in-block layer already factorized and the first-step gradient is
   still zero** — the teacher cannot produce compensation, or the teacher
   moves with the tuned weights (the shipped defect signature). Abort with
   `tune_fp_vacuous_teacher` naming the block; continue with pristine
   weights and the event on record.

## Remaining informed assumption (narrow)

The corpus names the TuneFP call but not its internal loss formula. This
design uses the block-output squared error against the cached teacher
(`‖block_tuned(x) − y_fp‖²`), the direct reading of the prose ("minimize
the error introduced by..."). Marker recorded in code and artifacts as
`tune_fp_loss: informed_assumption` citing the corpus block.
Wrong-assumption surfacing: the vacuous-teacher abort (case 2) plus the
nonzero-effect acceptance criterion.

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
first." Consequence: sequential in-block factorization + per-layer TuneFP
are in scope here (the former v1 non-goal is withdrawn); the three fix dirs
land before any further diagnostic runs; the ladder fork fires only after a
paper-faithful run exists.
