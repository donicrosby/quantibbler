# Proposal: qk-fix-tunefp-teacher

## Why

[NQ] Sec 3.2 Step 1 ("Error Propagation Mitigation") is binding prose: tune
the full-precision weights of the current block to minimize the error
introduced by the quantization of preceding blocks and previously factorized
layers in the current block. The shipped objective never does that. The only
production call site (`block_recon.py` resume loop) passes
`y_b = block(x_running)` and `x_b = x_running.clone()` — the teacher is the
same module's own output on the same input, so `_tune_fp`'s objective
`‖block(x_b[idx]) − y_b[idx]‖²` is zero at init, its gradient is zero, and no
step is ever taken. Measured on real bytes by the stage trace
(`docs/rca/2026-10-21-stage-trace-verdict.md`): T0 bit-identical to pristine,
`rel_err_vs_pristine = 0.0`, output error 0.0, on all 7 scoped linears, both
families. The pipeline has never executed [NQ] Step 1 in any archived run.

It is a **compounding** defect, not the sole cause: each block's fp weights
never compensate quantized-prefix drift, but the weight-space audit shows the
depth-2 cliff needs more than a missing pre-tune.

**Citation, verified on the pinned corpus bytes (2026-10-21):** the
Algorithm 1 *figure text* IS carried by the pinned conversion
(`docs/papers/nanoquant-2602.06694v3.md`, algorithm block). It pins the
Step-1 teacher construction: `X_b ← M̂_{<b}(X_cal)` (input after the
already-compressed prefix) and `Y_b ← B^FP_b(X_b)` (teacher = the pristine
FP block's output on that compressed-prefix stream), then
`TuneFP(b, X_b, Y_b; T_pre)`. The teacher is therefore **corpus-pinned,
not an informed assumption**. The one narrow remainder: the corpus names
the TuneFP call but not its internal loss formula — that stays an
informed-assumption marker, recorded in code and artifacts.

## What Changes

- `_tune_fp` minimizes against a **corpus-pinned cached teacher that does not
  move with the tuned weights**: `Y_b = B^FP_b(X_b)` — the pristine FP
  block's outputs on the compressed-prefix stream `X_b`, captured once per
  block before any tuning, detached, gradients disabled ([NQ] Algorithm 1,
  carried by the pinned conversion). The block's remaining FP weights are
  then tuned so its output on the quantized-prefix stream reproduces `Y_b`.
- In-block factorization is **sequential** ([NQ] Sec 3.2: "sequentially
  compress each linear layer in each transformer block"), and TuneFP fires
  **once per block** ([NQ] Algorithm 1 line 12) at the **Figure 3
  placement**: after the attention projections (Q/K/V) are factorized and
  before the remaining linears (O/G/U/D) are, tuning the still-FP
  remainder so it compensates previously-factorized layers in the current
  block — the objective is nonzero by construction. The figure's left
  panel encodes exactly this (Q/K/V green, tuning flames on the still-FP
  remainder); the
  line-12-before-everything placement — objective identically zero at
  init — is the measured defect and is removed.
- A **runtime contract catches vacuous teachers**: if the pre-tune objective
  gradient norm is zero at the first step, the stage aborts with a recorded
  event naming the vacuous teacher and carrying BOTH the gradient norm and
  the initial residual norm (broken-teacher vs detached-student); a run
  claiming paper fidelity fails on any vacuous event — never a silent no-op
  (the same fail-closed class as the fp-resume guard change).
- The teacher block's weights load from the **pinned pristine safetensors**
  (or a frozen reference copy), never live module state; the teacher cache
  is **memory-only in v1** and re-captured on resume; the artifact records
  the source fingerprint + capture index.
- The recorder/artifact records `tune_fp_effect` honestly (weight/output
  delta vs pristine) plus teacher provenance and the informed-assumption
  marker.
- Defaults (T_pre, lr, cosine schedule) stay [NQ] App. C; no budget change.

## Impact

- **Affected specs**: `qk-ptq`, one ADDED requirement only (sibling changes
  `qk-nq-recreation-bar`, `qk-divergence-guards` etc. carry live deltas under
  the same capability — no MODIFIED/REMOVED, per the parallel-changes rule).
- **Affected code**: `src/quantibbler/ptq/block_recon.py` (`_tune_fp`, capture
  path, call site), the driver's calibration capture (FP-prefix pass), tests.
- **Not invariant-touching**: no qpk1 format, kernels, or server change.

## Non-goals

- No change to Step 2 (LB-ADMM) or Step 3 (STE) budgets in this change.
- No stage budget/lr retuning in this change.
- No change to `factor_mode=fp` behavior (see `qk-fix-fp-resume-guard`).

## User directive (2026-10-21)

User: "Follow the paper TO THE LETTER; prove the paper's method is possible
first." Consequence: sequential in-block factorization + the Figure 3
TuneFP placement move INTO scope here (the former v1 non-goal is
withdrawn); this change and its siblings (`qk-fix-ste-sign-adaptation`,
`qk-fix-fp-resume-guard`) land before any further diagnostic runs; the
ladder fork fires only after a paper-faithful run exists.
