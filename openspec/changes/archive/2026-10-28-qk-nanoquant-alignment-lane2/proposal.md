# qk-nanoquant-alignment-lane2 — proposal

## Why

The single-stream teacher capture makes TuneFP structurally unable to do the one
job the pinned paper names for it. [NQ] Step 1 is "Error Propagation Mitigation":
"We tune the full-precision weights of the current block to minimize the error
introduced by the quantization of preceding blocks, as well as previously
factorized layers in the current block" ([NQ] :525). Under today's capture —
`Y_b = B^FP_b(X_b)` with `X_b` the compressed-prefix activation ([NQ] :735,
:730) — the teacher target already *contains* the prefix's error, so the
preceding-block error term has zero representation in the TuneFP objective.
TuneFP can only ever chase in-block factorization error.

The same single stream is why target-1 TuneFP is vacuous: with nothing
factorized in-block, student ≡ teacher on the same `X_b`, the objective is
identically zero, and the qk-fix-tunefp-teacher ruling (a) simply skips the
call. That vacuity is an artifact of the capture, not of the paper's placement.

Lane 2 re-pins the capture to the paper's Step-1 intent: block reconstruction
advances TWO streams — an fp stream through pristine blocks (produces the
teacher target `Y_b = B^FP_b(X_fp)`) and a compressed stream through quantized
blocks (produces the student input `X_b`) — and the student sees `X_b`.

## What Changes

1. **Dual-stream advancement** — `reconstruct_model` advances an fp activation
   stream `x_fp` through the pristine frozen-reference blocks alongside the
   existing compressed stream `x_running`. Per block, `Y_b = B^FP_b(x_fp)`;
   the student still consumes `x_running` ([NQ] :730, :735 — feed re-pinned
   per Step-1 prose :525/:577; deviation recorded in design.md).
2. **Target-1 TuneFP vacuity disappears by construction** — at every block
   after the first the streams have diverged, so the objective at a block's
   first target is nonzero even with zero in-block factorization. Block 1
   (identity prefix) keeps its skip, now documented as the single precise
   case instead of a general vacuity.
3. **Guard stays a detector** — `tune_fp_vacuous_teacher` behavior is
   unchanged (zero-grad + zero-residual still means broken plumbing,
   fail-closed under `paper_faithful`); its documented rationale is rewritten
   to the dual-stream reading.
4. **`paper_faithful` re-pinned** — fp-stream teacher provenance becomes a
   label requirement; a legacy compressed-stream (or unprovenanced) capture
   denies the label. All other verdict conditions unchanged.
5. **Stream provenance in recorder events** — capture/tune_fp/ste/vacuous
   events name the stream that produced each referenced tensor.
6. **No new solver state** — probe signature and checkpoint fingerprint shape
   unchanged; the lane boundary changes code identity only, which existing
   stale-state protection already refuses.

## Impact

- `src/quantibbler/ptq/block_recon.py` — `reconstruct_model` capture loop gains
  the fp stream; `reconstruct_block` and `_tune_fp` accept and record stream
  provenance; `_tune_fp` docstring rationale rewritten.
- `src/quantibbler/ptq/config.py` — `paper_faithful` comment re-pinned. No new
  config field (fingerprint shape unchanged).
- `src/quantibbler/qk_report.py` — capture stage-map description reworded
  (fp-stream teacher outputs); verdict gains the fp-stream-provenance
  condition.
- `tests/ptq/test_dual_stream.py` (new), `tests/ptq/test_tunefp_teacher.py`,
  `tests/test_qk_report.py`, `tests/ptq/test_progress_probe.py`,
  `tests/ptq/test_checkpoints.py` — fixtures prove stream separation; guard,
  label, and shape assertions.
- Fidelity label: re-pinned to dual-stream semantics (see design.md).

## Non-goals

- Does NOT touch the kernels contract or the qpk1 storage format — not
  invariant-touching; no compatibility impact.
- Does NOT change the ADMM solver, TuneLatentSTE budget, or the per-target
  Figure 3 factorization cycle beyond the TuneFP capture feed.
- Does NOT add a config knob for stream choice — dual-stream is the one
  capture, not a run option.
- Does NOT re-derive prefix-probe constants (collapse line, pedestals) — probe
  shape and semantics unchanged.
- Does NOT re-order lanes: lane 2 is independent of lanes 1 and 3, and lands
  before any new production run.
