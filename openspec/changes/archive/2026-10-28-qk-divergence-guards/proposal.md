# Proposal: qk-divergence-guards

## Why

The first production-rank run (Qwen3-0.6B, rank 137, 2026-10-19) completed
mechanically clean in 59 minutes but produced a broken artifact:
`ppl_quant = 33.2M` against a baseline of 28.67. The per-layer audit of the
exported qpk1 scales localized all of the damage to **block 0**: 5 of 7
linears exported scale vectors ~4 orders of magnitude above the healthy
population (s1 median ~21 vs ~1e-4; reconstructed `|Wq|` entries ~250x the
fp weights), while layers 1-26 are structurally coherent. Downstream blocks
rebuild on block 0's output, so the explosion compounded into
random-logits perplexity.

Root cause chain (measured + code-read):

1. `robust_diag(tau=None)` — the driver's default — clips raw diagonal
   statistics to `max(D)` **per vector**, which is a mathematical no-op.
   A-1's "absolute cap" protection is disabled on the default path.
2. Block 0's calibration inputs are raw unnormalized embedding outputs
   (the trunk's first block sees no prior normalization of the calibration
   stream); its raw statistics are heavy-tailed (p99/median ratios recorded
   in the probe, see design.md).
3. Under such statistics the ADMM solve lands a diverged basin; the
   magnitude-balance step (Eq. 7-9) then amplifies the diverged proxies
   into enormous scales.
4. Nothing in the pipeline checks the reconstructed weight against the
   fp target, so the layer shipped, the export ran to completion, and the
   failure only surfaced at minute 60 in the perplexity stage.

Rank-8 evidence: the archived r8 artifact's block 0 shows the same failure
mode in milder form (scales elevated ~10x, artifact PPL garbage at its
0.0345 BPW smoke point), so this is not a rank-137-only defect — block 0
is the fragile end of the pipeline at every rank.

## What Changes

- `robust_diag`'s default ceiling policy changes from the per-vector
  no-op (`max(D)`) to a finite robust ceiling derived per layer from the
  raw statistics, so A-1's absolute cap actually engages on heavy tails.
  The estimator is pinned from the block-0 treatment screen (design.md).
- The pipeline gains a **post-balance sanity gate**: after magnitude
  balancing each layer, the reconstructed binary weight's median absolute
  entry is compared against the fp weight's; a divergence beyond a stated
  factor band aborts the run naming the layer — at minutes, not at the
  perplexity stage.
- The gate failure is a hard error (nonzero exit, layer named in the
  message, telemetry emitted) BEFORE any export or dataset stage runs.
- `tools/run-ptq-container.sh` thread pinning is re-benchmarked against
  the container's `--cpus 24` quota (16 vs 24 BLAS threads) and pinned to
  the measured winner.

## Impact

- **Affected specs**: `qk-ptq` (preconditioner requirement MODIFIED — the
  tau policy; new post-balance gate requirement ADDED).
- **Affected code**: `src/quantibbler/ptq/calibration.py` (tau default),
  `src/quantibbler/ptq/block_recon.py` (gate invocation after balance),
  `src/quantibbler/ptq/config.py` (gate band knob), `tools/run-ptq-container.sh`
  (thread pin), `tools/quantize_hf_model.py` (error surfacing only).
- **Not invariant-touching**: the qpk1 container, the packed bitplane
  layout, and the kernels contract are untouched. This is compressor-
  internal numerics and pipeline control flow.
- **Operational impact**: an r137 rerun is required to produce the real
  artifact; the wounded artifact at `artifacts/qwen3-0.6B-r137` remains on
  disk as the divergence evidence copy.

## Non-goals

- No change to the qpk1 format, the bitplane packing, or any reader.
- No adaptive-rho schedule work: the linear penalty schedule (A-2) is
  paper-pinned; the tau cap addresses the measured mechanism. Adaptive
  rho stays a recorded open question for a future change if the tau
  treatment underperforms.
- No treatment of the flat-spectrum quality floor (down_proj
  rank-approximability): that is a modeling-expectation problem for the
  rank/BPW target choice, not a divergence defect.
- No attempt to make block 0's input distribution different (no extra
  normalization of calibration inputs) — that would change the algorithm's
  contract with the teacher, not just its numerics.
