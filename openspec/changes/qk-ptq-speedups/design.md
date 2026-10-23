# qk-ptq-speedups — design

## Sequencing mandate (user directive, 2026-10-23)

GPU acceleration lands PRIOR to the next production run. The user has ruled the numerics question settled by the measured A/B campaign (fp32 GPU ≡ fp64 CPU within tolerance on tested layers). This is a SEQUENCING constraint, not a tolerance waiver: the fp32-GPU-vs-fp64-CPU parity scenarios remain the acceptance gate, and the torch-autograd stages become the production path only after per-stage parity passes.

## min_eigs gating rationale

- Benchmark (measured, not comment-inferred): 2× `eigvalsh(512×512)` ≈ 142 ms/iteration at 16 OpenBLAS threads vs 62 ms for the 2× matmul + 2× Cholesky it sits next to; the code's "cheap r × r" comment is wrong at production thread settings.
- Redundancy: `_cholesky_solve` immediately below IS the SPD proof, and `trace.n_jitter` already records every non-SPD event — so the gated instrument guards nothing the always-on path misses.
- Scale note: fixed per-linear tax that does not shrink while GEMMs do — wrong cost to carry into a 4B run.

## STE/TuneFP GPU port rationale

- Pure torch autograd, zero device references in `block_recon.py` — GPU-movable with no algorithm change (sanctioned by config.yaml GPU policy: compressor/trainer on PyTorch).
- fp32 on gfx1151 is the earning configuration; fp64 throughput is ~1/16 of fp32, making fp64 GPU a wash against the 32-core CPU.

## Deviation log

- None of the changes in this set alter solver arithmetic or optimizer configuration; the gating is behavior-neutral by scenario (bit-identical fingerprint, flag off vs on).
- The measured per-stage wall-time deltas are recorded in the run report and the commit body at landing time, superseding the proposal's cost-model estimates.
