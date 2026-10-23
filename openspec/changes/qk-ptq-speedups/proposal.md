# qk-ptq-speedups — proposal (DRAFT, research-grounded, lands only after the fire's verdict)

## Why

The paper-faithful run costs ~21.8h wall at paper geometry (measured stage
constants, seq 2048/calib 128). Cost model from probe + fire constants:

| stage | cost/unit | units | total | share |
|---|---|---|---|---|
| TuneFP | ~133 s/fire | 168 (N−1 × 28 blocks) | ~6.2h | 29% |
| STE | ~154 s/linear | 196 (N × 28 blocks) | ~8.4h | 39% |
| ADMM | ~87 s/linear | 196 | ~4.7h | 22% |
| capture/stats/eval/report | — | — | ~2.5h | 11% |

ADMM is weight-space and geometry-independent; STE/TuneFP/eval are
activation-space and scale with seq_len.

## Findings (paper-grounded)

1. **STE + TuneFP are pure torch autograd** (`block_recon.py` — no CUDA/device
   references exist anywhere in `src/quantibbler/ptq/`). They are GPU-movable
   with zero algorithm change. Sanctioned by config.yaml GPU policy:
   "Compressor/trainer: PyTorch (training and QAT autograd)."
2. **The paper ablates its own speed levers** ([NQ] App D.4, Figure 9):
   - "Reducing the number of ADMM iterations gives faster convergence but
     leads to a higher final reconstruction error."
   - "Linear scheduling converges more slowly but achieves lower final error
     than more aggressive schedules."
   An opt-in speed profile (fewer iterations + aggressive schedule) is the
   paper's own documented tradeoff, not an invention.
3. **ADMM is dense small-matrix linear algebra in float64** (r×r Cholesky +
   trsm, O(r²·d_out) per iteration). GPU speedup depends on FP64 hardware
   class: datacenter NVIDIA (A100/H100) 20–50×; consumer NVIDIA/AMD (FP64
   crippled at 1/32–1/64 rate) ~5–10×.
4. **The per-target cycle is strictly serial** — TuneFP(i+1) computes
   residuals through packed targets 1..i, so no block-internal overlap;
   ADMM(i) consumes TuneFP(i)'s tuned weight. Parallelism across blocks is
   impossible (prefix dependency). Speedups must come from device or
   per-iteration cost, not scheduling.
5. **Batch-1 STE is latency-bound on GPU** — expected 10–30× may degrade to
   ~5–10× from kernel-launch overhead. Still the largest single lever
   (8.4h).

## Proposed levers (gated on fire verdict)

- **L1 (GPU autograd):** device plumbing for capture, TuneFP, STE, eval.
  float32 activations stay; solver untouched. Est. 14.6h → ~0.7–1.5h.
- **L2 (speed profile):** `speed_profile: "paper" | "fast"` — fast = fewer
  ADMM iterations + aggressive schedule per [NQ] D.4; named deviation in the
  run record with the paper's own quality curve cited. Est. 4.7h → ~2.3h
  (CPU), combined with GPU ~5–15min.
- **L3 (paper-matched rank):** ADMM scales r². Confirm the paper's 0.6B/1B
  rank (Table 1 BPW rows 0.80/0.55 exist; per-model rank is in figure text)
  — if it is below 512, matching it cuts solver cost 4× and storage
  proportionally.

## Non-goals

- No algorithm changes to STE/TuneFP/balance/pack semantics.
- No float32 default for the solver (config.yaml invariant).
- No reduction of STE/TuneFP step budgets (App. C prescribes them).
- Nothing lands until the fire completes and its verdict is read.

## Open questions

- Fire verdict: does the run survive depth 2 at paper geometry? Speedups
  are worthless if fidelity fails first.
- Target GPU hardware: homelab AMD consumer vs rented datacenter NVIDIA
  changes L2's expected gain by an order of magnitude. Owner: @user.
