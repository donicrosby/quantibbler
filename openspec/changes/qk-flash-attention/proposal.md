# Proposal: qk-flash-attention — tiled exact attention face (CubeCL)

## Why

Server decode is attention-bound as context grows: the scalar CPU attention
materializes S = QKᵀ per query position and reads/writes it per layer. The
GPU evidence from 2026-10-18 reverses the solver verdict for this workload:
fp32 sgemm on the Radeon 8060S measures 4.80 TFLOP/s (rocBLAS via the
sanctioned rocm/pytorch container) vs 0.44 TF fp64 — engine activations are
f32, so attention runs at FULL RDNA3 rate, unlike the fp64 solver. [FA]
(§3.1) defines the exact algorithm that converts that rate into wall-clock:
tile Q/K/V into SRAM-sized blocks, carry online softmax statistics
(m, ℓ) across blocks with the rescaling identity, never materialize the
N×N score matrix. [FA] Theorem 2 bounds the IO win.

## What Changes

- Add a tiled, online-softmax, EXACT attention forward to
  `quantibbler-kernels` as a CubeCL kernel (f32 compute, per the paper's
  Algorithm 1: tiling + running max/sum statistics + output rescaling),
  with a NumPy tiled reference as the executable specification per config.
- Add a GPU attention face to the server engine behind the `gpu` feature +
  an env opt-in, reusing the frozen-decision dispatch pattern proven for
  matvec (qk-attention 3.4): face selected once at model load, soft-fallback
  to the scalar face on init failure, honest `/health` label.
- Parity regime shifts per the 3.5 crystallization: this kernel REASSOCIATES
  (blockwise accumulation), so the gate is a bounded relative error on f32
  (≤ 1e-5) against the existing scalar face — NOT bit-identity, which is
  achievable only for order-preserving faces.

## Non-goals

- No backward pass, dropout, or training path — the server is inference
  only; [FA] §3.1 "Recomputation" is cited for context, not implemented.
- No quantized attention here — that is the sibling change
  `qk-sage-attention` ([SA] §4), which reuses this kernel's tiling scaffold
  and parity harness.
- No C/C++/CUDA-C++: config bans them from every build graph. This is a
  CubeCL reimplementation of the paper's algorithm, never a port of any
  reference kernel (evidence rule: papers only).
- No change to engine forward conventions: the kernel consumes exactly what
  `gqa_attention` consumes today (post-RoPE q/k, post-norm v, GQA groups,
  causal mask, KV-cache read) — conventions stay pinned in qk-server spec.
- Not default-on: the scalar face stays the default until the parity gate
  AND a measured decode-attention win on gfx1151 both exist.

## Impact

- Specs: qk-kernels +2 requirements (tiled reference, CubeCL kernel),
  qk-server +2 requirements (attention face dispatch, serving parity).
- Code: `crates/quantibbler-kernels/src/gpu_attention.rs` (+tests),
  `src/quantibbler/kernels/attention_tiled.py` (reference), engine wiring.
- Invariant note (per rules): the engine forward conventions are
  invariant-adjacent — this change must PROVE preservation (rel ≤ 1e-5
  logits on the fixture pipeline), not modify them. No storage-format or
  qpk1 impact.
