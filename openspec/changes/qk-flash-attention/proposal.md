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
  `quantibbler-kernels` as a NumPy tiled reference (f64) — the executable
  specification and parity oracle, per config's reference-first rule.
- Adopt `cubek-attention` 0.3.0 (tracel-ai/cubek — the CubeCL team's
  kernel collection; same `cubecl 0.11` we are pinned to, registry
  evidence 2026-10-06, 447k recent downloads, Apache-2.0, pure Rust) as
  the GPU implementation substrate behind the `gpu` feature: its forward
  pass ships causal masking, ragged tails, per-tensor element types, an
  accumulator-precision option, and a `cpu-reference` feature whose
  independent CPU implementation becomes a THIRD parity face.
- Two engine adapters bridge cubek's contract to ours: (1) GQA — cubek's
  `AttentionDims` carries ONE `num_heads` (no separate kv-heads), so the
  adapter materializes repeated K/V heads (correct output; the 2× KV
  traffic cost is part of what the perf probe must beat); (2) no softmax
  scale field — the adapter pre-scales Q by 1/√d in f32 (exact to
  round-off; verify no hidden scale at integration before trusting).
- Add the GPU attention face to the server engine (opt-in env, frozen
  decision at load, soft fallback to scalar, honest `/health` label) per
  the qk-attention 3.4 pattern.
- Parity regime per the 3.5 crystallization: this face REASSOCIATES, so
  the gate is bounded relative error on f32 (≤ 1e-5) against our NumPy
  reference — NOT bit-identity.

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
