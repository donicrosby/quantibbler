# qk-flash-attention — Design

## Method provenance (rule: every claim cited)

- Tiling + online softmax + rescaling identity: [FA] §3.1, Algorithm 1 —
  running statistics m (block max) and ℓ (block sum of exp), each new block
  rescales prior accumulators by exp(m_old − m_new). Implemented verbatim;
  no algorithmic novelty is claimed.
- IO-complexity motivation: [FA] §3.2, Theorem 2.
- Numerical-stability form of softmax (subtract max before exp): [FA] §3.1.
- f32 compute substrate: engine activations are f32 (pinned qk-server
  convention); RDNA3 fp32 measures 4.80 TFLOP/s (2026-10-18 rocBLAS probe)
  vs 0.44 TF fp64 — the rate argument that does NOT exist for the solver.

## Discarded alternatives

- **Port FlashAttention CUDA/ROCm reference kernels** — banned: config
  forbids C/C++/CUDA-C++ in the build graph; evidence rule says papers
  only. Discarded, not deferred.
- **rocBLAS-based split (two sgemms + fused softmax kernel)** — simpler
  kernels but materializes S per block and doubles global-memory traffic;
  [FA] §3.1's fusion is precisely the win being specified. Revisit only if
  the tiled kernel misses its perf goal.
- **f64 attention for parity bit-identity** — would throw away 11× of
  measured hardware rate to satisfy the wrong gate; per the 3.5 lesson the
  gate follows the arithmetic class (reassociating ⇒ bounded rel error),
  not the other way round.
- **kv-split / flash-decoding variants** — speculative scheduling beyond
  [FA] v2's pinned text; out of scope by the evidence rule.

## Informed assumptions

- **Assumption:** a naive-tiled first kernel (one workgroup per Q block,
  K-loop inside) reaches ≥ 1.5× the scalar CPU face on decode-shaped
  attention (seq ≤ 2k, d=128, GQA 8:1) on gfx1151. Rationale: 4.8 TF fp32
  vs measured ~0.46 TF effective CPU, with attention's arithmetic intensity
  amortizing launch overhead. **Wrong-if:** it surfaces as the perf probe
  (task 5) failing to clear 1.5× — then the face stays opt-in and the
  change records the miss instead of shipping a regression.
- **Assumption:** CubeCL f32 `Tensor` indexing/`ABSOLUTE_POS` forms from
  the landed gpu.rs (tasks 3.1–3.4) extend to 2D tiling without new API
  surface (shared-memory arrays via `cubecl::std` comptime-sized arrays).
  **Wrong-if:** compile failures force new forms — captured as
  crystallization entries, cost bounded by the proven 0.11 pin.

## Open Questions

- OQ-1 (owner: orchestrator, probe = task 5): does the tiled face clear
  1.5× CPU decode attention on gfx1151 at production shapes? Status: OPEN
  — resolved only by the timed probe, never by argument.
- OQ-2 (owner: orchestrator): workgroup/tiling constants (B_c, B_r, d
  specialization) — opened during implementation, closed by measurement
  over the fixture battery; defaults follow [FA] Algorithm 1 line 1's
  SRAM-size formula with gfx1151 LDS limits substituted.
