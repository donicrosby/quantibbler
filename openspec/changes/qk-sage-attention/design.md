# qk-sage-attention — Design

## Method provenance (rule: every claim cited)

- γ(K) = K − mean(K) and the shift-invariance exactness argument: [SA]
  §4.2, Eq. 6. Derived property we rely on: the mean may be taken over the
  K-block currently in flight (any constant shift preserves softmax), so
  the kernel never needs a whole-cache pass — this is algebra, not
  assumption.
- Granularity and dtypes: [SA] §4.3 — Q, K per-token INT8 (per-channel is
  structurally impossible at QKᵀ's inner axis); P̃ per-block with STATIC
  scale 1/127 (row max of P̃ = exp(s − rowmax) is 1 by construction); V
  per-channel INT8; FP32 accumulator ([SA] §4.3, Table 4: fp32 acc is the
  most accurate of their variants).
- Quality metrics (Cosine Sim, Relative L1, RMSE vs fp attention):
  [SA] §4.3 "Accuracy metrics" — adopted verbatim as our gate metrics.
- Worst-case warning that motivates the real-activation gate: [SA] Table 3
  (INT8 rows degrading to 56–79% cos-sim on outlier layers).

## Discarded alternatives

- **FP8 Q/K or V** — [SA] §4.3 accuracy is worse than INT8 for Q/K, and
  RDNA3 exposes no fp8 dot instruction; the paper's fp8 rationale
  (2×-slower-than-int8 on NVIDIA) does not transfer. Discarded.
- **Whole-cache mean(K) (precomputed at load)** — breaks with KV-cache
  growth per token and adds a per-step recompute; per-block mean is exact
  by shift invariance and free. Discarded for the per-block form.
- **Post-hoc static calibration of Q/K scales** — [SA] uses dynamic
  per-token scales inside the kernel; static scales would deviate from the
  pinned text to save nothing measurable. Discarded.
- **INT8 PV accumulation (i32 accumulator into softmax)** — P̃·Vᵀ must
  dequant per-channel V in f32 and the online-softmax statistics are f32;
  integer accumulation there would corrupt m/ℓ. Discarded; i32 appears only
  as the intermediate of INT8 dot products before f32 scaling.

## Informed assumptions

- **Assumption:** RDNA3 (gfx1151) byte-domain INT8 dot (`v_dot4`) yields
  ≥ 2× the fp32 dot rate at attention shapes, making the quantized path
  faster than the exact tiled face, not just narrower. Rationale: RDNA3
  ISA exposes vector byte-dot instructions; [SA]'s own speed mechanism is
  INT8 dot-rate. **Wrong-if:** the task-4 microbench measures < 2× — then
  the change records the number and the face justifies itself (or not) on
  bandwidth alone; no speed claim ships unmeasured.
- **Assumption:** the quality gate on fixture + real activations
  (cos-sim ≥ 0.999, rel-L1 ≤ 0.02 class) predicts acceptable generation
  quality on the r137 artifact. **Wrong-if:** quantized-PPL delta vs the
  exact face exceeds the recorded bound — the face stays
  opt-in-experimental with the failure recorded, per the same discipline
  as the solver's fp32 discussion: accelerators are opt-in, never silent.
- **Assumption:** cubek-quant 0.3.0's symmetric quantization primitives
  (per-block / per-tensor, q8) cover or compose into [SA]'s per-token-Q/K,
  static-P̃, per-channel-V axes on device. **Wrong-if:** the task-2.1
  compile/run probe shows a granularity mismatch — fall back to hand-rolled
  quantize epilogues on the landed gpu.rs forms (the i8/u32-packed ladder),
  recorded as crystallization; the parity gates are substrate-agnostic.

## Open Questions

- OQ-1 (owner: orchestrator, probe = task 4.1): INT8 dot throughput on
  gfx1151 vs fp32 — resolves the speed mechanism. OPEN.
- OQ-2 (owner: orchestrator, probe = task 4.2): quality-gate numbers on
  real r137 activations + PPL delta — resolves ship/experimental status.
  OPEN.
