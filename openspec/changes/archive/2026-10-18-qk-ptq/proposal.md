# Change: qk-ptq — PTQ pipeline (NanoQuant path)

## Why

quantibbler ships two methods for sub-1-bit weight compression; the PTQ path ([NQ] Sec 3) is the one that needs no training loop, so it is the first path a new user can exercise end to end. The repo today has kernels scaffolding and a binding config contract but no specification for the pipeline that turns a caller-supplied fp model plus calibration data into a serialized qpk1 artifact. Without this change, implementation work on the PTQ path has no agreed requirements, no verification gate, and no recorded home for the paper-derived invariants — LB-ADMM initialization, block-wise reconstruction, global scale calibration ([NQ] Sec 3) — that the sibling capabilities (qk-representation, qk-kernels) must interoperate with.

## What Changes

- New capability `qk-ptq` (this change), specified as ADDED requirements:
  - PTQ entry point `quantize_model(model, calibration_data, config)` with caller-supplied model and calibration data (no-network rule).
  - Phase 1 global calibration: per-layer activation/gradient statistics and robust diagonal preconditioners ([NQ] Step 2-1).
  - Per-layer LB-ADMM initialization of (Ub, Vb, s1, s2): preconditioning, stabilized-Cholesky continuous updates, SVID proxies, linear penalty schedule, latent magnitude balancing ([NQ] Steps 2-2, 2-3).
  - Phase 2 block-wise reconstruction: quantized-prefix inputs, teacher targets, error-propagation mitigation, STE factorized refinement, sign extraction and packing ([NQ] Sec 3.2, Algorithm 1).
  - Phase 3 global scale calibration: KL objective on logits, only s1/s2 trainable, binaries bit-frozen ([NQ] Sec 3.3).
  - float64-default numerics policy; qpk1 serialize/reload round trip; tiny-model end-to-end gate (bounded logit delta, BPW < 1.0 measured from bytes).
- Consumes sibling capabilities BY NAME, without modifying them: `qk-representation` (qpk1 container, BPW-from-bytes accounting) and `qk-kernels` (pack/unpack, parity-gated dispatch). `sign_estimate` and `joint_itq` remain in the `qk-kernels`/QAT lane's scope and are not consumed here.
- No implementation in this change: proposal, design, tasks, and the delta spec only.

## Impact

- **Affected specs:** creates `specs/qk-ptq/spec.md` on archive. No existing spec is modified (the main `openspec/specs/` tree is empty today).
- **Affected code (future tasks):** new `quantibbler.ptq` subpackage (NumPy reference path); no changes to `quantibbler.kernels` dispatch contract.
- **Cross-lane:** downstream implementation consumes `qk-representation` accounting for the BPW gate and `qk-kernels` packing for bit identity; any drift in those capabilities will surface in this capability's round-trip and gate requirements.
- **Not invariant-touching:** this change does not alter the kernels contract or the storage format; it consumes the qpk1 container as specified by `qk-representation`.

## Non-Goals

- **QAT path** — SVD (Dual-SVID) / Joint-ITQ init, straight-through sign-estimation training, multi-scale compensation, distillation-recovered training ([LB] Sec 3; [LB2]) is a separate future change. This change covers PTQ only.
- **GPU acceleration** — kernels stay CPU-only in v0.1; GPU/array compute stays in PyTorch per config.yaml architecture invariants.
- **Model zoo / download integration** — no network access at runtime; fetching models and datasets is the caller's job, documented but not built.
- **Inference kernels and serving** — binary GEMV/GEMM kernels and a runtime inference engine are out of scope; this change produces quantized artifacts.
- **Evaluation harness** — wikitext-2 perplexity and zero-shot benchmark tooling are caller-driven; the library documents the recipe ([NQ] Sec 4.1) but ships no eval framework.
