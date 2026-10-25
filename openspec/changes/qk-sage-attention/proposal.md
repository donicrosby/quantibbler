# Proposal: qk-sage-attention — INT8 quantized attention face (CubeCL)

## Why

Once `qk-flash-attention` lands the tiled exact face, [SA] §4 defines how to
quantize the tiles for another large win: K's channel-wise outliers are a
per-key bias shared across tokens, so the transform γ(K) = K − mean(K)
(Eq. 6) removes them with NO loss — softmax shift invariance makes
σ(q(K−mean(K))ᵀ) = σ(qKᵀ) exactly ([SA] §4.2). The remaining signals
quantize to INT8 (per-token Q/K; per-block static-scale P̃ whose row max is 1
by construction; per-channel V) with an FP32 accumulator ([SA] §4.3, their
Table 4). Locally the mechanism to verify is NOT the paper's tensor-core
count but RDNA3's byte-domain INT8 dot instructions plus halved K/V memory
traffic — an assumption this change is built to probe, not to presume.
Substrate note (2026-10-19): `cubek` 0.3.0 — the CubeCL team's kernel
collection, verified to resolve on our exact `cubecl 0.11` pin — ships
`cubek-quant` with symmetric per-block/per-tensor quantization primitives
that [SA]'s scheme composes from; adopted for those primitives in the
sibling change, with the INT8 attention loop remaining our code on the
same scaffold.

## What Changes

- Add a quantized tiled attention REFERENCE to the Python kernels package:
  [SA] §4 scheme exactly — per-K-block γ, symmetric per-token INT8 Q/K,
  P̃ quantized at static scale 1/127, per-channel V, softmax statistics and
  output accumulation in the reference's f64 (emulating the kernel's f32
  accumulator semantics).
- Adopt the sibling's substrate path: [SA]'s quantization primitives map to
  `cubek-quant` 0.3.0 (symmetric quantization with per-block and
  per-tensor granularity, q2/q4/q8/fp4 variants — same crate family, same
  `cubecl 0.11` pin) for the device-side quantize/dequantize primitives,
  layered on the sibling change's attention scaffold. The tiled attention
  loop with online softmax stays ours (cubek-attention is fp-path only);
  its parity harness and gate patterns are reused unchanged. If
  `cubek-quant`'s granularity axes don't cover per-token-K/post-softmax-P̃
  at integration, hand-rolled quantize epilogues on the landed forms are
  the fallback — the gates do not care which substrate computes.
- Add the `gpu:sage` server attention face: opt-in env, frozen at load,
  soft fallback chain sage → tiled → scalar, honest `/health` label.
- Gate the face on MEASURED quality, per [SA] §4.3's own metrics (Cosine
  Sim / Relative L1 / RMSE vs the exact face) on captured REAL activations
  plus a quantized-PPL delta bound on the eval set — the paper's Table 3
  shows worst-case layers cratering to 56–79% cos-sim when outlier handling
  is wrong, so the gate is on our activations, not their tables.

## Non-goals

- No FP8 (E4M3/E5M2) variants — [SA] §4.3 records them as INT8
  alternatives; RDNA3 has no fp8 dot path. Discarded, not deferred.
- No per-channel Q/K — impossible at the inner axis of QKᵀ ([SA] §4.3,
  reason 1). Discarded by the paper's own argument.
- No port of SageAttention's kernels or its CUDA/ Triton schedule — banned
  by config (no C/C++/CUDA-C++); papers-only evidence rule.
- No training-time attention quantization / QAT — inference face only.
- No serialization format change: INT8 tiles and scales are ephemeral
  in-kernel; qpk1 is untouched (invariant noted per rules).
- This change does NOT land unless its sibling's scaffold exists: the tiled
  exact face and its parity harness are the foundation (dependency noted in
  tasks; do not start this change before `qk-flash-attention` lands).

## Impact

- Specs: qk-kernels +2 requirements (quantized reference, INT8 kernel),
  qk-server +4 requirements (sage face admission, PPL-clause-absent
  admission, quality-gate provenance, serving parity).
- Code: `src/quantibbler/kernels/attention_sage.py` (reference), the
  kernels crate's gpu_attention module (INT8 path), engine face enum.
- Invariant note: extends the attention-face surface pinned by
  `qk-flash-attention`; the EXACT faces stay the default and the fallback
  root. No storage-format impact.
