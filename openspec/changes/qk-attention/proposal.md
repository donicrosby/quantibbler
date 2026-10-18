# Change: qk-attention

## Why

The server's engine is a linear stack: it loads qpk1 factorized layers and
projects through them, but a real transformer needs attention (RoPE, causal
masking, GQA) and normalization (RMSNorm, SwiGLU) around those layers. The
first full-pipeline acquisition (Qwen3-0.6B r8, 2026-10-18) proved the
artifact contract for factorized weights AND exposed what is missing:
per-layer RMSNorm weights, attention geometry, and rope/eps hyperparameters
are not in the exported model directory, so no correct forward pass can be
assembled from the artifact alone.

## What Changes

- **Model directory contract (BREAKING delta to qk-server model-dir):**
  export and loader carry a `norms.npz`-style fp32 sidecar with the per-layer
  input/post-attention norms, the final norm, and a `config.toml` extension
  (`n_heads`, `n_kv_heads`, `head_dim`, `rms_norm_eps`, `rope_theta`,
  `n_layers`), all validated fail-closed at load.
- **Engine (qk-server):** token-streaming decode over the full transformer
  block semantics — RMSNorm, GQA causal attention with RoPE, SwiGLU MLP,
  residuals — against the packed factorized projections, in f32, greedy and
  temperature sampling, with a KV cache.
- **GPU kernels (qk-kernels):** CubeCL HIP kernels for the hot packed
  operations behind a cargo feature `gpu`, CPU fallback bit- preserved;
  runtime-validated against the rlib implementations (the parity gate goes
  through the GPU when the feature is on and a device is present).

## Impact

- Affected specs: `qk-server` (model directory + engine), `qk-kernels`
  (dispatch adds a HIP backend), `qk-representation` (untouched).
- Affected code: `tools/quantize_hf_model.py` (export), `crates/quantibbler-qpk1`
  (loader), `crates/quantibbler-server/src/engine/`, `crates/quantibbler-kernels`.
- Execution proof (2026-10-18): CubeCL 0.11 HIP JIT kernel passes on the
  gfx1151 device in-container; recipe in agent memory and in the probe
  sources at `/opt/data/tmp/cubecl-probe`.
