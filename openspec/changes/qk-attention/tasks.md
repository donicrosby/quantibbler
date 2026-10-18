# qk-attention Tasks

## 1. Export + loader contract (norm sidecar + geometry)

- [x] 1.1 Driver export: write `norms.bin` (fp32 LE: per-block input_layernorm,
      post_attention_layernorm, then final norm) and extend `config.toml` with
      `n_layers`, `n_heads`, `n_kv_heads`, `head_dim`, `rms_norm_eps`,
      `rope_theta`; read them off the HF config/weights, never hardcoded.
- [x] 1.2 qpk1 loader: parse + fail-closed validate the new fields (sidecar
      count/byte length `(2*n_layers+1)*d_model` floats, GQA consistency
      `n_kv*head_dim==d_model`, `n_heads % n_kv == 0`, zero/negative/overflow
      geometry guards); unit tests for each rejection path + server e2e
      corrupted-variant coverage for all seven new fail-closed fixtures.
- [ ] 1.3 Re-export the r8 artifact and confirm the server still reports
      ready (contract migration proof, old-artifact rejection covered).

## 2. Engine transformer forward (CPU-correct first)

- [x] 2.1 Write failing parity test: tiny Qwen3 fixture exported through the
      driver, engine block forward vs HF reference forward, max |Δlogit| ≤
      1e-2 (RED before implementation). RED observed 2026-10-18 after the
      inventory contract (A1) landed: acquisition + declared-inventory
      cross-check pass, forward fails at the engine's `unimplemented!`
      prefill — the budget assertion itself stays unreachably RED by
      construction until 2.2/2.3 wire it (see design.md crystallization).
- [ ] 2.2 RMSNorm (f32, eps from config), RoPE (theta from config, f32
      cos/sin tables computed once at load), GQA causal attention with KV
      cache over packed Q/K/V projections.
- [ ] 2.3 SwiGLU MLP over gate/up/down packed projections; residual wiring;
      final norm + LM head; greedy decode parity test GREEN.
- [ ] 2.4 End-to-end: serve the r8 artifact, `/v1/chat/completions` greedy
      stream; sample outputs reviewed for sanity (expectation: r8 quality is
      garbage — correctness, not quality, is the gate).

## 3. CubeCL GPU kernels (parity-gated, opt-in)

- [ ] 3.1 `gpu` cargo feature on quantibbler-kernels; runtime probe module
      (device detection, staged SDK dir per the validated recipe); fallback
      semantics tested on GPU-less builder.
- [ ] 3.2 `packed_matvec` CubeCL kernel (per-row bitplane decode + f64-
      accumulation semantics emulated deterministically on device — pair
      accumulation order documented in kernel doc comment).
- [ ] 3.3 Parity harness: random ±1 bitplanes + scales, GPU vs CPU rlib,
      rel 1e-12, repeated-run bit-identity; autotune-free fixed launch
      geometry first (correctness before performance).
- [ ] 3.4 Dispatch wiring behind the feature flag + feature-gated tests in
      CI-with-GPU lane; server purity gate (default features) stays green.

## 4. Landing

- [ ] 4.1 `openspec validate qk-attention --strict` green; spec deltas
      folded only at archive time.
- [ ] 4.2 Gates: pytest, cargo test (kernels + server), clippy both crates,
      pre-commit; scoped commits per repo discipline (leaves never commit —
      orchestrator lands).
