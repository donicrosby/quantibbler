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
- [x] 1.3 Re-export the r8 artifact and confirm the server still reports
      ready (contract migration proof, old-artifact rejection covered).
      Done 2026-10-18: full driver re-run (baseline ppl 28.6696, matching
      the prior run's regime) exported under the declared-inventory
      contract (`norm_slots = ["input","post_attention","q_norm","k_norm"]`,
      genuinely-tied head — real Qwen3-0.6B IS tied; only the tiny config
      defaults untied), release binary reports ready on the 28-layer /
      151936-vocab artifact. Old-contract artifact kept at
      `artifacts/qwen3-0.6B-r8-20261018-old-contract`; the loader's
      fail-closed inventory validation rejects it (missing norm_slots).

## 2. Engine transformer forward (CPU-correct first)

- [x] 2.1 Write failing parity test: tiny Qwen3 fixture exported through the
      driver, engine block forward vs HF reference forward, max |Δlogit| ≤
      1e-2 (RED before implementation). RED observed 2026-10-18 after the
      inventory contract (A1) landed: acquisition + declared-inventory
      cross-check pass, forward fails at the engine's `unimplemented!`
      prefill — the budget assertion itself stays unreachably RED by
      construction until 2.2/2.3 wire it (see design.md crystallization).
- [x] 2.2 RMSNorm (f32, eps from config), RoPE (theta from config, f32
      cos/sin tables computed once at load), GQA causal attention with KV
      cache over packed Q/K/V projections. Landed with the inventory-driven
      norm application (E-2.3); RoPE computes per-position angles from
      theta (rotate_half convention) rather than precomputed tables —
      same math, see design.md.
- [x] 2.3 SwiGLU MLP over gate/up/down packed projections; residual wiring;
      final norm + LM head; greedy decode parity test GREEN (max |Δlogit|
      = 4.1e-5 vs fp32-of-artifact goldens; budget 1e-2 — E-2.2).
- [x] 2.4 End-to-end: serve the r8 artifact, `/v1/chat/completions` greedy
      stream; sample outputs reviewed for sanity (expectation: r8 quality is
      garbage — correctness, not quality, is the gate). Demonstrated
      2026-10-18 on the real 28-layer artifact: /health ready,
      /v1/completions + /v1/chat/completions (non-stream and SSE stream)
      greedy-decode through the transformer engine; determinism confirmed
      (byte-identical "anced"*8 across runs — E-2.1 on a real artifact);
      quality is garbage by construction at 0.0345 BPW.

## 3. CubeCL GPU kernels (parity-gated, opt-in)

- [x] 3.1 `gpu` cargo feature on quantibbler-kernels; runtime probe module
      (device detection, staged SDK dir per the validated recipe); fallback
      semantics tested on GPU-less builder. Done 2026-10-18: `gpu =
      ["dep:cubecl"]` (cubecl 0.11, hip); `GpuContext::init` fail-closed
      with GPU_INIT_HINT carrying the staged-SDK recipe
      (LD_LIBRARY_PATH=/opt/data/tmp/rocm-sdk/lib +
      HSA_OVERRIDE_GFX_VERSION=11.5.1 — without it the probe reports
      NoSuchDevice and the server falls back to CPU). Fallback semantics:
      inert flag on non-gpu builds (`gpu_flag_is_inert_without_the_feature`),
      CPU selection when unset, soft log-and-fall-back on init failure
      (`select_from_env`).
- [x] 3.2 `packed_matvec` CubeCL kernel (per-row bitplane decode + f64
      accumulation semantics emulated deterministically on device — pair
      accumulation order documented in kernel doc comment). Done: one
      thread per row, LSB-first decode, STRICTLY SEQUENTIAL f64
      accumulation in element order (same pair order as the CPU rlib);
      host-side validate + u32 index ceiling before any launch.
- [x] 3.3 Parity harness: random ±1 bitplanes + scales, GPU vs CPU rlib,
      rel 1e-12, repeated-run bit-identity; autotune-free fixed launch
      geometry first (correctness before performance). Done:
      tests/gpu_parity.rs behind QUANTIBBLER_GPU_TESTS=1; hardware run on
      gfx1151 — worst rel err 0.000e0 (bit-identical incl. pad-byte and
      degenerate shapes; rejection messages mirror the CPU face); two
      GPU runs byte-identical. Launch geometry fixed: 256-thread cubes,
      ceil-div grid.
- [x] 3.4 Dispatch wiring behind the feature flag + feature-gated tests in
      CI-with-GPU lane; server purity gate (default features) stays green.
      Done: MatvecFace/FaceDecision frozen at ModelWeights construction
      (E-2.1 lifetime determinism — never swapped mid-stream);
      QUANTIBBLER_GPU=1 requests HIP with soft init-failure fallback;
      /health reports `matvec_face`; tests/gpu_dispatch.rs hardware run —
      face=gpu:hip:gfx, full fixture transformer generates on device
      matvecs, token stream identical to the CPU face. Default-feature
      server suite green incl. binary_purity (124 tests); clippy -D
      warnings + fmt green on both crates in both feature states.
- [x] 3.5 CPU fast path (user-directed): SIMD inner product for
      packed_matvec/packed_matmul (±1 decode via byte LUT/shuffle into
      f32xN/f64xN ±1 vectors) + rayon row-parallelism behind an opt-in
      feature. Done 2026-10-18: `fast-cpu` feature (wide 0.7 f64x4 + rayon
      1.10); byte-LUT sign decode, lane-sequential f64x4 partials, fixed
      reduce + scalar tail. Release measurement: 512×(1024×137) matvec
      scalar 234 ms → fast 9.3 ms (**25.1×**). Parity gate is the
      BACKWARD error (≤1e-12 · Σ|terms|) — forward-relative is
      condition-bound and hit 1.53e-12 on a cancellation row while the
      backward error stayed ~1e-15 of term scale (crystallized in
      design.md); run-to-run bit-identity asserted exactly; parallelism
      across 64-row blocks only (per-row tasks measured 1.2× in dev —
      overhead-bound). Server wired via `cpu_matvec` shim; /health label
      "cpu:fast" under the feature; clippy green in all four feature
      combos; full suites green in default, fast-cpu, gpu, and both.

## 4. Landing

- [x] 4.1 `openspec validate qk-attention --strict` green; spec deltas
      folded only at archive time. Green at every spec edit this session
      (latest: 2026-10-18, before the 3.5 landing commit).
- [x] 4.2 Gates: pytest, cargo test (kernels + server), clippy both crates,
      pre-commit; scoped commits per repo discipline (leaves never commit —
      orchestrator lands). Done 2026-10-18: pytest 66/66 after
      `uv sync --reinstall` of the changed kernels extension; cargo test
      green on kernels + server in default, gpu, fast-cpu, and both
      feature states (incl. binary_purity and the hardware-gated
      gpu_parity/gpu_dispatch arms on gfx1151); clippy -D warnings green
      in all four combos on both crates; pre-commit ran on every scoped
      commit (hooks all Passed).
