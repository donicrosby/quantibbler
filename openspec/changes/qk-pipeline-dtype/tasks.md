# Tasks: qk-pipeline-dtype

## Phase 1: config + validation (dtype knob)

- [ ] 1.1 **Add bf16 to PTQConfig.compute_dtype and change default.**
  `config.py`: `_COMPUTE_DTYPES = ("bf16", "float32", "float64")`, default
  `"bf16"`. Validation accepts all three. `np_compute_dtype` mapping: `"bf16"`
  → **raises `NumPyBackendUnsupportedDtypeError`** (NumPy backend is
  float64-only; bf16 and float32 require torch backend). Driver
  `--compute-dtype` adds `bf16` as default. Run
  `uv run pytest tests/ptq/test_config.py -v`. Expected: new test covering
  bf16 validation + default + error on NumPy bf16 attempt.

- [ ] 1.2 **Accumulation dtype constants.** Define named per-site
  accumulation dtype in a new `accumulation.py` (or inline in each module).
  Each site: `ACCUM_DTYPE = torch.float32 if compute_dtype == torch.bfloat16
  else compute_dtype`. The mapping is a pure function of compute_dtype, not
  a separate knob. Run `uv run pytest tests/ptq/ -k accum`.

## Phase 2: FactorizedLinear dtype generic

- [ ] 2.1 **Remove hardcoded float64 from FactorizedLinear.__init__.**
  `_factorized.py`: latents stored at `compute_dtype` (not `.double()`).
  Scales `s1`/`s2` at `compute_dtype`. Bias at `compute_dtype`. Constructor
  takes `compute_dtype` parameter. Run `uv run pytest tests/ptq/test_factorized.py -v`.

- [ ] 2.2 **FactorizedLinear.from_weight dtype-generic.**
  SVD init: on GPU, use `torch.linalg.svd` at **float32** accumulation
  dtype (justified accumulation site — bf16 SVD is platform-dependent); on
  CPU, use NumPy float64 path then cast. The NumPy path stays the reference
  at the same declared dtype. Run `uv run pytest tests/ptq/test_factorized.py -v`.

- [ ] 2.3 **_factor_terms at compute_dtype.**
  The activation_dtype parameter equals `compute_dtype`. The STE passthrough
  operates at compute_dtype (identical algebra). Verify STE gradient parity
  at float64 vs old path. Run
  `uv run pytest tests/ptq/test_root_cause_ablation.py -v`.

## Phase 3: Solver dtype generic + GPU

- [ ] 3.1 **solver_backend.py: dtype-aware to_host_np.**
  `to_host_np` parameterized by compute_dtype. Default maps to `np.float32`
  for bf16/float32, `np.float64` for float64. The NumPy parity backend stays
  float64-only. Run `uv run pytest tests/ptq/test_solver_backend.py -v`.

- [ ] 3.2 **lb_admm.py: dtype-generic ADMM solve.**
  All tensor creations at `compute_dtype`. Normal equations at accumulation
  dtype (float32 for bf16). Cholesky at accumulation dtype. Latents at
  compute_dtype. Dead-channel scale floor at `compute_dtype`. Run
  `uv run pytest tests/ptq/test_lb_admm.py -v`.
  Expected: per-dtype tolerance assertions (1e-2 bf16, 1e-4 float32, 1e-12 float64).

- [ ] 3.3 **lb_admm.py: torch device tensors.**
  ADMM entry: tensors stay on the module's device (no `.cpu().numpy()`).
  `torch.linalg.cholesky`, `torch.linalg.solve_triangular` on device tensors.
  SVD init: `torch.linalg.svd` on device tensors at float32. The two
  host↔device round-trips per layer are eliminated. Run
  `uv run pytest tests/ptq/test_lb_admm.py -v`.

- [ ] 3.4 **Parity suite: dtype×device matrix.**
  Extend parity tests across `{bf16, float32, float64} × {CPU, GPU}`.
  Per-dtype tolerances: bf16 1e-2, float32 1e-4, float64 1e-12.
  CPU path runs always. GPU path runs only on the designated GPU hardware
  lane (`quantibbler-ptq:rocm-official` container, digest-pinned, on Strix
  Halo/gfx1151), with exact skip-count discipline — every skipped GPU cell
  MUST produce a documented skip event. Run `uv run pytest tests/ptq/ -k parity -v`.

## Phase 4: Pipeline stages dtype generic

- [ ] 4.1 **calibration.py: dtype-generic statistics.**
   `collect_stats`: activations captured at `compute_dtype`. Chunked
   statistics: sum-of-squares accumulator at accumulation dtype (float32
   under bf16). `to_host_np` at compute_dtype. Chunk path stays bit-identical
   to unchunked path at float64. Run
   `uv run pytest tests/ptq/test_calibration.py -v`.

- [ ] 4.2 **robust_diag: dtype-generic preconditioner.**
   Diagonal entries at `compute_dtype`. Clipping and shrinkage at
   `compute_dtype`. The `DEFAULT_SCALE_FLOOR` at `compute_dtype`. Run
   `uv run pytest tests/ptq/test_robust_diag.py -v`.

- [ ] 4.3 **block_recon.py: dtype-generic block reconstruction.**
   TuneFP: teacher output Y_b, quantized prefix throughput at
   `compute_dtype`. Loss and backward at `compute_dtype`. STE refinement:
   `_actor_terms` at `compute_dtype`. Run
   `uv run pytest tests/ptq/test_block_recon.py -v`.

- [ ] 4.4 **bit_polish.py: dtype-generic polish.**
   Greedy polish at `compute_dtype`. Bit-flip proposals and objective
   evaluation at `compute_dtype`. Run
   `uv run pytest tests/ptq/test_bit_polish.py -v`.

- [ ] 4.5 **TuneScalesKD: dtype-generic.**
   Scales are already at `compute_dtype` (from FactorizedLinear). KL loss
   computation at `compute_dtype`. Teacher logit cache at `compute_dtype`.
   Run `uv run pytest tests/ptq/test_tune_scales_kd.py -v`.

## Phase5: Device placement

- [ ] 5.1 **Per-stage device placement map.**
   Every stage reads the device from the owning module's parameters
   (`next(m.parameters()).device`) — never from a global or threaded-out
   state. Accelerator stages: capture, TuneFP, solver (ADMM/SVD), STE,
   TuneScalesKD, eval. Host-side stages: sign/extract/pack (latent-to-bit
   conversion boundary — `x >= 0` convention, not `torch.sign`), qpk1 export
   (disk write). Run `uv run pytest tests/ptq/ -k device -v`.

- [ ] 5.2 **Device-mismatch auditor.**
   Opt-in `nn.Module.__call__` wrapper that raises on tensor args whose
   device differs from the module's device. Enabled by `--device-audit` flag
   for debugging; off by default. Run
   `uv run pytest tests/ptq/test_device_audit.py -v`.

- [ ] 5.3 **CPU path: good-guest thread sizing.**
   At driver startup, BEFOR any import of torch or BLAS: execute sizing
   block — `sched_getaffinity` → n_cores, `/proc/loadavg` → 1-min load,
   compute `n_threads` per design.md formula (light/moderate/heavy branches).
   Set `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` in
   `os.environ`. After torch import, call `torch.set_num_threads(n_threads)`.
   Assert `torch.get_num_threads()` reflects computed value. `--max-threads N`
   override. `--no-good-guest` disables. Run
   `uv run pytest tests/ptq/test_good_guest.py -v`.

- [ ] 5.4 **GPU memory fraction cap.**
   `torch.cuda.set_per_process_memory_fraction(0.85)` at startup when discrete
   GPU present. On APU/iGPU devices (Strix Halo/gfx1151), additionally enforce
   absolute-GB cap (`--gpu-memory-gb G`) to prevent host-RAM drain. Record
   `max_memory_reserved()` in report. `--gpu-memory-fraction F` and
   `--gpu-memory-gb G` overrides. Run
   `uv run pytest -k gpu_memory` (GPU-present gate).

- [ ] 5.5 **Per-stage timing rows.**
   Every pipeline stage (capture, TuneFP, solver, STE, sign/extract/pack,
   TuneScalesKD, eval, export) records elapsed-wall seconds as a JSONL row,
   appended as each stage produces its timing. The per-layer aggregate enables
   comparison against the ~8h v6-class baseline. Run
   `uv run pytest tests/ptq/ -k timing`.

## Phase6: Hardcoded-dtype elimination

- [ ] 6.1 **Grep-evidenced elimination in PTQ pipeline.**
   Every `.double()`, `torch.float32`, `torch.float64`, `np.float32`,
   `np.float64`, and `astype(…float…)` call site in `src/quantibbler/ptq/`
   and `tools/quantize_hf_model.py` is either made parametric or justified.
   The grep command (full pattern set):
   `grep -rnE '\.double\(\)|torch\.float32|torch\.float64|np\.float32|np\.float64|astype\(.*float' src/quantibbler/ptq/ tools/quantize_hf_model.py`
   Zero unjustified hits. Each site maps to one of three resolution classes:
   (a) parametric, (b) accumulation-site, (c) reference-path. Populate the
   enumeration in design.md's crystallization section. Run after Phase4.

- [ ] 6.2 **Driver: remove float64 hardcodes in export path.**
   `tools/quantize_hf_model.py`: embedding/lm_head export at remainder_dtype
   (no `.to(torch.float64).cpu().numpy()` intermediate). Norm weights at
   remainder_dtype. `remainder_dtype` default `"bf16"` (tracks compute_dtype
   when unset). Run `uv run pytest tests/ptq/test_e2e.py -v`.

- [ ] 6.3 **Container wapper: update defaults.**
   `tools/run-ptq-container.sh`: `COMPUTE_DTYPE:-bf16` (was `:-float64`).
   `OMP_NUM_THREADS` derived from container's visible cores, not hardcoded
   16. Run: wrapper DRY_RUN test.

- [ ] 6.4 **config.yaml method-invariant update.**
   Config line: "solvers compute in float64 by default" → "solvers compute in
   bf16 by default; float32 and float64 are selectable non-defaults."
   Run: `npx --yes @fission-ai/openspec validate qk-pipeline-dtype --strict`.

## Phase7: Spec delta + validation

- [ ] 7.1 **Spec delta: MODIFED requirements.**
   Two MODIFED requirements in `qk-ptq`:
   - ADMM solver (dtype-generic, device placement).
   - Factorized components STE refinement (dtype-generic).
   Every MODIFED preserves all existing scenarios + adds new ones.
   Default-numerics requirement was REMOVED + ADDED (not MODIFED).

- [ ] 7.2 **Spec delta: ADDED requirements.**
   Ten ADDED requirements (one in first ADDED section: bf16-default numerics;
   nine in second ADDED section):
   - Bf16 default numerics (replaces REMOVED float64 default).
   - Dtype-agnostic pipeline contract.
   - Accumulation dtype separation.
   - GPU solver device placement.
   - Per-stage device placement map.
   - Good-guest CPU thread sizing.
   - GPU memory fraction cap.
   - Hardcoded-dtype elimination.
   - Artifact remainder dtype default tracking.
   - Dtype-generic FactorizedLinear.
   Each has ≥1 scenario with WHEN/THEN bullets. The bias-path requirement is
   the ninth requirement of the second ADDED section; the Dtype-agnostic
   contract and Hardcoded-dtype elimination bullets both live under that
   section's first (grep-evidenced) requirement — 10 ADDED total, matching
   proposal.md's impact count.

- [ ] 7.3 **Validate spec.**
   `npx --yes @fission-ai/openspec validate qk-pipeline-dtype --strict`.
   Expected: "Change 'qk-pipeline-dtype' is valid" — zero warnings.

## Phase8: Bias-path check

- [x] 8.1 **Read parallel bug-hunt verdict.**
   t_59815b89 completed: bias zeros are legit-zero math (bias-free
   Qwen3-0.6B config). OQ-1 resolved as "not dtype-related; bias path
   rewritten to dtype-agnostic form as defense-in-depth." Recorded in
   design.md. Two peripheral findings routed to t_373bdeff — no scope
   expansion for this change.

- [ ] 8.2 **Bias dtype path in FactorizedLinear.**
   Bias tensor stored at `compute_dtype` (not `.double()`). Bias carried
   through solver, STE, and export at `compute_dtype`. Zero-bias assertion
   test at bf16 compute: `uv run pytest tests/ptq/ -k bias`.

## Phase9: Gates

- [ ] 9.1 **Full gate suite.**
   `uv sync --all-groups && uv run pytest -x -q &&
   cargo test --manifest-path crates/quantibbler-kernels/Cargo.toml &&
   cargo clippy --manifest-path crates/quantibbler-kernels/Cargo.toml --all-targets -- -D warnings &&
   cargo fmt --manifest-path crates/quantibbler-kernels/Cargo.toml -- --check &&
   maturin develop --manifest-path crates/quantibbler-kernels/Cargo.toml &&
   pre-commit run --all-files`
   All gates green. Record per-dtype tolerance evidence in the test output.
