# Tasks: qk-ptq — PTQ pipeline

Task numbering is `N.M` under numbered `## N.` groups. Tests precede
implementation within each group; test names reference scenario IDs from
`specs/qk-ptq/spec.md` (e.g. `test_scn_caller_supplied_model_and_data` for
scenario `caller_supplied_model_and_data`). All gates
are run through `uv` per `AGENTS.md` (`uv run pytest`, `uv run maturin
build`); Rust gates run in `crates/quantibbler-kernels/`.

## 1. Scaffolding and configuration

- [x] 1.1 Write failing test `tests/ptq/test_config.py::test_scn_config_defaults` (scenarios `config_defaults_paper_values`, `default_numerics_float64`, `float32_opt_in`): `PTQConfig` exists with float64-default numerics (`compute_dtype == "float64"`), `seed`, rank, robust-diagonal (tau, gamma), ADMM (K, rho endpoints, lam, epsilon), stage steps/lr/batch per [NQ] App. C defaults. Expected: test fails with ImportError/AttributeError.
- [x] 1.2 Implement `quantibbler/ptq/config.py` with the `PTQConfig` dataclass; float32 is opt-in only, never a silent default. Verify: `uv run pytest tests/ptq/test_config.py` passes.

## 2. Global calibration (Phase 1)

- [x] 2.1 Write failing test `tests/ptq/test_calibration.py::test_scn_preconditioner_properties` (scenario `preconditioner_shape_robustness`): on a random linear layer fed synthetic activations, `robust_diag` returns nonnegative diagonal preconditioners of the right shapes, entries clipped to tau (assumption A-1) before shrinkage, shrinkage applied per Eq. 3 of [NQ] Step 2-1; a layer with zero-variance activations yields finite, positive entries. Expected: fails until implemented.
- [x] 2.2 Write failing test `tests/ptq/test_calibration.py::test_scn_stats_all_layers` (scenario `stats_cover_all_layers`): one forward (and backward) pass over calibration batches records statistics for every linear layer of a 2-block tiny model, in module declaration order. Expected: fails until implemented.
- [x] 2.3 Implement `quantibbler/ptq/calibration.py` (statistics collection + `RobustDiag` with (tau, gamma), float64 accumulators). Verify: `uv run pytest tests/ptq/test_calibration.py` passes.

## 3. LB-ADMM initialization (per layer)

- [x] 3.1 Write failing test `tests/ptq/test_lb_admm.py::test_scn_admm_spd_solves` (scenario `admm_spd_solves`): for random W~ and rank r, each continuous update system matrix `H = VᵀV + (ρ+λ)I` is symmetric positive definite (Lemma 2, [NQ] App. B.4) and solved via Cholesky; the stabilized fallback path triggers deterministically when the diagonal shift is perturbed. Expected: fails until implemented.
- [x] 3.2 Write failing test `tests/ptq/test_lb_admm.py::test_scn_admm_consensus_convergence` (scenarios `admm_consensus_convergence`, `penalty_schedule_stopping`): on a rank-r ground-truth matrix, ADMM residual `max(||U−Z_U||, ||V−Z_V||)` decreases to below epsilon (or hits the K = 400 cap), the rho value increases linearly per iteration, and the SVID proxy preserves the sign structure of its input each iteration (assumption A-5 semantics asserted on a closed-form case). Expected: fails until implemented.
- [x] 3.3 Write failing test `tests/ptq/test_lb_admm.py::test_scn_scale_balancing` (scenario `magnitude_balancing_scales`): after magnitude balancing, `||η·Û||_F ≈ ||η⁻¹·V̂||_F` ratio matches Eq. 7–9 of [NQ] Step 2-3 in float64; scales are per-row mean absolute values; dead rows produce the clamped scale floor, not NaN (assumption A-11). Expected: fails until implemented.
- [x] 3.4 Implement `quantibbler/ptq/lb_admm.py`: preconditioned target construction, Cholesky-based continuous updates (float64), SVID proxy, dual updates, linear penalty schedule (A-2), ridge (A-3), tolerance (A-4), truncated-SVD warm start (A-6), magnitude balancing. Verify: `uv run pytest tests/ptq/test_lb_admm.py` passes.

## 4. Block reconstruction (Phase 2)

- [x] 4.1 Write failing test `tests/ptq/test_block_recon.py::test_scn_block_loop_order` (scenario `quantized_prefix_inputs`): with a 2-block tiny model, the loop consumes quantized-prefix inputs `X_b` (mock forward records that block 1 sees block 0's *quantized* output) and teacher targets `Y_b`; TuneFP runs before LB-ADMM before TuneLatentSTE per block ([NQ] Algorithm 1 order). Expected: fails until implemented.
- [x] 4.2 Write failing test `tests/ptq/test_block_recon.py::test_scn_ste_refinement` (scenario `ste_refinement_improves_mse`): during refinement, gradients flow to (U, V, s1, s2) with straight-through sign (grad of sign treated as passthrough) and block output MSE decreases over T_post steps on a fixed toy block. Expected: fails until implemented.
- [x] 4.3 Write failing test `tests/ptq/test_block_recon.py::test_scn_sign_extraction_parity` (scenario `sign_extraction_packing_parity`): extracted `U±1 = sign(U)`, `V±1 = sign(V)` match the packed bits after `qk-kernels` pack/unpack (bit identity, parity-gated dispatch per config.yaml), including the sign(0) = +1 convention. Expected: fails until implemented.
- [x] 4.4 Implement `quantibbler/ptq/block_recon.py`: quantized-prefix input capture, fp-weight TuneFP step, per-layer LB-ADMM wiring, STE refinement (App. C lr/batch/epochs defaults), sign extraction and packing via `qk-kernels`. Verify: `uv run pytest tests/ptq/test_block_recon.py` passes.

## 5. Global scale calibration (Phase 3)

- [x] 5.1 Write failing test `tests/ptq/test_scale_calib.py::test_scn_kl_scale_calibration` (scenario `kl_scale_calibration_frozen_bits`): with binaries frozen (packed bits bit-identical before/after), optimizing s1/s2 decreases KL(softmax(z_fp/T) || softmax(z_q/T)) on cached teacher logits for a tiny model; only scale parameters carry `.grad` (T = 1.0 default, assumption A-7). Expected: fails until implemented.
- [x] 5.2 Implement `quantibbler/ptq/scale_calib.py`: teacher-logit caching, KL objective, scale-only optimization with cosine schedule (App. C: lr 1e-6, bs 1, 8 epochs). Verify: `uv run pytest tests/ptq/test_scale_calib.py` passes.

## 6. Public API and serialization round trip

- [x] 6.1 Write failing test `tests/ptq/test_api.py::test_scn_entry_point_signature` (scenario `caller_supplied_model_and_data`): `quantibbler.ptq.quantize_model(model, calibration_data, config)` accepts a caller-constructed tiny `nn.Module` and iterable of token batches; no network access anywhere in the path (import-site audit: no urllib/requests/huggingface imports in `quantibbler/ptq/`). Expected: fails until implemented.
- [x] 6.2 Write failing test `tests/ptq/test_api.py::test_scn_qpk1_round_trip` (scenarios `qpk1_round_trip_byte_exact`, `rust_matches_reference_probes`): `QuantizedModel` serializes to a `qk-representation` qpk1 container and reloads with byte-exact packed bits and float64 scales; reload-then-forward equals serialize-time forward on probe inputs (uses `qk-kernels` unpack; Rust path must match NumPy reference at 1e-12 rel on float64 probes when the extension is present, per config.yaml). Expected: fails until implemented.
- [x] 6.3 Write failing test `tests/ptq/test_api.py::test_scn_float32_opt_in` (scenario `float32_opt_in`): default config runs float64; setting `compute_dtype="float32"` is honored and the test asserts the solver entry dtype flips — proving float32 is opt-in, not default. Expected: fails until implemented.
- [x] 6.4 Implement `quantibbler/ptq/api.py` (`quantize_model`, `QuantizedModel`, serialize/reload via qpk1). Verify: `uv run pytest tests/ptq/test_api.py` passes.

## 7. End-to-end gate and full suite

- [x] 7.1 Write failing test `tests/ptq/test_e2e_gate.py::test_scn_e2e_tiny_gate` (scenarios `e2e_tiny_gate_passes`, `e2e_bpw_and_parity`, `e2e_gate_detects_corruption`): seeded tiny random transformer block (d≈32, seq≈32, 8 calibration samples × 64 tokens per OQ-4 proposal) → `quantize_model` → serialize qpk1 → reload → forward; assert (a) max |logit_fp − logit_q| ≤ 1.0 absolute (assumption A-8; a deliberately bit-flipped artifact must FAIL this assertion — corruption self-test), (b) effective BPW < 1.0 computed from serialized bytes via the `qk-representation` accounting (never parameter counts), (c) pack/unpack parity (bit identity). Expected: fails until implemented.
- [x] 7.2 Implement `quantibbler/ptq/e2e.py` (tiny-model builder + gate harness) and wire `quantize_model` end to end across Phases 1–3. Verify: `uv run pytest tests/ptq/test_e2e_gate.py` passes.
- [x] 7.3 Run the full gate suite and record results (config.yaml verification gates; AGENTS.md ordering): `uv run pre-commit run --all-files` (ruff + `cargo fmt --check` + `cargo clippy --all-targets -- -D warnings`), `uv run pytest`, `uv run maturin build`, `cargo test` (in `crates/quantibbler-kernels/`). All must pass before this change lands; paste tails into the PR description. Command: `cd /opt/data/quantibbler && uv run pre-commit run --all-files && uv run pytest && uv run maturin build && (cd crates/quantibbler-kernels && cargo test)`. Expected: every gate exits 0.
