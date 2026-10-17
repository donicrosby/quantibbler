# Design: qk-ptq — PTQ pipeline (NanoQuant path)

Follows the method invariants of `openspec/config.yaml`: the authoritative source is [NQ] = nanoquant-2602.06694v3, cited by section/equation. Every method-design claim below cites [NQ] or is explicitly flagged as an informed assumption (config.yaml evidence hierarchy, level 2). Where sibling capabilities are consumed, they are named, not specified: `qk-representation` (qpk1 container, BPW-from-bytes accounting) and `qk-kernels` (pack/unpack, sign_estimate, joint_itq, parity-gated dispatch).

## 1. Pipeline structure

The PTQ pipeline mirrors Algorithm 1 of [NQ] Sec 3.3 ("The NanoQuant algorithm") as three phases:

1. **Phase 1 — Global calibration.** Run calibration data through the fp teacher once, collecting per-linear-layer input/output activation statistics; from these build the robust diagonal preconditioners (`RobustDiag` with clipping ceiling `tau` and shrinkage `gamma`, Eq. 3) ([NQ] Step 2-1, Algorithm 1 lines 2–5). The preconditioners approximate the Hessian-weighted distortion objective of Eq. 2 via K-FAC-style diagonal factors. Gradient statistics enter `D_out`; computing them costs one backward-style pass over calibration data ([NQ] Step 2-1, App. C: memory-efficient cross-entropy + gradient checkpointing are noted as implementation aids). Layer order: statistics are collected for all layers before any block is quantized.
2. **Phase 2 — Block reconstruction loop** (Sec 3.2), per transformer block b, sequential:
   - **Inputs from the quantized prefix:** `X_b` is produced by feeding calibration data through the already-quantized blocks `<b`; the teacher target `Y_b` is the fp block's output on those inputs ([NQ] Algorithm 1 lines 9–10). This is the error-propagation-mitigation formulation: later blocks see the inputs they will actually see at inference.
   - **Step 1, error-propagation mitigation (TuneFP):** briefly tune the fp weights of the current block against (X_b, Y_b) with T_pre steps before factorizing ([NQ] Step 1, Algorithm 1 line 12).
   - **Step 2, per-linear-layer LB-ADMM initialization** (Steps 2-1/2-2/2-3, detailed below): preconditioned target `W~ = D~_out @ W @ D~_in`, ADMM to convergence, magnitude balancing to extract (Ub, Vb, s1, s2).
   - **Step 3, factorized component refinement (TuneLatentSTE):** jointly tune continuous latents U, V and scales s1, s2 with the Straight-Through Estimator against the block-level objective of Eq. 10; then extract `U±1 = sign(U)`, `V±1 = sign(V)` and pack ([NQ] Step 3, Algorithm 1 lines 19–22).
3. **Phase 3 — Global scale calibration (TuneScalesKD).** With binaries bit-frozen and packed, tune only the scale vectors of all layers S_global = {s1, s2} to minimize KL divergence between teacher and student logits under temperature T (Eq. 11) ([NQ] Sec 3.3, Algorithm 1 lines 26–27).

Discarded alternative — *decouple initialization from the block loop* (run LB-ADMM for all layers first, then reconstruct): rejected because [NQ] Sec 3.2 explicitly integrates initialization as a subroutine inside the block reconstruction loop ("Unlike methods that treat initialization as a separate pre-processing phase…"), and preconditioned targets must be built from the *updated* fp weights of the current block (Step 1 runs before Step 2 within a block).

Discarded alternative — *defer binary optimization to a global PV-tuning stage* (the DBF/[Boža and Macko 2026] organization [NQ] contrasts itself with): rejected; [NQ] Step 3 refines factors locally per block during reconstruction, which the paper credits for its PTQ-budget efficiency.

## 2. LB-ADMM initialization (per linear layer)

Objective (Eq. 4 / Eq. 27): minimize `½||W~ − U Vᵀ||_F² + λ/2(||U||_F² + ||V||_F²)` subject to `U = Z_U`, `V = Z_V`, where the proxy families are the sign-structured sets feeding latent binary factors. Updates (App. B.3):

- **Continuous update** (Eq. 5 / 32–34): `(VᵀV + (ρ+λ)I) Uᵀ = VᵀW~ᵀ + ρ(Z_U − Λ_U)ᵀ`, symmetrically for V. The system matrix `H_U = VᵀV + (ρ+λ)I` is SPD for ρ>0, λ≥0 (Lemma 2), so we solve by **stabilized Cholesky** — [NQ] Sec 2-2 states this reduces cost to O(r³/3) vs O(2r³/3) for LU and is what makes 70B-scale factorization tractable. "Stabilized" here means: attempt Cholesky; on a non-positive-definite or ill-conditioned system (conditioning bound of Cor. 2 degrades when σ_min(V)² → 0 relative to ρ+λ), add a small relative jitter to the diagonal shift and retry, in float64 (informed assumption on the exact fallback mechanics; the paper names the method, not the failure path — see Assumptions).
- **Proxy update** (Eq. 6 / 35): `Z ← SVID(U + Λ)`, the Sign-Value Independent Decomposition [Pouransari et al. 2020; Xu et al. 2024] of the consensus variable: the best rank-r approximation that preserves the sign structure of its input. Implementation-wise this is computed per column, not as a full SVD of the (dout × r) consensus matrix (informed assumption; see Assumptions). Note this SVID proxy in LB-ADMM is an *operator inside initialization* and is distinct from LittleBit's SVD initialization stage ([LB]); it does not make this change depend on the QAT lane.
- **Dual update** (Eq. 36): `Λ ← Λ + U − Z`.
- **Penalty schedule:** linear ADMM penalty scheduler over K = 400 outer iterations ([NQ] App. C). The paper states the schedule is linear but not its endpoints; we ramp rho linearly across the K iterations (see Assumptions). Convergence knob `epsilon` (Algorithm 1 input list) bounds the consensus residual `max(||U − Z_U||, ||V − Z_V||)`; K is a hard cap, epsilon an early stop (informed assumption on the exact stopping semantics).
- **Initialization of the very first iterate:** W~ = D~_out W D~_in is rank-≤ r approximated; we seed U, V from a truncated SVD of W~ (informed assumption; [NQ] does not state the warm start. It is the natural rank-r start compatible with the O(r³) solve cost, and the SVID operator makes the first proxy well-defined).

**Latent magnitude balancing** (Step 2-3, Eqs. 7–9, App. A): after convergence, recover unscaled proxies `Û = D~_out⁻¹ P_U^(K)`, `V̂ = D~_in⁻¹ P_V^(K)` where `P_U = U + Λ_U`; equilibrium factor `η = sqrt(||V̂||_F / ||Û||_F)` (Eq. 7 — note: Frobenius *norms*, per Eq. 7 as written, which Appendix A's magnitude-balancing analysis is consistent with); scales from mean absolute row values `[s1]_i = mean(|η û_i|)`, `[s2]_j = mean(|η⁻¹ v̂_j|)` (Eq. 8); final latents `U := η Û`, `V := η⁻¹ V̂` (Eq. 9). This balancing removes the inherent scale ambiguity between the factors (App. A, Proposition 1) and conditions the subsequent STE refinement.

Zero-scale guard: if a row of Û or V̂ is identically zero (dead channel), its scale entry would be 0 and sign() is undefined there; we clamp scale floors to a small positive constant in float64 and define sign(0) = +1 at extraction (informed assumption; [NQ] is silent on dead channels — how a wrong choice would surface: a zero row in packed bits or a NaN in the forward pass of the gate).

## 3. Numerics policy

- All solvers (statistics accumulation, preconditioner construction, Cholesky solves, ADMM, magnitude balancing) compute in **float64 by default** (config.yaml method invariants). Accumulators for mean-absolute scale extraction are float64.
- **float32 is opt-in** via `PTQConfig.compute_dtype = "float32"`; the config must state it explicitly, never silently default. The Rust kernels must match the NumPy reference within 1e-12 relative tolerance on float64 probes (config.yaml verification gates), so the reference path is the contract.
- Packing maps −1 → 0, +1 → 1 ([NQ] Fig. 2 stage c); bit identity after pack/unpack is asserted through the `qk-kernels` capability (parity-gated dispatch) in the round-trip requirement.

## 4. Block reconstruction details

- **Which layers:** every linear layer in the block — q/k/v/o projections and MLP up/gate/down — following [NQ] Sec 3.2 ("We sequentially compress each linear layer in each transformer block") and the "all linear layers" strategy of [NQ] Step 1. Embeddings, the final LM head, and norms stay full precision (informed assumption: [NQ] quantizes transformer linear layers; keeping embeddings/lm_head dense is the standard PTQ scope and avoids catastrophic first-token error — see Assumptions).
- **Weighted MSE:** during block reconstruction the loss is a weighted MSE between quantized and fp block outputs ([NQ] App. C), consistent with Eq. 10's ||·||²_F objective.
- **Ordering within a block:** [NQ] does not pin whether q/k/v/o are factorized before MLP within Step 2. We process layers in module declaration order (informed assumption; how a wrong choice would surface: only as small gate-level logit-delta shifts, since Step 3 refinement and Phase 3 recalibration absorb ordering effects).
- **STE refinement:** Adam-family updates on U, V, s1, s2 (Eq. 10) with straight-through sign; hyperparameters per App. C.

## 5. Global scale calibration

- Objective: KL(softmax(z_M/T) || softmax(z_M̂/T)) over calibration logits (Eq. 11), teacher logits computed once and cached; only s1/s2 receive gradients; packed binaries are frozen ([NQ] Sec 3.3).
- The temperature T is a config input in Algorithm 1's formulation via the KD objective but its value is not stated in [NQ] (informed assumption, T = 1.0 default — see Assumptions).
- Memory: teacher logits cache is O(samples × seq_len × vocab) in float64; for the tiny-model gate this is trivial, for large models the caller batches (documented constraint, no code change needed in this spec).

## 6. API shape

Entry point (Python, per config.yaml architecture; caller-supplied model, no network):

```
quantibbler.ptq.quantize_model(model, calibration_data, config: PTQConfig) -> QuantizedModel
```

- `model`: a caller-constructed PyTorch `nn.Module` (fp teacher). The pipeline reads weights and registers quantized replacements; it never downloads anything.
- `calibration_data`: an iterable of token-id batches (caller-supplied; [NQ] uses 128 wikitext-2 samples × 2048 tokens, App. C — this is a recipe, not a library requirement).
- `PTQConfig` (dataclass): `rank r`, `robust-diagonal (tau, gamma)`, `admm (K, rho_schedule, lam, epsilon)`, `steps (T_pre, T_post, T_glob)` with lr/batch per stage (App. C defaults), `compute_dtype` ("float64" default, "float32" opt-in), `seed`, `temperature`, plus serialization knobs.
- `QuantizedModel` carries packed binaries and float scales per layer and knows how to **serialize to / reload from** the `qk-representation` qpk1 container; forward pass runs through `qk-kernels` dispatch (dequantize-on-the-fly reference path or Rust kernels when present).
- Progress/verbosity hooks are kept minimal: a `verbose` flag; no callback protocol in v0.1 (informed assumption on scope).

Discarded alternative — *generator-style streaming API* (`quantize_iter(blocks)`) to allow resumable pipelines: rejected for v0.1; checkpointing mid-pipeline is an Open Question (below) and would complicate the state contract without a demonstrated need.

Discarded alternative — *per-layer functional API* (`quantize_layer(W, hessian_stats)`) as the primary public surface: rejected as the sole entry point because the block-level loop (Step 1 TuneFP + Step 3 STE against block outputs) needs activations from the whole block, not a single layer. The per-layer factorizer remains internal.

## 7. End-to-end gate (tiny model)

Mirrors the verification-gates section of config.yaml: construct a **tiny random transformer block** (deterministic seed), quantize with the PTQ pipeline, serialize to qpk1, reload into a fresh process-object, forward the same probe inputs, and assert:

- max logit delta vs the fp tiny model is bounded (threshold from Assumption A-8's calibration rationale);
- **effective BPW < 1.0 measured from serialized bytes** via the `qk-representation` accounting (bytes on disk / weight count — never from parameter counts);
- pack/unpack parity: packed bits identical across the round trip (this is also the Rust-vs-NumPy sign-agreement invariant of config.yaml).

The tiny model is built in-test from `torch.nn` primitives only (no download, honoring the no-network rule). The gate is deliberately the *last* task in tasks.md, matching the config gate suite.

## 8. Informed assumptions

Each entry: assumption → rationale → how a wrong choice would surface. Paper-stated values are *not* assumptions (they are cited): calibration 128 samples × seq 2048 from wikitext-2, seed 0 ([NQ] App. C); K = 400 ADMM steps with linear penalty schedule ([NQ] App. C); stage-wise lr/batch/epochs: 1e-4 / bs 4 (T_pre), 1e-5 / bs 1 (T_post), 1e-6 / bs 1 (T_glob), 8 epochs each, cosine schedule ([NQ] App. C); γ ≈ 0.2 small / 0.6 large models ([NQ] Step 2-1); [LB] and [LB2] values are out of scope here.

- **A-1 `RobustDiag` clipping ceiling τ and its meaning.** [NQ] Algorithm 1 lists robust-diagonal parameters (τ, γ) but never states τ's default or whether it is a quantile, an absolute cap, or τ_max of App. B.2's Lemma 1. Assumption: τ is an absolute cap on diagonal entries, applied before shrinkage, with default τ = 99.9th-percentile-equivalent scaled to each layer's statistic max (implemented as: clip entries at τ = 1000 × median(D) if that is smaller). Rationale: Lemma 1 (App. B.2) needs a bounded spectral norm for the stability argument, which any finite cap satisfies; the exact value only shifts conditioning. *Wrong-choice signature:* LB-ADMM linear solves become ill-conditioned (Cholesky retry path fires often) or preconditioning loses all outlier sensitivity (degraded reconstruction, visible as elevated block MSE at fixed rank) — both observable in unit tests before any model-scale run.
- **A-2 Penalty schedule endpoints.** Linear scheduler for ρ over K = 400 iterations is stated ([NQ] App. C; D.4 shows linear converges to lower final error than aggressive schedules), endpoints are not. Assumption: ρ from 1e-2 to 1e+2 (log-spaced endpoints, linear interpolation in iteration index). *Wrong-choice signature:* consensus residual stalls early (endpoints too low) or continuous/proxy variables oscillate without consensus (too high); both visible in the residual trace of a rank-8 toy factorization unit test.
- **A-3 Ridge λ default.** λ ≥ 0 is an Algorithm 1 input; no value stated. Assumption: λ = 1e-6 — small enough to keep Lemma 2's SPD guarantee dominant via ρ while still bounding Cor. 2's conditioning when V is near-rank-deficient. *Wrong-choice signature:* too large → biased factors (block MSE floor above the rank-limited bound); too small → Cholesky jitter path fires on rank-deficient factors. Surfaced by the same toy-factorization test.
- **A-4 ADMM tolerance ε semantics.** Algorithm 1 lists ε but no value or definition. Assumption: early-stop when consensus residual `max(||U−Z_U||_F, ||V−Z_V||_F) / (1 + ||U||_F)` < ε with default ε = 1e-7 (else run all K = 400). *Wrong-choice signature:* too large → initialization quality varies run-to-run at fixed seed count (noisy sign patterns across seeds); too small → wasted iterations, no quality change. Checked by comparing two ε settings on a fixed toy matrix.
- **A-5 SVID implementation.** [NQ] cites SVID [Pouransari et al. 2020; Xu et al. 2024] without an algorithm block. Assumption: column-wise operation — for each column z of the consensus matrix, compute the sign pattern s = sign(z) and the value vector as the rank-1-in-sign least-squares fit; equivalently, project onto sign pattern then renormalize to preserve the column norm of (U + Λ). *Wrong-choice signature:* proxy quality degrades → ADMM consensus residual plateaus high; would show up as poor final block MSE relative to the paper's qualitative claims, and as unstable sign patterns across ADMM iterations in the D.3-style latent-dynamics test.
- **A-6 Warm start for LB-ADMM.** Not stated in [NQ]. Assumption: truncated rank-r SVD of the preconditioned target W~ (economic SVD, top-r), U = U_r, V = V_r. Rationale: natural, cheap relative to O(r³/3) solves, and dual/consensus variables start at zero. *Wrong-choice signature:* slower ADMM convergence (more iterations to hit ε); visible as iteration-count inflation in tests, not as a wrong final result.
- **A-7 τ (temperature) for Eq. 11 = 1.0.** [NQ] states the KD objective with T in Eq. 11 but gives no value; App. C's unified lr table suggests a plain-KD setup. Assumption: T = 1.0 default, config-overridable. *Wrong-choice signature:* softened (T > 1) would spread probability mass and slightly slow scale convergence — surfaced as slower Phase-3 loss decrease; hard to detect otherwise because Phase 3 optimizes scales to convergence either way.
- **A-8 Gate logit-delta threshold.** config.yaml says "max logit delta bounded" without a number. Assumption: on a tiny 1-block model (d ≈ 32, seq ≈ 32) with float64 pipeline and rank ≈ 8, require max |logit_fp − logit_q| ≤ 1.0 absolute, plus a relative bound of 25% on the max-logit magnitude, with the exact threshold recorded in the test and revisited against the first real-model run. Rationale: a freshly-initialized random tiny block carries little redundancy; a delta above ~1.0 at these dimensions indicates a real pipeline defect (packing bug, scale mis-serialization), not quantization noise. *Wrong-choice signature:* too tight → flaky gate; too loose → masks serialization bugs. Both are caught by running the gate against a deliberately corrupted artifact (flip one packed bit; the gate must fail).
- **A-9 Embeddings / LM head / norms stay fp.** [NQ] Sec 3.2 scopes compression to transformer linear layers; it does not enumerate exclusions. Assumption: embedding tables, final LM head, RMSNorm/LayerNorm parameters remain full precision in v0.1. *Wrong-choice signature:* quantizing the LM head at sub-1-bit would collapse the tiny-model gate's logit delta far beyond any threshold; first end-to-end run reveals immediately.
- **A-10 Number of reconstruction/refinement epochs on tiny models.** App. C's 8 epochs per stage is stated for large models; the tiny gate uses the same stage structure but may override steps down for test speed (documented in the test, default config keeps paper values). *Wrong-choice signature:* none on correctness — gate asserts bounded delta, not quality; only runtime.
- **A-11 Dead-channel handling** (scale floor + sign(0) = +1): see §2. *Wrong-choice signature:* NaNs in forward or a zero row of packed bits — trivially visible in the round-trip test.
- **A-12 In-block layer order** = module declaration order: see §4. *Wrong-choice signature:* small gate-level logit-delta shifts only.

## 9. Open Questions

- **OQ-1 Checkpoint/resume of the block loop.** Status: OPEN. A 70B-scale block loop is hours-long; mid-pipeline checkpointing would need a defined artifact format. Owner: qk-ptq lane (proposal author). Blocked on: whether `qk-representation` extends qpk1 with a partial-artifact mode (cross-lane decision).
- **OQ-2 Gradient-statistics cost model.** [NQ] Step 2-1 needs gradient statistics for D~_out; a full backward per calibration sample is expensive. Status: OPEN whether we mirror the paper's exact estimator or a cheaper diagonal proxy, decided at implementation time with unit tests on reconstruction quality. Owner: qk-ptq lane implementer.
- **OQ-3 Default rank r per layer type.** [NQ] experiments report results at effective 1.00 / 0.80 / 0.55 bits but the per-layer rank mapping is not spelled out in the text read for this change. Status: OPEN; v0.1 exposes `rank` as a config knob with a single global default, no per-layer schedule. Owner: qk-ptq lane; needs a paper re-read of Sec 4.3/App. F before defaults ship.
- **OQ-4 Calibration-set size floor for tiny models.** [NQ] D.1 shows quality saturates ≥ 256 samples for 7B; tiny-model gates will use far fewer samples. Status: OPEN whether the gate fixes a sample count or ties it to a quality assertion; proposed: fixed 8 samples × 64 tokens for the gate, recorded in the test. Owner: qk-ptq lane implementer.

## 10. Cross-lane contract

- **Consumes `qk-representation`:** the qpk1 container (serialize/reload of packed binaries + float scales) and the BPW-from-bytes accounting used by the end-to-end gate. Required semantics, in prose: serialization must be byte-exact across round trips; BPW accounting must count serialized bytes (header + payload), not parameter counts; effective BPW = total serialized bytes × 8 / total represented weight entries, and the gate requires this < 1.0.
- **Consumes `qk-kernels`:** pack/unpack (−1→0, +1→1 bit mapping, bit identity across round trips), parity-gated dispatch (NumPy reference path is the executable specification; Rust path matches at 1e-12 rel on float64 probes per config.yaml). `sign_estimate` and `joint_itq` belong to the QAT lane ([LB], [LB2]) and are *not* consumed by this change; the LB-ADMM SVID proxy (A-5) is internal to this pipeline's initialization.
- **No function signatures are invented here** for sibling capabilities; the qpk1 and dispatch contracts are owned by their lanes. Any change this pipeline needs from them is a cross-lane negotiation recorded as an Open Question (OQ-1).
