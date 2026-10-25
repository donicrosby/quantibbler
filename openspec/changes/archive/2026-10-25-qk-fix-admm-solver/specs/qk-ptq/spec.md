# qk-ptq delta: qk-fix-admm-solver

## MODIFIED Requirements

### Requirement: LB-ADMM SHALL iterate continuous updates, SVID proxies, and scaled duals to a stated stopping rule

The continuous update SHALL solve `(VᵀV + (ρ+λ)I)Uᵀ = VᵀW~ᵀ + ρ(Z_U − Λ_U)ᵀ` (Eq. 5) via stabilized Cholesky decomposition, symmetrically for V ([NQ] App. B.4, Lemma 2). The proxy update SHALL apply the two-sided optimal rank-1 SVID operator to the consensus variables (Eq. 6): `Z = sign(P) ⊙ (g hᵀ)` where (g, h) is the top rank-1 SVD of `sign(P)∘P` ([NQ] :592, "optimal rank-1 approximation that preserves the sign structure"). The dual update SHALL accumulate `Λ ← Λ + U − Z` ([NQ] App. B.3).

#### Scenario: admm_consensus_convergence

- **WHEN** LB-ADMM factorizes a matrix with an exact rank-r structure
- **THEN** the consensus residual `max(||U−Z_U||_F, ||V−Z_V||_F)` decreases across iterations and terminates below epsilon or at the K-iteration cap
- **AND** the SVID proxy preserves the sign structure of its input at every proxy update

#### Scenario: svid_is_two_sided_optimum

- **WHEN** the SVID proxy processes a consensus variable `P = U + Λ` with sign structure `S = sign(P)`
- **THEN** the returned Z is `S ⊙ (g hᵀ)` where (g, h) are the top rank-1 SVD factors of `S∘P`
- **AND** for a column-constant input the two-sided optimum is exactly the column-only solution — the fix is a strict superset, never a regression on the existing test fixtures

#### Scenario: svid_sign_preserved

- **WHEN** SVID processes any input matrix
- **THEN** `sign(Z) == sign(input)` entrywise, with sign(0) = +1 per the A-11 convention
- **AND** dead columns (all-zero input) receive the positive `DEFAULT_SCALE_FLOOR`

### Requirement: The ADMM penalty SHALL follow a linear schedule over K iterations with epsilon as an early stop

The penalty parameter SHALL follow a linear schedule across K outer iterations ([NQ] App. C :3008), and ADMM SHALL stop early when the consensus residual falls below the tolerance epsilon; otherwise it SHALL run to the K-iteration cap. The rho endpoint SHALL stay within the stated endpoint interval so the consensus term does not overwhelm the unit-scale data term: in the normalized regime (target divided by its Frobenius norm at entry per [NQ] Step 2), `rho_end` SHALL lie in the closed interval `[0.0, 1.0]`, with the schema refusing `rho_end < rho_start`. The paper prescribes the linear shape; the endpoints are an implementation choice, and an endpoint that prevents convergence is a defect.

#### Scenario: penalty_schedule_stopping

- **WHEN** LB-ADMM runs with default ADMM parameters on a fixed target
- **THEN** the penalty value increases linearly per iteration up to K, and the solver halts at whichever comes first: residual below epsilon or the K-th iteration

#### Scenario: rho_end_bounded_for_convergence

- **WHEN** LB-ADMM factorizes a planted rank-32 target at rank-64 factorization (`gamma=0.2`; warm-start cos ≥0.99 — the fixture the convergence gate exercises) with the paper's `tau`/`gamma` preconditioning; real-layer evidence: all 20 recorded `layers.0` probe rows in `artifacts/lane0-fire-r3-20261024/rows.jsonl` are non-monotone (2–66 rises each; one row — arm B `self_attn.o_proj` — does not net-decrease over the first 200), confirming the shape is structural, not a fixture artifact
- **THEN** the consensus residual makes net progress across the first 200 iterations (net decrease from start of the second half, with the first half dominated by the consensus-term pull onto the ±1 proxy before the data term reasserts itself) and the solver terminates at the K-iteration cap — `converged=False` is expected with a ±1-structured proxy and does not indicate solver failure
- **AND** the post-iteration continuous reconstruction satisfies `cos(UVᵀ, W) > 0.5`, proving the ADMM produces a usable initializer (cos loss from the unconstrained truncated-SVD warm start is expected; the SVID proxy pushes latents into the ±1-signed rank-1 subspace, and the paper's pipeline compensates in STE refinement per [NQ] Step 3)
- **AND** `cos(UVᵀ, W) > cos(sign(zu) @ sign(zv)ᵀ, W)` — the continuous reconstruction beats a naive sign-only assignment, proving the ADMM iteration does meaningful work beyond trivial binarization

### Requirement: LB-ADMM SHALL conclude with latent magnitude balancing that yields Ub, Vb, s1, s2

Upon ADMM convergence, the pipeline SHALL recover unscaled proxies `Û = D~_out⁻¹ P_U`, `V̂ = D~_in⁻¹ P_V`, compute the equilibrium factor `η = sqrt(||V̂||_F / ||Û||_F)` (Eq. 7), extract scales `[s1]_i = mean(|η û_i|)`, `[s2]_j = mean(|η⁻¹ v̂_j|)` (Eq. 8), and set final latents `U = η Û`, `V = η⁻¹ V̂` (Eq. 9) ([NQ] Step 2-3). The ADMM exit SHALL rescale factors from the normalized space back to the target's scale with `factor_scale = sqrt(target_scale)` applied to each of `(U, V, P_U, P_V)` — a single factor of `target_scale` on the product `UVᵀ`, maintaining scale equivariance: scaling the entry target by `c` SHALL scale the returned product by `c`, not `c²`. Initialization of the layer is complete only when all of (Ub, Vb, s1, s2) are produced.

#### Scenario: magnitude_balancing_scales

- **WHEN** ADMM converges and magnitude balancing executes on the recovered proxies
- **THEN** the balanced factor norms satisfy the η relation of Eq. 7, the extracted scales equal per-row mean absolute values per Eq. 8, and no scale entry is NaN; an all-zero row receives the clamped scale floor rather than an undefined value

#### Scenario: exit_rescale_is_equivariant

- **WHEN** the ADMM entry target is scaled by a nonzero factor c
- **THEN** the returned product `UVᵀ` scales by exactly c — not c² — verified within relative tolerance 1e-5
- **AND** the iteration trajectory in the normalized space is identical regardless of the entry scale
