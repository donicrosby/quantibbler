# qk-ptq spec delta — qk-fix-admm-factor-scale

## MODIFIED Requirements

### Requirement: LB-ADMM SHALL conclude with latent magnitude balancing that yields Ub, Vb, s1, s2

Upon ADMM convergence, the pipeline SHALL recover unscaled proxies `Û = D~_out⁻¹ P_U`, `V̂ = D~_in⁻¹ P_V`, compute the equilibrium factor `η = sqrt(||V̂||_F / ||Û||_F)` (Eq. 7), extract scales `[s1]_i = mean(|η û_i|)`, `[s2]_j = mean(|η⁻¹ v̂_j|)` (Eq. 8), and set final latents `U = η Û`, `V = η⁻¹ V̂` (Eq. 9) ([NQ] Step 2-3). The stage SHALL then apply the serialization-faithfulness correction; the returned scales are the corrected scales. Initialization of the layer is complete only when all of (Ub, Vb, s1, s2) are produced.

#### Scenario: magnitude_balancing_scales

- **WHEN** ADMM converges and magnitude balancing executes on the recovered proxies
- **THEN** the balanced factor norms satisfy the η relation of Eq. 7, the scales are first extracted as per-row mean absolute values per Eq. 8 and then corrected by the serialization-faithfulness factor γ, and no scale entry is NaN; an all-zero row receives the clamped scale floor rather than an undefined value

## ADDED Requirements

### Requirement: The magnitude balancing stage SHALL apply a target-free serialization-faithfulness correction to the binary product

After Eq. 8 extraction, the stage SHALL compute the serialized product `M = diag(s1) @ Ub @ Vbᵀ @ diag(s2)` and the continuous-product norm `||UVᵀ||_F` from the balanced latents, then apply `γ = sqrt(||UVᵀ||_F / ||M||_F)` symmetrically to `s1` and `s2` and record it in `BalanceResult.gamma`. The correction SHALL NOT consume `target`; on a zero or non-finite norm it SHALL raise `ValueError`. The correction SHALL replace the LS α refit against `W`.

#### Scenario: correction_preserves_continuous_product_norm

- **WHEN** magnitude balancing applies the correction at rank 3 on the `_SevenLinearBlock` ADMM fixture
- **THEN** the corrected serialized product norm equals the continuous-product norm: `||M||_F / ||UVᵀ||_F = 1.0` within relative tolerance `1e-6`

#### Scenario: correction_survives_target_removal

- **WHEN** the serialized product is formed with no `target`/refit input (the follow-up change's end-state)
- **THEN** `||M||_F / ||W||_F` stays in the healthy band `[0.5, 2.0]` — LS-refit removal is a semantic no-op

#### Scenario: correction_is_noop_when_already_scale_faithful

- **WHEN** the Eq. 8 serialized product is already norm-faithful to the continuous product — a constructed faithful fixture where `||M||_F = ||UVᵀ||_F` exactly (rank-r outer product U = a⊗1, V = b⊗1 with a,b > 0 — γ = 1.0 exactly, robust to the preconditioner)
- **THEN** `gamma == 1.0` within relative tolerance `1e-3` — the correction applies no renormalization

#### Scenario: correction_is_recorded_finite_and_lands_in_band_at_production_rank

- **WHEN** the balance stage runs at production-like rank (r ≥ 64)
- **THEN** the corrected product `||M||_F / ||W||_F` lands in the healthy band `[0.5, 2.0]`, and `gamma` is recorded, finite, and `> 0`
- **AND** `gamma` is not asserted ≈ 1.0 at production rank: measured γ is shape-dependent ≈ 1.01–1.17 (128×128 r32 1.0916 → 128×128 r64 1.1165 → 512×512 r64 1.1738) — the raw Eq. 8 product is under-normed at 512×512 r64 (`||M_unc||_F / ||W||_F ≈ 0.79–0.81`), which is exactly what the correction addresses

#### Scenario: correction_raises_on_degenerate_input

- **WHEN** `||M||_F` is zero or the ratio `||UVᵀ||_F / ||M||_F` is non-finite
- **THEN** the stage raises `ValueError` naming the degenerate condition rather than silently skipping the correction

#### Scenario: correction_recorded_in_result

- **WHEN** the balance stage completes with correction factor γ
- **THEN** `BalanceResult.gamma` equals the applied γ

### Requirement: The serialized factor-scale overshoot SHALL be discriminated by test

A test SHALL assert that, at rank 3 on the `_SevenLinearBlock` ADMM fixture, the uncorrected Eq. 8 serialized product overshoots the continuous-product norm while the corrected product's norm equals it within a tight relative tolerance, so a no-op `gamma = 1.0` mutant cannot pass.

#### Scenario: discrimination_test_reds_on_uncorrected_product

- **WHEN** the discrimination test forms the uncorrected Eq. 8 serialized product on the rank-3 fixture
- **THEN** `||M||_F / ||UVᵀ||_F` is far from 1.0 (the measured overshoot mode), failing the tight-tolerance assertion on current trunk bytes

#### Scenario: discrimination_test_green_on_corrected_product

- **WHEN** the discrimination test runs on the corrected stage
- **THEN** `||M||_F / ||UVᵀ||_F ≈ 1.0` within the tight tolerance, and `||M||_F / ||W||_F ∈ [0.5, 2.0]` on every linear layer

#### Scenario: discrimination_test_rejects_gamma_stub

- **WHEN** the balance stage returns `gamma = 1.0` while retaining the LS α refit (the no-op-mutant shape)
- **THEN** `||M||_F / ||UVᵀ||_F` equals the α-refit ratio (measured `0.555–0.931`), which fails the tight tolerance, while `||M||_F / ||W||_F = cos(W,M)` stays in `[0.70, 0.93]` (measured `0.7016–0.9276`) — the α-refit shrink signature, so the test discriminates the correction from a stub
