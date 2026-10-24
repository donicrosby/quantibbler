# qk-fix-admm-solver — design

## T² exit rescale (landed at `df1e6cd`)

Entry: `w = w / target_scale` (`lb_admm.py:290`). Exit: was `× target_scale` on each of u, v, P_u, P_v. For a two-factor product `UVᵀ`, scaling both factors by `t` scales the product by `t²`. Fix: `factor_scale = sqrt(target_scale)` on all four returns (`:356-361`). Entry comment corrected to state the real invariant. Red-first test (`test_lb_admm_scale_equivariance`) verifies: scale target by c, returned product scales by c.

## SVID proxy — two-sided optimum

### Paper definition

[NQ] :592: "We apply Sign-Value Independent Decomposition (SVID) (Pouransari et al., 2020; Xu et al., 2024) to derive the **optimal rank-1 approximation that preserves the sign structure**."

The SVID of matrix P with sign structure S = sign(P) is: Z = S ⊙ (g hᵀ) where (g, h) minimize ‖P − S ⊙ (g hᵀ)‖_F. Since S² = 1 entrywise, this is equivalent to finding (g, h) as the top rank-1 SVD of (S∘P). Then Z = S ⊙ (g hᵀ). For the consensus variable P = U + Λ ∈ R^{d_out×r}, the two vectors g ∈ R^{d_out}, h ∈ R^r are computed from the top singlet of the d_out×r matrix S∘P via power iteration (O(d_out·r) per call, strictly cheaper than the existing O(r³) Cholesky).

### Current implementation

Column-only: `out[:, j] = s[:, j] * (s[:, j]ᵀ z[:, j] / ‖s[:, j]‖²)`. One scalar per column, row vector h fixed at all-ones. The docstring records three column-only variants tried; the two-sided optimum was never in the set.

### Fix

Replace the column-wise projection with the two-sided optimum: compute `M = sign(P) ∘ P` (entrywise), compute top rank-1 SVD of M → σ, u₁, v₁, set `g = u₁ * √σ`, `h = v₁ * √σ` (or equivalent split), return `Z = sign(P) ⊙ (g hᵀ)`. The SVD is O(r³) — same complexity class as the existing Cholesky solves (r ≤ 512). Dead columns (all-zero input): Z gets the positive `DEFAULT_SCALE_FLOOR`.

## Rho schedule — scale-aware endpoints

### Paper

[NQ] App. C :3008: "We employed a linear ADMM penalty scheduler for 400 factorization steps, for each weight matrix across all models." The line does NOT specify the rho range [1e-2, 1e2] — that range was an informed assumption calibrated on r137-scale targets.

### Defect

The iteration runs in the normalized space (`w/T`, O(1) norm). At rho=1e2, the consensus RHS `rho*(zu−lu)` dominates the data RHS `vᵀwᵀ` (O(1)) by two orders of magnitude, crushing U onto the ±1-shaped proxy before the continuous fit converges. Measured: capping rho at 1e-1 lifted raw cos on 6/7 linears (mean +0.11). The solver reports `converged=False` after every measured run — the tolerance is unreachable within 400 iterations at the shipped endpoints.

### Fix

Keep the linear schedule shape; cap `rho_end` at `1e-1` in the normalized regime. `rho_start` stays at `1e-2`. The data term is O(1) by construction (target divided by its own Frobenius norm at entry), so `rho_end = 1e-1` keeps the consensus term at ≤10% of the data term magnitude at the final iteration — sufficient to nudge latents toward the binary proxy while preserving the continuous fit.

The paper's schedule is a shape prescription, not an endpoint prescription. The endpoints are an implementation choice; a `rho_end` that prevents convergence is a defect, not a deviation.

### Discarded alternatives

- **Scale-relative rho (rho × ‖W̃‖_F):** "converges" at iteration 1 with product unmoved — trivial dominance, not adjustment.
- **Scaling epsilon:** bit-identical to capping rho in measured probe (the tolerance is not the binding constant).
- **Adaptive rho:** adds a tunable heuristic the paper doesn't describe; the fixed linear schedule with constrained endpoints is simpler and stays closer to the paper's algorithm shape.

### Open Questions

None — endpoints are an implementation choice the paper leaves unspecified, and the measured defect is the rho_end that prevents convergence.

## Interaction with pack-stage bottleneck

The decomposition showed post-pack cos clamped to 0.34–0.45 regardless of raw fit quality (0.32–0.84). Fixing rho and SVID improves the raw continuous fit; the pack stage remains the binding constraint, and its fix is a separate change. This change improves the solver's input to the pack stage; it does not claim to move the shipped number alone.
