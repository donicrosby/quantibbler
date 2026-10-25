# qk-fix-admm-solver — proposal

## Why

Three defects in the LB-ADMM solver deviate the implementation from [NQ] NanoQuant:

1. **Exit rescale shoots the product by `‖W̃‖_F`.** The entry normalization divides the target by `target_scale` once (`lb_admm.py:290`), but the exit rescale applied `× target_scale` to *each* factor of a two-factor product — overshooting the reconstructible matrix by a full factor of `‖W̃‖_F` (fixed at `df1e6cd`, `factor_scale = sqrt(target_scale)`).

2. **Absolute rho schedule [1e-2→1e2] never converges** — 400/400 iterations, epsilon never met — and the 100× consensus dominance at iteration end crushes the continuous latents onto the ±1-shaped SVID proxy, destroying the truncated-SVD warm start (raw cos 0.84 → 0.32, measured on real `layers.0` at rank 512).

3. **SVID proxy is column-only, not the paper's optimal two-sided rank-1.** The paper defines SVID as the "optimal rank-1 approximation that preserves the sign structure" ([NQ] :592). The closed form is the top rank-1 SVD of `sign(P)∘P`, yielding two vectors (g, h). Our `svid_proxy` (`lb_admm.py:162`) fits one scalar per column with row vector fixed at 1 — optimal only in that restricted class.

## What Changes

1. **ADMM exit rescale** — already landed. Spec captures the invariant for audit.

2. **Rho schedule** — constrain `rho_end` so the consensus term does not overwhelm the O(1) data term in the normalized space. The paper prescribes a linear schedule over 400 steps ([NQ] App. C :3008); the endpoints were an informed assumption. `rho_end` SHALL not exceed `1e-1` in the normalized regime (where the target is divided by its Frobenius norm at entry).

3. **SVID proxy** — implement the two-sided optimal rank-1: `Z = sign(P) ⊙ (g hᵀ)` where `(g, h)` is the top rank-1 SVD of `sign(P)∘P`, computed via a small `r×r` SVD already O(r³) on the existing Cholesky path.

## Impact

- `src/quantibbler/ptq/lb_admm.py` — `svid_proxy` rewired; `rho_end` default updated in `config.py`; existing `factor_scale = sqrt(target_scale)` confirmed.
- `src/quantibbler/ptq/config.py` — `rho_end` default lowered.
- `tests/ptq/` — new scale-equivariance test already landed in `df1e6cd`; SVID-optimality probe added.
- Fidelity label: unchanged — these are implementation corrections, not method deviations. The paper's own SVID definition is the two-sided optimum; the paper's linear schedule is shape only, endpoints are implementation.

## Non-goals

- Does NOT change the ADMM iteration algebra (Eq. 5, dual update, Cholesky) — those were verified line-for-line against [NQ] App. B.
- Does NOT add or remove solver stages. The paper's three-stage reconstruction is unaltered.
- Does NOT touch `magnitude_balance`, `block_recon.py`, STE, or the pack path.
- Does NOT change the paper-faithful gate semantics.
