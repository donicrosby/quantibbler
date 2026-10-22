"""LB-ADMM per-layer initialization (tasks 3.1-3.4; [NQ] Step 2-2/2-3,
Eqs. 4-9, App. B.3/B.4).

Solves, per linear layer, the ridge-regularized ADMM problem (Eq. 4)

    min 1/2 ||W~ - U Vᵀ||_F² + λ/2 (||U||_F² + ||V||_F²)  s.t.  U = Z_U, V = Z_V

with the continuous update of Eq. 5 solved by stabilized Cholesky
(``(VᵀV + (ρ+λ)I) Uᵀ = VᵀW~ᵀ + ρ(Z_U - Λ_U)ᵀ``, symmetrically for V), the
SVID proxy update of Eq. 6 (sign-structure-preserving, column-wise —
assumption A-5), the scaled dual update of Eq. 36, and a linear penalty
schedule over K iterations (A-2). Iteration stops early when the relative
consensus residual falls below ``epsilon`` (A-4) or at the K cap. Warm
start: truncated rank-r SVD of the preconditioned target (A-6).

Latent magnitude balancing (Step 2-3, Eqs. 7-9) recovers the unscaled
proxies Û = D~_out⁻¹ P_U and V̂ = D~_in⁻¹ P_V, computes η = sqrt(||V̂||_F /
||Û||_F), extracts per-row mean-absolute scales, and fixes the final
latents U := η Û, V := η⁻¹ V̂. Dead channels (all-zero rows) receive a
clamped scale floor and sign(0) = +1 at extraction (A-11).

All computation is float64 by default; float32 only via an explicit config
(compute_dtype), honored at the solver entry.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .config import PTQConfig

__all__ = [
    "ADMMResult",
    "ADMMTrace",
    "BalanceResult",
    "DEFAULT_SCALE_FLOOR",
    "lb_admm",
    "magnitude_balance",
    "svid_proxy",
]

# Dead-channel scale floor (A-11): clamped positive constant in float64.
DEFAULT_SCALE_FLOOR = float(np.finfo(np.float64).eps) * 64.0
# Relative jitter added to the diagonal shift when Cholesky hits a
# non-PD system (design section 2, stabilized Cholesky fallback).
_JITTER = 1e-10


@dataclass
class ADMMTrace:
    """Per-iteration diagnostics of one LB-ADMM run."""

    residuals: list[float] = field(default_factory=list)
    rhos: list[float] = field(default_factory=list)
    min_eigs: list[float] = field(default_factory=list)
    n_cholesky: int = 0
    n_jitter: int = 0
    svid_sign_preserved: list[bool] = field(default_factory=list)


@dataclass
class ADMMResult:
    """Outcome of LB-ADMM on one preconditioned target.

    Attributes:
        U: continuous factor ``(dout, r)`` (post-balanced latents live in
            :class:`BalanceResult`; this is the raw consensus iterate).
        V: continuous factor ``(din, r)``.
        P_u / P_v: final consensus proxies ``U + Λ_U`` / ``V + Λ_V`` in
            preconditioned space (the magnitude-balancing inputs).
        n_iter: iterations executed (early stop or K).
        converged: True when the residual fell below epsilon.
        trace: per-iteration diagnostics.
    """

    U: np.ndarray
    V: np.ndarray
    P_u: np.ndarray
    P_v: np.ndarray
    n_iter: int
    converged: bool
    trace: ADMMTrace


@dataclass
class BalanceResult:
    """Magnitude-balancing outputs (Eqs. 7-9) for one layer.

    Attributes:
        U / V: final latents ``η Û`` / ``η⁻¹ V̂`` (float64).
        Ub / Vb: extracted binary factors ``sign(U)`` / ``sign(V)`` with
            sign(0) = +1 (int8, entries exactly ±1).
        s1 / s2: scale vectors, per-row mean absolute values, dead rows
            clamped to ``DEFAULT_SCALE_FLOOR`` (float64).
        eta: equilibrium factor of Eq. 7.
    """

    U: np.ndarray
    V: np.ndarray
    Ub: np.ndarray
    Vb: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    eta: float


def _np_dtype(compute_dtype: str) -> np.dtype:
    return np.dtype(compute_dtype)


def _cholesky_factor(h: np.ndarray, trace: ADMMTrace | None = None) -> np.ndarray:
    """Stabilized Cholesky factorization of an SPD system matrix.

    Attempts ``numpy.linalg.cholesky``; on a non-positive-definite system
    (LinAlgError) or a factor with non-finite entries, adds a small relative
    jitter to the diagonal shift and retries (design section 2). The jitter
    path is counted in ``trace.n_jitter``; every successful first-try
    factorization counts in ``trace.n_cholesky``.
    """
    attempt = h.copy()
    diag_scale = max(float(np.abs(np.diag(h)).max()), 1.0)
    jitter = _JITTER * diag_scale
    for k in range(60):
        try:
            factor = np.linalg.cholesky(attempt)
            if np.all(np.isfinite(factor)):
                if k > 0 and trace is not None:
                    trace.n_jitter += 1
                elif k == 0 and trace is not None:
                    trace.n_cholesky += 1
                return factor
        except np.linalg.LinAlgError:
            pass
        # Stabilization: relative jitter on the diagonal shift, escalated
        # geometrically so genuinely indefinite systems recover deterministically.
        attempt = h + jitter * np.eye(h.shape[0])
        jitter *= 2.0
    raise np.linalg.LinAlgError("stabilized Cholesky failed to produce an SPD factor")


def _cholesky_solve(h: np.ndarray, b: np.ndarray, trace: ADMMTrace | None = None):
    """Solve ``H X = B`` via stabilized Cholesky.

    Returns ``(solution, jittered)`` where ``jittered`` marks that the
    fallback engaged (exposed for tests; production callers pass a trace).
    """
    factor = _cholesky_factor(h, trace)
    y = np.linalg.solve(factor, b)  # forward substitution equivalent
    x = np.linalg.solve(factor.T, y)  # back substitution equivalent
    jittered = trace is not None and trace.n_jitter > 0
    # A direct jittered flag for the deterministic test path: recompute.
    try:
        np.linalg.cholesky(h)
        jittered = False
    except np.linalg.LinAlgError:
        jittered = True
    return x, jittered


def _top_rank1(mag: np.ndarray, *, steps: int = 100, tol: float = 1e-14):
    """Top singular triplet of a NON-NEGATIVE matrix, deterministically.

    ``mag`` is ``|P|``, so it is entrywise non-negative and the Perron-
    Frobenius theorem puts the leading singular vectors in the non-negative
    orthant. Power iteration from the all-ones vector therefore converges to
    the leading pair without any sign ambiguity, and a planted rank-1 matrix
    is recovered in a single step. Returns ``(sigma, u, v)`` with unit ``u``
    and ``v``; an all-zero input returns ``(0.0, None, None)``.

    An explicit top-rank-1 solve rather than a full ``np.linalg.svd``: the
    proxy is called twice per ADMM iteration (up to 800 times per layer), so a
    dense SVD of the ``d_out x r`` latents would dominate the whole solve.
    Same optimum, O(d_out * r) per step.
    Bound, recorded rather than hidden: the stopping rule watches the singular
    value only (``|sigma_new - sigma| <= tol * sigma``). ``sigma`` converges
    faster than the singular vectors, so on a near-degenerate spectrum the
    vectors can still be short of their limit when the value has settled, and
    the cap (100 steps) binds with the vectors unconverged. The result stays
    deterministic -- fixed start, fixed cap -- but in exact-tie cases it is a
    slightly suboptimal projection rather than the optimum. For the planted
    rank-1 fixtures and the latents this proxy actually sees, the spectrum is
    well separated and one to a few steps reach machine precision.
    """
    n = mag.shape[1]
    v = np.full(n, 1.0 / np.sqrt(n), dtype=np.float64)
    u = mag @ v
    sigma = float(np.linalg.norm(u))
    if not np.isfinite(sigma) or sigma <= 0.0:
        return 0.0, None, None
    u = u / sigma
    for _ in range(steps):
        v_new = mag.T @ u
        nv = float(np.linalg.norm(v_new))
        if nv <= 0.0:
            break
        v_new = v_new / nv
        u_new = mag @ v_new
        nu = float(np.linalg.norm(u_new))
        if nu <= 0.0:
            break
        u_new = u_new / nu
        moved = abs(nu - sigma)
        u, v, sigma = u_new, v_new, nu
        if moved <= tol * max(nu, 1.0):
            break
    return sigma, u, v


def svid_proxy(z: np.ndarray) -> np.ndarray:
    """SVID operator (Eq. 6): the OPTIMAL two-sided rank-1 that preserves the
    sign structure.

    [NQ] :592 defines SVID as "the optimal rank-1 approximation that preserves
    the sign structure". With ``S = sign(P)`` (sign(0) = +1) and ``S**2 = 1``
    entrywise,

        min_{g,h} || P - S o (g h^T) ||_F  ==  min_{g,h} || |P| - g h^T ||_F

    because ``S o P = |P|``, so the optimum is the top rank-1 of ``|P|``: take
    the leading singular triplet ``(sigma, u1, v1)`` of ``|P|``, split it as
    ``g = u1 * sqrt(sigma)`` and ``h = v1 * sqrt(sigma)``, and return
    ``Z = S o (g h^T)``.

    This replaces a column-only proxy that fixed the row vector at 1 (one
    scalar per column). That variant is optimal only inside its restricted
    class; the old docstring recorded three column-only variants tried and the
    two-sided optimum was never among them. The "A-5" label retires with it --
    the paper has exactly one SVID definition and this is it.

    Output is float64, same shape as the input. Signs of the input survive with
    sign(0) = +1, and no output entry is ever zero: dead columns (all-zero
    input) are clamped to the positive ``DEFAULT_SCALE_FLOOR`` (matching
    kernels.sign_estimate), and isolated exact zeros of the rank-1 fit get the
    same floor so the sign convention holds on every entry.
    """
    zf = np.asarray(z, dtype=np.float64)
    signs = np.sign(zf)
    signs[signs == 0.0] = 1.0  # sign(0) = +1 (A-11 convention)

    sigma, u1, v1 = _top_rank1(np.abs(zf))
    if u1 is None:
        out = np.zeros_like(zf)
    else:
        root = np.sqrt(sigma)
        out = signs * np.outer(u1 * root, v1 * root)

    # No output value is ever zero (sign(0) = +1 must hold on every entry).
    out = np.where(out == 0.0, DEFAULT_SCALE_FLOOR, out)

    # Dead columns (all-zero input) would be exactly zero, which carries no
    # sign; clamp them to the positive floor so sign(0) = +1 holds and no
    # output value is ever zero (A-11, matching kernels.sign_estimate).
    col_norm = np.linalg.norm(zf, axis=0, keepdims=True)
    dead = (col_norm == 0.0).ravel()
    if np.any(dead):
        out[:, dead] = DEFAULT_SCALE_FLOOR
    return out


def _warm_start(w: np.ndarray, r: int, dt: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """Truncated rank-r SVD warm start (A-6): U = U_r, V = V_r (economic).

    ``gesdd`` (numpy's default SVD driver) is known to declare
    non-convergence on finite but extreme-dynamic-range inputs; after
    scale normalization the target is O(1) in norm, so a single retry on
    the transpose (different reduction order, the classical gesdd
    workaround) plus a ``gesvd``-style eigendecomposition fallback covers
    the flaky case. A target with non-finite entries is a pipeline bug,
    not an SVD problem: it fails here with a loud, diagnosable error.
    """
    if not np.all(np.isfinite(w)):
        bad = int(np.count_nonzero(~np.isfinite(w)))
        raise ValueError(
            f"warm start: target has {bad} non-finite entries "
            f"(max|w|={np.nanmax(np.abs(w)):.3e}) — TuneFP/calibration "
            "produced NaN/Inf upstream"
        )
    try:
        u_mat, s_vec, vt_mat = np.linalg.svd(w, full_matrices=False)
    except np.linalg.LinAlgError:
        try:
            u_mat, s_vec, vt_mat = np.linalg.svd(w.T, full_matrices=False)
            vt_mat, s_vec, u_mat = (
                u_mat.T,
                s_vec,
                vt_mat.T,
            )
        except np.linalg.LinAlgError:
            # symmetric eig on the Gram matrix: slower, unconditionally
            # stable for finite inputs.
            s2_vec, v_mat = np.linalg.eigh(w.T @ w)
            order = np.argsort(s2_vec)[::-1][:r]
            s_vec = np.sqrt(np.maximum(s2_vec[order], 0.0))
            v0 = v_mat[:, order]
            u0 = w @ v0
            nrm = np.linalg.norm(u0, axis=0, keepdims=True)
            u0 = u0 / np.maximum(nrm, 1e-300)
            return (
                np.ascontiguousarray(u0 * s_vec[None, :], dtype=dt),
                np.ascontiguousarray(v0 * s_vec[None, :], dtype=dt),
            )
    u0 = u_mat[:, :r] * np.sqrt(s_vec[:r])
    v0 = vt_mat[:r].T * np.sqrt(s_vec[:r])
    return np.ascontiguousarray(u0, dtype=dt), np.ascontiguousarray(v0, dtype=dt)


def _rho_end(cfg: PTQConfig) -> float:
    """`cfg.rho_end`, with a diagnostic env override for attribution runs.

    [NQ] App. C :3008 prescribes the rho schedule SHAPE but not its endpoints,
    so isolating the two-sided SVID fix from the endpoint cap needs two runs
    differing in exactly one number. A config default cannot express "this run
    used the previous endpoint", and there is no CLI flag for it, so the
    override is an env var -- the same pattern as the QK_RHO_CAP diagnostic.

    Fail-closed on a malformed value: silently falling back to the default
    would produce a run whose provenance claims an override it never applied.
    """
    raw = os.environ.get("QK_RHO_END")
    if raw is None:
        return cfg.rho_end
    try:
        return float(raw)
    except ValueError:
        msg = f"QK_RHO_END must be a float, got {raw!r}"
        raise ValueError(msg) from None


def lb_admm(w_tilde: np.ndarray, cfg: PTQConfig) -> ADMMResult:
    """Run LB-ADMM on the preconditioned target ``W~`` ([NQ] Step 2-2).

    Args:
        w_tilde: ``(dout, din)`` preconditioned target (float64 unless the
            config opts into float32).
        cfg: pipeline configuration (rank, ADMM parameters, dtype).

    Returns:
        :class:`ADMMResult` with the final continuous factors, consensus
        proxies, stopping state, and per-iteration trace.
    """
    dt = _np_dtype(cfg.compute_dtype)
    w = np.asarray(w_tilde, dtype=dt)
    if w.ndim != 2:
        raise ValueError(f"preconditioned target must be 2-D, got shape {w.shape}")
    dout, din = w.shape
    r = cfg.rank
    if r > min(dout, din):
        raise ValueError(f"rank r={r} exceeds min(dout, din)={min(dout, din)}")

    # Scale normalization: the rho schedule and epsilon are ABSOLUTE
    # constants (A-2/A-3), so the solve must happen in the O(1) scale
    # class they were calibrated in. Robust preconditioning on real
    # calibration statistics can crush ||W~||_F to ~1e-4 (r137: s1~2e-4,
    # s2~0.3), where the consensus term dominates the data term by ~1e8
    # and the solve "converges" to mush in a handful of iterations
    # (measured: 4/400 iterations, binary cos 0.02 — every production
    # artifact ever exported had cos ~0.001 per layer). Solving on
    # W~/||W~||_F and rescaling the factors on exit IS scale-equivariant only
    # because the exit rescale splits the factor across the pair (see the
    # sqrt below); applying target_scale to each factor overshoots by
    # target_scale. Fixture-scale targets (||W~||~O(1)) are no-ops.
    target_scale = float(np.linalg.norm(w))
    if not (np.isfinite(target_scale) and target_scale > 0.0):
        target_scale = 1.0  # degenerate all-zero target: keep legacy behavior
    if target_scale != 1.0:
        w = w / target_scale

    trace = ADMMTrace()
    u = np.zeros((dout, r), dtype=dt)
    v = np.zeros((din, r), dtype=dt)
    u, v = _warm_start(w, r, dt)
    zu = svid_proxy(u).astype(dt)
    zv = svid_proxy(v).astype(dt)
    lu = np.zeros((dout, r), dtype=dt)
    lv = np.zeros((din, r), dtype=dt)
    eye_r = np.eye(r, dtype=dt)

    def residual(u_cur, v_cur, zu_cur, zv_cur) -> float:
        du = float(np.linalg.norm(u_cur - zu_cur, "fro"))
        dv = float(np.linalg.norm(v_cur - zv_cur, "fro"))
        return max(du, dv) / (1.0 + float(np.linalg.norm(u_cur, "fro")))

    converged = False
    prev_res = np.inf
    denom = max(cfg.admm_max_iter - 1, 1)
    rho_end = _rho_end(cfg)
    for k in range(cfg.admm_max_iter):
        # Linear penalty schedule spanning [rho_start, rho_end] across the
        # K iterations (A-2).
        rho = float(cfg.rho_start + (rho_end - cfg.rho_start) * k / denom)
        shift = rho + cfg.admm_lam

        # Continuous update, Eq. 5 via stabilized Cholesky (Lemma 2 SPD).
        hu = v.T @ v + shift * eye_r
        rhs_u = v.T @ w.T + rho * (zu - lu).T
        # min-eig check (documented SPD probe; cheap at r x r)
        trace.min_eigs.append(float(np.linalg.eigvalsh(hu).min()))
        u = _cholesky_solve(hu, rhs_u, trace)[0].T
        hv = u.T @ u + shift * eye_r
        rhs_v = u.T @ w + rho * (zv - lv).T
        trace.min_eigs.append(float(np.linalg.eigvalsh(hv).min()))
        v = _cholesky_solve(hv, rhs_v, trace)[0].T

        # Proxy update, Eq. 6: SVID of the consensus variable.
        pu = u + lu
        pv = v + lv
        zu_new = svid_proxy(pu).astype(dt)
        zv_new = svid_proxy(pv).astype(dt)
        trace.svid_sign_preserved.append(
            bool(np.array_equal(np.sign(zu_new), np.where(pu == 0, 1, np.sign(pu))))
        )
        zu, zv = zu_new, zv_new

        # Dual update (Eq. 36).
        lu = lu + u - zu
        lv = lv + v - zv

        res = residual(u, v, zu, zv)
        trace.residuals.append(res)
        trace.rhos.append(rho)
        if cfg.admm_epsilon > 0 and res < cfg.admm_epsilon:
            converged = True
            break
        prev_res = res
    else:
        prev_res = trace.residuals[-1] if trace.residuals else prev_res

    # The entry normalization divides the TARGET by target_scale exactly
    # once, but a rank factorization W ~ U V^T has TWO factors: scaling both
    # by target_scale scales the reconstructible matrix by target_scale^2 and
    # overshoots by a full factor of ||W~||_F. Split it across the pair so the
    # product carries it once -- the equivariance the entry comment claims.
    factor_scale = float(np.sqrt(target_scale))
    return ADMMResult(
        U=u * factor_scale,
        V=v * factor_scale,
        P_u=(u + lu) * factor_scale,
        P_v=(v + lv) * factor_scale,
        n_iter=len(trace.residuals),
        converged=converged,
        trace=trace,
    )


def magnitude_balance(
    p_u: np.ndarray,
    p_v: np.ndarray,
    d_out_tilde: np.ndarray,
    d_in_tilde: np.ndarray,
    *,
    scale_floor: float = DEFAULT_SCALE_FLOOR,
    target: np.ndarray | None = None,
) -> BalanceResult:
    """Latent magnitude balancing ([NQ] Step 2-3, Eqs. 7-9).

    Recovers unscaled proxies ``Û = D~_out⁻¹ P_U`` and ``V̂ = D~_in⁻¹ P_V``,
    computes ``η = sqrt(||V̂||_F / ||Û||_F)`` (Eq. 7), extracts scales
    ``[s1]_i = mean(|η û_i|)`` and ``[s2]_j = mean(|η⁻¹ v̂_j|)`` (Eq. 8),
    sets the final latents ``U := η Û``, ``V := η⁻¹ V̂`` (Eq. 9), and fixes
    the binaries ``Ub = sign(U)``, ``Vb = sign(V)`` with sign(0) = +1.
    Dead rows (all-zero) receive the clamped ``scale_floor`` (A-11).

    ``target``: the ORIGINAL (un-preconditioned) weight ``W``. Eq. 7 fixes
    only the U/V norm *ratio*, not the absolute scale of the factorized
    product ``Ub Vbᵀ ⊙ s``; the initialization-scale analysis (design
    implementation notes, alpha refit) shows the raw balanced product can
    sit orders of magnitude off ``W``. When ``target`` is given, the
    remaining scalar degree of freedom is closed by the least-squares
    global refit ``α = <W, Ub Vbᵀ> / ||Ub Vbᵀ||²_F``, folded into ``s1``
    (informed implementation choice; Phase 3 recalibrates the scales
    afterwards). Without ``target`` the Eq. 7 scale is kept as-is.
    """
    du = np.asarray(d_out_tilde, dtype=np.float64)
    di = np.asarray(d_in_tilde, dtype=np.float64)
    pu = np.asarray(p_u, dtype=np.float64)
    pv = np.asarray(p_v, dtype=np.float64)
    if pu.shape[0] != du.shape[0] or pv.shape[0] != di.shape[0]:
        raise ValueError("proxy rows must match preconditioner lengths")
    if pu.shape[1] != pv.shape[1]:
        raise ValueError("proxy ranks disagree")
    if np.any(du <= 0.0) or np.any(di <= 0.0):
        raise ValueError("preconditioner entries must be positive")

    u_hat = pu / du[:, None]
    v_hat = pv / di[:, None]
    n_u = float(np.linalg.norm(u_hat, "fro"))
    n_v = float(np.linalg.norm(v_hat, "fro"))
    if n_u <= 0.0 or n_v <= 0.0:
        raise ValueError("magnitude balancing is undefined for an all-zero factor")

    eta = float(np.sqrt(n_v / n_u))
    u = eta * u_hat
    v = v_hat / eta

    ub = np.where(u >= 0.0, np.int8(1), np.int8(-1)).astype(np.int8)
    vb = np.where(v >= 0.0, np.int8(1), np.int8(-1)).astype(np.int8)

    s1 = np.mean(np.abs(u), axis=1)
    s2 = np.mean(np.abs(v), axis=1)

    if target is not None:
        w = np.asarray(target, dtype=np.float64)
        if w.shape != (ub.shape[0], vb.shape[0]):
            raise ValueError("target must have shape (dout, din)")
        # Weighted global refit: the balanced product already carries the
        # s1/s2 magnitudes, so the scalar degree of freedom is fitted
        # against the CURRENT product M = (s1 ⊙ Ub)(s2 ⊙ Vb)ᵀ — fitting
        # the raw sign matrix Ub Vbᵀ instead would charge the scales'
        # magnitude twice and shrink the product orders of magnitude below
        # W (observed as a ~0.99 relative reconstruction error on an exact
        # rank-r target; crystallized in design.md implementation notes).
        b = (s1[:, None] * ub.astype(np.float64)) @ (s2[:, None] * vb.astype(np.float64)).T
        denom = float((b * b).sum())
        if denom > 0.0:
            alpha = float((w * b).sum()) / denom
            if np.isfinite(alpha) and alpha > 0.0:
                s1 = s1 * alpha

    # Dead-channel guard (A-11): clamp floors, never NaN or 0.
    s1 = np.maximum(s1, scale_floor)
    s2 = np.maximum(s2, scale_floor)
    return BalanceResult(U=u, V=v, Ub=ub, Vb=vb, s1=s1, s2=s2, eta=eta)
