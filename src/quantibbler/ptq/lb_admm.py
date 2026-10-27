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

The solver computes in the DECLARED compute dtype (``PTQConfig.compute_dtype``,
bf16 by default). Sites that SUM over the design -- the normal equations, the
Cholesky factorization, the SVID rank-1 fit, the balancing norms -- execute at
the declared accumulation dtype (float32 under bf16) and are named as such at
their call sites; the latents and scales are stored at ``compute_dtype``. The
NumPy backend is the float64 parity oracle and reaches the same code at the
float64 grade, which is what the fixtures pin.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .accumulation import accumulation_spec
from .config import PTQConfig
from .progress import ProgressReporter
from .solver_backend import get_backend, solver_backend_provenance

__all__ = [
    "ADMMResult",
    "ADMMTrace",
    "BalanceResult",
    "DEFAULT_SCALE_FLOOR",
    "SOLVER_PROBE_ENV",
    "admm_reg_provenance",
    "lb_admm",
    "magnitude_balance",
    "rho_end_provenance",
    "svid_proxy",
]

#: Solver-probe flag. Enables the per-linear diagnostics: the ``min_eigs``
#: eigenvalue probe in the solve loop below, and ``block_recon``'s per-linear
#: JSONL writer (which imports this name). It is declared HERE because the gate
#: is the solver's own -- one name for both readers, so a gate and a writer can
#: never drift apart. Off (unset) by default: a production run pays no
#: diagnostic tax.
SOLVER_PROBE_ENV = "QK_SOLVER_PROBE"

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
    #: Smallest eigenvalue of each solved system matrix. DIAGNOSTIC: populated
    #: only when the solver probe is enabled (``QK_SOLVER_PROBE``) -- the SPD
    #: proof a production run relies on is the Cholesky solve itself, recorded
    #: always-on in :attr:`n_jitter`, so an empty list here is not a missing
    #: signal. Read it as "no diagnostic was requested", never as "not SPD".
    min_eigs: list[float] = field(default_factory=list)
    n_cholesky: int = 0
    n_jitter: int = 0
    svid_sign_preserved: list[bool] = field(default_factory=list)
    # ``mean(diag(Xn^T Xn))`` of the row/col-normalized design, one entry
    # per continuous half-update (two per iteration: U-update then V-update).
    # This is the scale the ridge is anchored to: with a unit-diagonal design it
    # is ~1 regardless of layer scale, which is the regime's whole claim. Recorded
    # rather than inferred so a reader can see the anchor the solve used.
    ridge_scale: list[float] = field(default_factory=list)


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


def _cholesky_factor(xp, h: np.ndarray, trace: ADMMTrace | None = None) -> np.ndarray:
    """Stabilized Cholesky factorization of an SPD system matrix.

    Attempts ``numpy.linalg.cholesky``; on a non-positive-definite system
    (LinAlgError) or a factor with non-finite entries, adds a small relative
    jitter to the diagonal shift and retries (design section 2). The jitter
    path is counted in ``trace.n_jitter``; every successful first-try
    factorization counts in ``trace.n_cholesky``.
    """
    attempt = xp.clone(h)
    diag_scale = max(float(xp.abs(xp.diag(h)).max()), 1.0)
    jitter = _JITTER * diag_scale
    for k in range(60):
        try:
            factor = xp.linalg.cholesky(attempt)
            if xp.all(xp.isfinite(factor)):
                if k > 0 and trace is not None:
                    trace.n_jitter += 1
                elif k == 0 and trace is not None:
                    trace.n_cholesky += 1
                return factor
        except xp.linalg.LinAlgError:
            pass
        # Stabilization: relative jitter on the diagonal shift, escalated
        # geometrically so genuinely indefinite systems recover deterministically.
        attempt = h + jitter * xp.eye(h.shape[0])
        jitter *= 2.0
    raise xp.linalg.LinAlgError("stabilized Cholesky failed to produce an SPD factor")


def _cholesky_solve(xp, h: np.ndarray, b: np.ndarray, trace: ADMMTrace | None = None):
    """Solve ``H X = B`` via stabilized Cholesky.

    Returns ``(solution, jittered)`` where ``jittered`` marks that the
    fallback engaged (exposed for tests; production callers pass a trace).
    """
    factor = _cholesky_factor(xp, h, trace)
    y = xp.linalg.solve(factor, b)  # forward substitution equivalent
    x = xp.linalg.solve(factor.T, y)  # back substitution equivalent
    jittered = trace is not None and trace.n_jitter > 0
    # A direct jittered flag for the deterministic test path: recompute.
    try:
        xp.linalg.cholesky(h)
        jittered = False
    except xp.linalg.LinAlgError:
        jittered = True
    return x, jittered


def _top_rank1(xp, mag: np.ndarray, *, steps: int = 100, tol: float = 1e-14, dtype=None):
    """Top singular triplet of a NON-NEGATIVE matrix, deterministically.

    ``mag`` is ``|P|``, so it is entrywise non-negative and the Perron-
    Frobenius theorem puts the leading singular vectors in the non-negative
    orthant. Power iteration from the all-ones vector therefore converges to
    the leading pair without any sign ambiguity, and a planted rank-1 matrix
    is recovered in a single step. Returns ``(sigma, u, v)`` with unit ``u``
    and ``v``; an all-zero input returns ``(0.0, None, None)``.

    An explicit top-rank-1 solve rather than a full ``xp.linalg.svd``: the
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
    # The power iteration runs at ``dtype`` -- the caller's accumulation dtype.
    # Defaulting to the float64 parity grade keeps the NumPy oracle arm
    # bit-identical; a bf16/float32 run passes its accumulation dtype so the
    # norms and the matvecs do not silently widen past the declared regime.
    vec_dtype = xp.dtype("float64") if dtype is None else dtype
    v = xp.full(n, 1.0 / xp.sqrt(n), dtype=vec_dtype)
    u = mag @ v
    sigma = float(xp.linalg.norm(u))
    if not xp.isfinite(sigma) or sigma <= 0.0:
        return 0.0, None, None
    u = u / sigma
    for _ in range(steps):
        v_new = mag.T @ u
        nv = float(xp.linalg.norm(v_new))
        if nv <= 0.0:
            break
        v_new = v_new / nv
        u_new = mag @ v_new
        nu = float(xp.linalg.norm(u_new))
        if nu <= 0.0:
            break
        u_new = u_new / nu
        moved = abs(nu - sigma)
        u, v, sigma = u_new, v_new, nu
        if moved <= tol * max(nu, 1.0):
            break
    return sigma, u, v


def svid_proxy(xp, z: np.ndarray, *, dtype=None) -> np.ndarray:
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
    # accumulation-site: the rank-1 fit sums over the d_out x r entries of |P|,
    # so it runs at the declared accumulation dtype (float32 under bf16 compute);
    # ``None`` is the float64 parity grade the NumPy oracle arm keeps.
    work_dtype = xp.dtype("float64") if dtype is None else dtype
    zf = xp.asarray(z, dtype=work_dtype)
    signs = xp.sign(zf)
    signs[signs == 0.0] = 1.0  # sign(0) = +1 (A-11 convention)

    sigma, u1, v1 = _top_rank1(xp, xp.abs(zf), dtype=work_dtype)
    if u1 is None:
        out = xp.zeros_like(zf)
    else:
        root = xp.sqrt(sigma)
        out = signs * xp.outer(u1 * root, v1 * root)

    # No output value is ever zero (sign(0) = +1 must hold on every entry).
    out = xp.where(out == 0.0, DEFAULT_SCALE_FLOOR, out)

    # Dead columns (all-zero input) would be exactly zero, which carries no
    # sign; clamp them to the positive floor so sign(0) = +1 holds and no
    # output value is ever zero (A-11, matching kernels.sign_estimate).
    col_norm = xp.linalg.norm(zf, axis=0, keepdims=True)
    dead = (col_norm == 0.0).ravel()
    if xp.any(dead):
        out[:, dead] = DEFAULT_SCALE_FLOOR
    return out


def _warm_start(xp, w: np.ndarray, r: int, dt: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """Truncated rank-r SVD warm start (A-6): U = U_r, V = V_r (economic).

    ``gesdd`` (numpy's default SVD driver) is known to declare
    non-convergence on finite but extreme-dynamic-range inputs; after
    scale normalization the target is O(1) in norm, so a single retry on
    the transpose (different reduction order, the classical gesdd
    workaround) plus a ``gesvd``-style eigendecomposition fallback covers
    the flaky case. A target with non-finite entries is a pipeline bug,
    not an SVD problem: it fails here with a loud, diagnosable error.
    """
    if not xp.all(xp.isfinite(w)):
        bad = int(xp.count_nonzero(~xp.isfinite(w)))
        raise ValueError(
            f"warm start: target has {bad} non-finite entries "
            f"(max|w|={xp.nanmax(xp.abs(w)):.3e}) — TuneFP/calibration "
            "produced NaN/Inf upstream"
        )
    try:
        u_mat, s_vec, vt_mat = xp.linalg.svd(w, full_matrices=False)
    except xp.linalg.LinAlgError:
        try:
            u_mat, s_vec, vt_mat = xp.linalg.svd(w.T, full_matrices=False)
            vt_mat, s_vec, u_mat = (
                u_mat.T,
                s_vec,
                vt_mat.T,
            )
        except xp.linalg.LinAlgError:
            # symmetric eig on the Gram matrix: slower, unconditionally
            # stable for finite inputs.
            s2_vec, v_mat = xp.linalg.eigh(w.T @ w)
            order = xp.argsort(s2_vec)[::-1][:r]
            s_vec = xp.sqrt(xp.maximum(s2_vec[order], 0.0))
            v0 = v_mat[:, order]
            u0 = w @ v0
            nrm = xp.linalg.norm(u0, axis=0, keepdims=True)
            u0 = u0 / xp.maximum(nrm, 1e-300)
            return (
                xp.ascontiguousarray(u0 * s_vec[None, :], dtype=dt),
                xp.ascontiguousarray(v0 * s_vec[None, :], dtype=dt),
            )
    u0 = u_mat[:, :r] * xp.sqrt(s_vec[:r])
    v0 = vt_mat[:r].T * xp.sqrt(s_vec[:r])
    return xp.ascontiguousarray(u0, dtype=dt), xp.ascontiguousarray(v0, dtype=dt)


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
    return rho_end_provenance(cfg.rho_end)[0]


def rho_end_provenance(default: float) -> tuple[float, str]:
    """Effective ``rho_end`` and where it came from: ``"env"`` or ``"config"``.

    The value alone is not enough to audit a run. Two attribution passes differ
    in exactly this number, and neither the report nor the launch line is
    guaranteed to survive beside the artifact -- so the source travels with the
    value, and a run cannot be silently mislabelled about which arm it was.
    """
    raw = os.environ.get("QK_RHO_END")
    if raw is None:
        return default, "config"
    try:
        return float(raw), "env"
    except ValueError:
        msg = f"QK_RHO_END must be a float, got {raw!r}"
        raise ValueError(msg) from None


def admm_reg_provenance(default: float) -> tuple[float, str]:
    """Effective ``admm_reg`` and where it came from: ``"env"`` or ``"config"``.

    Exact mirror of :func:`rho_end_provenance` for the normalized regime's
    stabilizer floor. ``QK_ADMM_REG`` is a solve-time override, so it must be
    resolved where the solve happens (not at ``PTQConfig()`` construction: a
    dataclass cannot see the env var at the moment the solve reads it, and
    freezing it at construction would let a config built before the env was set
    solve under a value its provenance never names).

    Fail-closed on a malformed value: silently falling back to the default
    would produce a run whose provenance claims an override it never applied.
    """
    raw = os.environ.get("QK_ADMM_REG")
    if raw is None:
        return default, "config"
    try:
        return float(raw), "env"
    except ValueError:
        msg = f"QK_ADMM_REG must be a float, got {raw!r}"
        raise ValueError(msg) from None


def _admm_reg(cfg: PTQConfig) -> float:
    """`cfg.admm_reg`, with the solve-time ``QK_ADMM_REG`` override applied."""
    return admm_reg_provenance(cfg.admm_reg)[0]


def _col_scales(xp, x: np.ndarray) -> np.ndarray:
    """Per-column L2 norms of a design matrix, floored so the inverse is finite.

    A dead column (all-zero, e.g. a collapsed latent) would make the map-back a
    division by zero; the floor keeps it finite instead of poisoning the solve
    with NaN. Live latents are nowhere near the float64 floor, so this is a
    guard, not a regularizer.
    """
    s = xp.linalg.norm(x, axis=0)
    return xp.maximum(s, DEFAULT_SCALE_FLOOR)


def _ridge_shift(xp, design_n: np.ndarray, rho: float, reg: float) -> tuple[float, float]:
    """Scale-relative stabilizer: ``rho * mean(diag(Xn^T Xn)) + reg``.

    ``design_n`` is the column-normalized fixed partner factor, so its Gram
    diagonal is unit-scale by construction and the returned ``scale`` is ~1 for
    every layer: the ridge no longer depends on raw layer magnitude, which is
    what lets one rho schedule serve all layers. Returns ``(shift, scale)`` with
    ``scale`` recorded on the trace.
    """
    # mean(diag(Xn^T Xn)) == ||Xn||_F^2 / r exactly; one norm instead of a Gram
    # build, and it stays inside the backend surface the solver vets.
    r = int(design_n.shape[1])
    scale = float(xp.linalg.norm(design_n)) ** 2 / r
    return rho * scale + reg, scale


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
    # Resolved at the point of consumption, mirroring _rho_end/_admm_reg: the
    # SOLVER_BACKEND env knob must be able to select the arm for THIS solve, and
    # the run must not be mislabelled about it. An explicit device env is honoured
    # (diagnostic path); implicit torch on a CPU box fails closed.
    backend_name, _backend_source = solver_backend_provenance(cfg.solver_backend)
    xp = get_backend(backend_name, device=os.environ.get("QUANTIBBLER_DEVICE"))
    dt = xp.dtype(cfg.compute_dtype)
    # The accumulation dtype follows from the declared compute dtype (float32
    # under bf16). Sites that SUM over the design widen to it explicitly; the
    # latents stay at ``dt``. torch has no bf16 CPU SVD kernel at all
    # ("linalg_svd_cpu not implemented for 'BFloat16'"), so the warm start is one
    # of those sites rather than an exception.
    adt = xp.dtype(accumulation_spec(cfg.compute_dtype))
    w = xp.asarray(w_tilde, dtype=dt)
    if w.ndim != 2:
        raise ValueError(f"preconditioned target must be 2-D, got shape {w.shape}")
    dout, din = w.shape
    r = cfg.rank
    if r > min(dout, din):
        raise ValueError(f"rank r={r} exceeds min(dout, din)={min(dout, din)}")

    # Scale normalization: the rho schedule and epsilon are ABSOLUTE
    # constants (A-2/A-3), so the solve must happen in the O(1) scale
    # class they were calibrated in. Robust preconditioning on real
    # calibration statistics can crush ||W~||_F to ~1e-4 (measured:
    # s1~2e-4, s2~0.3), where the consensus term dominates the data term by ~1e8
    # and the solve "converges" to mush in a handful of iterations
    # (measured: 4/400 iterations, binary cos 0.02 — every production
    # artifact ever exported had cos ~0.001 per layer). Solving on
    # W~/||W~||_F and rescaling the factors on exit IS scale-equivariant only
    # because the exit rescale splits the factor across the pair (see the
    # sqrt below); applying target_scale to each factor overshoots by
    # target_scale. Fixture-scale targets (||W~||~O(1)) are no-ops.
    target_scale = float(xp.linalg.norm(w))
    if not (xp.isfinite(target_scale) and target_scale > 0.0):
        target_scale = 1.0  # degenerate all-zero target: keep legacy behavior
    if target_scale != 1.0:
        w = w / target_scale

    trace = ADMMTrace()
    # The ITERATE lives at the accumulation dtype and is cast to the compute
    # dtype at the return boundary — that is the storage contract ("latents
    # stored at compute_dtype") applied where storage actually happens. Rounding
    # the iterate to bf16 on every one of up to 400 iterations would discard the
    # accumulator's guard bits at each step, which is the failure the widening
    # exists to prevent. At float32/float64 compute adt == dt, so those arms are
    # bit-unchanged.
    w_a = xp.astype(w, adt)
    # SVD warm start at the ACCUMULATION dtype: a bf16 SVD is both
    # platform-dependent and, on torch CPU, unimplemented.
    u, v = _warm_start(xp, w_a, r, adt)
    zu = xp.astype(svid_proxy(xp, u, dtype=adt), adt)
    zv = xp.astype(svid_proxy(xp, v, dtype=adt), adt)
    lu = xp.zeros((dout, r), dtype=adt)
    lv = xp.zeros((din, r), dtype=adt)
    eye_r = xp.eye(r, dtype=adt)

    def residual(u_cur, v_cur, zu_cur, zv_cur) -> float:
        du = float(xp.linalg.norm(u_cur - zu_cur, "fro"))
        dv = float(xp.linalg.norm(v_cur - zv_cur, "fro"))
        return max(du, dv) / (1.0 + float(xp.linalg.norm(u_cur, "fro")))

    # Progress is wired here because this loop is the spec's own worst case: 400
    # iterations at ~0.2s each, where per-iteration output would be 400 lines
    # around the handful that matter. The first tick emits immediately, so a
    # reader sees the solve start rather than inferring it from silence.
    reporter = ProgressReporter("admm", cfg.admm_max_iter)

    converged = False
    prev_res = xp.inf
    denom = max(cfg.admm_max_iter - 1, 1)
    rho_end = _rho_end(cfg)
    # Resolved once at solve time, mirroring rho_end: the effective stabilizer
    # floor is env-overridable and must be the value the iteration actually used.
    reg = _admm_reg(cfg)
    # Diagnostic gate, resolved once from the env at the solve site (like rho/reg,
    # and NEVER from the config, which the checkpoint fingerprint hashes). The
    # per-iteration eigenvalue probe is a fixed per-linear tax -- two dense
    # `eigvalsh` on the r x r system matrices, which do NOT shrink while the
    # GEMMs around them do -- and it duplicates a proof the always-on path
    # already carries: `_cholesky_solve` below is the SPD factorization, and
    # `trace.n_jitter` records every non-SPD event. So the solve runs without it
    # unless the solver probe asks for the column.
    probe = bool(os.environ.get(SOLVER_PROBE_ENV))
    for k in range(cfg.admm_max_iter):
        reporter.tick(k + 1)
        # Linear penalty schedule spanning [rho_start, rho_end] across the
        # K iterations (A-2).
        rho = float(cfg.rho_start + (rho_end - cfg.rho_start) * k / denom)

        # Continuous update for U (Eq. 5), regressed against the COLUMN-NORMALIZED
        # design X = V. With W ~ U V^T and V = Vn D (D = diag(column norms)), the
        # normalized-basis unknown is Y = U D, so the constraint U = Z_U becomes
        # Y = Z_U D and the dual pair maps by the same D. Solving in that basis
        # gives X^T X a unit diagonal, so the scale-relative stabilizer
        # rho*mean(diag(X^T X)) + reg is O(1) for every layer. Y D^-1 maps the
        # iterate back, keeping U V^T a valid factorization of the target.
        # Accumulation sites: the column norms and the normal equation below sum
        # over din/dout terms, so they execute at ``adt``; ``vn`` is the widened
        # design (the iterate stays at adt, so this is a no-op above bf16).
        sv = _col_scales(xp, v)
        vn = v / sv
        shift_u, scale_u = _ridge_shift(xp, vn, rho, reg)
        trace.ridge_scale.append(scale_u)
        hu = vn.T @ vn + shift_u * eye_r
        rhs_u = vn.T @ w_a.T + rho * ((zu * sv) - (lu * sv)).T
        # min-eig check: the documented SPD probe, gated on the solver probe
        # (`probe` above) -- NOT on being cheap, which it is not at production
        # thread counts. The Cholesky solve on the next line is the SPD proof
        # the production path keeps.
        if probe:
            trace.min_eigs.append(float(xp.linalg.eigvalsh(hu).min()))
        u = _cholesky_solve(xp, hu, rhs_u, trace)[0].T / sv

        # Symmetric update for V, regressed against the column-normalized design
        # X = U (unknown Y = V D_u, mapped back by D_u^-1).
        su = _col_scales(xp, u)
        un = u / su
        shift_v, scale_v = _ridge_shift(xp, un, rho, reg)
        trace.ridge_scale.append(scale_v)
        hv = un.T @ un + shift_v * eye_r
        rhs_v = un.T @ w_a + rho * ((zv * su) - (lv * su)).T
        if probe:
            trace.min_eigs.append(float(xp.linalg.eigvalsh(hv).min()))
        v = _cholesky_solve(xp, hv, rhs_v, trace)[0].T / su

        # Proxy update, Eq. 6: SVID of the consensus variable.
        pu = u + lu
        pv = v + lv
        # Both the proxy and the iterate live at ``adt``; only the RETURN
        # boundary casts to the declared compute dtype (see the loop-head note).
        zu_new = xp.astype(svid_proxy(xp, pu, dtype=adt), adt)
        zv_new = xp.astype(svid_proxy(xp, pv, dtype=adt), adt)
        trace.svid_sign_preserved.append(
            bool(xp.array_equal(xp.sign(zu_new), xp.where(pu == 0, 1, xp.sign(pu))))
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

    # The count reached, not the budget: an early-converged solve must not report
    # iterations it never ran.
    reporter.complete(len(trace.residuals))

    # The entry normalization divides the TARGET by target_scale exactly
    # once, but a rank factorization W ~ U V^T has TWO factors: scaling both
    # by target_scale scales the reconstructible matrix by target_scale^2 and
    # overshoots by a full factor of ||W~||_F. Split it across the pair so the
    # product carries it once -- the equivariance the entry comment claims.
    factor_scale = float(xp.sqrt(target_scale))
    # Storage boundary: the iterate is returned at the DECLARED compute dtype
    # (see the loop-head note). No-op when adt == dt.
    return ADMMResult(
        U=xp.astype(u * factor_scale, dt),
        V=xp.astype(v * factor_scale, dt),
        P_u=xp.astype((u + lu) * factor_scale, dt),
        P_v=xp.astype((v + lv) * factor_scale, dt),
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
    backend: str | None = None,
    dtype: str = "float64",
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
    # The device is resolved here, exactly as at the solve site: an explicit
    # QUANTIBBLER_DEVICE is honoured (the diagnostic/parity path a CPU-box driver
    # exports), while implicit torch on a box with no accelerator still fails
    # closed in resolve_device. A balance that silently ran somewhere other than
    # the solve it refits would be a device mismatch nothing records.
    xp = get_backend(backend, device=os.environ.get("QUANTIBBLER_DEVICE"))
    # accumulation-site: the Frobenius norms (Eq. 7) and the per-row mean-absolute
    # scale extraction (Eq. 8) both SUM over many entries, so the stage runs at
    # the declared accumulation dtype -- float32 under bf16 compute. ``dtype``
    # defaults to the float64 grade, which is what the NumPy oracle arm and every
    # float64 fixture keep.
    work = xp.dtype(dtype)
    du = xp.asarray(d_out_tilde, dtype=work)
    di = xp.asarray(d_in_tilde, dtype=work)
    pu = xp.asarray(p_u, dtype=work)
    pv = xp.asarray(p_v, dtype=work)
    if pu.shape[0] != du.shape[0] or pv.shape[0] != di.shape[0]:
        raise ValueError("proxy rows must match preconditioner lengths")
    if pu.shape[1] != pv.shape[1]:
        raise ValueError("proxy ranks disagree")
    if xp.any(du <= 0.0) or xp.any(di <= 0.0):
        raise ValueError("preconditioner entries must be positive")

    u_hat = pu / du[:, None]
    v_hat = pv / di[:, None]
    n_u = float(xp.linalg.norm(u_hat, "fro"))
    n_v = float(xp.linalg.norm(v_hat, "fro"))
    if n_u <= 0.0 or n_v <= 0.0:
        raise ValueError("magnitude balancing is undefined for an all-zero factor")

    eta = float(xp.sqrt(n_v / n_u))
    u = eta * u_hat
    v = v_hat / eta

    ub = xp.where(u >= 0.0, xp.scalar(1, xp.dtype("int8")), xp.scalar(-1, xp.dtype("int8")))
    vb = xp.where(v >= 0.0, xp.scalar(1, xp.dtype("int8")), xp.scalar(-1, xp.dtype("int8")))

    s1 = xp.mean(xp.abs(u), axis=1)
    s2 = xp.mean(xp.abs(v), axis=1)

    if target is not None:
        w = xp.asarray(target, dtype=work)
        if w.shape != (ub.shape[0], vb.shape[0]):
            raise ValueError("target must have shape (dout, din)")
        # Weighted global refit: the balanced product already carries the
        # s1/s2 magnitudes, so the scalar degree of freedom is fitted
        # against the CURRENT product M = (s1 ⊙ Ub)(s2 ⊙ Vb)ᵀ — fitting
        # the raw sign matrix Ub Vbᵀ instead would charge the scales'
        # magnitude twice and shrink the product orders of magnitude below
        # W (observed as a ~0.99 relative reconstruction error on an exact
        # rank-r target).
        b = (s1[:, None] * xp.astype(ub, work)) @ (s2[:, None] * xp.astype(vb, work)).T
        denom = float((b * b).sum())
        if denom > 0.0:
            alpha = float((w * b).sum()) / denom
            if xp.isfinite(alpha) and alpha > 0.0:
                s1 = s1 * alpha

    # Dead-channel guard (A-11): clamp floors, never NaN or 0.
    s1 = xp.maximum(s1, scale_floor)
    s2 = xp.maximum(s2, scale_floor)
    return BalanceResult(U=u, V=v, Ub=ub, Vb=vb, s1=s1, s2=s2, eta=eta)
