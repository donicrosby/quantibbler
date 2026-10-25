"""ADMM convergence and non-destruction of its own initialization.

The shipped schedule `[1e-2 -> 1e2]` never converged in any measured run
(`converged=False` at 400/400) and the iterations actively degraded their own
truncated-SVD warm start (continuous cos 0.84 -> 0.32 on real layers.0 at rank
512). `rho_end` is now capped at `1e-1` in the normalized regime, where the
target is divided by its own Frobenius norm at ADMM entry.

Scope note on what is NOT gated here. `converged` is a MEASUREMENT, not a gate:
with a +-1-structured SVID proxy the consensus term pulls the latents toward a
discrete class while the continuous solve works in R, so the 1e-7 consensus
tolerance is unreachable by construction and the solver finishes at the K cap.
Likewise the raw product is not required to beat the truncated-SVD warm start:
that warm start is an UNCONSTRAINED optimum, while the proxy deliberately pushes
the latents into the signed rank-1 subspace, and the paper's pipeline compensates
in STE refinement (Step 3). The gated claims are that the iteration does
meaningful work (residual falls; the continuous product beats a sign-only
assignment) and that it yields a usable initializer (cos > 0.5).

Assertion provenance (`qk-fix-admm-solver`, planner ruling, tasks.md Phase 2).
`converged is True` and `cos >= warm_start cos` are both dropped; the gates are
`cos(U V^T, W) > 0.5` and `cos(U V^T, W) > cos(sign(Z_u) sign(Z_v)^T, W)`. One
adverb in that ruling is deliberately NOT asserted, because it is measurably
false: the residual does not fall MONOTONICALLY across the first 200 iterations
-- 67 of the 199 steps rise on this fixture, and every one of the 20 recorded
real layers.0 probe rows (artifacts/lane0-fire-r3-20261024/rows.jsonl) is
non-monotone too, 2-66 rises each. The requirement of record folded in
`openspec/specs/qk-ptq/spec.md` says only that the residual "decreases across
iterations"; the kill-only form of that is what is gated below (net decrease
over the first half, terminal value at the schedule's own floor). Asserting the
adverb would be asserting a convex descent this non-convex iteration does not
have.

Target is preconditioned per [NQ] Step 2-1 (`robust_diag`, gamma = 0.2) as the
production path does, so the test measures the regime the defect was found in.
"""

import numpy as np
import pytest

from quantibbler.ptq.calibration import robust_diag
from quantibbler.ptq.config import PTQConfig
from quantibbler.ptq.lb_admm import lb_admm


def _res_of(solved):
    w_tilde, cfg, res = solved
    return np.asarray(res.trace.residuals, dtype=np.float64), cfg, res


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.sum(a * b)) / (na * nb)


@pytest.fixture(scope="module")
def preconditioned_target():
    """Planted rank-32 target, rank-64 factorization, preconditioned gamma=0.2.

    The ruling's fixture premise is measured here rather than assumed: the
    unconstrained truncated-SVD warm start must already represent this target at
    cos >= 0.99, so post-ADMM cos > 0.5 is a floor on how far the proxy may drag
    the fit, not a low bar. Measured: warm-start cos 0.999715, post-ADMM
    cos(U V^T, W) 0.933478.
    """
    dout, din, rank = 256, 128, 64
    rng = np.random.default_rng(0)
    b = rng.normal(size=(dout, 32))
    c = rng.normal(size=(32, din))
    w = (b @ c) / np.sqrt(32.0) + 0.05 * rng.normal(size=(dout, din))
    gamma = 0.2
    d_in_t, d_out_t = robust_diag(np.linalg.norm(w, axis=0), np.linalg.norm(w, axis=1), gamma=gamma)
    w_tilde = (d_out_t[:, None] * w) * d_in_t[None, :]
    ub, sb, vb = np.linalg.svd(w_tilde, full_matrices=False)
    warm_cos = _cos(w_tilde, (ub[:, :rank] * sb[:rank]) @ vb[:rank, :])
    assert warm_cos >= 0.99, (
        f"fixture premise violated: warm start represents the target at cos={warm_cos:.6f} < 0.99"
    )
    return rank, w_tilde, gamma


@pytest.fixture(scope="module")
def solved(preconditioned_target):
    """One 400-iteration solve, shared by the three assertions below.

    All three tests use identical inputs, so solving per-test tripled the
    module's wall time (~45s per solve) for zero extra coverage -- enough on
    its own to stop the gate suite fitting in one foreground window.
    """
    rank, w_tilde, gamma = preconditioned_target
    cfg = PTQConfig(solver_backend="numpy", rank=rank, gamma=gamma)
    return w_tilde, cfg, lb_admm(w_tilde, cfg)


def test_consensus_residual_decreases_over_the_first_half(solved):
    """The iteration SHALL make real progress before it hits the K cap.

    `converged` is diagnostics only; the gates are that the schedule reaches
    its own residual floor, that it makes net progress over the first half of
    the budget, and that the run terminates at epsilon or at the cap.

    Measured at HEAD (numpy backend, OPENBLAS_NUM_THREADS=1, K=400, rho in
    [0.0, 1.0], admm_reg 3e-2): first-half residual 0.4243 -> 0.3443, all-400
    terminal 0.01738 == trace min, `converged=False` at 400/400.
    """
    residuals, cfg, res = _res_of(solved)

    residuals = np.asarray(res.trace.residuals, dtype=np.float64)
    half = max(1, cfg.admm_max_iter // 2)
    window = residuals[:half]
    assert window.size >= 2
    # The schedule drives the residual to its own floor. Monotonicity is NOT
    # asserted: measured shape rises to 0.877 by iteration 5 before descending,
    # with 67 of the first-half steps rising, because the warm start is pulled
    # onto the SVID proxy before the data term reasserts itself. That is a
    # feature of a non-convex iteration, not noise, and requiring zero rises
    # would assert a purely convex descent this solver does not have -- refuted
    # by measurement on this fixture and on all 20 real layers.0 rows.
    assert residuals[-1] <= 1.1 * residuals.min(), (
        f"schedule did not reach its own floor: terminal {residuals[-1]:.6g} vs "
        f"min {residuals.min():.6g} ({residuals[-1] / residuals.min():.4f}x)"
    )
    assert window[-1] < window[0], (
        f"no net progress over the first {half} iterations: {window[0]:.6g} -> {window[-1]:.6g}"
    )

    assert res.converged or res.n_iter == cfg.admm_max_iter
    if res.converged:
        assert residuals[-1] < cfg.admm_epsilon
        assert res.n_iter < cfg.admm_max_iter
    assert isinstance(res.converged, bool)


def test_raw_product_is_a_usable_initializer(solved):
    """The solve SHALL yield a usable starting point for STE refinement.

    The ruling names `cos(U V^T, W)`: U/V are the continuous factors and the
    product is what reconstructs the target. Measured 0.933478 (the consensus
    proxies that feed `magnitude_balance`, `P_u P_v^T`, measure 0.941705).
    """
    w_tilde, cfg, res = solved
    raw = np.asarray(res.U) @ np.asarray(res.V).T
    cos_raw = _cos(w_tilde, raw)
    assert cos_raw > 0.5, f"raw product cos={cos_raw:.6f} is not a usable initializer"


def test_continuous_product_beats_a_sign_only_assignment(solved):
    """The continuous latents SHALL carry more than their signs alone.

    Binarize-and-go on the final consensus proxies is the baseline; the
    continuous product must beat it, which is what shows the iteration did
    meaningful work rather than collapsing onto the proxy. The ruling writes the
    baseline as `sign(zu) sign(zv)^T`; `zu = svid_proxy(P_u)` preserves its
    input's sign by construction, so `sign(P_u)` is that operand. Measured:
    continuous 0.933478 vs sign-only 0.883145.
    """
    w_tilde, cfg, res = solved

    su = np.sign(np.asarray(res.P_u, dtype=np.float64))
    su[su == 0.0] = 1.0
    sv = np.sign(np.asarray(res.P_v, dtype=np.float64))
    sv[sv == 0.0] = 1.0
    cos_raw = _cos(w_tilde, np.asarray(res.U) @ np.asarray(res.V).T)
    cos_sign_only = _cos(w_tilde, su @ sv.T)
    assert cos_raw > cos_sign_only, (
        f"continuous cos={cos_raw:.6f} does not beat the sign-only baseline cos={cos_sign_only:.6f}"
    )
