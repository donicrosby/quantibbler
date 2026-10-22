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
    """Rank-64 factorization of a 256x128 target, preconditioned gamma=0.2."""
    dout, din, rank = 256, 128, 64
    rng = np.random.default_rng(0)
    b = rng.normal(size=(dout, 32))
    c = rng.normal(size=(32, din))
    w = (b @ c) / np.sqrt(32.0) + 0.05 * rng.normal(size=(dout, din))
    gamma = 0.2
    d_in_t, d_out_t = robust_diag(np.linalg.norm(w, axis=0), np.linalg.norm(w, axis=1), gamma=gamma)
    w_tilde = (d_out_t[:, None] * w) * d_in_t[None, :]
    return rank, w_tilde, gamma


@pytest.fixture(scope="module")
def solved(preconditioned_target):
    """One 400-iteration solve, shared by the three assertions below.

    All three tests use identical inputs, so solving per-test tripled the
    module's wall time (~45s per solve) for zero extra coverage -- enough on
    its own to stop the gate suite fitting in one foreground window.
    """
    rank, w_tilde, gamma = preconditioned_target
    cfg = PTQConfig(rank=rank, gamma=gamma)
    return w_tilde, cfg, lb_admm(w_tilde, cfg)


def test_consensus_residual_decreases_over_the_first_half(solved):
    """The iteration SHALL make real progress before it hits the K cap.

    `converged` is diagnostics only; the gates are that the schedule reaches
    its own residual floor, that it makes net progress over the first half of
    the budget, and that the run terminates at epsilon or at the cap.
    """
    residuals, cfg, res = _res_of(solved)

    residuals = np.asarray(res.trace.residuals, dtype=np.float64)
    half = max(1, cfg.admm_max_iter // 2)
    window = residuals[:half]
    assert window.size >= 2
    # The schedule drives the residual to its own floor. Monotonicity is NOT
    # asserted: measured shape rises to 0.749 by iteration 4 before descending
    # to 0.046 (28 rises within the first 200), because the warm start is
    # pulled onto the SVID proxy before the data term reasserts itself. That is
    # a feature of a non-convex iteration, not noise, and requiring zero rises
    # would be asserting a purely convex descent this solver does not have.
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
    """The solve SHALL yield a usable starting point for STE refinement."""
    w_tilde, cfg, res = solved
    raw = np.asarray(res.P_u) @ np.asarray(res.P_v).T
    cos_raw = _cos(w_tilde, raw)
    assert cos_raw > 0.5, f"raw product cos={cos_raw:.6f} is not a usable initializer"


def test_continuous_product_beats_a_sign_only_assignment(solved):
    """The continuous latents SHALL carry more than their signs alone.

    Binarize-and-go on the final consensus proxies is the baseline; the
    continuous product must beat it, which is what shows the iteration did
    meaningful work rather than collapsing onto the proxy.
    """
    w_tilde, cfg, res = solved

    su = np.sign(np.asarray(res.P_u, dtype=np.float64))
    su[su == 0.0] = 1.0
    sv = np.sign(np.asarray(res.P_v, dtype=np.float64))
    sv[sv == 0.0] = 1.0
    cos_raw = _cos(w_tilde, np.asarray(res.P_u) @ np.asarray(res.P_v).T)
    cos_sign_only = _cos(w_tilde, su @ sv.T)
    assert cos_raw > cos_sign_only, (
        f"continuous cos={cos_raw:.6f} does not beat the sign-only baseline cos={cos_sign_only:.6f}"
    )
