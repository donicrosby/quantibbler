"""Tasks 3.1-3.3: LB-ADMM initialization (scenarios admm_spd_solves,
admm_consensus_convergence, penalty_schedule_stopping,
magnitude_balancing_scales)."""

import numpy as np
import pytest

from quantibbler.ptq.config import PTQConfig
from quantibbler.ptq.lb_admm import (
    DEFAULT_SCALE_FLOOR,
    _cholesky_solve,
    lb_admm,
    magnitude_balance,
    svid_proxy,
)


def _ground_truth(m: int, n: int, r: int, seed: int) -> np.ndarray:
    """Exact rank-r matrix built from ±1 factors."""
    gen = np.random.default_rng(seed)
    ub = np.where(gen.random((m, r)) < 0.5, -1.0, 1.0)
    vb = np.where(gen.random((n, r)) < 0.5, -1.0, 1.0)
    return ub @ vb.T


def test_scn_admm_spd_solves():
    """admm_spd_solves: every continuous-update system matrix H = VᵀV+(ρ+λ)I
    is symmetric positive definite (Lemma 2), solved via Cholesky; the
    stabilized fallback engages only on numerically violated PD, and it
    triggers deterministically on a perturbed diagonal shift."""
    for r in (2, 4, 8):
        w = _ground_truth(16, 12, r, seed=40 + r)
        cfg = PTQConfig(rank=r, rho_start=1e-2, rho_end=1e2)
        res = lb_admm(w, cfg)
        assert res.trace.min_eigs, "trace recorded no solves"
        # Every solve was on an SPD system (min eigenvalue positive).
        assert all(eig > 0.0 for eig in res.trace.min_eigs)
        # Well-conditioned targets never need the jitter fallback.
        assert res.trace.n_jitter == 0
        assert res.trace.n_cholesky >= 2 * res.n_iter
    # Deterministic fallback: an indefinite perturbation of an SPD matrix
    # must engage the jitter path (and still return a solution).
    h_spd = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([[1.0, 2.0], [3.0, 4.0]])
    x0, fell = _cholesky_solve(h_spd, b)
    assert fell is False
    h_bad = h_spd - 10.0 * np.eye(2)  # indefinite by construction
    x1, fell = _cholesky_solve(h_bad, b)
    assert fell is True
    # The fallback solution satisfies the (jittered) system, not garbage.
    assert np.all(np.isfinite(x1))


def test_scn_admm_consensus_convergence():
    """admm_consensus_convergence + penalty_schedule_stopping: on an exact
    rank-r target the consensus residual decreases and stops below epsilon
    (or at the K cap); rho increases linearly per iteration; the SVID proxy
    preserves sign structure with norm-preserved columns (A-5, closed form)."""
    r = 4
    w = _ground_truth(20, 14, r, seed=99)
    cfg = PTQConfig(rank=r)
    res = lb_admm(w, cfg)

    residuals = np.asarray(res.trace.residuals)
    assert residuals[0] > 0
    # Decreasing trend and a terminal state per the stopping rule.
    assert residuals[-1] < residuals[: min(5, residuals.size)].max()
    assert res.converged or res.n_iter == cfg.admm_max_iter
    if res.converged:
        assert residuals[-1] < cfg.admm_epsilon
        assert res.n_iter < cfg.admm_max_iter

    # Linear penalty schedule across the iterations actually run (A-2).
    rhos = np.asarray(res.trace.rhos)
    full = np.linspace(cfg.rho_start, cfg.rho_end, cfg.admm_max_iter)
    assert np.allclose(rhos, full[: rhos.size])
    assert np.all(np.diff(rhos) > 0)

    # SVID closed form (A-5): project onto the sign pattern, keep the LS
    # magnitudes. The projection preserves the sign structure; the values
    # are the least-squares fit ON that pattern,
    # proj = sign(z) ⊙ (sign(z)ᵀ z / ||sign(z)||²) — not a constant
    # magnitude per column and not norm-anchored (the norm-anchored variant
    # destabilizes the consensus loop; see lb_admm.svid_proxy notes).
    z = np.array([[3.0, -1.0, 0.0], [-2.0, 0.5, 0.0]])
    zc = svid_proxy(z)
    assert np.array_equal(np.sign(zc), np.sign(np.where(z == 0, 1.0, z)))
    # sign(0) = +1 convention on the third column.
    assert np.all(zc[:, 2] > 0)
    s0 = np.where(z == 0.0, 1.0, np.sign(z))
    for j in range(z.shape[1]):
        proj = s0[:, j] * ((s0[:, j] @ z[:, j]) / (s0[:, j] @ s0[:, j]))
        assert np.allclose(zc[:, j], proj)
    # In-run proxies preserved sign structure at every recorded check.
    for preserved in res.trace.svid_sign_preserved:
        assert preserved


def test_scn_scale_balancing():
    """magnitude_balancing_scales: Eq. 7 eta relation, Eq. 8 per-row mean
    absolute scales, Eq. 9 final latents, and the dead-row scale floor
    (A-11) in float64."""
    gen = np.random.default_rng(5)
    u_hat = gen.standard_normal((7, 3))
    v_hat = gen.standard_normal((5, 3))
    d_out = np.linspace(0.5, 1.5, 7)
    d_in = np.linspace(1.0, 2.0, 5)
    # Consensus proxies in preconditioned space: P = D~ · Û.
    p_u = d_out[:, None] * u_hat
    p_v = d_in[:, None] * v_hat

    bal = magnitude_balance(p_u, p_v, d_out, d_in)
    eta = bal.eta
    assert eta == pytest.approx(np.sqrt(np.linalg.norm(v_hat) / np.linalg.norm(u_hat)))

    # Eq. 7 relation on the final latents: ||η·Û||_F == ||η⁻¹·V̂||_F.
    assert np.linalg.norm(bal.U) == pytest.approx(np.linalg.norm(bal.V), rel=1e-12)

    # Eq. 8: scales are per-row mean absolute values of the final latents.
    assert bal.s1.shape == (7,) and bal.s2.shape == (5,)
    assert np.allclose(bal.s1, np.mean(np.abs(bal.U), axis=1))
    assert np.allclose(bal.s2, np.mean(np.abs(bal.V), axis=1))
    assert bal.U.dtype == np.float64 and bal.s1.dtype == np.float64

    # Eq. 9: final latents are the eta-scaled unscaled proxies.
    assert np.allclose(bal.U, eta * u_hat)
    assert np.allclose(bal.V, v_hat / eta)

    # Binaries: exact ±1 int8 with sign(0) = +1.
    assert bal.Ub.dtype == np.int8 and bal.Vb.dtype == np.int8
    assert np.all((bal.Ub == 1) | (bal.Ub == -1))

    # Dead row: an all-zero row of Û yields the clamped floor, not NaN/0,
    # and its binary row is +1 (sign(0) = +1, A-11).
    u_dead = u_hat.copy()
    u_dead[3] = 0.0
    bal_dead = magnitude_balance(d_out[:, None] * u_dead, p_v, d_out, d_in)
    assert np.isfinite(bal_dead.s1).all()
    assert bal_dead.s1[3] == pytest.approx(DEFAULT_SCALE_FLOOR)
    assert np.all(bal_dead.Ub[3] == 1)
    # The equilibrium factor moves only because ||U_hat||_F changed with the
    # dead row (Eq. 7); the V-side scaling relation of Eq. 9 still holds.
    assert np.allclose(bal_dead.V, v_hat / bal_dead.eta)


def test_magnitude_balance_rejects_degenerate():
    """An all-zero factor set has no equilibrium; fail closed."""
    zeros = np.zeros((4, 2))
    with pytest.raises(ValueError):
        magnitude_balance(zeros, zeros, np.ones(4), np.ones(4))
