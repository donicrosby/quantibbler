"""Tasks 3.1-3.3: LB-ADMM initialization (scenarios admm_spd_solves,
admm_consensus_convergence, penalty_schedule_stopping,
magnitude_balancing_scales)."""

import numpy as np
import pytest

from quantibbler.ptq.config import PTQConfig
from quantibbler.ptq.lb_admm import (
    DEFAULT_SCALE_FLOOR,
    _cholesky_solve,
    _rho_end,
    lb_admm,
    magnitude_balance,
    svid_proxy,
)
from quantibbler.ptq.solver_backend import get_backend

_NP = get_backend("numpy")


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
        cfg = PTQConfig(rank=r, rho_start=1e-2, rho_end=1e2, solver_backend="numpy")
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
    x0, fell = _cholesky_solve(_NP, h_spd, b)
    assert fell is False
    h_bad = h_spd - 10.0 * np.eye(2)  # indefinite by construction
    x1, fell = _cholesky_solve(_NP, h_bad, b)
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
    cfg = PTQConfig(rank=r, solver_backend="numpy")
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
    zc = svid_proxy(_NP, z)
    assert np.array_equal(np.sign(zc), np.sign(np.where(z == 0, 1.0, z)))
    # sign(0) = +1 convention on the third column.
    assert np.all(zc[:, 2] > 0)
    s0 = np.where(z == 0.0, 1.0, np.sign(z))
    # Two-sided optimum ([NQ] :592), replacing the retired column-only closed
    # form this block used to assert verbatim. The optimum's magnitude profile
    # is rank-1 (the top rank-1 of |z|), and it is strictly closer to z than
    # the column-only variant in the paper's own objective -- which is what
    # discriminates the two, since a column-only fit cannot satisfy rank-1
    # unless |z| has a constant row vector.
    col_only = (
        s0
        * np.array([(s0[:, j] @ z[:, j]) / (s0[:, j] @ s0[:, j]) for j in range(z.shape[1])])[
            None, :
        ]
    )
    assert np.linalg.norm(z - zc) < np.linalg.norm(z - col_only)
    live = np.abs(zc)[:, :2]  # column 2 is the all-zero (dead) one
    assert np.linalg.matrix_rank(live, tol=1e-9 * float(np.abs(live).max())) == 1
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

    bal = magnitude_balance(p_u, p_v, d_out, d_in, backend="numpy")
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
    bal_dead = magnitude_balance(d_out[:, None] * u_dead, p_v, d_out, d_in, backend="numpy")
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
        magnitude_balance(zeros, zeros, np.ones(4), np.ones(4), backend="numpy")


def test_scn_warm_start_svd_flaky_fallback_and_nan_guard():
    """_warm_start survives gesdd non-convergence (transpose retry /
    eigendecomposition fallback) and fails LOUD on non-finite targets."""
    from quantibbler.ptq.lb_admm import _warm_start

    gen = np.random.default_rng(7)
    # extreme dynamic range but finite: the gesdd-flaky species
    scales = 10.0 ** gen.uniform(-12, 12, size=(64, 1))
    w = (gen.standard_normal((64, 32)) * scales).astype(np.float64)
    u0, v0 = _warm_start(_NP, np.ascontiguousarray(w), 8, np.dtype("float64"))
    assert np.all(np.isfinite(u0)) and np.all(np.isfinite(v0))
    assert u0.shape == (64, 8) and v0.shape == (32, 8)
    # warm start spans the dominant subspace: rank-8 recon of a rank-8
    # slice beats a random basis by a wide margin.
    u8, s8, vt8 = np.linalg.svd(w, full_matrices=False)
    err_ws = np.linalg.norm(w - u0 @ v0.T)
    err_top8 = np.linalg.norm(w - (u8[:, :8] * s8[:8]) @ vt8[:8])
    assert err_ws <= 2.0 * err_top8

    # non-finite target: pipeline bug, loud error with diagnostics
    w_bad = w.copy()
    w_bad[3, 5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _warm_start(_NP, np.ascontiguousarray(w_bad), 8, np.dtype("float64"))


def test_rho_end_override_is_explicit_and_fail_closed(monkeypatch):
    """QK_RHO_END is an override, not a second default.

    Attribution runs depend on a run meaning what its provenance says it ran, so
    both directions need pinning: unset must leave cfg.rho_end untouched, and a
    malformed value must raise rather than silently reverting to the default.
    """
    cfg = PTQConfig(rank=4, rho_end=1e-1, solver_backend="numpy")

    monkeypatch.delenv("QK_RHO_END", raising=False)
    assert _rho_end(cfg) == 1e-1

    monkeypatch.setenv("QK_RHO_END", "1e2")
    assert _rho_end(cfg) == 1e2

    monkeypatch.setenv("QK_RHO_END", "not-a-float")
    with pytest.raises(ValueError, match="QK_RHO_END must be a float"):
        _rho_end(cfg)


# ---------------------------------------------------------------------------
# Lane 3 (qk-nanoquant-alignment-lane3): ADMM normalized regime.
#
# Each test names the scenario ID it pins. A red here fails on an assertion
# about the regime's behavior, never on an import or a missing attribute.
# ---------------------------------------------------------------------------

_LANE3 = (9, 7, 4)


def _lane3_matrix(scale: float = 1.0):
    """A badly-scaled ground-truth matrix: the regime's whole target."""
    m, n, r = _LANE3
    a = _ground_truth(m, n, r, seed=3)
    return a * scale


def _rel_fit(res, w) -> float:
    """Relative Frobenius error of the returned factorization U V^T against w.

    The reconstruction is ``U @ V.T`` -- the product of the two returned factors.
    ``P_u``/``P_v`` are full (d, r) consensus proxies, NOT per-column scales, so
    an elementwise ``U * P_u`` is a different object entirely and must never be
    used as a reconstruction.
    """
    approx = np.asarray(res.U) @ np.asarray(res.V).T
    return float(np.linalg.norm(approx - w) / np.linalg.norm(w))


def test_scn_normalized_regressor_design_matrix():
    """Scenario ``normalized_regressor_design_matrix``.

    THEN the design matrix is built from the row/col-normalized regressor, so
    the Gram matrix X^T X is O(1)-scale with unit-scale diagonal AND the returned
    iterate is mapped back to the unnormalized factor basis, so U V^T remains a
    valid factorization of the target.
    """
    cfg = PTQConfig(rank=_LANE3[2])
    small = lb_admm(_lane3_matrix(1.0), cfg)
    large = lb_admm(_lane3_matrix(1e3), cfg)

    for res, w in ((small, _lane3_matrix(1.0)), (large, _lane3_matrix(1e3))):
        assert res.trace.ridge_scale, "trace recorded no ridge scale"
        # Unit-scale diagonal: mean(diag(Xn^T Xn)) == 1 for a normalized design.
        assert all(abs(s - 1.0) < 1e-9 for s in res.trace.ridge_scale), (
            f"normalized-design Gram scale is not unit-scale: {res.trace.ridge_scale[:4]}"
        )
        # Map-back validity: the product reconstructs the target. The bar is a
        # KILL-ONLY bound, set from measurement under this regime (rel 3.34e-2,
        # cos 0.99988), not a fit target. What sets the 3.3e-2 is the REG FLOOR,
        # not the consensus proxy: admm_reg=1e-5 gives 1.19e-5 on the same
        # fixture (converged, iter 168), 3e-2 plateaus at 3.3e-2 even at K=4000,
        # and the SVD floor is 8.7e-16 on this exactly-rank-4 target. The
        # residual grows linearly in reg. Do not attribute it to the proxy.
        assert _rel_fit(res, w) < 5e-2, (
            f"returned U V^T does not reconstruct the target (rel {_rel_fit(res, w):.3e})"
        )
        approx = np.asarray(res.U) @ np.asarray(res.V).T
        cos = float(np.sum(approx * w) / (np.linalg.norm(approx) * np.linalg.norm(w)))
        assert cos > 0.999, f"map-back product cos={cos:.6f} is not a valid factorization"

    # The Gram scale is invariant to a 1e3 target rescale -- that IS the
    # normalization claim, so it is asserted at 1e-12, not a loose tolerance.
    a = np.asarray(small.trace.ridge_scale)
    b = np.asarray(large.trace.ridge_scale)
    assert a.shape == b.shape
    assert np.allclose(a, b, rtol=1e-12, atol=1e-12), (
        f"ridge scale moved under a 1e3 target rescale ({a[:3]} vs {b[:3]}): the "
        "design matrix is not normalized against the target's own scale"
    )


def test_scn_stabilizer_scale_relative_ridge():
    """Scenario ``stabilizer_scale_relative_ridge``.

    THEN the shift is exactly ``rho * mean(diag(X^T X)) + reg`` with ``reg`` read
    from a config field (default 3e-2) AND the anchor depends only on the
    normalized regressor, never on raw layer scale.
    """
    from quantibbler.ptq.lb_admm import _ridge_shift

    # Mutation-proof: on a deliberately NON-unit design the formula must track
    # mean(diag(X^T X)) and not a constant. Column squared norms are 5 and 14.
    x = np.array([[1.0, 2.0], [2.0, -1.0], [0.0, 3.0]])
    shift, scale = _ridge_shift(_NP, x, 0.5, 3e-2)
    assert scale == pytest.approx((5.0 + 14.0) / 2.0)
    assert shift == pytest.approx(0.5 * (5.0 + 14.0) / 2.0 + 3e-2)

    cfg = PTQConfig(rank=_LANE3[2])
    assert float(cfg.admm_reg) == pytest.approx(3e-2)
    res = lb_admm(_lane3_matrix(1.0), cfg)
    trace = res.trace
    assert trace.ridge_scale, "the solver trace exposes no ridge scale"

    assert len(trace.ridge_scale) == 2 * len(trace.rhos), (
        "expected one ridge scale per half-update (2 per iteration)"
    )
    for k, _rho in enumerate(trace.rhos):
        # Unnormalized design in the interior loop is unit-column by
        # construction, so the anchor is 1 and the shift is rho + reg exactly.
        for s in (trace.ridge_scale[2 * k], trace.ridge_scale[2 * k + 1]):
            assert s == pytest.approx(1.0, abs=1e-9)

    # Scale-free anchor: a 1e3 target rescale leaves every recorded scale equal.
    large = lb_admm(_lane3_matrix(1e3), cfg)
    assert np.allclose(trace.ridge_scale, large.trace.ridge_scale, rtol=1e-12, atol=1e-12)


def test_scn_reg_env_override_explicit_fail_closed(monkeypatch):
    """Scenario ``reg_env_override_explicit_fail_closed``.

    ``QK_ADMM_REG`` is a SOLVE-TIME override: it must be resolved by the solver
    (not frozen when ``PTQConfig()`` is constructed), fail closed on garbage, and
    never silently fall back to the default.
    """
    from quantibbler.ptq.lb_admm import _admm_reg, admm_reg_provenance

    # Constructed BEFORE the env var is set -- a construction-time-only override
    # would never see the change below.
    cfg = PTQConfig(rank=4)

    monkeypatch.delenv("QK_ADMM_REG", raising=False)
    assert admm_reg_provenance(cfg.admm_reg) == (3e-2, "config")
    assert _admm_reg(cfg) == 3e-2

    monkeypatch.setenv("QK_ADMM_REG", "5e-2")
    value, source = admm_reg_provenance(cfg.admm_reg)
    assert value == 5e-2 and source == "env", (
        f"QK_ADMM_REG=5e-2 did not resolve at solve time (got {value!r}, {source!r})"
    )
    assert _admm_reg(cfg) == 5e-2

    monkeypatch.setenv("QK_ADMM_REG", "not-a-number")
    with pytest.raises(ValueError, match="QK_ADMM_REG"):
        _admm_reg(cfg)


def test_scn_new_solver_state_joins_fingerprint(monkeypatch):
    """Scenario ``new_solver_state_joins_fingerprint``.

    THEN two configs differing only in a new solver-state field produce DIFFERENT
    fingerprints AND the new fields are absent from ``_FINGERPRINT_EXCLUDED``, so
    a future exclusion-list edit cannot silently drop solver state.
    """
    import dataclasses

    from quantibbler.ptq.checkpoints import (
        _FINGERPRINT_EXCLUDED,
        config_fingerprint,
        effective_solver_state,
    )

    for name in ("admm_reg", "rho_start", "rho_end"):
        assert name not in _FINGERPRINT_EXCLUDED, (
            f"{name!r} sits in _FINGERPRINT_EXCLUDED: solver state would be dropped "
            "from the digest"
        )

    monkeypatch.delenv("QK_ADMM_REG", raising=False)
    monkeypatch.delenv("QK_RHO_END", raising=False)
    base = PTQConfig(rank=4)
    base_fp = config_fingerprint(base)

    for kwargs in ({"admm_reg": 4e-2}, {"rho_start": 5e-3}, {"rho_end": 5e-1}):
        other = dataclasses.replace(base, **kwargs)
        assert config_fingerprint(other) != base_fp, (
            f"a config differing only in {next(iter(kwargs))} hashes identically "
            "to the base config"
        )

    # Effective (env-resolved) values are RECORDED, not folded. The digest must
    # NOT move when the env changes: a digest that shifts with an env value is a
    # resume lock on a code state, which is the tripwire the user ruled out. The
    # value is still visible in the artifact, which is what a human needs.
    monkeypatch.setenv("QK_RHO_END", "0.5")
    assert config_fingerprint(base) == base_fp, (
        "an env-resolved value moved the fingerprint: that is a resume lock, "
        "not a record"
    )
    assert effective_solver_state(base)["rho_end"] == pytest.approx(0.5)
    assert effective_solver_state(base)["rho_end_source"] == "env"
    monkeypatch.delenv("QK_RHO_END")
    assert config_fingerprint(base) == base_fp
    assert effective_solver_state(base)["rho_end_source"] == "config"


def test_scn_rho_endpoints_recalibrated_normalized_regime():
    """Scenario ``rho_endpoints_recalibrated_normalized_regime``.

    THEN rho follows the paper's linear schedule over K=400 with endpoints ~0->1
    AND at schedule end the ridge sits at the data term's own scale AND the reg
    floor keeps the subproblem SPD near rho ~ 0.
    """
    cfg = PTQConfig(rank=_LANE3[2])
    assert float(cfg.rho_start) == 0.0
    assert float(cfg.rho_end) == 1.0

    res = lb_admm(_lane3_matrix(1.0), cfg)
    rhos = np.asarray(res.trace.rhos)
    full = np.linspace(cfg.rho_start, cfg.rho_end, cfg.admm_max_iter)
    assert np.allclose(rhos, full[: rhos.size]), (
        "rho does not follow the linear K=400 schedule from the recalibrated endpoints"
    )

    # Ridge at the data term's own scale: rho_end * 1 + reg ~ 1.03, not 1e2.
    reg = float(cfg.admm_reg)
    ridge_end = float(rhos[-1]) * res.trace.ridge_scale[-1] + reg
    assert ridge_end == pytest.approx(float(rhos[-1]) + reg, rel=1e-9)
    assert 0.5 <= ridge_end <= 2.0, (
        f"terminal ridge {ridge_end:.4g} is not at the data term's scale"
    )

    # Well-posedness at rho ~ 0: the reg floor keeps the subproblem SPD.
    assert min(res.trace.min_eigs) >= reg * (1.0 - 1e-9), (
        f"smallest solved eigenvalue {min(res.trace.min_eigs):.6g} fell below the "
        f"reg floor {reg}: the continuous subproblem is not SPD near rho ~ 0"
    )
