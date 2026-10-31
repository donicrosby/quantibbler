"""qk-fix-admm-factor-scale: the ADMM serialized binary factor pair is
renormalized to the continuous product's norm by a target-free correction γ.

The continuous factors U, V reconstruct W well, but the ±1-bitplane serialized
product ``diag(s1) Ub Vbᵀ diag(s2)`` loses the continuous factors' cancellation
structure and drifts far from ``||UVᵀ||_F``. The correction
``γ = sqrt(||UVᵀ||_F / ||M||_F)`` applied symmetrically to s1 and s2 makes the
serialized product norm-faithful to the continuous product the balance stage
already holds — with no reference to any target, so it survives the removal of
the LS-α refit that used to mask the defect.

Scenarios pinned (``openspec/changes/qk-fix-admm-factor-scale/specs/qk-ptq/spec.md``):

* ``correction_preserves_continuous_product_norm``
* ``correction_survives_target_removal``
* ``correction_is_noop_when_already_scale_faithful``
* ``correction_is_recorded_finite_and_lands_in_band_at_production_rank``
* ``correction_raises_on_degenerate_input``
* ``correction_recorded_in_result``
* ``discrimination_test_reds_on_uncorrected_product``
* ``discrimination_test_green_on_corrected_product``
* ``discrimination_test_rejects_gamma_stub``

The primary discrimination red is the exact-norm ratio
``||M_ser||_F / ||M_cont||_F`` computed from the RETURNED s1, s2 BEFORE any
``BalanceResult.gamma`` access: on the pre-fix stage the returned α-refit scales
put that ratio at the α-refit value (≠ 1.0), so the red is the ratio failure and
never a missing-attribute error.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from quantibbler.ptq.block_recon import check_layer_magnitude
from quantibbler.ptq.config import PTQConfig
from quantibbler.ptq.lb_admm import lb_admm, magnitude_balance

SEVEN = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


class _SevenLinearBlock(nn.Module):
    """A Qwen3-shaped block: 7 scoped linears, GQA k/v (out-dim != d_model)."""

    def __init__(self, d: int = 8, dkv: int = 4, dff: int = 16, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.self_attn = nn.ModuleDict(
            {
                "q_proj": nn.Linear(d, d, bias=False, dtype=torch.float64),
                "k_proj": nn.Linear(d, dkv, bias=False, dtype=torch.float64),
                "v_proj": nn.Linear(d, dkv, bias=False, dtype=torch.float64),
                "o_proj": nn.Linear(dkv, d, bias=False, dtype=torch.float64),
            }
        )
        self.mlp = nn.ModuleDict(
            {
                "gate_proj": nn.Linear(d, dff, bias=False, dtype=torch.float64),
                "up_proj": nn.Linear(d, dff, bias=False, dtype=torch.float64),
                "down_proj": nn.Linear(dff, d, bias=False, dtype=torch.float64),
            }
        )
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight.copy_(
                        torch.randn(m.weight.shape, generator=gen, dtype=torch.float64) * 0.4
                    )


def _fro(x) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64), "fro"))


def _serialized(s1, ub, vb, s2) -> np.ndarray:
    """``diag(s1) @ Ub @ Vbᵀ @ diag(s2)`` in host float64 (±1 products are exact)."""
    return (s1[:, None] * np.asarray(ub, np.float64)) @ (s2[:, None] * np.asarray(vb, np.float64)).T


def _gaussian(dout: int, din: int, rank: int, seed: int) -> np.ndarray:
    """Deterministic rank-``rank`` gaussian target (the reviewer's anchor rig)."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(dout, rank)) @ rng.normal(size=(din, rank)).T


@pytest.fixture(scope="module")
def seven_solved():
    """The 7 ``_SevenLinearBlock`` linears solved once at rank 3 (ADMM arm).

    Yields ``(name, W, P_u, P_v)`` per linear with unit preconditioners, matching
    the balance call shape the reviewer's anchor rig used. The solve runs once;
    each scenario then calls ``magnitude_balance`` on the same proxies.
    """
    block = _SevenLinearBlock(seed=31)
    cfg = PTQConfig(rank=3, compute_dtype="float64", solver_backend="numpy")
    out = []
    for name in SEVEN:
        w = block.get_submodule(name).weight.detach().to(torch.float64).cpu().numpy()
        dout, din = w.shape
        res = lb_admm(w, cfg)
        out.append((name, w, np.asarray(res.P_u), np.asarray(res.P_v), dout, din))
    return out


def test_scn_correction_preserves_continuous_product_norm(seven_solved):
    """correction_preserves_continuous_product_norm: after the correction the
    serialized product's norm equals the continuous product's to float64
    round-off (relative tolerance 1e-6) on every rank-3 linear.
    """
    for name, _W, pu, pv, dout, din in seven_solved:
        bal = magnitude_balance(pu, pv, np.ones(dout), np.ones(din), backend="numpy")
        m_ser = _serialized(bal.s1, bal.Ub, bal.Vb, bal.s2)
        m_cont = np.asarray(bal.U) @ np.asarray(bal.V).T
        ratio = _fro(m_ser) / _fro(m_cont)
        assert ratio == pytest.approx(1.0, rel=1e-6), (
            f"{name}: ||M_ser||_F / ||M_cont||_F = {ratio!r} is not norm-faithful"
        )


def test_scn_correction_survives_target_removal(seven_solved):
    """correction_survives_target_removal: the correction consumes NO target, so
    the result is identical with and without one, and the serialized product
    stays in the healthy band ``||M||_F / ||W||_F ∈ [0.5, 2.0]`` — LS-refit
    removal is a semantic no-op.
    """
    for name, W, pu, pv, dout, din in seven_solved:
        without = magnitude_balance(pu, pv, np.ones(dout), np.ones(din), backend="numpy")
        with_t = magnitude_balance(pu, pv, np.ones(dout), np.ones(din), target=W, backend="numpy")
        # Passing a target must not change a single returned scale: the
        # correction is anchored to the continuous product, not to W.
        assert np.array_equal(np.asarray(without.s1), np.asarray(with_t.s1)), name
        assert np.array_equal(np.asarray(without.s2), np.asarray(with_t.s2)), name
        m_ser = _serialized(without.s1, without.Ub, without.Vb, without.s2)
        ratio_w = _fro(m_ser) / _fro(W)
        assert 0.5 <= ratio_w <= 2.0, f"{name}: ||M||_F / ||W||_F = {ratio_w!r} outside band"


def test_scn_correction_is_noop_when_already_scale_faithful():
    """correction_is_noop_when_already_scale_faithful: on a constructed faithful
    fixture (rank-r outer product ``U = a⊗1``, ``V = b⊗1``, a,b > 0) the Eq. 8
    serialized product already equals the continuous product, so γ == 1.0 within
    1e-3 — robust to the preconditioner.
    """
    dout, din, rank = 8, 16, 4
    rng = np.random.default_rng(7)
    a = rng.uniform(0.5, 2.0, size=dout)
    b = rng.uniform(0.5, 2.0, size=din)
    pu = np.outer(a, np.ones(rank))
    pv = np.outer(b, np.ones(rank))
    legs = {
        "unit-prec": (np.ones(dout), np.ones(din)),
        "rand-prec": (rng.uniform(0.5, 2.0, size=dout), rng.uniform(0.5, 2.0, size=din)),
        "skew-prec": (
            np.exp(rng.normal(0.0, 1.0, size=dout)),
            np.exp(rng.normal(0.0, 1.0, size=din)),
        ),
    }
    for leg, (d_out, d_in) in legs.items():
        bal = magnitude_balance(pu, pv, d_out, d_in, backend="numpy")
        assert bal.gamma == pytest.approx(1.0, rel=1e-3), f"{leg}: gamma = {bal.gamma!r}"
        m_ser = _serialized(bal.s1, bal.Ub, bal.Vb, bal.s2)
        m_cont = np.asarray(bal.U) @ np.asarray(bal.V).T
        assert _fro(m_ser) == pytest.approx(_fro(m_cont), rel=1e-12), leg


def test_scn_correction_is_recorded_finite_and_lands_in_band_at_production_rank():
    """correction_is_recorded_finite_and_lands_in_band_at_production_rank: at
    rank 64 the corrected product lands in ``[0.5, 2.0]`` and γ is recorded,
    finite and > 0. γ is NOT ≈ 1.0 here: the raw Eq. 8 product is under-normed at
    production rank, which is exactly what the correction addresses.
    """
    dout, din, rank = 128, 128, 64
    w = _gaussian(dout, din, rank, 0)
    cfg = PTQConfig(rank=rank, compute_dtype="float64", solver_backend="numpy")
    res = lb_admm(w, cfg)
    bal = magnitude_balance(res.P_u, res.P_v, np.ones(dout), np.ones(din), backend="numpy")
    m_ser = _serialized(bal.s1, bal.Ub, bal.Vb, bal.s2)
    ratio_w = _fro(m_ser) / _fro(w)
    assert 0.5 <= ratio_w <= 2.0, f"||M||_F / ||W||_F = {ratio_w!r} outside band at rank 64"
    assert np.isfinite(bal.gamma) and bal.gamma > 0.0, f"gamma = {bal.gamma!r}"
    assert bal.gamma > 1.0 + 1e-3, (
        f"gamma = {bal.gamma!r} is ≈ 1.0 at production rank; the raw Eq. 8 product "
        "is under-normed there and the correction must scale it up"
    )


def test_scn_correction_raises_on_degenerate_input():
    """correction_raises_on_degenerate_input: a zero serialized-product norm (the
    ratio is then non-finite) raises ValueError rather than silently skipping.
    """
    # row of U = [1, 1], row of V = [1, -1] -> sum_k Ub[0,k] Vb[0,k] = 0, so the
    # serialized product is exactly zero while both factors are non-zero.
    pu = np.array([[1.0, 1.0]])
    pv = np.array([[1.0, -1.0]])
    with pytest.raises(ValueError, match="serialization-faithfulness"):
        magnitude_balance(pu, pv, np.ones(1), np.ones(1), backend="numpy")


def test_scn_correction_recorded_in_result(seven_solved):
    """correction_recorded_in_result: ``BalanceResult.gamma`` equals the applied
    correction γ = sqrt(||UVᵀ||_F / ||M_unc||_F) computed from the returned
    latents and the uncorrected Eq. 8 scales.
    """
    for name, _W, pu, pv, dout, din in seven_solved:
        bal = magnitude_balance(pu, pv, np.ones(dout), np.ones(din), backend="numpy")
        s1_raw = np.mean(np.abs(np.asarray(bal.U)), axis=1)
        s2_raw = np.mean(np.abs(np.asarray(bal.V)), axis=1)
        m_unc = _serialized(s1_raw, bal.Ub, bal.Vb, s2_raw)
        m_cont = np.asarray(bal.U) @ np.asarray(bal.V).T
        expected = (_fro(m_cont) / _fro(m_unc)) ** 0.5
        assert bal.gamma == pytest.approx(expected, rel=1e-12), name


def test_scn_discrimination_test_reds_on_uncorrected_product(seven_solved):
    """discrimination_test_reds_on_uncorrected_product: the UNCORRECTED Eq. 8
    serialized product overshoots the continuous-product norm by orders of
    magnitude on the rank-3 fixture. This is a defect canary — it holds on both
    the pre-fix and post-fix stage (the overshoot is a property of the data, not
    the code), documenting the defect without discriminating.
    """
    for name, _W, pu, pv, dout, din in seven_solved:
        bal = magnitude_balance(pu, pv, np.ones(dout), np.ones(din), backend="numpy")
        s1_raw = np.mean(np.abs(np.asarray(bal.U)), axis=1)
        s2_raw = np.mean(np.abs(np.asarray(bal.V)), axis=1)
        m_unc = _serialized(s1_raw, bal.Ub, bal.Vb, s2_raw)
        m_cont = np.asarray(bal.U) @ np.asarray(bal.V).T
        ratio = _fro(m_unc) / _fro(m_cont)
        assert ratio > 5.0, (
            f"{name}: uncorrected ||M||_F / ||UVᵀ||_F = {ratio!r} is not the overshoot mode"
        )


def test_scn_discrimination_test_green_on_corrected_product(seven_solved):
    """discrimination_test_green_on_corrected_product: on the corrected stage the
    serialized product is norm-faithful (ratio ≈ 1.0 within 1e-6), every linear
    lands in ``||M||_F / ||W||_F ∈ [0.5, 2.0]``, and the divergence guard passes.
    """
    for name, W, pu, pv, dout, din in seven_solved:
        bal = magnitude_balance(pu, pv, np.ones(dout), np.ones(din), backend="numpy")
        m_ser = _serialized(bal.s1, bal.Ub, bal.Vb, bal.s2)
        m_cont = np.asarray(bal.U) @ np.asarray(bal.V).T
        assert _fro(m_ser) / _fro(m_cont) == pytest.approx(1.0, rel=1e-6), name
        assert 0.5 <= _fro(m_ser) / _fro(W) <= 2.0, name
        cfg = PTQConfig(rank=3, compute_dtype="float64", solver_backend="numpy")
        ratio = check_layer_magnitude(
            name, np.asarray(bal.U), np.asarray(bal.V), bal.s1, bal.s2, W, cfg
        )
        assert 1e-4 <= ratio <= 1e4, f"{name}: divergence gate ratio {ratio!r}"


def test_scn_discrimination_test_rejects_gamma_stub(seven_solved):
    """discrimination_test_rejects_gamma_stub: the no-op-mutant shape — keep the
    LS-α refit, add ``gamma = 1.0`` — leaves the serialized product at the α-refit
    ratio, far from 1.0, so it fails the tight-tolerance assertion; its
    ``||M||_F / ||W||_F`` equals ``cos(W, M_unc)`` (the α-refit shrink signature,
    strictly < 1, measured 0.588–0.863 on this fixture).
    """
    for name, W, pu, pv, dout, din in seven_solved:
        bal = magnitude_balance(pu, pv, np.ones(dout), np.ones(din), backend="numpy")
        m_cont = np.asarray(bal.U) @ np.asarray(bal.V).T
        s1_raw = np.mean(np.abs(np.asarray(bal.U)), axis=1)
        s2_raw = np.mean(np.abs(np.asarray(bal.V)), axis=1)
        m_unc = _serialized(s1_raw, bal.Ub, bal.Vb, s2_raw)
        alpha = float((W * m_unc).sum()) / float((m_unc * m_unc).sum())
        m_stub = alpha * m_unc
        stub_ratio = _fro(m_stub) / _fro(m_cont)
        assert abs(stub_ratio - 1.0) > 1e-3, (
            f"{name}: the alpha-refit stub reaches norm-faithfulness "
            f"({stub_ratio!r}); the test does not discriminate the correction"
        )
        stub_over_w = _fro(m_stub) / _fro(W)
        assert 0.4 <= stub_over_w < 0.95, (
            f"{name}: alpha-refit shrink signature missing: ||M_stub||_F / ||W||_F = "
            f"{stub_over_w!r}"
        )
