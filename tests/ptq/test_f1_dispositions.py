"""Disposition F1 contract pins (2026-10-28 operator PPL audit).

F1 - stage lrs are [NQ] App. C ABSOLUTE values when the stage input is the
BALANCED class (``_ste``/``scale_calib``); the per-tensor relative form
``lr * median|p|`` survives only as the pathological else-branch, and the
stage boundary collapses an s1/s2 class split product-invariantly.

(F2's Eq. 8 refit-removal pins are split out with the F2 removal itself --
preserved on the F2 refit-removal preservation branch -- because F2 lands only
after the ADMM serialized factor-scale defect is fixed.)
"""

from __future__ import annotations

import inspect

import pytest
import torch

from quantibbler.ptq._stage_lr import (
    BALANCED_MEDIAN_HIGH,
    BALANCED_MEDIAN_LOW,
    derive_stage_lr_groups,
    rebalance_scale_pair,
)
from quantibbler.ptq.config import PTQConfig, StageHyper

# --------------------------------------------------------------------------
# F1 - App. C absolute lrs under the balanced class
# --------------------------------------------------------------------------


def _fac(dout=12, din=10, r=4, u_scale=0.5, v_scale=0.5, s1_scale=1.0, s2_scale=1.0):
    """A FactorizedLinear with every parameter class set to a chosen magnitude."""
    from quantibbler.ptq.block_recon import FactorizedLinear

    gen = torch.Generator().manual_seed(0)
    u = (torch.randn(dout, r, generator=gen, dtype=torch.float64) * u_scale).contiguous()
    v = (torch.randn(din, r, generator=gen, dtype=torch.float64) * v_scale).contiguous()
    fac = FactorizedLinear(u, v, torch.zeros(dout, dtype=torch.float64))
    with torch.no_grad():
        fac.s1.mul_(s1_scale)
        fac.s2.mul_(s2_scale)
    return fac


def _run_ste(fac, steps=0, lr=1e-5):
    from quantibbler.ptq.block_recon import _tune_latent_ste

    gen = torch.Generator().manual_seed(1)
    block = torch.nn.Linear(10, 12, dtype=torch.float64)
    x_b = torch.randn(8, 3, 10, generator=gen, dtype=torch.float64)
    y_b = torch.randn(8, 3, 12, generator=gen, dtype=torch.float64)
    cfg = PTQConfig(
        solver_backend="numpy",
        compute_dtype="float64",
        rank=4,
        stage_latent=StageHyper(lr, 1, 8),
    )
    return _tune_latent_ste(block, {"": fac}, x_b, y_b, cfg, 4, steps=steps)


def test_derive_groups_absolute_under_balanced():
    tensors = [
        torch.full((6,), 0.5, dtype=torch.float64),
        torch.full((5,), 0.8, dtype=torch.float64),
    ]
    groups, prov = derive_stage_lr_groups(tensors, 1e-5)
    assert prov["class"] == "balanced"
    assert prov["effective_lr"] == pytest.approx(1e-5)
    assert all(g["lr"] == pytest.approx(1e-5) for g in groups)
    assert all(g["lr_scale"] == 1.0 for g in groups)


def test_derive_groups_relative_for_pathological():
    tensors = [
        torch.full((6,), 1e-6, dtype=torch.float64),
        torch.full((5,), 3e-6, dtype=torch.float64),
    ]
    groups, prov = derive_stage_lr_groups(tensors, 1e-5)
    assert prov["class"] == "pathological"
    # else-branch: lr_rel * the tensor's OWN median
    assert groups[0]["lr"] == pytest.approx(1e-5 * 1e-6)
    assert groups[1]["lr"] == pytest.approx(1e-5 * 3e-6)


def test_balanced_band_edges_are_the_declared_constants():
    assert BALANCED_MEDIAN_LOW == 0.01
    assert BALANCED_MEDIAN_HIGH == 1.0
    for inside in (0.01, 0.1, 1.0):
        _, prov = derive_stage_lr_groups([torch.full((4,), inside, dtype=torch.float64)], 1e-5)
        assert prov["class"] == "balanced", inside
    for outside in (1e-6, 1.01, 5.0):
        _, prov = derive_stage_lr_groups([torch.full((4,), outside, dtype=torch.float64)], 1e-5)
        assert prov["class"] == "pathological", outside


def test_scn_factorized_stage_lr_is_absolute_under_balanced_inputs():
    """factorized_stage_lr_is_absolute_under_balanced_inputs: balanced latents
    AND balanced s1/s2 -> each group's lr is the App. C absolute 1e-5."""
    fac = _fac(u_scale=0.5, v_scale=0.5, s1_scale=1.0, s2_scale=1.0)
    stats = _run_ste(fac, steps=0, lr=1e-5)
    assert stats["lr_class"] == "balanced"
    assert stats["lr_effective"] == pytest.approx(1e-5)
    assert all(g["lr"] == pytest.approx(1e-5) for g in stats["lr_groups"])


def test_scn_pathological_class_survivor_uses_relative_form():
    """pathological_class_survivor_uses_relative_form: a stage whose input sits
    outside the balanced band keeps the relative form and RECORDS the class,
    the median magnitude, and the effective lr."""
    fac = _fac(u_scale=1e-6, v_scale=1e-6, s1_scale=1e-6, s2_scale=1e-6)
    stats = _run_ste(fac, steps=0, lr=1e-5)
    assert stats["lr_class"] == "pathological"
    assert stats["lr_median"] == pytest.approx(1e-6, rel=1e-6)
    assert stats["lr_effective"] > 0.0
    assert stats["lr_effective"] < 1e-5  # relative form is strictly below the absolute value


def test_scn_scale_class_anchoring_at_stage_boundary():
    """scale_class_anchoring_at_stage_boundary: a multi-decade s1/s2 split is
    collapsed by a PRODUCT-INVARIANT gamma rescale before lr derivation, and
    the factor travels in the stage record."""
    fac = _fac(u_scale=0.5, v_scale=0.5, s1_scale=1e-6, s2_scale=1e0)
    s1_pre = fac.s1.detach().clone()
    s2_pre = fac.s2.detach().clone()
    product_pre = s1_pre[:, None] * s2_pre[None, :]

    stats = _run_ste(fac, steps=0, lr=1e-5)

    gamma = stats["rebalance_factor"][""]
    assert gamma == pytest.approx(1e3, rel=0.2)
    # the stored weight is untouched by the rescale
    product_post = fac.s1.detach()[:, None] * fac.s2.detach()[None, :]
    assert torch.allclose(product_pre, product_post, rtol=1e-12, atol=0.0)
    # the two classes now sit at the same median
    m1 = float(fac.s1.detach().abs().median())
    m2 = float(fac.s2.detach().abs().median())
    assert m1 == pytest.approx(m2, rel=1e-9)


def test_scn_scale_class_anchoring_at_stage_boundary_balanced_mid():
    """scale_class_anchoring_at_stage_boundary: a split whose geometric mid is
    IN the balanced band collapses to the balanced class at the stage boundary,
    so the optimizer sees BALANCED parameters at the App. C ABSOLUTE lr (the
    composite THEN). s1/s2 straddle the band so the collapse lands both classes
    on unit median (mid = sqrt(1e-2 * 1e2) = 1.0) -- the path F1 exists for,
    which the multi-decade split fixture above (mid 1e-3 -> pathological) never
    exercises."""
    fac = _fac(u_scale=0.5, v_scale=0.5, s1_scale=1e-2, s2_scale=1e2)
    s1_pre = fac.s1.detach().clone()
    s2_pre = fac.s2.detach().clone()
    product_pre = s1_pre[:, None] * s2_pre[None, :]

    stats = _run_ste(fac, steps=0, lr=1e-5)

    # gamma = sqrt(median|s2| / median|s1|) = sqrt(1e2 / 1e-2)
    gamma = stats["rebalance_factor"][""]
    assert gamma == pytest.approx(1e2, rel=1e-9)
    # product invariance: the rescale cannot move the stored weight
    product_post = fac.s1.detach()[:, None] * fac.s2.detach()[None, :]
    assert torch.allclose(product_pre, product_post, rtol=1e-12, atol=0.0)
    # the rebalanced input IS the balanced class...
    assert stats["lr_class"] == "balanced"
    # ...so every group steps at the App. C ABSOLUTE lr, not lr * median|p|
    assert stats["lr_effective"] == pytest.approx(1e-5)
    assert all(g["lr"] == pytest.approx(1e-5) for g in stats["lr_groups"])


def test_rebalance_is_a_noop_for_an_already_balanced_pair():
    s1 = torch.full((6,), 0.5, dtype=torch.float64)
    s2 = torch.full((5,), 0.5, dtype=torch.float64)
    gamma = rebalance_scale_pair(s1, s2)
    assert gamma == pytest.approx(1.0)
    assert torch.equal(s1, torch.full((6,), 0.5, dtype=torch.float64))


def test_scn_global_scale_lr_is_absolute_under_balanced_scales():
    """global_scale_lr_is_absolute_under_balanced_scales: the KD stage derives
    its groups from the App. C ABSOLUTE 1e-6 under balanced scale vectors."""
    import quantibbler.ptq.scale_calib as sc

    src = inspect.getsource(sc.calibrate_scales)
    assert "derive_stage_lr_groups" in src, (
        "TuneScalesKD must derive its optimizer groups through the shared "
        "App. C lr contract, not a local per-tensor relative anchor"
    )
    assert "cfg.stage_global.lr" in src
    assert "median()" not in src, "a local median-anchored lr survived in TuneScalesKD"
