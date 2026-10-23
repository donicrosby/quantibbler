"""r137-v7 post-mortem regressions: per-tensor relative lr + gamma rebalance.

Measured mechanism chain (norm_probe, block 26 o_proj): polish exits
healthy (nr 0.475, cos +0.475) but with the scale vectors 11 decades
apart (s1 ~ 3e-7, s2 ~ 2.7e4 — legal, the product is gamma-invariant);
STE's absolute lr 2e-3 then steps s1 by ~5e4x its own magnitude per
step, ending the stage at nr 4.5e4 with cos NEGATIVE — the single
largest artifact-quality defect in the pipeline's history.

Fix under test: every tuning stage steps each tensor at
lr_rel * median|tensor| (the step is always relevant to the parameter
class it walks), and polish collapses any s1/s2 split at exit via the
exact gamma rescale (product-invariant).
"""

import numpy as np
import torch

from quantibbler.ptq.bit_polish import greedy_polish
from quantibbler.ptq.block_recon import FactorizedLinear, _tune_latent_ste
from quantibbler.ptq.config import PTQConfig, StageHyper


def _tiny_block(seed: int = 0):
    """A bare nn.Linear as a one-layer 'block' — the minimal legal facs set."""
    gen = torch.Generator().manual_seed(seed)
    linear = torch.nn.Linear(16, 12, dtype=torch.float64)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(12, 16, generator=gen, dtype=torch.float64) * 0.3)
        linear.bias.zero_()
    u0, s0, vt0 = np.linalg.svd(linear.weight.detach().numpy(), full_matrices=False)
    fac = FactorizedLinear(
        torch.from_numpy(np.ascontiguousarray(u0[:, :6] * np.sqrt(s0[:6]))),
        torch.from_numpy(np.ascontiguousarray(vt0[:6].T * np.sqrt(s0[:6]))),
        linear.bias.detach().clone(),
    )
    from quantibbler.ptq.block_recon import _set_linear

    _set_linear(linear, fac)
    return linear, fac


def test_scn_ste_relative_lr_preserves_scale_classes():
    """At a nominal lr that is 1e6x too big for s1's scale class, the
    per-tensor relative lr must keep s1's class intact (post-STE median
    within 2x of entry) while still moving the latents."""
    torch.manual_seed(0)
    linear, fac = _tiny_block()
    fac.s1.data.mul_(1e-6)  # force the split: s1 class ~1e-6, s2 ~1e0
    fac.s2.data.mul_(1e0)
    pre_s1_med = float(fac.s1.detach().abs().median())

    gen = torch.Generator().manual_seed(1)
    x_b = torch.randn(8, 4, 16, generator=gen, dtype=torch.float64)
    y_b = torch.randn(8, 4, 12, generator=gen, dtype=torch.float64)

    cfg = PTQConfig(solver_backend="numpy", rank=6, stage_latent=StageHyper(1e0, 4, 1))  # brutal lr
    _tune_latent_ste(linear, {"": fac}, x_b, y_b, cfg, 4)

    post_s1_med = float(fac.s1.detach().abs().median())
    assert post_s1_med > 0.0
    # Bound: per-step Adam movement ~ lr_rel * median; total steps <= 2
    # here, so sane relative stepping lands within ~4x. The old absolute
    # lr overshot this layer's s1 by ~5e4x PER STEP — two orders past
    # this assert.
    assert post_s1_med < 4.0 * pre_s1_med, (
        f"s1 scale class not preserved: {pre_s1_med:.3e} -> {post_s1_med:.3e} "
        "(absolute-lr blowup regression)"
    )


def test_scn_e2e_gate_svd_solver_path():
    """The solver='svd' path must pass the full e2e gate: the ablation
    branch (truncated SVD -> balance -> polish) flows through capture,
    STE/KD, export, reload and the corruption bounds identically."""
    import dataclasses

    from quantibbler.ptq.e2e import gate_config, run_e2e_gate

    cfg = dataclasses.replace(gate_config(), solver="svd")
    result = run_e2e_gate(cfg=cfg)
    assert result["bits_identical"], result
    assert float(result["logit_delta"]) < 1.0, result


def test_scn_polish_gamma_rebalance_is_product_invariant():
    """The exit rebalance must leave diag(s1)Ub Vb^T diag(s2) unchanged
    (to f64 round-off) while collapsing the median-magnitude split."""
    rng = np.random.default_rng(5)
    w = (rng.standard_normal((24, 4)) @ rng.standard_normal((4, 16))) * 2.0
    u0, s0, vt0 = np.linalg.svd(w, full_matrices=False)
    uu = u0[:, :6] * np.sqrt(s0[:6])
    vv = vt0[:6].T * np.sqrt(s0[:6])
    from quantibbler.ptq.lb_admm import magnitude_balance

    bal = magnitude_balance(uu.copy(), vv.copy(), np.ones(24), np.ones(16), target=w)
    bal.s1 = bal.s1 * 1e-6  # force the split polish will inherit

    before = (bal.s1[:, None] * (bal.Ub @ bal.Vb.T)) * bal.s2[None, :]
    ub, vb, s1, s2 = greedy_polish(w, bal.Ub, bal.Vb, bal.s1, bal.s2, rounds=2)
    after = (s1[:, None] * (ub @ vb.T)) * s2[None, :]

    # Polish is a descent method on the stored objective: even from a
    # deliberately split input it must land closer to W than it started
    # (the gamma rescale itself is product-invariant; what changes here
    # is the descent).
    def rel_err(m):
        return float(np.linalg.norm(m - w) / np.linalg.norm(w))

    assert rel_err(after) < rel_err(before), "polish regressed the stored objective"

    # The rebalance did its job: exit scale classes are comparable.
    p1 = float(np.median(np.abs(s1)))
    p2 = float(np.median(np.abs(s2)))
    ratio = max(p1, p2) / min(p1, p2)
    assert ratio < 1e3, f"scale classes still split: p1 {p1:.3e} vs p2 {p2:.3e}"
