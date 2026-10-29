"""r137-v7 post-mortem regressions: stage-boundary gamma rebalance.

Measured mechanism chain (norm_probe, block 26 o_proj): polish exits
healthy (nr 0.475, cos +0.475) but with the scale vectors 11 decades
apart (s1 ~ 3e-7, s2 ~ 2.7e4 — legal, the product is gamma-invariant);
STE's bare absolute lr then steps s1 by ~5e4x its own magnitude per
step, ending the stage at nr 4.5e4 with cos NEGATIVE — the single
largest artifact-quality defect in the pipeline's history.

Fix under test (F1 disposition, 2026-10-28): scale-class anchoring is the
STAGE BOUNDARY's job — any s1/s2 split is collapsed by the exact,
product-invariant gamma rescale BEFORE lr derivation, so the stage steps
both classes at the [NQ] App. C absolute lr. The polish-era per-tensor
relative form (``lr * median|p|``) this file originally pinned survives
only as the pathological else-branch; it no longer governs the balanced
class (see ``tests/ptq/test_f1_dispositions.py``).
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


def test_scn_scale_class_split_is_collapsed_at_the_stage_boundary():
    """F1 disposition (2026-10-28): scale-class anchoring moved from the lr to
    the STAGE BOUNDARY. A multi-decade s1/s2 split is collapsed
    product-invariantly BEFORE lr derivation, so an absolute lr can no longer
    strangle one class while blowing the other up: after the stage both classes
    sit within ~4x of the rebalanced entry median.

    Superseded shape (per-tensor relative lr, ``lr * median|p|``) was the
    polish-era cure this replaces; the relative form survives only as the
    pathological else-branch (``tests/ptq/test_f1_dispositions.py``)."""
    torch.manual_seed(0)
    linear, fac = _tiny_block()
    fac.s1.data.mul_(1e-6)  # force the split: s1 class ~1e-6, s2 ~1e0
    fac.s2.data.mul_(1e0)
    pre_s1_med = float(fac.s1.detach().abs().median())

    gen = torch.Generator().manual_seed(1)
    x_b = torch.randn(8, 4, 16, generator=gen, dtype=torch.float64)
    y_b = torch.randn(8, 4, 12, generator=gen, dtype=torch.float64)

    cfg = PTQConfig(
        solver_backend="numpy", compute_dtype="float64", rank=6, stage_latent=StageHyper(1e0, 4, 1)
    )  # brutal lr
    stats = _tune_latent_ste(linear, {"": fac}, x_b, y_b, cfg, 4)

    # The boundary rebalance collapsed the split: gamma == sqrt(median|s2|/median|s1|)
    # takes s1's median to the geometric mid of the two classes, so
    # ``pre_s1_med * gamma`` IS the post-rebalance s1 entry median.
    gamma = stats["rebalance_factor"][""]
    rebalanced_entry = pre_s1_med * gamma

    post_s1_med = float(fac.s1.detach().abs().median())
    post_s2_med = float(fac.s2.detach().abs().median())
    assert post_s1_med > 0.0 and post_s2_med > 0.0
    # Both classes leave the stage within ~4x of the rebalanced entry. The
    # polish-era blowup this replaces was s1 overshooting by ~5e4x PER STEP
    # under a bare absolute lr; the boundary rebalance is what removes it.
    assert post_s1_med < 4.0 * rebalanced_entry, (
        f"s1 class blown past the rebalanced entry: {rebalanced_entry:.3e} -> {post_s1_med:.3e}"
    )
    assert post_s2_med < 4.0 * rebalanced_entry, (
        f"s2 class blown past the rebalanced entry: {rebalanced_entry:.3e} -> {post_s2_med:.3e}"
    )


def test_scn_e2e_gate_svd_solver_path():
    """The solver='svd' path must pass the full e2e gate: the ablation
    branch (truncated SVD -> balance -> polish) flows through capture,
    STE/KD, export, reload and the corruption bounds identically."""
    import dataclasses

    from quantibbler.ptq.e2e import gate_config, run_e2e_gate

    # ``solver_backend="numpy"`` by name, for the same reason the sibling pins
    # in this file do: the production default is torch, and on a box with no
    # accelerator ``resolve_device`` refuses an IMPLICIT device by design. The
    # arm under test is the SVD ablation, not the backend. Float64 is declared
    # with it because the NumPy backend IS the float64 parity oracle: it has no
    # bf16 scalar type, so a numpy arm must name the dtype it runs in.
    cfg = dataclasses.replace(
        gate_config(), solver="svd", solver_backend="numpy", compute_dtype="float64"
    )
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

    # Pin the parity reference explicitly: NumPy is the unit-test arm (and the
    # only arm runnable on a box with no accelerator), while the production
    # default is torch and fails closed there by design. Every sibling call
    # site in the tier pins it the same way.
    bal = magnitude_balance(
        uu.copy(), vv.copy(), np.ones(24), np.ones(16), target=w, backend="numpy"
    )
    bal.s1 = bal.s1 * 1e-6  # force the split polish will inherit

    # bal.Ub/Vb are solver artifacts and may be device int8 tensors; int8 `@`
    # is not implemented on the accelerator ('addmm_cuda' for 'Char'), so the
    # product is taken in host float64 where +-1 products are exact.
    from quantibbler.ptq.solver_backend import to_host_np

    ub_h, vb_h = to_host_np(bal.Ub), to_host_np(bal.Vb)
    s1_h, s2_h = to_host_np(bal.s1), to_host_np(bal.s2)
    # Hand scales too: multiplying a DEVICE tensor by a host array converts
    # the device tensor, which raises the same way the copies above did.
    before = (s1_h[:, None] * (ub_h @ vb_h.T)) * s2_h[None, :]
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
