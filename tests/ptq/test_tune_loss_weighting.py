"""qk-nanoquant-alignment-lane4 phases 2-4: the tuning losses are weighted MSE.

Scenario under test (TuneFP half, phase 2):
``weighting_reaches_the_loss_poisoned_weight_changes_loss``.

Spec letter: the TuneFP and TuneLatentSTE losses SHALL be weighted MSE over the
block-output residual against the cached teacher -- ``mean(w * (B(X) - Y_b)**2)``
per output feature -- with ``w`` the detached, non-trainable robust
diagonalization of the block's terminal linear's Phase-1 output statistic
(``d_out_raw``), per [NQ] App. C :3012 and Algorithm 1 lines 2-5 with Step 2-1
Eq. 2. The weight SOURCE is the change's informed assumption IA-1 (design.md);
the spec does not claim the paper dictates where the statistic is consumed, and
neither does this file.

The discriminator is deliberately narrow: two runs of ``_tune_fp`` on identical
inputs that differ ONLY in the weight vector handed in. While the loss ignores
its weight (plain MSE at ``block_recon.py:705``), the two runs are identical and
the test fails. When the weight reaches the loss, poisoning one entry changes the
step-0 gradient and the two runs separate. The weight arrives as an ARGUMENT --
it is resolved once, at the call site, from the block's statistics book -- so this
fixture keeps testing the loss site and not a re-derivation buried inside it.

Observable: ``stats["gradient_norm"]`` is recorded at step 0 from the loss
gradient (``block_recon.py:719``). No first-step LOSS VALUE is recorded anywhere
in the stage record, so the gradient norm is what this fixture can honestly read;
the loss itself is not exposed and this test does not pretend it is.
"""

import copy

import numpy as np
import pytest
import torch
from torch import nn

from quantibbler.ptq.block_recon import (
    FactorizedLinear,
    _set_linear,
    _tune_fp,
    _tune_latent_ste,
)
from quantibbler.ptq.config import PTQConfig, StageHyper


class _TwoLinearBlock(nn.Module):
    """fc1 -> tanh -> fc2, with fc2 the terminal (tuned) linear."""

    def __init__(self, d: int = 8, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.fc1 = nn.Linear(d, d, dtype=torch.float64)
        self.fc2 = nn.Linear(d, d, dtype=torch.float64)
        with torch.no_grad():
            for m in (self.fc1, self.fc2):
                m.weight.copy_(torch.randn(d, d, generator=gen, dtype=torch.float64) * 0.4)
                m.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(x)))


def _fixture(d: int = 8):
    """Toy block, input batch, and a target that is NOT the block's own output.

    The target must differ from ``block(x_b)`` or the objective is zero at init
    and the fail-closed teacher guard (``tune_fp_vacuous_teacher``) fires instead
    of the loss ever being evaluated -- the measured defect qk-fix-tunefp-teacher
    pinned shut. ``factorized_in_block=1`` keeps the call on the post-factorization
    side of the guard.
    """
    cfg = PTQConfig(solver_backend="numpy", rank=4, stage_pre=StageHyper(1e-4, 4, 2))
    block = _TwoLinearBlock(d, seed=0)
    gen = torch.Generator().manual_seed(11)
    x_b = torch.randn(4, d, generator=gen, dtype=torch.float64)
    with torch.no_grad():
        y_b = block(x_b) + 0.3 * torch.randn(4, d, generator=gen, dtype=torch.float64)
    return cfg, block, x_b, y_b


def _run(block, x_b, y_b, cfg, *, loss_weight):
    """One TuneFP stage with an explicit weight vector (the argument under test)."""
    return _tune_fp(
        block,
        x_b,
        y_b,
        cfg,
        cfg.stage_pre.batch_size,
        label="b0",
        tune_names=["fc2"],
        factorized_in_block=1,
        loss_weight=loss_weight,
    )


def test_scn_weighting_reaches_the_loss_poisoned_weight_changes_loss():
    """Poisoned statistic entry -> different step-0 gradient. TuneFP half."""
    cfg, block, x_b, y_b = _fixture()
    d = x_b.shape[1]
    clean_w = np.ones(d, dtype=np.float64)

    stats_clean = _run(copy.deepcopy(block), x_b, y_b, cfg, loss_weight=clean_w)
    poisoned_w = clean_w.copy()
    poisoned_w[0] *= 1e6  # sentinel magnitude on one output feature
    stats_poisoned = _run(copy.deepcopy(block), x_b, y_b, cfg, loss_weight=poisoned_w)

    # Guard sanity: both runs actually tuned. A zero-gradient run would satisfy a
    # naive "the two readings differ" assertion for the wrong reason.
    assert stats_clean["vacuous"] is False, stats_clean
    assert stats_poisoned["vacuous"] is False, stats_poisoned
    assert stats_clean["steps"] > 0 and stats_poisoned["steps"] > 0

    assert stats_clean["gradient_norm"] != pytest.approx(
        stats_poisoned["gradient_norm"], rel=1e-12
    ), (
        "the poisoned weight vector did not change the step-0 reading "
        f"(clean {stats_clean['gradient_norm']!r}, "
        f"poisoned {stats_poisoned['gradient_norm']!r}) -- the loss is not "
        "consuming its weight"
    )


# ---------------------------------------------------------------------------
# Phase 3: the same poisoned-weight discriminator, TuneLatentSTE half
# ---------------------------------------------------------------------------


def _ste_fixture(d_in: int = 16, d_out: int = 12, rank: int = 6):
    """Identical-by-construction STE fixture: same weights, inputs and target.

    Rebuilt from fixed seeds per run, so the only difference between the two runs
    below is the weight vector handed in. The target is the factorized module's
    own output PLUS noise: a fixture starting at its own optimum has no gradient
    and would satisfy the assertion for the wrong reason. The factors are
    installed with ``_set_linear`` exactly as the production caller does, so the
    stage sees the shipped arrangement rather than a bare module.
    """
    linear = nn.Linear(d_in, d_out, dtype=torch.float64)
    gen = torch.Generator().manual_seed(21)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(d_out, d_in, generator=gen, dtype=torch.float64) * 0.4)
        linear.bias.zero_()
    fac = FactorizedLinear.from_weight(linear.weight.detach(), rank)
    _set_linear(linear, fac)
    cfg = PTQConfig(solver_backend="numpy", rank=rank, stage_latent=StageHyper(1e-3, 1, 4))
    x_b = torch.randn(8, 4, d_in, generator=gen, dtype=torch.float64)
    with torch.no_grad():
        y_b = linear(x_b).detach() + (
            torch.randn(8, 4, d_out, generator=gen, dtype=torch.float64) * 0.5
        )
    return linear, fac, x_b, y_b, cfg


def _ste_run(loss_weight) -> dict:
    linear, fac, x_b, y_b, cfg = _ste_fixture()
    stats = _tune_latent_ste(linear, {"": fac}, x_b, y_b, cfg, 4, steps=1, loss_weight=loss_weight)
    return stats["per_linear"][""]


def test_scn_weighting_reaches_the_loss_poisoned_weight_changes_loss_ste():
    """Poisoned statistic entry -> different step-0 gradient. STE half."""
    d_out = 12
    clean = _ste_run(np.ones(d_out, dtype=np.float64))
    poisoned_w = np.ones(d_out, dtype=np.float64)
    poisoned_w[0] *= 1e6  # sentinel magnitude on one output feature
    poisoned = _ste_run(poisoned_w)

    assert clean["gradient_norm"] > 0.0, clean
    assert poisoned["gradient_norm"] > 0.0, poisoned
    assert clean["gradient_norm"] != pytest.approx(poisoned["gradient_norm"], rel=1e-12), (
        "the poisoned weight vector did not change the STE step-0 reading "
        f"(clean {clean['gradient_norm']!r}, poisoned {poisoned['gradient_norm']!r}) "
        "-- the STE loss is not consuming its weight"
    )
