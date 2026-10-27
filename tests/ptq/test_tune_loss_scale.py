"""qk-tune-loss-scale: unit-mean loss weight, resolution floor, declared inertia.

Spec letter (`openspec/changes/qk-tune-loss-scale/specs/qk-ptq/spec.md`):

- ``weight_is_normalized_before_the_loss`` — the terminal-statistic tuning loss
  weight SHALL be normalized to unit mean before use, and both stage records
  SHALL carry the pre/post-normalization means.
- ``below_floor_gradient_declared_vacuous`` — the vacuous-stage guard SHALL test
  a resolution-derived floor (optimizer eps x parameter-dtype relative
  resolution), with an identically-zero gradient a special case.
- ``full_budget_without_movement_is_declared_inert`` — a stage that runs its
  budget and moves less than the dtype resolution SHALL record ``inert: true``.

The v5 signature this reproduces (``qk-pertarget-cycle`` design.md 23): on bf16
parameters a raw-scale terminal weight (mean ~1e-10) puts the step-0 gradient at
~1e-11, below the 7.8125e-11 floor, and every step rounds back — the stage ran
256 steps and moved exactly nothing while ``vacuous_events`` stayed 0.
"""

import numpy as np
import pytest
import torch
from torch import nn

from quantibbler.ptq.block_recon import (
    LOSS_WEIGHT_MEAN_NORMALIZATION,
    _normalize_weight_to_unit_mean,
    _optimizer_gradient_floor,
    _tune_fp,
    _tune_latent_ste,
)
from quantibbler.ptq.config import PTQConfig, StageHyper


class _TwoLinearBlock(nn.Module):
    """fc1 -> tanh -> fc2, with every parameter in the stage's own ``dtype``."""

    def __init__(self, d: int = 8, seed: int = 0, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.fc1 = nn.Linear(d, d, dtype=dtype)
        self.fc2 = nn.Linear(d, d, dtype=dtype)
        with torch.no_grad():
            for m in (self.fc1, self.fc2):
                m.weight.copy_((torch.randn(d, d, generator=gen) * 0.4).to(dtype))
                m.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(x)))


def _fixture(dtype: torch.dtype, d: int = 8):
    """Toy block, batch and a target that is NOT the block's own output.

    The target must differ from ``block(x_b)`` or the objective is zero at init
    and the fail-closed guard fires for a different reason than the one under
    test here. ``factorized_in_block=1`` keeps the call on the post-factorization
    side of the placement rule.
    """
    cfg = PTQConfig(
        solver_backend="numpy", compute_dtype="float64", rank=4, stage_pre=StageHyper(1e-4, 4, 2)
    )
    block = _TwoLinearBlock(d, seed=0, dtype=dtype)
    gen = torch.Generator().manual_seed(11)
    x_b = torch.randn(4, d, generator=gen).to(dtype)
    with torch.no_grad():
        y_b = (block(x_b) + 0.3 * torch.randn(4, d, generator=gen)).to(dtype)
    return cfg, block, x_b, y_b


def _tune(block, x_b, y_b, cfg, *, loss_weight, recorder=None) -> dict:
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
        recorder=recorder,
    )


def _recorder(sink: list):
    def record(event: str, **payload) -> None:
        sink.append((event, payload))

    return record


# ---------------------------------------------------------------------------
# The floor's derivation: optimizer eps x the COARSEST parameter dtype
# ---------------------------------------------------------------------------


def test_gradient_floor_is_derived_from_the_optimizer_eps_and_the_dtype():
    """Both factors are read live, and the coarsest dtype governs a mixed set."""
    f64 = nn.Parameter(torch.zeros(2, dtype=torch.float64))
    f32 = nn.Parameter(torch.zeros(2, dtype=torch.float32))
    bf16 = nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))

    floor, name, rel = _optimizer_gradient_floor(1e-8, [f64])
    assert floor == pytest.approx(1e-8 * float(torch.finfo(torch.float64).eps))
    assert name == "torch.float64"
    assert rel == pytest.approx(float(torch.finfo(torch.float64).eps))

    mixed, mixed_name, mixed_rel = _optimizer_gradient_floor(1e-8, [f64, f32, bf16])
    assert mixed_name == "torch.bfloat16", "a bf16 parameter is the one that rounds back"
    assert mixed == pytest.approx(1e-8 * float(torch.finfo(torch.bfloat16).eps))
    assert mixed_rel == pytest.approx(float(torch.finfo(torch.bfloat16).eps))

    # The floor scales with the optimizer's own eps rather than a module constant.
    other, _, _ = _optimizer_gradient_floor(1e-6, [bf16])
    assert other == pytest.approx(100.0 * mixed)


# ---------------------------------------------------------------------------
# weight_is_normalized_before_the_loss
# ---------------------------------------------------------------------------


def test_scn_weight_is_normalized_before_the_loss():
    """Unit mean, relative structure preserved, both means returned."""
    raw = np.array([1.3, 2.1, 2.9, 3.7], dtype=np.float64)
    norm, pre, post = _normalize_weight_to_unit_mean(raw)

    assert pre == pytest.approx(float(raw.mean()))
    assert post == pytest.approx(1.0)
    assert float(norm.mean()) == pytest.approx(1.0)
    assert np.allclose(norm, raw / raw.mean()), "the normalization is not a pure rescale"
    for i in range(len(raw) - 1):
        assert norm[i + 1] / norm[i] == pytest.approx(raw[i + 1] / raw[i]), (
            "the relative per-feature structure (the statistical content) moved"
        )

    # A degenerate weight is handed back untouched with equal means: the record
    # must be able to say the normalization was a no-op rather than divide by zero.
    flat, flat_pre, flat_post = _normalize_weight_to_unit_mean(np.zeros(4, dtype=np.float64))
    assert flat_pre == 0.0 and flat_post == 0.0
    assert np.array_equal(flat, np.zeros(4))


class _DownProjBlock(nn.Module):
    """A block whose terminal linear carries the canonical name ``down_proj``."""

    def __init__(self, d: int = 8, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(d, d, dtype=torch.float64, bias=False)
        self.mlp.down_proj = nn.Linear(d, d, dtype=torch.float64, bias=False)
        with torch.no_grad():
            for m in (self.mlp.up_proj, self.mlp.down_proj):
                m.weight.copy_(torch.randn(d, d, generator=gen, dtype=torch.float64) * 0.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp.down_proj(torch.tanh(self.mlp.up_proj(x)))


def _calib(d: int, n: int = 4, seq: int = 8, seed: int = 23) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(n, seq, d, generator=gen, dtype=torch.float64) for _ in range(2)]


def _fixed_cot(module, batches) -> torch.Tensor:
    x = torch.cat(batches, dim=0)
    with torch.no_grad():
        probe = module(x)
    return torch.randn(probe.shape, generator=torch.Generator().manual_seed(9), dtype=probe.dtype)


def test_scn_weight_is_normalized_before_the_loss_pipeline():
    """Both stage records of a terminal-weighted block carry the two means."""
    from quantibbler.ptq.block_recon import reconstruct_block

    block = _DownProjBlock(8)
    batches = _calib(8)
    sink: list = []
    reconstruct_block(
        block,
        batches,
        PTQConfig(
            solver_backend="numpy",
            compute_dtype="float64",
            rank=4,
            stage_pre=StageHyper(1e-4, 4, 2),
            stage_latent=StageHyper(1e-3, 4, 4),
        ),
        cotangent=_fixed_cot(block, batches),
        recorder=_recorder(sink),
    )

    for stage in ("tune_fp", "ste"):
        recs = [payload for event, payload in sink if event == stage]
        assert recs, f"no {stage} record emitted"
        for rec in recs:
            assert rec["loss_weight"] == "terminal_grad", (stage, rec)
            assert rec["loss_weight_normalization"] == LOSS_WEIGHT_MEAN_NORMALIZATION, (stage, rec)
            assert rec["loss_weight_mean_after"] == pytest.approx(1.0), (
                f"{stage}: the weight handed to the loss is not unit mean"
            )
            assert rec["loss_weight_mean_before"] > 0.0, (
                f"{stage}: the pre-normalization scale is not on the record"
            )


# ---------------------------------------------------------------------------
# below_floor_gradient_declared_vacuous
# ---------------------------------------------------------------------------


def test_scn_below_floor_gradient_declared_vacuous():
    """The v5 signature: raw-scale weight -> nonzero step-0 gradient below floor."""
    cfg, block, x_b, y_b = _fixture(torch.bfloat16)
    raw = np.full(8, 1e-10, dtype=np.float64)  # the raw robust-diagonal scale
    sink: list = []
    stats = _tune(block, x_b, y_b, cfg, loss_weight=raw, recorder=_recorder(sink))

    bf16_rel = float(torch.finfo(torch.bfloat16).eps)
    assert stats["floor_param_dtype"] == "torch.bfloat16"
    assert stats["floor_dtype_rel_resolution"] == pytest.approx(bf16_rel)
    assert stats["gradient_floor"] == pytest.approx(1e-8 * bf16_rel)

    assert stats["gradient_norm"] > 0.0, "the fixture must produce a NONZERO dead gradient"
    assert stats["gradient_norm"] < stats["gradient_floor"], (
        f"grad {stats['gradient_norm']!r} is not below the floor {stats['gradient_floor']!r}"
        " -- this fixture no longer reproduces the v5 regime"
    )
    assert stats["vacuous"] is True
    assert stats["vacuous_reason"] == "below_resolution_gradient_floor"
    assert stats["steps"] == 0
    assert stats["inert"] is True

    events = [info for event, info in sink if event == "tune_fp_vacuous_teacher"]
    assert len(events) == 1, "the guard must emit exactly one event per vacuous call"
    assert events[0]["reason"] == "below_resolution_gradient_floor"
    assert events[0]["gradient_floor"] == stats["gradient_floor"]
    assert events[0]["floor_optimizer_eps"] == 1e-8


def test_scn_exact_zero_gradient_is_a_special_case_of_the_floor():
    """An identically-zero gradient still lands on the guard, by the same floor."""
    cfg, block, x_b, _ = _fixture(torch.bfloat16)
    with torch.no_grad():
        y_own = block(x_b).clone()  # the target IS the block's own output
    stats = _tune(block, x_b, y_own, cfg, loss_weight=np.full(8, 1e-10, dtype=np.float64))

    assert stats["gradient_norm"] == 0.0
    assert stats["gradient_norm"] < stats["gradient_floor"]
    assert stats["vacuous"] is True
    assert stats["vacuous_reason"] == "zero_first_step_gradient"


# ---------------------------------------------------------------------------
# full_budget_without_movement_is_declared_inert
# ---------------------------------------------------------------------------


def test_scn_full_budget_without_movement_is_declared_inert():
    """bf16 rounds the (now live) step back -> inert; f64 clears it -> moved."""
    unit = np.ones(8, dtype=np.float64)

    cfg, block, x_b, y_b = _fixture(torch.bfloat16)
    bf = _tune(block, x_b, y_b, cfg, loss_weight=unit)
    assert bf["vacuous"] is False, bf
    assert bf["steps"] > 0
    assert bf["gradient_norm"] > bf["gradient_floor"], "a live gradient is not a vacuous stage"
    assert bf["weight_delta_rel"] < bf["floor_dtype_rel_resolution"], (
        f"the fixture moved {bf['weight_delta_rel']!r} on bf16, above its"
        f" {bf['floor_dtype_rel_resolution']!r} resolution -- it no longer reproduces the"
        " rounding-back regime"
    )
    assert bf["inert"] is True

    cfg64, block64, x64, y64 = _fixture(torch.float64)
    f64 = _tune(block64, x64, y64, cfg64, loss_weight=unit)
    assert f64["steps"] > 0
    assert f64["weight_delta_rel"] > f64["floor_dtype_rel_resolution"], (
        "the f64 fixture must move above its own resolution: it is the positive control"
        " that the floor is not flagging every stage"
    )
    assert f64["inert"] is False, "declared, never absent: an effective stage says so"


def test_scn_full_budget_without_movement_is_declared_inert_ste():
    """TuneLatentSTE declares the floor and the per-linear inertia too."""
    linear, fac, x_b, y_b, cfg = _ste_fixture()

    live = _tune_latent_ste(
        linear, {"": fac}, x_b, y_b, cfg, 4, steps=2, loss_weight=np.ones(12, dtype=np.float64)
    )
    rec = live["per_linear"][""]
    assert live["gradient_floor"] == pytest.approx(1e-8 * float(torch.finfo(torch.float64).eps))
    assert live["floor_param_dtype"] == "torch.float64"
    assert rec["gradient_norm"] > live["gradient_floor"]
    assert rec["inert"] is False, "a live f64 refinement row moved"
    assert live["inert"] is False

    # A weight so small the step-0 gradient falls under the f64 floor: the stage
    # declares itself vacuous and inert and restores the entry latents.
    linear2, fac2, x2, y2, cfg2 = _ste_fixture()
    entry = fac2.u_latent.detach().clone()
    dead = _tune_latent_ste(
        linear2, {"": fac2}, x2, y2, cfg2, 4, steps=2, loss_weight=np.full(12, 1e-30)
    )
    dead_rec = dead["per_linear"][""]
    assert dead_rec["gradient_norm"] < dead["gradient_floor"]
    assert dead["vacuous"] is True
    assert dead["vacuous_reason"] == "below_resolution_gradient_floor"
    assert dead["inert"] is True
    assert dead_rec["inert"] is True
    assert dead_rec["steps"] == 0
    assert torch.equal(fac2.u_latent.detach(), entry), (
        "a vacuous stage must restore the entry latents, never leave a half step"
    )


def _ste_fixture(d_in: int = 16, d_out: int = 12, rank: int = 6):
    """The shipped STE arrangement: factors installed with ``_set_linear``."""
    from quantibbler.ptq.block_recon import FactorizedLinear, _set_linear

    linear = nn.Linear(d_in, d_out, dtype=torch.float64)
    gen = torch.Generator().manual_seed(21)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(d_out, d_in, generator=gen, dtype=torch.float64) * 0.4)
        linear.bias.zero_()
    fac = FactorizedLinear.from_weight(linear.weight.detach(), rank)
    _set_linear(linear, fac)
    cfg = PTQConfig(
        solver_backend="numpy",
        compute_dtype="float64",
        rank=rank,
        stage_latent=StageHyper(1e-3, 1, 4),
    )
    x_b = torch.randn(8, 4, d_in, generator=gen, dtype=torch.float64)
    with torch.no_grad():
        y_b = linear(x_b).detach() + (
            torch.randn(8, 4, d_out, generator=gen, dtype=torch.float64) * 0.5
        )
    return linear, fac, x_b, y_b, cfg
