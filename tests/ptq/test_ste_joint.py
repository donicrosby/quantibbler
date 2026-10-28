"""qk-ste-joint: TuneLatentSTE refines a block's whole factorized set in one pass.

Scenarios (``openspec/changes/qk-ste-joint/specs/qk-ptq/spec.md``):

* ``ste_refines_all_factorized_linears_in_one_block_pass`` — the refinement is
  ONE pass per block over every factorized linear, one optimizer and one loss.
* ``ste_record_is_one_block_record_with_per_linear_rows`` — one block-scope
  record carrying the per-linear rows and a scope marker.
* ``dead_columns_are_recorded_per_linear``.
* ``joint_pass_uses_the_normalized_weight_and_the_floor``.
* ``joint_budget_is_the_appendix_c_regime_once_per_block``.

The per-linear movement figure's honesty is pinned beside that scenario set:
``test_zero_baseline_movement_records_no_value`` (an entry baseline of exactly
zero is no denominator, so the figure is `None`, never a confident `0.0`) and
its discriminator ``test_nonzero_baseline_records_zero_when_nothing_moves``.

The measured defect these pin shut: the shipped per-target structure emitted one
STE record per block whose seven rows carried seven DIFFERENT objective readings
(0.2098, 0.2321, 0.5454, 0.5727, 1.0080, 1.4611, 1.5653), i.e. one block
objective measured seven times at seven different contexts — each linear refined
against neighbours already committed. The rows also cannot say which of the two
structures produced them, which is what the scope marker is for.
"""

import contextlib
import io

import pytest
import torch
from torch import nn

from quantibbler.ptq.block_recon import (
    STE_JOINT_SCOPE,
    FactorizedLinear,
    _dead_columns,
    _set_linear,
    _tune_latent_ste,
    reconstruct_block,
)
from quantibbler.ptq.config import PTQConfig, StageHyper

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
        self.dkv = dkv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.self_attn["q_proj"](x)
        k = self.self_attn["k_proj"](x)
        v = self.self_attn["v_proj"](x)
        groups = q.shape[-1] // k.shape[-1]
        qq = q.reshape(*q.shape[:-1], k.shape[-1], groups).mean(dim=-1)
        attn = torch.sigmoid(qq * k)
        h = x + self.self_attn["o_proj"](attn * v)
        mlp = self.mlp["down_proj"](
            torch.nn.functional.gelu(self.mlp["gate_proj"](h)) * self.mlp["up_proj"](h)
        )
        return h + mlp


def _calib(d: int = 8, n: int = 4, seq: int = 8, seed: int = 1) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(n, seq, d, generator=gen, dtype=torch.float64) for _ in range(2)]


def _fixed_cot(module, batches):
    """Caller-supplied output cotangent (spec: caller_supplied_head_for_bare_block)."""
    x = torch.cat(batches, dim=0) if len(batches) > 1 else batches[0]
    with torch.no_grad():
        probe = module(x)
    return torch.randn(probe.shape, generator=torch.Generator().manual_seed(9), dtype=probe.dtype)


def _recorder(sink: list):
    def rec(event: str, **info) -> None:
        sink.append((event, info))

    return rec


@contextlib.contextmanager
def _quiet():
    """Swallow the stage's progress lines: they are not what these tests read."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _block_cfg(*, stage_latent: StageHyper | None = None) -> PTQConfig:
    return PTQConfig(
        solver_backend="numpy",
        compute_dtype="float64",
        rank=3,
        stage_latent=stage_latent or StageHyper(1e-3, 4, 8),
    )


def _run_block(seed: int = 31, cfg: PTQConfig | None = None) -> dict:
    block = _SevenLinearBlock(seed=seed)
    calib = _calib(8, seed=seed)
    sink: list = []
    with _quiet():
        reconstruct_block(
            block,
            calib,
            cfg or _block_cfg(),
            cotangent=_fixed_cot(block, calib),
            recorder=_recorder(sink),
        )
    ste = [info for event, info in sink if event == "ste"]
    assert len(ste) == 1, f"expected exactly one STE record per block; got {len(ste)}"
    return ste[0]


# --------------------------------------------------------------------------
# ste_refines_all_factorized_linears_in_one_block_pass
# --------------------------------------------------------------------------


def test_scn_ste_refines_all_factorized_linears_in_one_block_pass(monkeypatch):
    """ONE invocation per block, holding the block's ENTIRE factorized set.

    The refinement unit is the block: a per-target call can only ever offer one
    linear as free parameters, so a correction needing two linears to move
    together is not in its reachable set. The count and the argument are the
    discriminator between the two structures, and they are read from the call
    rather than inferred from the rows.
    """
    from quantibbler.ptq import _block_stage as stage

    calls: list[list[str]] = []
    real = stage._tune_latent_ste

    def spy(block, facs, x_b, y_b, cfg, batch_size, *, steps=None, label="", **kwargs):
        calls.append(sorted(facs))
        return real(block, facs, x_b, y_b, cfg, batch_size, steps=steps, label=label, **kwargs)

    monkeypatch.setattr(stage, "_tune_latent_ste", spy)
    block = _SevenLinearBlock(seed=31)
    calib = _calib(8, seed=31)
    with _quiet():
        reconstruct_block(
            block,
            calib,
            _block_cfg(),
            cotangent=_fixed_cot(block, calib),
            recorder=_recorder([]),
        )

    assert len(calls) == 1, f"the refinement ran {len(calls)} times for one block: {calls}"
    assert calls[0] == sorted(SEVEN), (
        f"the joint pass did not receive the block's whole factorized set: {calls[0]}"
    )


def test_scn_joint_budget_is_the_appendix_c_regime_once_per_block():
    """The App. C factorized-stage regime, consumed once for the whole block.

    `epochs x (rows // batch_size)` — the same regime the per-target cycle handed
    EACH linear, now spent once. Every row books that one budget, so a row cannot
    be read as a per-linear share of it.
    """
    cfg = _block_cfg(stage_latent=StageHyper(1e-3, 4, 8))
    rec = _run_block(cfg=cfg)
    expected = cfg.stage_latent.epochs * (8 // cfg.stage_latent.batch_size)
    assert rec["total_steps"] == expected
    assert rec["steps"] == expected, "the stage did not run its recorded budget"
    assert "allocation" not in rec, "a block-pooled allocation is not part of the regime"
    for name, row in rec["per_linear"].items():
        assert row["scheduled_steps"] == expected, name
        assert row["steps"] == expected, name


# --------------------------------------------------------------------------
# ste_record_is_one_block_record_with_per_linear_rows
# --------------------------------------------------------------------------


def test_scn_ste_record_is_one_block_record_with_per_linear_rows():
    """One record, per-linear rows, and ONE measurement behind every row.

    The shared-per-row objective is the determinism the per-target structure
    cannot produce: there, each row measured the block objective at its own
    point in the cycle (and the values differ), so equal readings across rows are
    only possible when a single pass measured the block once.
    """
    rec = _run_block()

    assert rec["scope"] == STE_JOINT_SCOPE, (
        "a joint firing must declare its scope; the rows alone cannot say whether "
        "one pass or N produced them"
    )
    per = rec["per_linear"]
    assert sorted(per) == sorted(SEVEN), "the record must cover every factorized linear"

    for key in ("residual_before", "residual_after", "gradient_norm"):
        values = {row[key] for row in per.values()}
        assert len(values) == 1, (
            f"{key} differs across rows ({sorted(values)}): the rows were not produced "
            "by one pass over one block objective"
        )
    row = per["mlp.down_proj"]
    assert row["residual_after"] <= row["residual_before"], "the joint pass made the block worse"
    assert rec["elapsed_s"] is not None and rec["elapsed_s"] > 0.0, (
        "the block's own wall time is the stage's row time; per-linear shares are null"
    )
    for name, row in per.items():
        assert {
            "dead_columns",
            "flips_u",
            "flips_v",
            "per_step_flips",
            "u_rel_move",
            "v_rel_move",
            "s1_rel_move",
            "s2_rel_move",
        } <= set(row), name
        assert row["elapsed_s"] is None, (
            f"{name}: a per-linear share of a joint pass would name a share no clock observed"
        )


def test_scn_dead_columns_are_recorded_per_linear():
    """A rank component with a zero factor column carries no sign, and is counted.

    The count is the only thing that can distinguish an effective joint pass from
    one whose rank budget is partly inert by construction.
    """
    linear = nn.Linear(16, 12, bias=False, dtype=torch.float64)
    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(12, 16, generator=gen, dtype=torch.float64) * 0.4)
    fac = FactorizedLinear.from_weight(linear.weight.detach(), rank=6)

    assert _dead_columns(fac) == 0, "a freshly factorized linear has no dead column"
    with torch.no_grad():
        fac.u_latent.data[:, 0] = 0.0
    assert _dead_columns(fac) == 1, "an all-zero U column is a dead rank component"
    with torch.no_grad():
        fac.v_latent.data[:, 1] = 0.0
    assert _dead_columns(fac) == 2, (
        "a dead column in EITHER factor kills the rank component; the counts must union"
    )

    # ... and the row reports it, measured after the pass (a component the pass
    # did not revive stays dead).
    _set_linear(linear, fac)
    with torch.no_grad():
        fac.u_latent.data[:, 0] = 0.0
        fac.v_latent.data[:, 1] = 0.0
    _set_linear(linear, fac)
    x_b = torch.randn(8, 4, 16, generator=gen, dtype=torch.float64)
    y_b = torch.randn(8, 4, 12, generator=gen, dtype=torch.float64)
    with _quiet():
        stats = _tune_latent_ste(
            linear, {"": fac}, x_b, y_b, _block_cfg(), 4, steps=0, label="probe"
        )
    assert stats["per_linear"][""]["dead_columns"] == 2


# --------------------------------------------------------------------------
# joint_pass_uses_the_normalized_weight_and_the_floor
# --------------------------------------------------------------------------


def test_scn_joint_pass_uses_the_normalized_weight_and_the_floor():
    """The joint pass consumes the resolved unit-mean weight and the floor guard."""
    from quantibbler.ptq.block_recon import LOSS_WEIGHT_MEAN_NORMALIZATION

    rec = _run_block()
    assert rec["loss_weight"] == "terminal_grad"
    assert rec["loss_weight_normalization"] == LOSS_WEIGHT_MEAN_NORMALIZATION
    assert rec["loss_weight_mean_after"] == pytest.approx(1.0), (
        "the weight handed to the joint loss is not unit mean"
    )
    assert rec["loss_weight_mean_before"] > 0.0, "the pre-normalization scale is not on the record"
    assert rec["gradient_floor"] is not None and rec["gradient_floor"] > 0.0, (
        "the joint pass must carry the resolution-derived floor it guarded on"
    )
    assert rec["inert"] is not None, "a completed joint pass declares its inertia"


# --------------------------------------------------------------------------
# The provable separation, on a fixture where the correction needs BOTH linears
# --------------------------------------------------------------------------


class _Chain(nn.Module):
    """``out = proj_out(tanh(proj_in(x)))`` — the two linears compose."""

    def __init__(self, d: int = 8, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.proj_in = nn.Linear(d, d, bias=False, dtype=torch.float64)
        self.proj_out = nn.Linear(d, d, bias=False, dtype=torch.float64)
        with torch.no_grad():
            for m in (self.proj_in, self.proj_out):
                m.weight.copy_(torch.randn(d, d, generator=gen, dtype=torch.float64) * 0.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(torch.tanh(self.proj_in(x)))


def _chained_factors(seed: int = 11, d: int = 8, rank: int = 4, near_zero: bool = True):
    """A two-linear chain with both linears factorized, or the same fresh pair.

    The target comes from a DIFFERENT teacher, so the block objective cannot be
    closed by either linear alone; the latents sit near the sign boundary, which
    is what makes the pass's effect measurable through the sign-mapped forward.
    """
    gen = torch.Generator().manual_seed(seed)
    x_b = torch.randn(8, 4, d, generator=gen, dtype=torch.float64)
    teacher = _Chain(d, seed=seed + 100)
    with torch.no_grad():
        y_b = teacher(x_b).detach()

    blk = _Chain(d, seed=seed)
    fa = FactorizedLinear.from_weight(blk.proj_in.weight.detach(), rank)
    fb = FactorizedLinear.from_weight(blk.proj_out.weight.detach(), rank)
    if near_zero:
        g = torch.Generator().manual_seed(seed + 7)
        with torch.no_grad():
            for f in (fa, fb):
                f.u_latent.data.copy_(
                    torch.randn(f.u_latent.shape, generator=g, dtype=torch.float64) * 0.05
                )
                f.v_latent.data.copy_(
                    torch.randn(f.v_latent.shape, generator=g, dtype=torch.float64) * 0.05
                )
    _set_linear(blk.proj_in, fa, "binary")
    _set_linear(blk.proj_out, fb, "binary")
    return blk, fa, fb, x_b, y_b


def _params(fac: FactorizedLinear):
    return (fac.u_latent, fac.v_latent, fac.s1, fac.s2)


def test_joint_pass_frees_every_factorized_linear_at_once():
    """A correction needing BOTH linears is reachable only when both are free.

    Provable, and that is why this is the fixture: a per-target call's optimizer
    holds ONE linear's parameters, so the neighbour's parameters are bit-identical
    after it runs — the neighbour is not merely unmoved, it is not in the
    optimization at all. The joint pass holds both, moves both, and lowers the
    block objective on the same fixture and the same budget.

    What is deliberately NOT claimed here: that the joint pass beats the
    SEQUENTIAL composition's residual. Measured across seeds and budgets that
    ordering is non-monotone, because the composition gets two bounded passes
    where the joint pass gets one.
    """
    cfg = PTQConfig(
        solver_backend="numpy", compute_dtype="float64", rank=4, stage_latent=StageHyper(0.3, 1, 1)
    )
    steps = 48

    # Control: the per-target unit's free set is one linear.
    blk_one, fa_one, fb_one, x_b, y_b = _chained_factors()
    neighbour = [p.detach().clone() for p in _params(fb_one)]
    with _quiet():
        one = _tune_latent_ste(blk_one, {"proj_in": fa_one}, x_b, y_b, cfg, 1, steps=steps)
    assert list(one["per_linear"]) == ["proj_in"], "a per-target call refines ONE linear"
    assert all(
        torch.equal(p.detach(), q) for p, q in zip(_params(fb_one), neighbour, strict=True)
    ), "a neighbour's parameters must be untouched by a per-target pass"

    # The joint pass: both linears are free, both move, the block objective drops.
    blk, fa, fb, x_b, y_b = _chained_factors()
    before_out = [p.detach().clone() for p in _params(fb)]
    with _quiet():
        joint = _tune_latent_ste(
            blk, {"proj_in": fa, "proj_out": fb}, x_b, y_b, cfg, 1, steps=steps
        )
    assert sorted(joint["per_linear"]) == ["proj_in", "proj_out"]
    for name in ("proj_in", "proj_out"):
        row = joint["per_linear"][name]
        assert row["u_rel_move"] > 0.0 and row["v_rel_move"] > 0.0, (
            f"{name}: a joint pass must have every factorized linear's latents moving"
        )
    assert not all(
        torch.equal(p.detach(), q) for p, q in zip(_params(fb), before_out, strict=True)
    ), "the second linear did not move: it was not a free parameter"
    shared = joint["per_linear"]["proj_in"]
    assert shared["residual_after"] < shared["residual_before"], (
        f"the joint pass left the block objective at {shared['residual_after']!r} "
        f"from {shared['residual_before']!r}"
    )


def test_zero_baseline_movement_records_no_value():
    """An all-zero entry baseline is no denominator: ``None``, never a ``0.0``.

    ``_rel_move`` divides the movement by the entry tensor's L2 norm, so a
    parameter whose entry baseline is exactly zero has no denominator and its
    relative movement is UNDEFINED. The row recorded ``0.0`` for it -- "the
    parameter never moved" -- about a latent the pass may have moved off zero,
    and that confident zero also entered ``max(moves) < floor_rel``, so a moved
    row could be declared inert on a figure nobody measured.

    Both degenerate arms are pinned here, each against movement recomputed from
    the held entry snapshot:

    * the zeroed latent MOVED off zero (the old row read exactly ``0.0``);
    * the zeroed latent was HELD at zero -- a zero denominator is no denominator
      whether or not anything moved, so this is still no value and not a
      legitimate measured ``0.0``.

    Reachability note: the driver refuses an all-zero factor at
    ``magnitude_balance`` ("undefined for an all-zero factor"), so this is a
    state of the STAGE BOUNDARY -- and that boundary is the contract this
    function owns, since its refinement unit is the factorized set it is handed.
    """
    cfg = PTQConfig(solver_backend="numpy", rank=4, stage_latent=StageHyper(1e-3, 1, 1))

    # (a) Moved off zero: in a one-linear pass a zeroed factor is the only
    # parameter left carrying a gradient -- its sign is zero, so every term of the
    # outer product that multiplies it dies -- and the STE passthrough steps it.
    blk, fa, _fb, x_b, y_b = _chained_factors()
    with torch.no_grad():
        fa.u_latent.data.zero_()
    entry_u = fa.u_latent.detach().clone()
    with _quiet():
        stats = _tune_latent_ste(blk, {"proj_in": fa}, x_b, y_b, cfg, 1, steps=16)
    rec = stats["per_linear"]["proj_in"]
    assert stats["steps"] == 16 and not stats["vacuous"], "the stage must run for this"
    assert float(entry_u.pow(2).sum()) == 0.0, "the entry baseline is exactly zero"
    assert float((fa.u_latent.detach() - entry_u).pow(2).sum()) > 0.0, (
        "the pass moved the zeroed latent off zero"
    )
    assert rec["u_rel_move"] is None, "a zero entry baseline has no ratio to report"
    assert rec["inert"] is None, "no movement figure -- no inertness verdict either"
    assert stats["inert"] is None, "the stage verdict inherits the row's no-answer"

    # (b) Held at zero: the zeroed factor is held out of the optimizer across the
    # joint pass. Its linear keeps a bias so the neighbour still sees a nonzero
    # input: a zeroed factor zeroes every OTHER gradient of its own linear, so
    # without that feed the sibling linear gets no gradient and the whole stage
    # goes vacuous rather than running over a held baseline.
    blk, fa, fb, x_b, y_b = _chained_factors()
    with torch.no_grad():
        fa.u_latent.data.zero_()
        fa.bias.data.fill_(0.3)
    fa.u_latent.requires_grad_(False)
    entry_u = fa.u_latent.detach().clone()
    with _quiet():
        stats = _tune_latent_ste(blk, {"proj_in": fa, "proj_out": fb}, x_b, y_b, cfg, 1, steps=16)
    rec = stats["per_linear"]["proj_in"]
    assert stats["steps"] == 16 and not stats["vacuous"], "the stage must run for this"
    assert float(entry_u.pow(2).sum()) == 0.0, "the entry baseline is exactly zero"
    assert float((fa.u_latent.detach() - entry_u).pow(2).sum()) == 0.0, (
        "the held factor stayed where the zero baseline was"
    )
    assert rec["u_rel_move"] is None, "held or moved, a zero baseline has no ratio"
    assert rec["inert"] is None, "no movement figure -- no inertness verdict either"
    assert stats["inert"] is None, "the stage verdict inherits the row's no-answer"


def test_nonzero_baseline_records_zero_when_nothing_moves():
    """The discriminator: a MEASURED zero is still ``0.0``, and not ``None``.

    With a non-zero entry baseline and the factor held out of the optimizer
    nothing moves, so the relative movement is a measured ``0.0``. It is the
    denominator that decides between this and the ``None`` arms above, not the
    movement: 0.0 / anything is a value, while a zero baseline has no ratio at
    all. The other factors stay tunable, so the stage runs a live gradient and
    this is not the vacuous path in disguise.
    """
    blk, fa, fb, x_b, y_b = _chained_factors()
    fa.u_latent.requires_grad_(False)
    entry_u = fa.u_latent.detach().clone()
    cfg = PTQConfig(solver_backend="numpy", rank=4, stage_latent=StageHyper(1e-3, 1, 1))
    with _quiet():
        stats = _tune_latent_ste(blk, {"proj_in": fa}, x_b, y_b, cfg, 1, steps=16)
    rec = stats["per_linear"]["proj_in"]

    assert stats["steps"] == 16 and not stats["vacuous"], "the stage must run its budget"
    assert float(entry_u.pow(2).sum()) > 0.0, "the baseline must be non-zero"
    assert float((fa.u_latent.detach() - entry_u).pow(2).sum()) == 0.0
    assert rec["u_rel_move"] == 0.0, "a real denominator makes this a measured 0.0"
