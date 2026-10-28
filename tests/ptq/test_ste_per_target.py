"""qk-pertarget-cycle: TuneLatentSTE fires per target, one linear at a time.

Scenarios (``openspec/changes/qk-pertarget-cycle/specs/qk-ptq/spec.md``):

* ``ste_refines_all_factorized_linears_in_one_block_pass`` — N fires per block,
  each on ONE target linear (the per-target reading; the once-per-block joint
  pass is the superseded shape this file's predecessors pinned).
* ``per_target_ste_refines_one_linear_at_a_time``.
* ``per_target_ste_cycle_sequence``.
* ``per_target_budget_is_appendix_c_per_linear``.
* ``ste_record_is_one_block_record_with_per_linear_rows``.
* ``dead_columns_are_recorded_per_linear``.
* ``joint_pass_uses_the_normalized_weight_and_the_floor``.
* ``joint_budget_is_the_appendix_c_regime_once_per_block``.

The per-linear movement figure's honesty is pinned beside that scenario set:
``test_zero_baseline_movement_records_no_value`` (an entry baseline of exactly
zero is no denominator, so the figure is `None`, never a confident `0.0`) and
its discriminator ``test_nonzero_baseline_records_zero_when_nothing_moves``.

Plus the harden step (Algorithm 1 lines 20-23) the cycle runs between targets:
sign extraction through the ONE latent-to-bit site, idempotent by construction —
a second extraction of an already-hardened target flips zero bits — and a freeze
that keeps a later fire's optimizer out of the committed target.
"""

import contextlib
import io

import numpy as np
import pytest
import torch
from torch import nn

from quantibbler.ptq.block_recon import (
    STE_PER_TARGET_SCOPE,
    FactorizedLinear,
    _dead_columns,
    _set_linear,
    _tune_latent_ste,
    extract_and_pack,
    harden_target,
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


def _run_events(seed: int = 31, cfg: PTQConfig | None = None) -> list[tuple[str, dict]]:
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
    return sink


def _ste_records(sink: list[tuple[str, dict]]) -> list[dict]:
    return [info for event, info in sink if event == "ste"]


def _run_block(seed: int = 31, cfg: PTQConfig | None = None) -> list[dict]:
    """The block's per-target STE fires — one record per target linear."""
    return _ste_records(_run_events(seed=seed, cfg=cfg))


# --------------------------------------------------------------------------
# ste_refines_all_factorized_linears_in_one_block_pass
# --------------------------------------------------------------------------


def test_scn_ste_refines_all_factorized_linears_in_one_block_pass(monkeypatch):
    """N fires per block, each holding EXACTLY ONE target linear.

    The record count and each call's free set are the discriminator between the
    per-target cycle and the superseded joint pass: the joint reading is ONE
    call holding all N linears; the cycle is N calls each holding one. A
    per-target call can only ever offer one linear as free parameters, so a
    correction needing two linears to move together is not in its reachable set
    — that is the reading the AMEND ruling restores, and this count is what
    separates it from the falsified block-joint shape.
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

    assert len(calls) == len(SEVEN), f"the refinement ran {len(calls)} times for one block: {calls}"
    assert all(len(c) == 1 for c in calls), f"a fire held more than one target linear: {calls}"
    assert sorted(name for c in calls for name in c) == sorted(SEVEN), (
        "the cycle did not cover every factorized linear exactly once"
    )


def test_scn_per_target_ste_refines_one_linear_at_a_time():
    """Each fire's record carries one row — the target's own objective reading.

    The joint pass could not do this: every row shared one block objective
    measured once, so a row could not be attributed to a linear. Here each fire
    is scored against the block output at its own point in the cycle and its
    record names the target and holds exactly that target's row.
    """
    recs = _run_block()
    assert len(recs) == len(SEVEN)
    targets = [r["target"] for r in recs]
    assert sorted(targets) == sorted(SEVEN), targets
    for r in recs:
        assert list(r["per_linear"]) == [r["target"]], (
            "a fire's record must carry exactly its target's row"
        )
        row = r["per_linear"][r["target"]]
        assert {"u_rel_move", "v_rel_move", "residual_before", "residual_after"} <= set(row)


def test_scn_per_target_ste_cycle_sequence():
    """The cycle order: factorize target i → STE(i) → harden → next target.

    Target 1 is factorized first with NO preceding TuneFP; every later target is
    preceded by its TuneFP(still-FP remainder) pass. So the interleaved
    TuneFP/STE record stream is a STE, then (TuneFP, STE) repeated N−1 times —
    the shape that shows STE runs INSIDE the factorization loop rather than once
    after it closes.
    """
    sink = _run_events()
    seq = [(event, info.get("target")) for event, info in sink if event in ("ste", "tune_fp")]
    events = [event for event, _ in seq]
    assert events == ["ste"] + ["tune_fp", "ste"] * (len(SEVEN) - 1), events
    ste_targets = [target for event, target in seq if event == "ste"]
    assert len(set(ste_targets)) == len(SEVEN), (
        f"an STE fire repeated a target instead of advancing: {ste_targets}"
    )
    assert all(target is not None for target in ste_targets), (
        "every STE fire must name the target it refined"
    )


# --------------------------------------------------------------------------
# ste_record_is_one_block_record_with_per_linear_rows
# --------------------------------------------------------------------------


def test_scn_ste_record_is_one_block_record_with_per_linear_rows():
    """One record per fire: its scope, its target, and the target's one row.

    A per-target fire CAN attribute its wall time to the single linear it
    refined, so the target's row carries the fire's own clock (the joint pass
    could not — its one measurement belonged to the whole set).
    """
    recs = _run_block()
    assert len(recs) == len(SEVEN)
    for r in recs:
        assert r["scope"] == STE_PER_TARGET_SCOPE, (
            "a per-target firing must declare its scope; the rows alone cannot "
            "say whether one pass or N produced them"
        )
        assert r["target"] in SEVEN
        assert list(r["per_linear"]) == [r["target"]]
        row = r["per_linear"][r["target"]]
        assert {
            "dead_columns",
            "flips_u",
            "flips_v",
            "per_step_flips",
            "u_rel_move",
            "v_rel_move",
            "s1_rel_move",
            "s2_rel_move",
        } <= set(row), r["target"]
        assert r["elapsed_s"] is not None and r["elapsed_s"] > 0.0, (
            "the fire's own wall time is the target's row time"
        )
        assert row["elapsed_s"] == r["elapsed_s"]


def test_scn_dead_columns_are_recorded_per_linear():
    """A rank component with a zero factor column carries no sign, and is counted.

    The count is the only thing that can distinguish an effective fire from one
    whose rank budget is partly inert by construction.
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
    """Every fire consumes the resolved unit-mean weight and the floor guard."""
    from quantibbler.ptq.block_recon import LOSS_WEIGHT_MEAN_NORMALIZATION

    recs = _run_block()
    assert recs
    for r in recs:
        assert r["loss_weight"] == "terminal_grad"
        assert r["loss_weight_normalization"] == LOSS_WEIGHT_MEAN_NORMALIZATION
        assert r["loss_weight_mean_after"] == pytest.approx(1.0), (
            "the weight handed to the loss is not unit mean"
        )
        assert r["loss_weight_mean_before"] > 0.0, (
            "the pre-normalization scale is not on the record"
        )
        assert r["gradient_floor"] is not None and r["gradient_floor"] > 0.0
        assert r["inert"] is not None, "a completed fire declares its inertia"


# --------------------------------------------------------------------------
# The App. C budget, per linear — consumed afresh for EVERY target
# --------------------------------------------------------------------------


def test_scn_per_target_budget_is_appendix_c_per_linear():
    """Each fire books the App. C regime for its ONE linear, undivided."""
    cfg = _block_cfg(stage_latent=StageHyper(1e-3, 4, 8))
    recs = _run_block(cfg=cfg)
    expected = cfg.stage_latent.epochs * (8 // cfg.stage_latent.batch_size)
    assert len(recs) == len(SEVEN)
    for r in recs:
        assert r["total_steps"] == expected, r["target"]
        assert r["steps"] == expected, f"{r['target']}: the stage did not run its recorded budget"
        assert "allocation" not in r, "a block-pooled allocation is not part of the regime"
        row = r["per_linear"][r["target"]]
        assert row["scheduled_steps"] == expected, r["target"]
        assert row["steps"] == expected, r["target"]


def test_scn_joint_budget_is_the_appendix_c_regime_once_per_block():
    """The block's TOTAL is N× the regime — one full budget per target.

    Where the superseded joint pass spent `epochs x (rows // batch)` ONCE for the
    block and handed every linear a share of that one pass, the cycle spends it
    afresh for each target; the sum across a block of N linears is N× the regime.
    """
    cfg = _block_cfg(stage_latent=StageHyper(1e-3, 4, 8))
    recs = _run_block(cfg=cfg)
    expected = cfg.stage_latent.epochs * (8 // cfg.stage_latent.batch_size)
    assert len(recs) == len(SEVEN)
    assert sum(r["total_steps"] for r in recs) == expected * len(SEVEN), (
        "the block's total STE budget is not N× the App. C factorized-stage regime"
    )


# --------------------------------------------------------------------------
# The per-target call holds ONE linear and leaves its neighbours bit-identical
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
    """A two-linear chain with both linears factorized, or the same fresh pair."""
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


def test_per_target_call_holds_one_linear_and_leaves_the_neighbour_bit_identical():
    """A fire's free set is ONE linear; its neighbour is not merely unmoved.

    Provable, and that is why this is the fixture: a per-target call's optimizer
    holds ONE linear's parameters, so the neighbour's parameters are
    bit-identical after it runs — the neighbour is not in the optimization at
    all. This is the property the cycle relies on when it hardens each target
    before the next fires.
    """
    cfg = PTQConfig(
        solver_backend="numpy", compute_dtype="float64", rank=4, stage_latent=StageHyper(0.3, 1, 1)
    )
    steps = 48
    blk_one, fa_one, fb_one, x_b, y_b = _chained_factors()
    neighbour = [p.detach().clone() for p in _params(fb_one)]
    with _quiet():
        one = _tune_latent_ste(blk_one, {"proj_in": fa_one}, x_b, y_b, cfg, 1, steps=steps)
    assert list(one["per_linear"]) == ["proj_in"], "a per-target call refines ONE linear"
    assert all(
        torch.equal(p.detach(), q) for p, q in zip(_params(fb_one), neighbour, strict=True)
    ), "a neighbour's parameters must be untouched by a per-target pass"


# --------------------------------------------------------------------------
# The harden step (Algorithm 1 lines 20-23) between targets
# --------------------------------------------------------------------------


def test_harden_is_idempotent_and_freezes_the_target():
    """Extraction is the ONE latent-to-bit site, and it commits the target.

    The bits written back are the bits :func:`extract_and_pack` emits later, so
    the export re-extraction is idempotent BY CONSTRUCTION: a second extraction
    of an already-hardened target flips zero bits. The freeze keeps a later
    fire's optimizer out of a value the cycle has already committed.
    """
    linear = nn.Linear(16, 12, bias=False, dtype=torch.float64)
    gen = torch.Generator().manual_seed(6)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(12, 16, generator=gen, dtype=torch.float64) * 0.4)
    fac = FactorizedLinear.from_weight(linear.weight.detach(), rank=6)

    harden_target(fac, "probe")
    assert all(not p.requires_grad for p in (fac.u_latent, fac.v_latent, fac.s1, fac.s2)), (
        "the harden step must freeze all four tensors"
    )

    st = extract_and_pack(fac.u_latent.detach(), fac.v_latent.detach(), name="probe")
    assert (np.abs(st.Ub) == 1).all() and (np.abs(st.Vb) == 1).all()
    # The hardened latent already holds its own sign: re-extraction flips nothing.
    assert torch.equal(torch.sign(fac.u_latent.detach()), fac.u_latent.detach())
    assert torch.equal(torch.sign(fac.v_latent.detach()), fac.v_latent.detach())

    before_u = fac.u_latent.detach().clone()
    before_v = fac.v_latent.detach().clone()
    harden_target(fac, "probe")  # a second harden must be a no-op
    assert torch.equal(fac.u_latent.detach(), before_u), "re-hardening moved the latents"
    assert torch.equal(fac.v_latent.detach(), before_v), "re-hardening moved the latents"


# --------------------------------------------------------------------------
# The provable separation: a zero baseline records no value
# --------------------------------------------------------------------------


def _chained_one(seed: int = 11, d: int = 8, rank: int = 4):
    blk, fa, fb, x_b, y_b = _chained_factors(seed=seed, d=d, rank=rank)
    return blk, fa, fb, x_b, y_b


def test_zero_baseline_movement_records_no_value():
    """An all-zero entry baseline is no denominator: ``None``, never a ``0.0``.

    ``_rel_move`` divides the movement by the entry tensor's L2 norm, so a
    parameter whose entry baseline is exactly zero has no denominator and its
    relative movement is UNDEFINED. The row recorded ``0.0`` for it -- "the
    parameter never moved" -- about a latent the pass may have moved off zero,
    and that confident zero also entered ``max(moves) < floor_rel``, so a moved
    row could be declared inert on a figure nobody measured.
    """
    cfg = PTQConfig(solver_backend="numpy", rank=4, stage_latent=StageHyper(1e-3, 1, 1))

    # (a) Moved off zero: in a one-linear fire a zeroed factor is the only
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


def test_nonzero_baseline_records_zero_when_nothing_moves():
    """The discriminator: a MEASURED zero is still ``0.0``, and not ``None``.

    With a non-zero entry baseline and the factor held out of the optimizer
    nothing moves, so the relative movement is a measured ``0.0``. It is the
    denominator that decides between this and the ``None`` arm above, not the
    movement: 0.0 / anything is a value, while a zero baseline has no ratio at
    all.
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
