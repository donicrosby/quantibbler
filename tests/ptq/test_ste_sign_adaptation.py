"""qk-fix-ste-sign-adaptation: effective sign adaptation, or a declaration.

Scenarios under test: ``flip_counts_are_recorded``,
``sign_adaptation_is_effective_or_declared``,
``refinement_covers_every_factorized_linear_with_paper_parameter_set``.

The measured defect this file pins shut: 4,096 STE steps flipped ZERO sign
bits on all 7 scoped linears while moving the latents 5-33% — the warm-start
latents sit far from the decision boundary, so the paper's continuous
refinement was inert in the configuration the pipeline actually ran, and a
zero-flip completion was indistinguishable from a healthy stage.
"""

import math

import pytest
import torch
from torch import nn

from quantibbler.ptq.block_recon import (
    STE_RESCOPE_MARKER,
    FactorizedLinear,
    QuantizationDivergenceError,
    _set_linear,
    _tune_latent_ste,
    reconstruct_block,
)
from quantibbler.ptq.config import PTQConfig, StageHyper
from quantibbler.qk_report import STEPS_EFFECT_EPS


def _fixed_cot(module, batches, *, seed: int = 9):
    """Caller-supplied output cotangent (spec: caller_supplied_head_for_bare_block).

    A bare block carries no scalar loss, so the local statistics pass takes
    the gradient at the module output from the caller. The probe runs the
    CONCATENATED batch stream — the local pass feeds ``[x_b]`` with
    ``x_b = cat(batches)``, so the cotangent must carry those rows. Fixed
    seed so runs asserting on recorded events stay reproducible.
    """
    x = torch.cat(batches, dim=0) if len(batches) > 1 else batches[0]
    with torch.no_grad():
        probe = module(x)
    return torch.randn(
        probe.shape, generator=torch.Generator().manual_seed(seed), dtype=probe.dtype
    )


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


def _explicit_linear(seed: int, *, scale: float = 0.3) -> nn.Linear:
    """The 12<-16 fixture Linear, with every draw generator-explicit.

    ``nn.Linear(...)`` consumes the GLOBAL stream for its default init, which the
    next line then overwrites: the weight is deterministic while the global RNG
    state is not, so the fixture silently shifts any later global draw depending
    on how many of these were built. Forking the construction leaves the global
    stream untouched, which is what lets a measured number in this file be cited
    from one CI run to the next. Shape and values are unchanged.
    """
    with torch.random.fork_rng():
        linear = nn.Linear(16, 12, bias=False, dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(12, 16, generator=gen, dtype=torch.float64) * scale)
    return linear


def _recorder(sink: list) -> object:
    def rec(event: str, **info) -> None:
        sink.append((event, info))

    return rec


def _fac_from_weight(weight: torch.Tensor, rank: int, bias: torch.Tensor | None = None):
    return FactorizedLinear.from_weight(weight, rank, bias)


def _push_latents_off_the_boundary(fac) -> None:
    """Give every entry a margin, so no step inside the budget can cross zero.

    The abort-path fixtures want a GEOMETRY where the finite steps flip nothing
    (so ``zero_crossings`` is True unless a rolled-back step is counted) and a
    value a poisoned entry can be flipped away from deterministically.
    """
    with torch.no_grad():
        for latent in (fac.u_latent, fac.v_latent):
            sign = torch.where(
                latent.data >= 0, torch.ones_like(latent.data), -torch.ones_like(latent.data)
            )
            latent.data.copy_(sign * latent.data.abs().clamp_min(0.5))


def _abort_on_the_second_step(monkeypatch, fac, *, at_call: int = 2) -> dict:
    """Make the ``at_call``-th optimizer step land non-finite AND sign-flipped.

    The defect is about ORDER — the flip counters are written before the
    finiteness gate, so a step the armor then rolls back has already voted in
    ``zero_crossings`` (the mechanism ladder's input). The abort therefore has
    to be engineered so the step is a genuine sign change as well as
    non-finite; hunting for that in an accidental numeric overflow would make
    the fixture's premise depend on the platform. The hook runs AFTER the real
    step and overwrites one latent entry with an infinity whose sign is
    opposite to the one it carried.

    Returns the counter of steps this hook observed, so a fixture can prove it
    reached the poisoned step rather than passing on a stage that never ran.
    """
    real_step = torch.optim.Adam.step
    seen = {"n": 0}

    def step(self, *args, **kwargs):
        out = real_step(self, *args, **kwargs)
        seen["n"] += 1
        if seen["n"] == at_call:
            with torch.no_grad():
                entry = fac.u_latent.data.reshape(-1)
                entry[0] = -math.copysign(1.0, float(entry[0])) * float("inf")
        return out

    monkeypatch.setattr(torch.optim.Adam, "step", step)
    return seen


def test_scn_ste_records_its_own_objective_effect():
    """The stage's effect is measured and recorded, not inferred from flips.

    ``zero_crossings`` is a restatement of the flip counters, so on its own the
    record cannot distinguish an effective stage from an inert one: the fatal
    fire booked a full 1024-step budget on all 14 linears, flipped nothing, and
    declared ``paper_ste`` anyway. The trio — ``residual_before``,
    ``residual_after``, ``gradient_norm`` — is what makes that budget auditable.
    """
    linear = _explicit_linear(11)
    fac = _fac_from_weight(linear.weight.detach(), rank=6)
    _set_linear(linear, fac)

    gen = torch.Generator().manual_seed(12)
    x_b = torch.randn(8, 4, 16, generator=gen, dtype=torch.float64)
    # A target the rank-6 factorization does NOT already reproduce: a fixture
    # starting at its own optimum would make the effect assertion vacuous.
    y_b = linear(x_b).detach() + torch.randn(8, 4, 12, generator=gen, dtype=torch.float64) * 0.5

    cfg = PTQConfig(
        solver_backend="numpy", compute_dtype="float64", rank=6, stage_latent=StageHyper(1e-2, 4, 4)
    )
    stats = _tune_latent_ste(linear, {"": fac}, x_b, y_b, cfg, 4, steps=256, label="probe")
    rec = stats["per_linear"][""]
    assert rec["gradient_norm"] > 0.0, (
        "the first-step gradient norm is identically zero: the stage is gradient-dead"
    )
    assert rec["residual_after"] <= rec["residual_before"] * (1.0 - STEPS_EFFECT_EPS), (
        f"STE moved its objective {rec['residual_before']!r} -> {rec['residual_after']!r}, "
        f"short of the shared STEPS_EFFECT_EPS {STEPS_EFFECT_EPS!r}: an inert stage"
    )


# --------------------------------------------------------------------------
# flip_counts_are_recorded + the paper mechanism actually crossing
# --------------------------------------------------------------------------


def test_scn_paper_ste_crosses_a_near_zero_latent_and_records_counts():
    """Latents AT the decision boundary must cross under the paper's own STE."""
    linear = _explicit_linear(3)
    fac = _fac_from_weight(linear.weight.detach(), rank=6)
    with torch.no_grad():
        fac.u_latent.data[0, 0] = 1e-9
        fac.u_latent.data[3, 1] = -1e-9
        fac.v_latent.data[1, 0] = 1e-9
    _set_linear(linear, fac)

    gen = torch.Generator().manual_seed(4)
    x_b = torch.randn(8, 4, 16, generator=gen, dtype=torch.float64)
    y_b = torch.randn(8, 4, 12, generator=gen, dtype=torch.float64)

    cfg = PTQConfig(
        solver_backend="numpy", compute_dtype="float64", rank=6, stage_latent=StageHyper(1e-3, 4, 4)
    )
    stats = _tune_latent_ste(linear, {"": fac}, x_b, y_b, cfg, 4, steps=8, label="probe")
    rec = stats["per_linear"][""]
    assert rec["steps"] == 8
    assert rec["flips_u"] >= 1 and rec["flips_v"] >= 1, (
        "the paper's STE did not flip a latent sitting 1e-9 from the boundary"
    )
    assert rec["first_flip_step"] is not None
    assert len(rec["per_step_flips"]) == 8, "per-step flip counts are missing"
    assert rec["zero_crossings"] is False
    # Both factors' movement and BOTH scale vectors are recorded (scales are
    # IN the paper's joint parameter set, Eq. 10).
    assert rec["u_rel_move"] > 0.0 and rec["v_rel_move"] > 0.0
    assert rec["s1_rel_move"] > 0.0 and rec["s2_rel_move"] > 0.0


def test_scn_zero_flip_completion_is_never_silent():
    """Far-from-zero latents: the stage declares scale-only, or the gate fails."""
    block = _SevenLinearBlock(seed=31)
    sink: list = []
    cfg = PTQConfig(
        solver_backend="numpy", compute_dtype="float64", rank=3, stage_latent=StageHyper(1e-8, 4, 8)
    )
    calib = _calib(8, seed=31)
    reconstruct_block(
        block, calib, cfg, cotangent=_fixed_cot(block, calib), recorder=_recorder(sink)
    )
    ste = [info for event, info in sink if event == "ste"][0]
    per = ste["per_linear"]
    assert per, "the STE stage produced no per-linear record at all"
    zero_flip = [n for n, r in per.items() if r["zero_crossings"]]
    assert zero_flip, "fixture did not reproduce the measured zero-flip geometry"
    for name in zero_flip:
        rec = per[name]
        assert rec["declared"] == "scale_only", f"{name}: zero-flip completion with no declaration"
        assert rec["rescope_marker"] == STE_RESCOPE_MARKER
        assert "flip_pass" not in rec, f"{name}: the deleted mechanism left a record"
    for rec in per.values():
        if rec["zero_crossings"] is False:
            assert rec["declared"] == "paper_ste"


# --------------------------------------------------------------------------
# The non-finite abort path: a rolled-back step must not vote
# --------------------------------------------------------------------------


def test_aborted_step_contributes_no_flips(monkeypatch, capsys):
    """A step the finiteness armor rolls back has not flipped anything.

    The counters are written BEFORE the gate, so a step that lands non-finite
    is counted and then undone: ``zero_crossings`` — a restatement of the flip
    counters, and the exact field the mechanism ladder reads — reads False on
    latents that were restored to the snapshot uncrossed, and
    ``stats["steps"]``/``per_linear[...]["steps"]`` disagree by one. Engineering
    the abort step to flip an entry is what makes the phantom visible: with the
    gate ahead of the counters the step contributes neither a flip nor a step.
    """
    linear = _explicit_linear(13)
    fac = _fac_from_weight(linear.weight.detach(), rank=6)
    _push_latents_off_the_boundary(fac)
    _set_linear(linear, fac)

    gen = torch.Generator().manual_seed(14)
    x_b = torch.randn(8, 4, 16, generator=gen, dtype=torch.float64)
    y_b = torch.randn(8, 4, 12, generator=gen, dtype=torch.float64)

    cfg = PTQConfig(
        solver_backend="numpy", compute_dtype="float64", rank=6, stage_latent=StageHyper(1e-3, 4, 4)
    )
    seen = _abort_on_the_second_step(monkeypatch, fac)
    entry_value = float(fac.u_latent.detach()[0, 0])
    stats = _tune_latent_ste(linear, {"": fac}, x_b, y_b, cfg, 4, steps=8, label="probe")
    rec = stats["per_linear"][""]

    # Precondition, not the claim: the fixture DID reach the non-finite abort
    # path, on the step the poison was aimed at (true before and after the fix).
    out = capsys.readouterr().out
    assert seen["n"] == 2, "the fixture never reached the poisoned optimizer step"
    assert "non-finite parameters after step 1; restored snapshot" in out, (
        "the fixture never took the non-finite abort path"
    )
    # The armor restored the entry the poison flipped (true before and after).
    assert abs(entry_value) >= 0.5, "the fixture's latents are not off the boundary"
    assert float(fac.u_latent.detach()[0, 0]) == entry_value, "the snapshot was not restored"

    assert rec["flips_u"] == 0 and rec["flips_v"] == 0, (
        f"a rolled-back step's sign changes were counted: "
        f"flips_u={rec['flips_u']} flips_v={rec['flips_v']}"
    )
    assert rec["per_step_flips"] == [0], (
        f"the per-step trace booked the rolled-back step: {rec['per_step_flips']}"
    )
    assert rec["steps"] == 1, f"the rolled-back step was booked as a step: {rec['steps']}"
    assert rec["first_flip_step"] is None
    assert rec["zero_crossings"] is True, "the ladder's input was poisoned by a rolled-back step"
    assert stats["steps"] == rec["steps"], (
        f"the stage and the per-linear record disagree on the step count: "
        f"{stats['steps']} vs {rec['steps']}"
    )


def test_aborted_step_does_not_skip_the_mechanism_ladder(monkeypatch):
    """A linear whose STE aborted on a phantom flip is not declared paper_ste.

    The consequence of the poisoned counters, end to end through
    ``reconstruct_block``: ``per["zero_crossings"]`` gates the bounded discrete
    flip pass, so a phantom flip declares the linear ``paper_ste`` and skips the
    ladder while its latents sit where the snapshot left them.
    """
    from quantibbler.ptq import _block_stage as stage
    from quantibbler.ptq import block_recon as br

    real = stage._tune_latent_ste
    poisoned: dict = {}

    # **kwargs, not a re-listed signature: a spy on a private stage function
    # must not break the day the real one gains a kwarg.
    def spy(block, facs, x_b, y_b, cfg, batch_size, *, steps=None, label="", **kwargs):
        name, fac = next(iter(facs.items()))
        if not poisoned:
            poisoned["name"] = name
            _push_latents_off_the_boundary(fac)
            _abort_on_the_second_step(monkeypatch, fac)
        return real(block, facs, x_b, y_b, cfg, batch_size, steps=steps, label=label, **kwargs)

    monkeypatch.setattr(stage, "_tune_latent_ste", spy)
    sink: list = []
    block = _SevenLinearBlock(seed=51)
    cfg = PTQConfig(
        solver_backend="numpy",
        compute_dtype="float64",
        rank=3,
        stage_latent=StageHyper(1e-8, 4, 8),
    )
    calib = _calib(8, seed=51)
    br.reconstruct_block(
        block, calib, cfg, cotangent=_fixed_cot(block, calib), recorder=_recorder(sink)
    )
    ste = [info for event, info in sink if event == "ste"][0]
    rec = ste["per_linear"][poisoned["name"]]

    assert rec["zero_crossings"] is True, (
        f"{poisoned['name']}: the rolled-back step's phantom flips reached the "
        f"ladder input (flips_u={rec['flips_u']}, flips_v={rec['flips_v']})"
    )
    assert rec["declared"] == "scale_only", (
        f"{poisoned['name']}: the ladder was skipped on rolled-back "
        f"flips — declared {rec['declared']!r}"
    )


# --------------------------------------------------------------------------
# refinement_covers_every_factorized_linear_with_paper_parameter_set
# --------------------------------------------------------------------------


def test_scn_refinement_covers_every_factorized_linear_with_full_budget():
    """Every scoped linear covers the recorded budget — no pooled split.

    Scope note (supersedes this test's original reading of `total_steps`): the
    refinement unit is now the BLOCK, not the linear. One joint pass consumes
    `epochs x (n // batch)` steps, and every per-linear row books every step of
    it — which is what these assertions check. What is NOT asserted any more is
    that each linear receives that budget SEVEN TIMES over; the per-target
    structure that did that is gone. What still holds, and is the point: the
    budget is never divided across linears (no pooled split), and no linear is
    left out of the refinement.
    """
    block = _SevenLinearBlock(seed=41)
    sink: list = []
    cfg = PTQConfig(
        solver_backend="numpy", compute_dtype="float64", rank=3, stage_latent=StageHyper(1e-3, 4, 8)
    )
    calib = _calib(8, seed=41)
    reconstruct_block(
        block, calib, cfg, cotangent=_fixed_cot(block, calib), recorder=_recorder(sink)
    )
    ste = [info for event, info in sink if event == "ste"][0]

    assert sorted(ste["per_linear"]) == sorted(SEVEN), (
        "refinement did not cover every factorized linear in the block"
    )
    # 8 epochs x (8 rows // batch 4) — the paper's per-linear budget, and it
    # is what EVERY linear receives.
    total = ste["total_steps"]
    assert total == cfg.stage_latent.epochs * (8 // cfg.stage_latent.batch_size)
    assert "allocation" not in ste, "a block-pooled allocation survived the re-ruling"
    for name in SEVEN:
        rec = ste["per_linear"][name]
        assert rec["scheduled_steps"] == total, name
        assert rec["steps"] == total, name
        assert {
            "flips_u",
            "flips_v",
            "per_step_flips",
            "u_rel_move",
            "v_rel_move",
            "s1_rel_move",
            "s2_rel_move",
        } <= set(rec)
        assert len(rec["per_step_flips"]) == total
        assert rec["s1_rel_move"] >= 0.0 and rec["s2_rel_move"] >= 0.0


def test_full_budget_is_per_linear_not_a_pooled_split():
    """The recorded budget is NOT divided across the block's linears.

    Scope note (supersedes this test's original reading): the budget belongs to
    the block's single joint pass, not to each linear separately — but the
    property under test is unchanged, that it is never divided BY the linear
    count, so a budget smaller than the linear count still runs in full and every
    linear books it.

    Seed note: this fixture family's divergence-gate fires were the ``lb_admm``
    exit double-rescale (both factors scaled by ``||W~||_F``, so the
    reconstructible product carried it twice). Pre-fix the family tripped the
    post-balance gate on 2/24 seeds (16 and 18 of 0..23); with the rescale fixed
    to split across the pair, ZERO of 0..23 trip and the family sits inside the
    shipped band. Earlier revisions of this note said "~4 in every 24 seeds and
    4/24 on the previous placement -- the rate is the fixture's, not the
    placement's": both halves were the defect, not a property of the family.
    Seed 47 stays deliberately off-band -- this test is testing the SCHEDULE,
    and a fixture that detonates an unrelated guard cannot test anything.
    """
    block = _SevenLinearBlock(seed=47)
    sink: list = []
    cfg = PTQConfig(
        solver_backend="numpy", compute_dtype="float64", rank=3, stage_latent=StageHyper(1e-3, 4, 1)
    )
    calib = _calib(8, seed=47)
    reconstruct_block(
        block, calib, cfg, cotangent=_fixed_cot(block, calib), recorder=_recorder(sink)
    )
    ste = [info for event, info in sink if event == "ste"][0]
    assert "allocation" not in ste
    total = ste["total_steps"]
    assert total == 1 * (8 // 4)  # epochs x (rows // batch), per linear
    for name in SEVEN:
        rec = ste["per_linear"][name]
        assert rec["scheduled_steps"] == total
        assert rec["steps"] == total
        # The zero-budget "not_scheduled" residual is gone: every linear runs.
        # The post-cut vocabulary is two-valued: `discrete_flip_pass` left it
        # with the mechanism, and this line is the canary if it re-enters.
        assert rec["declared"] in {"paper_ste", "scale_only"}, name


# --------------------------------------------------------------------------
# The paper regime: polish off, no extra discrete stages, paper_ste
# --------------------------------------------------------------------------


@pytest.fixture
def paper_regime(monkeypatch):
    """Make the block stage read this run as the paper regime.

    The regime conjunction requires bf16 compute on an accelerator, and these
    are the NumPy/float64 parity fixtures — so the VERDICT is stubbed and what
    these legs pin is the ladder's behavior UNDER it. The verdict itself is
    covered where it is computed (the report conditions, the driver's declared
    config). Patched on the module that HOLDS the call site: patching the
    re-export in ``block_recon`` would leave the stage reading the real one.
    """
    from quantibbler.ptq import _block_stage as stage

    monkeypatch.setattr(stage, "config_matches_paper_regime", lambda cfg: True)


def _paper_regime_cfg(**kwargs) -> PTQConfig:
    return PTQConfig(
        solver_backend="numpy",
        compute_dtype="float64",
        rank=3,
        stage_latent=StageHyper(1e-8, 4, 8),
        **kwargs,
    )


def test_scn_paper_faithful_hands_ste_continuous_latents(monkeypatch, paper_regime):
    """The faithful path must hand STE the Eq. 9 latents, not +-1 polish bits."""
    from quantibbler.ptq import _block_stage as stage
    from quantibbler.ptq import block_recon as br

    seen: dict[str, torch.Tensor] = {}
    real = stage._tune_latent_ste

    # **kwargs, not a re-listed signature: this spy intercepts a private stage
    # function, so every new stage kwarg (lane-4's loss_weight is the first)
    # would otherwise break the test with a TypeError that has nothing to do with
    # what it asserts.
    def spy(block, facs, x_b, y_b, cfg, batch_size, *, steps=None, label="", **kwargs):
        for name, fac in facs.items():
            seen[name] = fac.u_latent.detach().clone()
        return real(block, facs, x_b, y_b, cfg, batch_size, steps=steps, label=label, **kwargs)

    monkeypatch.setattr(stage, "_tune_latent_ste", spy)
    blk = _SevenLinearBlock(seed=91)
    calib = _calib(8, seed=91)
    br.reconstruct_block(blk, calib, _paper_regime_cfg(), cotangent=_fixed_cot(blk, calib))
    assert seen, "the faithful run never reached the STE stage"
    for name, u in seen.items():
        assert not torch.all(u.abs() == 1.0), (
            f"{name}: polish-binarized +-1 bits reached STE on the faithful path"
        )


def test_scn_paper_faithful_disables_greedy_polish(monkeypatch, paper_regime):
    """The regime path carries no polish; the production path is untouched."""
    from quantibbler.ptq import _block_stage as stage
    from quantibbler.ptq import block_recon as br

    calls: list[int] = []
    real = stage.greedy_polish

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(stage, "greedy_polish", spy)
    sink: list = []
    blk = _SevenLinearBlock(seed=91)
    calib = _calib(8, seed=91)
    br.reconstruct_block(
        blk,
        calib,
        _paper_regime_cfg(),
        cotangent=_fixed_cot(blk, calib),
        recorder=_recorder(sink),
    )
    assert calls == [], "the regime path ran the extra-paper greedy polish"
    lb = [info for event, info in sink if event == "lb_admm"][0]
    assert lb["polish_rounds"] == 0, "the artifact must PROVE zero polish rounds"


def test_greedy_polish_runs_on_the_production_binary_path(monkeypatch):
    """The non-regime binary path keeps the polish: the gate is the regime, not a flag.

    Separate from the regime leg on purpose: that one stubs the regime verdict
    for the whole test, so the production half has to run where the real verdict
    is in force — otherwise the "production path still polishes" claim would be
    asserted against a stubbed gate.
    """
    from quantibbler.ptq import _block_stage as stage
    from quantibbler.ptq import block_recon as br

    calls: list[int] = []
    real = stage.greedy_polish

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(stage, "greedy_polish", spy)
    sink2: list = []
    blk = _SevenLinearBlock(seed=91)
    calib = _calib(8, seed=91)
    br.reconstruct_block(
        blk,
        calib,
        PTQConfig(
            solver_backend="numpy",
            compute_dtype="float64",
            rank=3,
            stage_latent=StageHyper(1e-8, 4, 8),
        ),
        cotangent=_fixed_cot(blk, calib),
        recorder=_recorder(sink2),
    )
    assert calls == [1] * len(SEVEN), "the production path must polish every scoped linear"
    lb2 = [info for event, info in sink2 if event == "lb_admm"][0]
    assert lb2["polish_rounds"] == PTQConfig(solver_backend="numpy").polish_rounds


def test_scn_paper_faithful_records_measured_crossings_with_no_rescope(paper_regime):
    """`sign_adaptation_is_effective_or_declared`: the regime path is exempt.

    [NQ] prescribes sign+pack on whatever signs STE produced (Algorithm 1
    lines 20-23), so a zero-crossing outcome on the regime path is a
    MEASUREMENT, not a re-scope: no scale-only marker appears and `declared` is
    `paper_ste` on every linear.
    """
    sink: list = []
    block = _SevenLinearBlock(seed=71)
    cfg = _paper_regime_cfg()
    calib = _calib(8, seed=71)
    reconstruct_block(
        block, calib, cfg, cotangent=_fixed_cot(block, calib), recorder=_recorder(sink)
    )
    ste = [info for event, info in sink if event == "ste"][0]
    assert ste["per_linear"], "no per-linear STE record"
    for name, rec in ste["per_linear"].items():
        assert rec["declared"] == "paper_ste", name
        assert "flip_pass" not in rec, name
        assert "rescope_marker" not in rec, name
        assert rec["scheduled_steps"] == ste["total_steps"], name
        # The measurement still travels, so a zero-crossing geometry is on
        # record rather than hidden by the declaration.
        assert isinstance(rec["zero_crossings"], bool)


def test_no_layer_local_forward_form_in_the_block_stage_modules():
    """Source pin: every objective scores the BLOCK forward, not a layer residual.

    The dimensionally-incoherent layer-local form must not appear anywhere in
    the block-reconstruction stage modules: the forwards live in
    ``_factorized``/``_block_stage``, so the scan follows the code across the
    whole ptq package instead of one module's source.
    """
    import pathlib

    ptq_dir = pathlib.Path(__file__).resolve().parents[2] / "src" / "quantibbler" / "ptq"
    module_src = "\n".join(p.read_text() for p in sorted(ptq_dir.glob("*.py")))
    assert "Wq@x_b" not in module_src and "Wq @ x_b" not in module_src


def test_the_divergence_gate_fires_on_a_band_the_live_block_cannot_meet():
    """The guard's firing path is exercised against a live reconstruction.

    Pinning the schedule test OFF the band is only half a job: with nothing ON
    the band, the guard's own firing is untested, so a regression that widens it
    would flip no test at all. This test owns the band from both sides.

    Correction (this test previously claimed seeds 16/18 of the rank-3 family
    trip the SHIPPED band, 2/24, and asserted that firing). That was measured
    while ``lb_admm`` rescaled BOTH factors on exit, overshooting the
    reconstructed product by a full factor of ``||W~||_F``; the 2/24 "bad
    seeds" were that defect, not a property of the family. With the rescale
    fixed the family sits inside the shipped band and correctly no longer
    fires — so the old assertion had been riding the bug it was meant to
    catch. The guard's firing path is therefore pinned here against a band the
    live block genuinely cannot meet, and its silence is pinned against a
    generous one. Both sides run the real block end to end; neither is a mock.
    """
    block = _SevenLinearBlock(seed=16)
    calib = _calib(8, seed=16)

    # Silence side: a generous band must not fire on an honest reconstruction.
    generous = PTQConfig(
        solver_backend="numpy",
        compute_dtype="float64",
        rank=3,
        stage_latent=StageHyper(1e-3, 4, 1),
        divergence_band_factor=1e6,
    )
    assert (
        reconstruct_block(
            block, calib, generous, cotangent=_fixed_cot(block, calib), recorder=_recorder([])
        )
        is not None
    )

    # Firing side: a rank-3 balanced reconstruction of this block cannot have
    # ||Wq||_F/||W||_F within 1e-4 of 1.0, so a band that tight must fire.
    tight = PTQConfig(
        solver_backend="numpy",
        compute_dtype="float64",
        rank=3,
        stage_latent=StageHyper(1e-3, 4, 1),
        divergence_band_factor=1.0001,
    )
    with pytest.raises(QuantizationDivergenceError):
        reconstruct_block(
            block, calib, tight, cotangent=_fixed_cot(block, calib), recorder=_recorder([])
        )
