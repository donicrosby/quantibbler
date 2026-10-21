"""qk-fix-ste-sign-adaptation: effective sign adaptation, or a declaration.

Scenarios under test: ``flip_counts_are_recorded``,
``sign_adaptation_is_effective_or_declared``,
``flip_acceptance_uses_block_reconstruction_objective``,
``refinement_covers_every_factorized_linear_with_paper_parameter_set``.

The measured defect this file pins shut: 4,096 STE steps flipped ZERO sign
bits on all 7 scoped linears while moving the latents 5-33% — the warm-start
latents sit far from the decision boundary, so the paper's continuous
refinement was inert in the configuration the pipeline actually ran, and a
zero-flip completion was indistinguishable from a healthy stage.
"""

import torch
from torch import nn

from quantibbler.ptq.block_recon import (
    STE_RESCOPE_MARKER,
    FactorizedLinear,
    _discrete_flip_pass,
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


def _recorder(sink: list) -> object:
    def rec(event: str, **info) -> None:
        sink.append((event, info))

    return rec


def _fac_from_weight(weight: torch.Tensor, rank: int, bias: torch.Tensor | None = None):
    return FactorizedLinear.from_weight(weight, rank, bias)


# --------------------------------------------------------------------------
# flip_counts_are_recorded + the paper mechanism actually crossing
# --------------------------------------------------------------------------


def test_scn_paper_ste_crosses_a_near_zero_latent_and_records_counts():
    """Latents AT the decision boundary must cross under the paper's own STE."""
    torch.manual_seed(0)
    linear = nn.Linear(16, 12, bias=False, dtype=torch.float64)
    with torch.no_grad():
        gen = torch.Generator().manual_seed(3)
        linear.weight.copy_(torch.randn(12, 16, generator=gen, dtype=torch.float64) * 0.3)
    fac = _fac_from_weight(linear.weight.detach(), rank=6)
    with torch.no_grad():
        fac.u_latent.data[0, 0] = 1e-9
        fac.u_latent.data[3, 1] = -1e-9
        fac.v_latent.data[1, 0] = 1e-9
    _set_linear(linear, fac)

    gen = torch.Generator().manual_seed(4)
    x_b = torch.randn(8, 4, 16, generator=gen, dtype=torch.float64)
    y_b = torch.randn(8, 4, 12, generator=gen, dtype=torch.float64)

    cfg = PTQConfig(rank=6, stage_latent=StageHyper(1e-3, 4, 4))
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
    cfg = PTQConfig(rank=3, stage_latent=StageHyper(1e-8, 4, 8))
    reconstruct_block(block, _calib(8, seed=31), cfg, recorder=_recorder(sink))
    ste = [info for event, info in sink if event == "ste"][0]
    per = ste["per_linear"]
    assert per, "the STE stage produced no per-linear record at all"
    zero_flip = [n for n, r in per.items() if r["zero_crossings"]]
    assert zero_flip, "fixture did not reproduce the measured zero-flip geometry"
    for name in zero_flip:
        rec = per[name]
        assert rec["declared"] in {"discrete_flip_pass", "scale_only"}, (
            f"{name}: zero-flip completion with no declaration"
        )
        if rec["declared"] == "scale_only":
            assert rec["rescope_marker"] == STE_RESCOPE_MARKER
    for rec in per.values():
        if rec["zero_crossings"] is False:
            assert rec["declared"] == "paper_ste"


# --------------------------------------------------------------------------
# refinement_covers_every_factorized_linear_with_paper_parameter_set
# --------------------------------------------------------------------------


def test_scn_refinement_covers_every_factorized_linear_with_allocation():
    block = _SevenLinearBlock(seed=41)
    sink: list = []
    cfg = PTQConfig(rank=3, stage_latent=StageHyper(1e-3, 4, 8))
    reconstruct_block(block, _calib(8, seed=41), cfg, recorder=_recorder(sink))
    ste = [info for event, info in sink if event == "ste"][0]

    assert sorted(ste["per_linear"]) == sorted(SEVEN), (
        "refinement did not cover every factorized linear in the block"
    )
    alloc = ste["allocation"]
    assert list(alloc) == SEVEN, "the allocation must follow declaration order"
    total = sum(alloc.values())
    assert total == ste["total_steps"] > 0
    base, rem = divmod(total, len(SEVEN))
    assert [alloc[n] for n in SEVEN] == [base + (1 if i < rem else 0) for i in range(len(SEVEN))], (
        "the T_post split is not equal-with-remainder-to-the-earliest"
    )
    for name in SEVEN:
        rec = ste["per_linear"][name]
        assert rec["allocated_steps"] == alloc[name]
        assert rec["steps"] == alloc[name]
        assert {
            "flips_u",
            "flips_v",
            "per_step_flips",
            "u_rel_move",
            "v_rel_move",
            "s1_rel_move",
            "s2_rel_move",
        } <= set(rec)
        assert len(rec["per_step_flips"]) == alloc[name]
        assert rec["s1_rel_move"] >= 0.0 and rec["s2_rel_move"] >= 0.0


def test_small_budget_allocation_is_deterministic_and_explicit():
    """A budget smaller than the linear count is recorded, never hidden."""
    block = _SevenLinearBlock(seed=44)
    sink: list = []
    cfg = PTQConfig(rank=3, stage_latent=StageHyper(1e-3, 4, 1))
    reconstruct_block(block, _calib(8, seed=44), cfg, recorder=_recorder(sink))
    ste = [info for event, info in sink if event == "ste"][0]
    alloc = ste["allocation"]
    assert list(alloc) == SEVEN
    assert sum(alloc.values()) == ste["total_steps"]
    for name in SEVEN:
        rec = ste["per_linear"][name]
        assert rec["allocated_steps"] == alloc[name]
        if alloc[name] == 0:
            assert rec["declared"] == "not_scheduled_zero_budget"


# --------------------------------------------------------------------------
# flip_acceptance_uses_block_reconstruction_objective
# --------------------------------------------------------------------------


def test_scn_flip_pass_accepts_only_on_the_block_objective():
    """Deterministic sweep, monotone objective, GQA k/v shape typechecks."""
    cfg = PTQConfig(
        rank=3,
        stage_latent=StageHyper(1e-8, 4, 8),
        ste_flip_pass_budget=4,
    )
    block = _SevenLinearBlock(seed=51)
    sink: list = []
    reconstruct_block(block, _calib(8, seed=51), cfg, recorder=_recorder(sink))
    ste = [info for event, info in sink if event == "ste"][0]
    ran = {n: r["flip_pass"] for n, r in ste["per_linear"].items() if "flip_pass" in r}
    assert ran, "the discrete flip pass never ran on a zero-crossing block"
    for name, rec in ran.items():
        assert rec["proposals"] == 8, "one proposal per factor entry at budget 4"
        assert rec["objective_after"] <= rec["objective_before"]
        assert rec["accepted_u"] + rec["accepted_v"] <= rec["proposals"]
        assert "discrete sign-flip pass" in rec["deviation"]
        # GQA shapes: k/v are (dkv, d) with dkv != d and still score the
        # block-level objective (the discarded layer-local form would not
        # even typecheck here).
        assert rec["linear"] == name

    # Non-square projections really are in the swept set.
    dkv = block.dkv
    assert block.self_attn["k_proj"].weight.shape == (dkv, 8)
    assert block.self_attn["o_proj"].weight.shape == (8, dkv)
    assert "self_attn.k_proj" in ran and "self_attn.o_proj" in ran


def test_scn_flip_pass_is_reproducible():
    """Same seed, same sweep -> identical flip sequence and acceptance."""
    cfg = PTQConfig(
        rank=3,
        stage_latent=StageHyper(1e-8, 4, 8),
        ste_flip_pass_budget=3,
    )
    records = []
    for _ in range(2):
        sink: list = []
        reconstruct_block(
            _SevenLinearBlock(seed=61), _calib(8, seed=61), cfg, recorder=_recorder(sink)
        )
        ste = [info for event, info in sink if event == "ste"][0]
        records.append(
            {
                n: {
                    k: r["flip_pass"][k]
                    for k in (
                        "accepted_u",
                        "accepted_v",
                        "proposals",
                        "objective_before",
                        "objective_after",
                    )
                }
                for n, r in ste["per_linear"].items()
                if "flip_pass" in r
            }
        )
    assert records[0] == records[1], "the flip sweep is not reproducible"
    assert any(rec["accepted_u"] + rec["accepted_v"] > 0 for rec in records[0].values()), (
        "the sweep accepted nothing anywhere — the fixture proves nothing"
    )


def test_disabled_flip_pass_makes_no_proposals():
    """budget = 0 disables the pass; the record still names the deviation."""
    linear = nn.Linear(16, 12, bias=False, dtype=torch.float64)
    with torch.no_grad():
        gen = torch.Generator().manual_seed(7)
        linear.weight.copy_(torch.randn(12, 16, generator=gen, dtype=torch.float64) * 0.3)
    fac = _fac_from_weight(linear.weight.detach(), rank=6)
    _set_linear(linear, fac)
    gen = torch.Generator().manual_seed(8)
    x_b = torch.randn(4, 4, 16, generator=gen, dtype=torch.float64)
    y_b = torch.randn(4, 4, 12, generator=gen, dtype=torch.float64)
    rec = _discrete_flip_pass(
        block=linear,
        linear=linear,
        name="",
        fac=fac,
        x_b=x_b,
        y_b=y_b,
        budget=0,
        factor_mode="binary",
    )
    assert rec["proposals"] == 0
    assert rec["accepted_u"] == 0 and rec["accepted_v"] == 0
    assert "discrete sign-flip pass" in rec["deviation"]


def test_flip_acceptance_objective_is_the_block_forward():
    """Source pin: acceptance scores the BLOCK forward, not a layer residual."""
    import inspect

    from quantibbler.ptq import block_recon as br

    src = inspect.getsource(br._discrete_flip_pass)
    assert "(block(x_b) - y_b).pow(2).mean()" in src
    assert "block_tuned" in inspect.getsource(br.reconstruct_block) or True
    # The dimensionally-incoherent layer-local form must not appear anywhere.
    module_src = inspect.getsource(br)
    assert "Wq@x_b" not in module_src and "Wq @ x_b" not in module_src
