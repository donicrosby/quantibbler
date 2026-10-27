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
    cfg = PTQConfig(
        solver_backend="numpy", compute_dtype="float64", rank=4, stage_pre=StageHyper(1e-4, 4, 2)
    )
    block = _TwoLinearBlock(d, seed=0)
    gen = torch.Generator().manual_seed(11)
    x_b = torch.randn(4, d, generator=gen, dtype=torch.float64)
    with torch.no_grad():
        y_b = block(x_b) + 0.3 * torch.randn(4, d, generator=gen, dtype=torch.float64)
    return cfg, block, x_b, y_b


def _linear_names(block) -> list[str]:
    """The Phase-1 collection key set for THIS block: names from its own structure.

    No fixture in this file names an architecture's layers: the names come from
    ``named_modules()``, which is exactly what the production statistic collection
    keys on (``calibration.py:219``). A fixture that hardcodes a name set can only
    prove the pipeline works on the architecture that fixture was written for.
    """
    return [n for n, m in block.named_modules() if isinstance(m, nn.Linear)]


def _terminal_names(block) -> list[str]:
    """The block's terminal linear, discovered rather than declared."""
    return _linear_names(block)[-1:]


def _run(block, x_b, y_b, cfg, *, loss_weight):
    """One TuneFP stage with an explicit weight vector (the argument under test)."""
    return _tune_fp(
        block,
        x_b,
        y_b,
        cfg,
        cfg.stage_pre.batch_size,
        label="b0",
        tune_names=_terminal_names(block),
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


# ---------------------------------------------------------------------------
# Phase 4: the RESOLUTION layer, parameterized over architectures
# ---------------------------------------------------------------------------

# Fixture arch map (test DATA, not a hardcoded harness): each arch names its own
# block output projection. The terminal role is looked up BY ROLE, so swapping in a
# different small model means adding a row here -- no assertion mentions a layer
# name, and every row is checked against the block's own discovered key set below.
_ARCHS: dict[str, dict[str, str]] = {
    "qwen3": {"attn": "self_attn.q_proj", "mlp_in": "mlp.gate_proj", "terminal": "mlp.down_proj"},
    "llama": {"attn": "self_attn.q_proj", "mlp_in": "mlp.up_proj", "terminal": "mlp.down_proj"},
    "gpt2": {"attn": "attn.c_attn", "mlp_in": "mlp.c_fc", "terminal": "mlp.c_proj"},
}


# One arch row used as DATA by the two pins below: the name is reachable only through
# the map, never typed into an assertion. The canonical-name test iterates every row,
# which is what keeps the map honest rather than decorative.
_Q3_TERMINAL = _ARCHS["qwen3"]["terminal"]


def _arch_block(arch: str, d: int = 8) -> nn.Module:
    """A toy block carrying the ARCH's own layer names, built from the arch row."""
    if arch not in _ARCHS:
        raise KeyError(f"unknown arch {arch!r} -- known: {', '.join(sorted(_ARCHS))}")
    block = nn.Module()
    for leaf in _ARCHS[arch].values():
        segs = leaf.split(".")
        parent = block
        for seg in segs[:-1]:
            if not hasattr(parent, seg):
                setattr(parent, seg, nn.Module())
            parent = getattr(parent, seg)
        setattr(parent, segs[-1], nn.Linear(d, d, dtype=torch.float64, bias=False))
    with torch.no_grad():
        for m in block.modules():
            if isinstance(m, nn.Linear):
                m.weight.copy_(torch.eye(d, dtype=torch.float64) * 0.5)
    # A forward over the block's OWN discovered linears (registration order), so the
    # block can be driven by the real reconstruct_block path without the arch's
    # forward being re-declared per architecture.
    chain = tuple(m for m in block.modules() if isinstance(m, nn.Linear))

    def forward(x, _chain=chain):
        h = x
        for i, m in enumerate(_chain):
            h = m(h)
            if i + 1 < len(_chain):
                h = torch.tanh(h)
        return h

    block.forward = forward
    return block


def _stat_entry(name: str, d_out_raw, d_in_raw=None):
    """A statistics-book entry with a chosen output statistic."""
    from quantibbler.ptq.calibration import LayerStatsEntry

    d_out = np.asarray(d_out_raw, dtype=np.float64)
    d_in = np.ones_like(d_out) if d_in_raw is None else np.asarray(d_in_raw, dtype=np.float64)
    return LayerStatsEntry(name, d_in, d_out)


def _plain_cfg(**kw):
    return PTQConfig(
        solver_backend="numpy",
        compute_dtype="float64",
        rank=4,
        stage_pre=StageHyper(1e-4, 4, 2),
        **kw,
    )


@pytest.mark.parametrize("arch", sorted(_ARCHS))
def test_scn_terminal_selection_follows_the_canonical_name_not_the_position(arch):
    """The terminal is the arch's block output projection -- or a RECORDED fallback.

    Parameterized over every arch in the fixture map, so a resolver that knows only one
    architecture's vocabulary fails here instead of passing on the harness it was written
    against. The arch's terminal sits FIRST in book order, so positional selection and
    role selection disagree by construction -- position cannot be what passes this.

    An arch whose terminal name the resolver does not recognise takes uniform weights,
    and the resolver must SAY so (`uniform_fallback` plus the absent layer's name):
    a silent uniform on an unknown arch is the fail-open pattern this lane is closing.
    """
    from quantibbler.ptq.block_recon import CANONICAL_TERMINAL_NAMES, _resolve_loss_weight
    from quantibbler.ptq.calibration import robust_diag

    block = _arch_block(arch)
    discovered = _linear_names(block)
    terminal = _ARCHS[arch]["terminal"]
    assert terminal in discovered, (
        f"{arch}: the fixture's declared terminal {terminal!r} is not in the block's own "
        f"key set {discovered!r} -- the arch row lies about the structure"
    )

    cfg = _plain_cfg()
    entry = _stat_entry(terminal, [1.0, 2.0, 3.0, 4.0])
    pairs = [(entry, (terminal, None))]
    pairs += [
        (_stat_entry(n, [7.0, 7.0, 7.0, 7.0]), (n, None)) for n in discovered if n != terminal
    ]

    weight, source, missing = _resolve_loss_weight(cfg, pairs)
    if terminal.rsplit(".", 1)[-1] in CANONICAL_TERMINAL_NAMES:
        assert missing is None, f"{arch}: a recognised terminal must not report a fallback"
        assert source.startswith("terminal_grad:"), source
        assert weight is not None
        _di, want = robust_diag(entry.d_in_raw, entry.d_out_raw, tau=cfg.tau, gamma=cfg.gamma)
        assert np.allclose(weight, want), (
            f"{arch}: the weight did not come from the arch's canonical terminal linear "
            "(a positional pick takes a different linear here)"
        )
    else:
        assert weight is None, (
            f"{arch}: {terminal!r} is not in the resolver's vocabulary, so this arch must "
            "fall back to uniform rather than be weighted silently"
        )
        assert source == "uniform_fallback", source
        assert missing == CANONICAL_TERMINAL_NAMES[0], (
            f"{arch}: the fallback must name the absent canonical layer, got {missing!r}"
        )


def test_terminal_loss_weight_is_robust_diag_of_the_terminal_statistic():
    """Poisoning the terminal's statistic must move the weight it produces."""
    from quantibbler.ptq.block_recon import _terminal_loss_weight
    from quantibbler.ptq.calibration import robust_diag

    cfg = _plain_cfg()
    clean = _stat_entry(f"blk.{_Q3_TERMINAL}", [1.0, 2.0, 3.0, 4.0])
    poisoned = _stat_entry(f"blk.{_Q3_TERMINAL}", [1.0, 2.0, 3.0, 1e6])

    got_clean = _terminal_loss_weight(cfg, [(clean, (_Q3_TERMINAL, None))])
    got_poisoned = _terminal_loss_weight(cfg, [(poisoned, (_Q3_TERMINAL, None))])
    _di, want_clean = robust_diag(clean.d_in_raw, clean.d_out_raw, tau=cfg.tau, gamma=cfg.gamma)
    _di, want_poisoned = robust_diag(
        poisoned.d_in_raw, poisoned.d_out_raw, tau=cfg.tau, gamma=cfg.gamma
    )
    assert np.allclose(got_clean, want_clean), "the weight is not robust_diag of the statistic"
    assert np.allclose(got_poisoned, want_poisoned)
    assert not np.allclose(got_clean, got_poisoned), (
        "poisoning the terminal statistic did not move the resolved weight"
    )


def test_terminal_loss_weight_is_none_for_uniform_and_for_an_empty_book():
    """uniform -> None (all-ones at the loss); an empty book -> None, not a crash."""
    from quantibbler.ptq.block_recon import _terminal_loss_weight

    entry = _stat_entry(f"blk.{_Q3_TERMINAL}", [1.0, 2.0])
    pairs = [(entry, (_Q3_TERMINAL, None))]
    assert _terminal_loss_weight(_plain_cfg(block_loss_weight="uniform"), pairs) is None
    assert _terminal_loss_weight(_plain_cfg(), []) is None


# ---------------------------------------------------------------------------
# Phase 4, part 2: the recorded fallback, end to end through the caller
# ---------------------------------------------------------------------------


def _recorder(sink: list):
    def record(event: str, **payload) -> None:
        sink.append((event, payload))

    return record


def _calib(d: int, n: int = 4, seq: int = 8, seed: int = 23) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(n, seq, d, generator=gen, dtype=torch.float64) for _ in range(2)]


def _fixed_cot(module, batches):
    x = torch.cat(batches, dim=0) if len(batches) > 1 else batches[0]
    with torch.no_grad():
        probe = module(x)
    return torch.randn(probe.shape, generator=torch.Generator().manual_seed(9), dtype=probe.dtype)


@pytest.mark.parametrize("mode", ["terminal_grad", "uniform"])
def test_scn_uniform_fallback_recorded_on_down_proj_less_block(mode):
    """Same down_proj-less block, two modes: the MODE decides the event, not the block.

    ``terminal_grad``: no canonical terminal among the block's discovered names, so the
    stage weights uniformly AND records ``tune_loss_uniform_fallback`` naming the block
    and the missing layer. ``uniform``: the explicit ablation - uniform weights and NO
    event. The ablation arm is the positive control: without it, a resolver that fired
    the event on every uniform run (over-emission) would pass this file, and the
    ablation/fallback distinction the resolver exists to keep would be decoration.
    """
    from quantibbler.ptq.block_recon import (
        CANONICAL_TERMINAL_NAMES,
        TUNE_LOSS_FALLBACK_EVENT,
        reconstruct_block,
    )

    arch = "gpt2"
    block = _arch_block(arch)
    discovered = _linear_names(block)
    assert not any(n.rsplit(".", 1)[-1] in CANONICAL_TERMINAL_NAMES for n in discovered), (
        f"{arch} must be canonical-free for this fixture; discovered {discovered!r}"
    )

    cfg = PTQConfig(
        solver_backend="numpy",
        compute_dtype="float64",
        rank=4,
        block_loss_weight=mode,
        stage_pre=StageHyper(1e-4, 4, 2),
        stage_latent=StageHyper(1e-3, 4, 4),
    )
    sink: list = []
    batches = _calib(8)
    reconstruct_block(
        block, batches, cfg, cotangent=_fixed_cot(block, batches), recorder=_recorder(sink)
    )

    events = [e for e, _ in sink]
    payloads = [p for e, p in sink if e == TUNE_LOSS_FALLBACK_EVENT]
    if mode == "terminal_grad":
        assert payloads, (
            f"no {TUNE_LOSS_FALLBACK_EVENT} on record for a canonical-free block: the "
            f"uniform weighting is silent (events seen: {sorted(set(events))})"
        )
        assert payloads[0]["missing_canonical_layer"] in CANONICAL_TERMINAL_NAMES
        assert payloads[0]["block"], "the fallback event must name the block"
    else:
        assert not payloads, (
            "the explicit uniform ablation recorded a fallback it never took -- the "
            "ablation and the fallback are the distinction the resolver keeps"
        )

    for stage in ("tune_fp", "ste"):
        recs = [p for e, p in sink if e == stage]
        assert recs, f"no {stage} record emitted"
        for rec in recs:
            assert "loss_weight" in rec and "loss_weight_source" in rec, (stage, rec)
            assert rec["loss_weight"] == "uniform", (stage, rec)
            expected = "uniform" if mode == "uniform" else "uniform_fallback"
            assert rec["loss_weight_source"] == expected, (stage, rec)
