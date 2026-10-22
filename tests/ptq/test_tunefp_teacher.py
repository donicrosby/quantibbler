"""qk-fix-tunefp-teacher: the Figure 3 TuneFP placement, pristine-pinned teacher.

Scenarios under test: ``teacher_is_captured_before_tuning``,
``tunefp_fires_per_target_cycle``, ``vacuous_teacher_fails_closed``,
``tune_fp_effect_is_recorded_honestly``,
``teacher_cache_is_memory_only_and_recovered_on_resume``.

The measured defect this file pins shut: the only production call site passed
``y_b = block(x_running)`` and ``x_b = x_running.clone()`` — the teacher was
the same module's own output on the same input, so the objective was zero at
init, its gradient was identically zero, and TuneFP never executed in any
archived run (bit-identical T0 on all 7 scoped linears, both families).
"""

import ast
import dataclasses
import pathlib

import numpy as np
import pytest
import torch
from torch import nn

from quantibbler.ptq.block_recon import (
    TUNE_FP_LOSS_MARKER,
    VacuousTeacherError,
    _tune_fp,
    _weights_fingerprint,
    reconstruct_block,
    reconstruct_model,
)
from quantibbler.ptq.config import PTQConfig, StageHyper

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "src" / "quantibbler" / "ptq" / "block_recon.py"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


class _TwoLinearBlock(nn.Module):
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


class _ScopedBlock(nn.Module):
    """q_proj + gate_proj: the attention projection is the pre-TuneFP group."""

    def __init__(self, d: int = 8, seed: int = 2) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.self_attn = nn.ModuleDict({"q_proj": nn.Linear(d, d, dtype=torch.float64)})
        self.mlp = nn.ModuleDict({"gate_proj": nn.Linear(d, d, dtype=torch.float64)})
        with torch.no_grad():
            for m in (self.self_attn["q_proj"], self.mlp["gate_proj"]):
                m.weight.copy_(torch.randn(d, d, generator=gen, dtype=torch.float64) * 0.4)
                m.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp["gate_proj"](torch.tanh(self.self_attn["q_proj"](x)))


class _ThreeLinearBlock(nn.Module):
    """Three targets, so the per-target cycle and the once-per-block reading
    disagree on the TuneFP count (N-1 = 2 vs 1) — the discriminator."""

    def __init__(self, d: int = 8, seed: int = 5) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.self_attn = nn.ModuleDict({"q_proj": nn.Linear(d, d, dtype=torch.float64)})
        self.mlp = nn.ModuleDict(
            {
                "gate_proj": nn.Linear(d, d, dtype=torch.float64),
                "up_proj": nn.Linear(d, d, dtype=torch.float64),
            }
        )
        with torch.no_grad():
            for m in (*self.self_attn.values(), *self.mlp.values()):
                m.weight.copy_(torch.randn(d, d, generator=gen, dtype=torch.float64) * 0.4)
                m.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.self_attn["q_proj"](x)
        h = self.mlp["gate_proj"](torch.tanh(h))
        return self.mlp["up_proj"](torch.tanh(h))


class _DetachedStudent(nn.Module):
    """A tunable linear whose output is discarded: zero gradient, nonzero residual."""

    def __init__(self, d: int = 8, seed: int = 4) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.frozen = nn.Linear(d, d, bias=False, dtype=torch.float64)
        self.unused = nn.Linear(d, d, bias=False, dtype=torch.float64)
        with torch.no_grad():
            for m in (self.frozen, self.unused):
                m.weight.copy_(torch.randn(d, d, generator=gen, dtype=torch.float64) * 0.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.frozen(x)
        self.unused(x)  # computed, never reaches the output
        return h


def _calib(d: int = 8, n: int = 4, seq: int = 8, seed: int = 1) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(n, seq, d, generator=gen, dtype=torch.float64) for _ in range(2)]


def _recorder(sink: list) -> object:
    def rec(event: str, **info) -> None:
        sink.append((event, info))

    return rec


# --------------------------------------------------------------------------
# vacuous_teacher_fails_closed (A3): two-norm event + fidelity invalidation
# --------------------------------------------------------------------------


def test_scn_vacuous_teacher_carries_both_norms_and_invalidates_fidelity():
    """Broken teacher (grad 0 / residual 0) and detached student (grad 0 / residual > 0)."""
    d = 8
    x_b = torch.cat(_calib(d), dim=0)
    cfg = PTQConfig(rank=4, stage_pre=StageHyper(1e-4, 4, 2))

    # (1) The shipped shape: the target IS the block's own current output.
    block = _TwoLinearBlock(d, seed=0)
    with torch.no_grad():
        y_b = block(x_b)
    before = {
        n: m.weight.detach().clone() for n, m in block.named_modules() if isinstance(m, nn.Linear)
    }
    sink: list = []
    stats = _tune_fp(
        block,
        x_b,
        y_b,
        cfg,
        cfg.stage_pre.batch_size,
        label="b0",
        factorized_in_block=0,  # legacy pre-loop placement: nothing factorized
        recorder=_recorder(sink),
    )
    assert stats["vacuous"] is True and stats["steps"] == 0
    assert stats["gradient_norm"] == 0.0
    assert stats["initial_residual_norm"] == 0.0
    events = [info for event, info in sink if event == "tune_fp_vacuous_teacher"]
    assert len(events) == 1, "the guard must emit exactly one event per vacuous call"
    assert events[0]["reason"] == "no_in_block_layer_factorized"
    assert events[0]["gradient_norm"] == 0.0
    assert events[0]["initial_residual_norm"] == 0.0
    after = {n: m.weight.detach() for n, m in block.named_modules() if isinstance(m, nn.Linear)}
    assert all(torch.equal(before[n], after[n]) for n in before), (
        "a diagnostic run must continue with PRISTINE weights"
    )

    # (2) Detached student: the tunable set cannot reach the output.
    student = _DetachedStudent(d, seed=5)
    xs = torch.cat(_calib(d, seed=6), dim=0)
    with torch.no_grad():
        target = student.frozen(xs).clone()
    with torch.no_grad():
        student.frozen.weight.mul_(0.5)  # simulate a factorized in-block layer
    sink2: list = []
    stats2 = _tune_fp(
        student,
        xs,
        target,
        cfg,
        cfg.stage_pre.batch_size,
        label="b1",
        tune_names=["unused"],
        factorized_in_block=1,
        recorder=_recorder(sink2),
    )
    assert stats2["vacuous"] is True
    assert stats2["gradient_norm"] == 0.0
    assert stats2["initial_residual_norm"] > 0.0, (
        "gradient-zero with residual-nonzero is a detached student, not a broken teacher"
    )
    assert [i for e, i in sink2 if e == "tune_fp_vacuous_teacher"][0]["reason"] == (
        "zero_first_step_gradient"
    )

    # (3) A run claiming paper fidelity fails closed instead of continuing.
    faithful = dataclasses.replace(cfg, paper_faithful=True)
    with pytest.raises(VacuousTeacherError, match="vacuous teacher"):
        _tune_fp(
            _TwoLinearBlock(d, seed=0),
            x_b,
            y_b,
            faithful,
            faithful.stage_pre.batch_size,
            label="b0",
            factorized_in_block=0,
        )


def test_scn_diagnostic_run_records_the_vacuous_event_and_continues():
    """reconstruct_block on a degenerate block: the skip is recorded, never silent."""
    block = _TwoLinearBlock(8, seed=0)
    sink: list = []
    result = reconstruct_block(
        block,
        _calib(8, seed=9),
        PTQConfig(rank=4, stage_pre=StageHyper(1e-4, 4, 1)),
        recorder=_recorder(sink),
    )
    assert result is not None
    tune = [info for event, info in sink if event == "tune_fp"]
    assert len(tune) == 1
    assert tune[0]["marker"] == TUNE_FP_LOSS_MARKER
    assert tune[0]["tunable"] == ["fc2"]
    assert tune[0]["factorized_in_block"] == 1
    assert tune[0]["steps"] > 0


# --------------------------------------------------------------------------
# tunefp_fires_per_target_cycle (Figure 3, re-ruled): N-1 passes, N refinements
# --------------------------------------------------------------------------


def test_scn_tunefp_fires_per_target_cycle():
    """Figure 3's per-target cycle: TuneFP N-1 times, STE N times.

    The falsified once-per-block reading fired TuneFP exactly ONCE on this
    block; the cycle fires it once per target after the first. The count is
    therefore the discriminator between the two readings, and it is why this
    test could not stay green through the rework.
    """
    block = _ThreeLinearBlock(8, seed=5)
    sink: list = []
    reconstruct_block(
        block,
        _calib(8, seed=17),
        PTQConfig(rank=4, stage_pre=StageHyper(1e-3, 4, 2)),
        recorder=_recorder(sink),
    )
    names = [event for event, _ in sink]
    tune = [info for event, info in sink if event == "tune_fp"]
    assert len(tune) == 2, (
        f"the per-target cycle fires TuneFP N-1 times (N=3 targets here); got {len(tune)}"
    )
    assert [len(t["tunable"]) for t in tune] == [2, 1], (
        "each pass tunes exactly the still-FP remainder at that point in the cycle"
    )
    assert [t["factorized_in_block"] for t in tune] == [1, 2], (
        "the pass must see the targets already frozen ahead of it"
    )
    assert set(tune[1]["tunable"]) < set(tune[0]["tunable"]), (
        "the still-FP remainder must shrink as targets are factorized"
    )
    assert all(t["gradient_norm"] is not None and t["gradient_norm"] > 0.0 for t in tune), (
        "with >= 1 in-block linear factorized the objective cannot be identically zero"
    )
    # STE fires N times: every factorization is refined, target 1 included.
    ste = [info for event, info in sink if event == "ste"]
    assert len(ste) == 1
    assert len(ste[0]["per_linear"]) == 3, (
        "every factorized linear gets its refinement, target 1 included"
    )
    assert names.index("capture") < names.index("tune_fp")
    assert names.index("lb_admm") < names.index("ste")


def test_tunefp_call_site_precedes_the_targets_own_factorization():
    """Shape pin: one call site, inside the loop, BEFORE the target is frozen.

    Loop membership alone proves nothing — the falsified once-per-block
    placement also lived inside the loop — so the pin is the ORDERING against
    the target's own ``_set_linear`` write. A pin that only asserted
    membership would have stayed green straight through this rework, which is
    the same failure mode as the suite that certified the detonated fire.
    """
    src = MODULE_PATH.read_text()
    tree = ast.parse(src)

    def calls(node: ast.AST, fname: str) -> list[int]:
        return [
            n.lineno
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == fname
        ]

    module_tune = calls(tree, "_tune_fp")
    assert len(module_tune) == 1, f"expected exactly one _tune_fp call site; found {module_tune}"
    loop = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.For) and calls(n, "_tune_fp")),
        None,
    )
    assert loop is not None, "the TuneFP call site must sit inside the factorization loop"
    set_lines = calls(loop, "_set_linear")
    assert set_lines, "the loop must freeze each linear it factorizes"
    assert module_tune[0] < min(set_lines), (
        "the per-target cycle tunes the still-FP remainder BEFORE factorizing the "
        f"target: _tune_fp at {module_tune[0]} must precede the loop's first "
        f"_set_linear at {min(set_lines)}"
    )


# --------------------------------------------------------------------------
# teacher_is_captured_before_tuning (A1) + memory-only cache (A2)
# --------------------------------------------------------------------------


def test_scn_teacher_is_pinned_to_pristine_weights_with_provenance(tmp_path):
    model = nn.Sequential(_ScopedBlock(8, seed=7), _ScopedBlock(8, seed=8))
    pristine = [copy_weights(b) for b in model]
    batches = _calib(8, seed=13)
    sink: list = []
    reconstruct_model(
        model,
        batches,
        PTQConfig(rank=4, stage_pre=StageHyper(1e-4, 4, 0)),
        recorder=_recorder(sink),
        checkpoint_dir=tmp_path / "ck",
    )

    captures = [info for event, info in sink if event == "capture"]
    assert len(captures) == 2
    for idx, info in enumerate(captures):
        assert info["teacher_source"] == "frozen_reference_copy"
        assert info["capture_index"] == idx
        assert info["teacher_fingerprint"] == _weights_fingerprint(pristine[idx]), (
            "the recorded teacher fingerprint is not the pristine block's"
        )
        assert info["teacher_fingerprint"] != _weights_fingerprint(model[idx]), (
            "the teacher was read off LIVE (already written-back) module state"
        )
        # Memory-only (A2): detached, no graph, never a Parameter.
        assert info["y_b"].requires_grad is False
        assert info["y_b"].grad_fn is None

    # A2: the checkpoint writer persists no teacher tensor.
    for path in (tmp_path / "ck").glob("block-*.npz"):
        with np.load(path) as npz:
            assert not any("teacher" in key for key in npz.files), npz.files
    assert list((tmp_path / "ck").glob("*teacher*")) == []


def test_scn_tunefp_effect_is_recorded_honestly():
    """Real teacher: nonzero gradient, decreasing residual, nonzero fp delta."""
    block = _TwoLinearBlock(8, seed=12)
    x_b = torch.cat(_calib(8, seed=14), dim=0)
    with torch.no_grad():
        y_b = block(x_b)
    with torch.no_grad():
        block.fc1.weight.mul_(0.5)  # the factorized in-block layer's error
    sink: list = []
    stats = _tune_fp(
        block,
        x_b,
        y_b,
        PTQConfig(rank=4, stage_pre=StageHyper(1e-3, 4, 8)),
        4,
        label="b",
        tune_names=["fc2"],
        factorized_in_block=1,
        recorder=_recorder(sink),
    )
    assert stats["vacuous"] is False
    assert stats["gradient_norm"] > 0.0
    assert stats["steps"] > 0
    assert stats["weight_delta_rel"] > 0.0, "the fp weights did not move — Step 1 was inert"
    assert stats["residual_after"] < stats["residual_before"]
    assert stats["marker"] == TUNE_FP_LOSS_MARKER
    assert stats["teacher_fingerprint"] is None or isinstance(stats["teacher_fingerprint"], str)


def copy_weights(module: nn.Module) -> nn.Module:
    """A pristine structural copy (weights only) for fingerprint comparison."""
    import copy

    clone = copy.deepcopy(module)
    for param in clone.parameters():
        param.requires_grad_(False)
    return clone
