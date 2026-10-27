"""The PTQ stage's stdout progress lines: one per block, plus the phase
boundaries before the block loop.

Why they are pinned: a CI cancel destroys the runner, so the job log is the
ONLY witness of where a wedged PTQ stage stopped, and a stalled stage that
prints nothing between two stage banners cannot be localised at all.
"""

from __future__ import annotations

import re

import pytest
import torch
from torch import nn

from quantibbler.ptq.api import quantize_model
from quantibbler.ptq.block_recon import reconstruct_model
from quantibbler.ptq.calibration import collect_stats
from quantibbler.ptq.config import PTQConfig
from quantibbler.ptq.e2e import TrunkOutput

LINE = re.compile(r"^\[ptq\] (?P<note>.+?) wall=(?P<wall>\d+\.\d+)s(?: cgroup=\d+\.\d+G)?$")


class _TinyBlock(nn.Module):
    def __init__(self, d: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.fc1 = nn.Linear(d, d, dtype=torch.float64)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d, d, dtype=torch.float64)
        with torch.no_grad():
            for m in (self.fc1, self.fc2):
                m.weight.copy_(torch.randn(d, d, generator=gen, dtype=torch.float64) * 0.4)
                m.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _calib(d: int, seed: int = 1) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(2, 8, d, generator=gen, dtype=torch.float64) for _ in range(2)]


def _stats_book(model, batches):
    """A Phase-1 book for a bare ``Sequential`` (no trunk/head to take a loss
    from), so the block loop consumes it instead of collecting its own."""
    with torch.no_grad():
        probe = model(batches[0])
    cot = torch.randn(probe.shape, generator=torch.Generator().manual_seed(9), dtype=probe.dtype)
    return collect_stats(model, batches, cotangent=cot)


class _TinyTrunk:
    def __init__(self, embed: nn.Module, lin1: nn.Module, act: nn.Module, lin2: nn.Module) -> None:
        self._parts = (embed, lin1, act, lin2)

    def __call__(self, input_ids=None, tokens=None) -> TrunkOutput:  # noqa: ANN001
        embed, lin1, act, lin2 = self._parts
        x = embed(input_ids if input_ids is not None else tokens)
        return TrunkOutput(x + lin2(act(lin1(x))))


class _TinyWholeModel(nn.Module):
    """Whole-model shape ``collect_stats`` detects: a reachable trunk AND head."""

    def __init__(self, d: int = 8, vocab: int = 10) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(0)
        self.embed = nn.Embedding(vocab, d, dtype=torch.float64)
        self.lin1 = nn.Linear(d, d, dtype=torch.float64)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(d, d, dtype=torch.float64)
        self.head = nn.Linear(d, vocab, dtype=torch.float64)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(vocab, d, generator=gen, dtype=torch.float64) * 0.5)
            self.head.weight.copy_(torch.randn(vocab, d, generator=gen, dtype=torch.float64) * 0.1)
            self.head.bias.zero_()
            for m in (self.lin1, self.lin2):
                g = torch.Generator().manual_seed(7)
                a = torch.randn(d, 4, generator=g, dtype=torch.float64)
                b = torch.randn(4, d, generator=g, dtype=torch.float64)
                m.weight.copy_(a @ b * 0.08)
                m.bias.zero_()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        return self.head(x + self.lin2(self.act(self.lin1(x))))

    @property
    def trunk(self) -> _TinyTrunk:
        return _TinyTrunk(self.embed, self.lin1, self.act, self.lin2)


def _tokens(vocab: int = 10, seed: int = 5) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab, (2, 8), generator=gen) for _ in range(2)]


def _ptq_lines(out: str) -> list[re.Match[str]]:
    return [m for m in (LINE.match(line) for line in out.splitlines()) if m]


def test_block_progress_lines_are_one_per_block_with_wall_and_a_completion_marker(capsys):
    d = 8
    model = nn.Sequential(_TinyBlock(d, 0), _TinyBlock(d, 1))
    batches = _calib(d)
    reconstruct_model(
        model,
        batches,
        # NumPy arm: the port's config defaults to bf16, which this backend
        # refuses by name — its fixtures declare the float64 oracle grade.
        PTQConfig(solver_backend="numpy", rank=4, compute_dtype="float64"),
        stats_book=_stats_book(model, batches),
    )

    lines = _ptq_lines(capsys.readouterr().out)
    notes = [m.group("note") for m in lines]
    assert notes == [
        "block 0 (stem)",
        "block 1 0",
        "block 2 1",
        "block recon complete: 2 blocks",
    ]
    # Every line carries a measured wall, and the block count is monotone: the
    # reader of a cancelled log gets "how far" AND "how long" from the tail.
    walls = [float(m.group("wall")) for m in lines]
    assert all(wall >= 0.0 for wall in walls)
    assert walls == sorted(walls)


def test_a_block_that_raises_leaves_its_own_line_and_no_completion_marker(monkeypatch, capsys):
    """The forensic contract: the LAST line names the block that wedged."""
    d = 8
    model = nn.Sequential(_TinyBlock(d, 0), _TinyBlock(d, 1))
    batches = _calib(d)
    book = _stats_book(model, batches)

    from quantibbler.ptq import block_recon

    real = block_recon.reconstruct_block
    calls = {"n": 0}

    def explode_on_second(block, calibration_batches, cfg, **kwargs):  # noqa: ANN001, ANN002
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("solve wedge (simulated)")
        return real(block, calibration_batches, cfg, **kwargs)

    monkeypatch.setattr(block_recon, "reconstruct_block", explode_on_second)
    with pytest.raises(RuntimeError, match="simulated"):
        reconstruct_model(
            model,
            batches,
            PTQConfig(solver_backend="numpy", rank=4, compute_dtype="float64"),
            stats_book=book,
        )

    notes = [m.group("note") for m in _ptq_lines(capsys.readouterr().out)]
    assert notes == ["block 0 (stem)", "block 1 0", "block 2 1"]


def test_ptq_phase_lines_precede_the_first_block_line(capsys):
    model = _TinyWholeModel()
    batches = _tokens()
    quantize_model(
        model, batches, PTQConfig(solver_backend="numpy", rank=4, compute_dtype="float64")
    )

    notes = [m.group("note") for m in _ptq_lines(capsys.readouterr().out)]
    assert notes[:3] == [
        "phase 1 statistics: 2 batches",
        "phase 1 statistics done",
        "teacher snapshot done",
    ]
    assert notes[3].startswith("block 0 (stem)")
    assert notes[-1].startswith("block recon complete:")


def test_cgroup_figure_is_stated_or_absent_never_invented(monkeypatch, capsys):
    from quantibbler.ptq import _probe

    monkeypatch.setattr(_probe, "_cgroup_gb", lambda: None)
    _probe._note_ptq_progress("block 7 layers.6", 12.5)
    out = capsys.readouterr().out
    assert out == "[ptq] block 7 layers.6 wall=12.5s\n"

    monkeypatch.setattr(_probe, "_cgroup_gb", lambda: 3.5)
    _probe._note_ptq_progress("block 7 layers.6", 12.5)
    assert capsys.readouterr().out == "[ptq] block 7 layers.6 wall=12.5s cgroup=3.50G\n"
