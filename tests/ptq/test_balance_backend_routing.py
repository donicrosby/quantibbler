"""The balance stage runs on the RESOLVED solver arm, never on the bare default.

`qk-solver-backend`, scenario ``admm_algorithm_unchanged``: "WHEN either backend
factorizes a matrix THEN it executes the same continuous-update, SVID-proxy,
dual-update, and magnitude-balance steps". ``reconstruct_block`` holds the only
production call site of ``magnitude_balance``, so THAT call site -- not the
callee's ``DEFAULT_SOLVER_BACKEND`` fallback -- is what has to resolve the arm.

Omitted, the balance executed on torch regardless of ``cfg.solver_backend`` and
of the ``SOLVER_BACKEND`` / ``QUANTIBBLER_DEVICE`` knobs. Measured consequences:
on an accelerator-less host every ``reconstruct_block``-exercising test died at
the implicit-torch refusal, and on an accelerated host a ``SOLVER_BACKEND=numpy``
parity run solved on numpy while its balance ran on torch, under a probe row
that named numpy.

The pin watches ``lb_admm.get_backend`` -- the resolver the solve AND the balance
both reach through (``magnitude_balance`` lives in ``lb_admm``) -- so it asserts
the arm each stage actually constructed, not an argument spelling.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from quantibbler.ptq import block_recon, lb_admm
from quantibbler.ptq.config import PTQConfig


class _TinyBlock(nn.Module):
    """One transformer-ish block: attn linear + MLP linear (deterministic)."""

    def __init__(self, d: int = 8, seed: int = 0) -> None:
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


def _batches(d: int = 8, n: int = 4, seq: int = 8, seed: int = 1) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(n, seq, d, generator=gen, dtype=torch.float64) for _ in range(2)]


def _fixed_cot(module: nn.Module, batches, *, seed: int = 9) -> torch.Tensor:
    """Caller-supplied output cotangent (a bare block carries no scalar loss)."""
    x = torch.cat(batches, dim=0)
    with torch.no_grad():
        probe = module(x)
    return torch.randn(
        probe.shape, generator=torch.Generator().manual_seed(seed), dtype=probe.dtype
    )


def _watch_get_backend(monkeypatch) -> list[tuple[str | None, str | None]]:
    """Record every ``(name, device)`` the solve AND the balance resolve."""
    seen: list[tuple[str | None, str | None]] = []
    real = lb_admm.get_backend

    def spy(name=None, *, device=None):  # noqa: ANN001, ANN202
        seen.append((name, device))
        return real(name, device=device)

    monkeypatch.setattr(lb_admm, "get_backend", spy)
    return seen


def _no_knobs(monkeypatch) -> None:
    monkeypatch.delenv("SOLVER_BACKEND", raising=False)
    monkeypatch.delenv("QUANTIBBLER_DEVICE", raising=False)


def test_scn_admm_algorithm_unchanged_balance_follows_the_config(monkeypatch):  # noqa: ANN001
    """``cfg.solver_backend`` is the arm for BOTH stages of the algorithm.

    Before the fix the second resolver call per layer arrived with ``name=None``
    -- the bare ``DEFAULT_SOLVER_BACKEND`` -- which on this accelerator-less host
    is the refusal itself.
    """
    _no_knobs(monkeypatch)
    block = _TinyBlock(seed=2)
    batches = _batches(seed=3)
    cfg = PTQConfig(solver_backend="numpy", rank=4)
    seen = _watch_get_backend(monkeypatch)

    result = block_recon.reconstruct_block(
        block, batches, cfg, cotangent=_fixed_cot(block, batches)
    )

    assert len(result.layers) == 2
    assert seen, "the production path resolved no backend at all"
    assert all(name == "numpy" and device is None for name, device in seen), seen
    # one solve + one balance per factorized layer
    assert len(seen) == 2 * len(result.layers)


def test_scn_admm_algorithm_unchanged_balance_honors_the_env_knob(monkeypatch):  # noqa: ANN001
    """``SOLVER_BACKEND=numpy`` reaches the balance, not only the solve.

    A hard pin on this host: the config asks for torch, the env knob overrides
    it, and a balance that resolved the bare default instead would request
    implicit torch and die at the refusal while the solve ran on numpy.
    """
    monkeypatch.setenv("SOLVER_BACKEND", "numpy")
    monkeypatch.delenv("QUANTIBBLER_DEVICE", raising=False)
    block = _TinyBlock(seed=4)
    batches = _batches(seed=5)
    cfg = PTQConfig(solver_backend="torch", rank=4)
    seen = _watch_get_backend(monkeypatch)

    result = block_recon.reconstruct_block(
        block, batches, cfg, cotangent=_fixed_cot(block, batches)
    )

    assert len(result.layers) == 2
    assert all(name == "numpy" for name, _ in seen), seen


def test_scn_admm_algorithm_unchanged_probe_row_names_the_balance_arm(tmp_path, monkeypatch):  # noqa: ANN001
    """The row a numpy-arm run writes must name the arm its BALANCE ran on.

    The env knob exists for run identity and attribution: a row that says numpy
    while the balance executed on torch is exactly the mislabelling the omitted
    resolution caused. Value AND source, asserted against the arm the resolver
    actually constructed.
    """
    monkeypatch.setenv("SOLVER_BACKEND", "numpy")
    monkeypatch.setenv("QK_SOLVER_PROBE", "1")
    monkeypatch.setenv("QK_SOLVER_PROBE_PATH", str(tmp_path / "probe.jsonl"))
    monkeypatch.delenv("QUANTIBBLER_DEVICE", raising=False)
    block = _TinyBlock(seed=6)
    batches = _batches(seed=7)
    cfg = PTQConfig(solver_backend="torch", rank=4)
    seen = _watch_get_backend(monkeypatch)

    block_recon.reconstruct_block(block, batches, cfg, cotangent=_fixed_cot(block, batches))

    rows = [
        json.loads(line)
        for line in (tmp_path / "probe.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    assert len(rows) == 2, "one probe row per factorized layer"
    assert {row["solver_backend"] for row in rows} == {"numpy"}
    assert {row["solver_backend_source"] for row in rows} == {"env"}
    assert all(name == "numpy" for name, _ in seen), seen


@pytest.mark.skipif(torch.cuda.is_available(), reason="pins the accelerator-less refusal")
def test_scn_admm_algorithm_unchanged_balance_refuses_like_the_solve(monkeypatch):  # noqa: ANN001
    """Implicit torch without an accelerator fails closed at BOTH stages, with
    the same message: the balance is not a place a run can silently fall back to
    CPU, and the production path dies on the refusal rather than on a balance
    that would have executed somewhere other than its solve."""
    _no_knobs(monkeypatch)
    cfg = PTQConfig(solver_backend="torch", rank=4)

    with pytest.raises(RuntimeError, match="requires an accelerator") as solve_exc:
        lb_admm.lb_admm(np.eye(6, 4), cfg)

    with pytest.raises(RuntimeError, match="requires an accelerator") as bal_exc:
        lb_admm.magnitude_balance(
            np.ones((6, 4)), np.ones((4, 4)), np.ones(6), np.ones(4), backend="torch"
        )
    assert str(bal_exc.value) == str(solve_exc.value)

    block = _TinyBlock(seed=8)
    batches = _batches(seed=9)
    with pytest.raises(RuntimeError, match="requires an accelerator"):
        block_recon.reconstruct_block(block, batches, cfg, cotangent=_fixed_cot(block, batches))
