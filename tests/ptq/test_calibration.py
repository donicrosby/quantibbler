"""Task 2.1: RobustDiag preconditioner properties (scenario
preconditioner_shape_robustness). Task 2.2: statistics coverage (scenario
stats_cover_all_layers)."""

import numpy as np
import pytest
import torch
from torch import nn

from quantibbler.ptq.calibration import collect_stats, robust_diag


class _SingleLinear(nn.Module):
    """One linear layer exposed as a module (minimal calibration target)."""

    def __init__(self, dout: int, din: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.lin = nn.Linear(din, dout, dtype=torch.float64)
        with torch.no_grad():
            self.lin.weight.copy_(torch.randn(dout, din, generator=gen, dtype=torch.float64))
            self.lin.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def _synth_batches(n: int, seq: int, din: int, seed: int) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(n, seq, din, generator=gen, dtype=torch.float64) for _ in range(2)]


def test_scn_preconditioner_properties():
    """preconditioner_shape_robustness: shapes, clipping-then-shrinkage
    ordering (A-1, Eq. 3), and zero-variance robustness ([NQ] Step 2-1)."""
    dout, din = 6, 10
    layer = _SingleLinear(dout, din, seed=3)
    batches = _synth_batches(4, 5, din, seed=7)
    stats = collect_stats(layer, batches)

    tau = 1.5
    entry = stats[0]
    d_in, d_out = robust_diag(entry.d_in_raw, entry.d_out_raw, tau=tau, gamma=0.2)
    d_in2, d_out2 = robust_diag(entry.d_in_raw, entry.d_out_raw, tau=tau, gamma=0.2)

    # Deterministic, right shapes, finite, nonnegative, clipped to tau.
    assert d_in.shape == (din,) and d_out.shape == (dout,)
    assert d_in.dtype == np.float64 and d_out.dtype == np.float64
    assert np.all(np.isfinite(d_in)) and np.all(np.isfinite(d_out))
    assert np.all(d_in >= 0.0) and np.all(d_out >= 0.0)
    assert np.all(d_in <= tau) and np.all(d_out <= tau)
    assert np.array_equal(d_in, d_in2) and np.array_equal(d_out, d_out2)

    # Clip happens BEFORE shrinkage: shrinking an already-clipped vector
    # toward its mean keeps everything in [0, tau]; clipping the raw vector
    # first must reproduce the shrinkage form of Eq. 3 exactly.
    d_in_ref, _ = robust_diag(entry.d_in_raw, entry.d_out_raw, tau=tau, gamma=0.0)
    clipped = np.minimum(entry.d_in_raw, tau)
    expected = (1 - 0.2) * clipped + 0.2 * clipped.mean()
    assert np.allclose(d_in_ref, clipped)
    assert np.allclose(d_in, expected)

    # gamma = 0 with tau = inf returns the raw statistics untouched.
    d_raw = robust_diag(entry.d_in_raw, tau=np.inf, gamma=0.0)
    assert np.allclose(d_raw, entry.d_in_raw)

    # Zero-variance (constant) activations yield finite, positive entries.
    flat = [torch.full((2, 4, din), 0.25, dtype=torch.float64)]
    flat_stats = collect_stats(layer, flat)
    flat_entry = flat_stats[0]
    d_in_z, d_out_z = robust_diag(flat_entry.d_in_raw, flat_entry.d_out_raw, tau=tau, gamma=0.2)
    assert np.all(np.isfinite(d_in_z)) and np.all(np.isfinite(d_out_z))
    assert np.all(d_in_z > 0.0) and np.all(d_out_z > 0.0)


def test_scn_preconditioner_zero_stats():
    """All-zero statistics still produce finite positive preconditioners
    (the [0, tau] range of the spec's scenario, dead-channel robustness)."""
    zeros = np.zeros(7)
    d = robust_diag(zeros, tau=1.0, gamma=0.3)
    assert d.shape == (7,)
    assert np.all(np.isfinite(d)) and np.all(d > 0.0)


def test_scn_stats_all_layers():
    """stats_cover_all_layers: one forward+backward pass over calibration
    batches records statistics for every linear layer of a 2-block model,
    keyed in module declaration order."""
    d = 8
    gen = torch.Generator().manual_seed(11)
    w0 = nn.Linear(d, d, dtype=torch.float64)
    w1 = nn.Linear(d, d, dtype=torch.float64)
    b0 = nn.Sequential(w0, nn.GELU(), nn.Linear(d, d, dtype=torch.float64))
    b1 = nn.Sequential(w1, nn.GELU(), nn.Linear(d, d, dtype=torch.float64))
    model = nn.Sequential(b0, b1)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.weight.copy_(torch.randn(d, d, generator=gen, dtype=torch.float64) * 0.3)
                m.bias.zero_()

    batches = [torch.randn(2, 4, d, generator=gen, dtype=torch.float64) for _ in range(3)]
    book = collect_stats(model, batches)

    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert book.num_layers == len(linears) == 4
    assert book.layer_names == [
        name for name, _ in model.named_modules() if isinstance(_, nn.Linear)
    ]
    stats_list = list(book)
    for entry, _module in zip(stats_list, linears, strict=True):
        assert entry.d_in_raw.shape == (d,)
        assert entry.d_out_raw.shape == (d,)
        assert np.all(np.isfinite(entry.d_in_raw))
        assert np.all(np.isfinite(entry.d_out_raw))
    # Ordering assertion is exact: declaration order, no reordering.
    assert [e.name for e in stats_list] == book.layer_names


def test_stats_rejects_non_linear_branches():
    """Bookkeeping keys only nn.Linear modules; conv/norm layers are absent."""
    model = nn.Sequential(
        nn.Linear(4, 4, dtype=torch.float64), nn.LayerNorm(4, dtype=torch.float64)
    )
    batches = [torch.randn(1, 3, 4, dtype=torch.float64)]
    book = collect_stats(model, batches)
    assert book.num_layers == 1
    assert "0" in book.layer_names[0]


def test_robust_diag_validates_shape():
    """robust_diag validates the dimensionality of the raw statistics."""
    with pytest.raises(ValueError):
        robust_diag(np.ones((3, 3)), tau=1.0, gamma=0.2)
