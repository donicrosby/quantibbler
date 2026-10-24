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


def _ct(module, batch, head=lambda z: (z**2).sum()):
    """head -> loss -> backward -> cotangent, the caller's job under the contract."""
    from quantibbler.ptq.calibration import require_grad_or_name

    out = module(batch).detach().requires_grad_(True)
    loss = head(out)
    require_grad_or_name(loss)
    loss.backward()
    return out.grad


def test_scn_preconditioner_properties():
    """preconditioner_shape_robustness: shapes, clipping-then-shrinkage
    ordering (A-1, Eq. 3), and zero-variance robustness ([NQ] Step 2-1)."""
    dout, din = 6, 10
    layer = _SingleLinear(dout, din, seed=3)
    batches = _synth_batches(4, 5, din, seed=7)
    stats = collect_stats(layer, batches, cotangent=_ct(layer, batches[0]))

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
    flat_stats = collect_stats(layer, flat, cotangent=_ct(layer, flat[0]))
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
    book = collect_stats(model, batches, cotangent=_ct(model, batches[0]))

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
    book = collect_stats(model, batches, cotangent=_ct(model, batches[0]))
    assert book.num_layers == 1
    assert "0" in book.layer_names[0]


def test_robust_diag_validates_shape():
    """robust_diag validates the dimensionality of the raw statistics."""
    with pytest.raises(ValueError):
        robust_diag(np.ones((3, 3)), tau=1.0, gamma=0.2)


def test_scn_gradient_on_output_side() -> None:
    """scenario gradient_statistic_on_output_side.

    RED-FIRST against today's construction, which puts a uniform-cotangent
    gradient term on the INPUT side and nothing on the output side. Lane 1
    inverts that: d_out = E[g^2] from the caller's real loss gradient,
    d_in = E[x^2] only.

    The head is caller-supplied here because the layer is bare -- the alignment
    spec blesses exactly this form ("or documented caller-supplied
    head/cotangent for bare-block use"), and whole-model collection routes a CE
    backward through the model head instead.
    """
    lin = _SingleLinear(dout=4, din=3, seed=0)
    x = torch.randn(2, 5, 3, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    # Equal-norm tokens make the per-token clip a provable no-op, so this fixture
    # tests the gradient question and nothing else. It is not cosmetic: with 10
    # tokens the 99.9th-percentile order statistic clips one of them, which
    # silently moved d_in to 0.86x the analytic value and made a correct
    # implementation look wrong. The clip has its own test (task 3).
    target = torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(2), dtype=torch.float64)

    def head(z: torch.Tensor) -> torch.Tensor:
        # 0.5 * ||z - t||^2, so dLoss/dz == (z - t) exactly and
        # E[g^2] == mean((z - t)^2) with no chain-rule factor to lose.
        return 0.5 * (z - target).pow(2).sum()

    x = x / x.norm(dim=-1, keepdim=True)

    book = collect_stats(lin, [x], cotangent=_ct(lin, x, head))
    entry = next(iter(book))

    z = lin(x)
    expected_d_out = (z - target).pow(2).mean(dim=(0, 1)).detach().numpy()
    expected_d_in = x.pow(2).mean(dim=(0, 1)).detach().numpy()

    assert entry.d_out_raw.shape == (4,), entry.d_out_raw.shape
    assert np.allclose(entry.d_out_raw, expected_d_out, rtol=1e-9), (
        entry.d_out_raw,
        expected_d_out,
    )
    assert np.allclose(entry.d_in_raw, expected_d_in, rtol=1e-9), (
        entry.d_in_raw,
        expected_d_in,
    )


def test_scn_per_token_clip_before_squaring() -> None:
    """scenarios per_token_clip_before_squaring, clip_quantile_is_robust_max.

    RED-FIRST: no clip exists today, so the extreme token contributes its full
    squared norm and the statistic is dominated by a single row.

    Clip is on the per-token L2 NORM, applied BEFORE squaring, at the robust
    99.9th percentile (`_TOKEN_CLIP_QUANTILE = 0.999`). With 999 unit-norm
    tokens and one at norm 1000, p99.9 is 1.0, so the extreme token contributes
    exactly cap^2 = 1.0 rather than 1e6.
    """
    from quantibbler.ptq.calibration import _TOKEN_CLIP_QUANTILE

    assert _TOKEN_CLIP_QUANTILE == 0.999, _TOKEN_CLIP_QUANTILE

    din = 2
    x = torch.ones(1000, din, dtype=torch.float64)
    x = x / x.norm(dim=1, keepdim=True)  # 999 unit-norm rows... plus the last
    x[-1] = 1000.0  # ...and one extreme token

    lin = _SingleLinear(dout=3, din=din, seed=3)
    with torch.no_grad():
        lin.lin.weight.fill_(1.0)
    book = collect_stats(lin, [x], cotangent=_ct(lin, x))
    entry = next(iter(book))

    # Analytic: 999 rows contribute 1.0/din per feature; the clipped extreme row
    # contributes cap^2/din = 1.0/din. Unclipped it would be 1e6/din.
    expected = np.full(din, (999 * 1.0 + 1.0) / 1000.0 / din)
    assert np.allclose(entry.d_in_raw, expected, rtol=1e-6), (entry.d_in_raw, expected)
    assert entry.d_in_raw.max() < 10.0, "extreme token was not clipped"


def test_scn_dead_graph_at_the_loss_gets_the_named_failure() -> None:
    """A loss with no graph fails with the reason, not with autograd's raw error.

    Both places a caller can build such a loss: directly (require_grad_or_name),
    and through the whole-model path, where collect_stats runs the backward and a
    frozen model leaves the loss without a grad_fn. A frozen BARE-BLOCK caller is
    deliberately not this test's subject -- their cotangent comes from a detached
    leaf, which is a live graph regardless of the parameters' state, so the guard
    correctly does not fire for them.
    """
    from quantibbler.ptq.calibration import require_grad_or_name

    dead = torch.zeros((), dtype=torch.float64)
    with pytest.raises(ValueError, match="does not require grad"):
        require_grad_or_name(dead)

    live = torch.zeros((), dtype=torch.float64, requires_grad=True)
    require_grad_or_name(live)  # a live graph passes through unchanged
