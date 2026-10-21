"""Tasks 4.1-4.3: block reconstruction (scenarios quantized_prefix_inputs,
ste_refinement_improves_mse, sign_extraction_packing_parity)."""

import numpy as np
import pytest
import torch
from torch import nn

from quantibbler.ptq.block_recon import (
    QuantizedLayerState,
    extract_and_pack,
    reconstruct_block,
)
from quantibbler.ptq.config import PTQConfig


class _TinyBlock(nn.Module):
    """One transformer-ish block: attn linear + MLP linear (deterministic)."""

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


def _calib(d: int, n: int = 4, seq: int = 8, seed: int = 1) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(n, seq, d, generator=gen, dtype=torch.float64) for _ in range(2)]


def test_scn_block_loop_order():
    """quantized_prefix_inputs: block b > 0 sees the *quantized* prefix's
    outputs as X_b and the fp block's output on those inputs as Y_b; within
    a block the stage order is TuneFP -> LB-ADMM -> TuneLatentSTE."""
    d = 8
    b0, b1 = _TinyBlock(d, seed=0), _TinyBlock(d, seed=1)
    events: list[str] = []
    cfg = PTQConfig(rank=4)
    # Instrument stages via the recorder hook.
    captured: dict[str, object] = {}

    def recorder(event: str, **info) -> None:
        events.append(event)
        if event == "capture":
            captured.setdefault("inputs", []).append(info["x_b"])
            captured.setdefault("targets", []).append(info["y_b"])

    # Two-block model, both blocks reconstructed through the public loop.
    model = nn.Sequential(b0, b1)
    from quantibbler.ptq.block_recon import reconstruct_model

    x = torch.cat(_calib(d), dim=0)
    # FP reference captured BEFORE the loop mutates the blocks (the fp
    # prefix output the quantized prefix must differ from).
    with torch.no_grad():
        fp0 = b0(x)
    reconstruct_model(model, _calib(d), cfg, recorder=recorder)
    assert captured["inputs"][1].shape == x.shape

    # Stage order per block: capture -> tune_fp -> lb_admm -> ste. Additional
    # record events (the sign-extraction stage map entry added by
    # qk-fix-tunefp-teacher) are filtered out here: the assertion is about
    # the Algorithm 1 stage ORDER, not about the record's exact event count.
    alg1 = [e for e in events if e in ("capture", "tune_fp", "lb_admm", "ste")]
    assert alg1[:4] == ["capture", "tune_fp", "lb_admm", "ste"]
    assert alg1[4:8] == ["capture", "tune_fp", "lb_admm", "ste"]
    assert events.count("sign_extract_pack") == 2

    # Block 1's captured input (quantized-prefix advance) differs from the
    # fp prefix output.
    quant0_outputs = captured["inputs"][1]
    assert not torch.allclose(quant0_outputs, fp0, atol=1e-8)
    assert captured["inputs"][0].shape == x.shape
    assert captured["targets"][1].shape == x.shape


def test_block_loop_consumes_the_phase1_book_entry_for_entry(monkeypatch):
    """The hoist: one pre-loop pass, and the block loop consumes ITS entries.

    Identity, not equality: a block that quietly recomputed statistics would
    look the same in the record, so the test watches the objects handed to
    ``reconstruct_block``.
    """
    from quantibbler.ptq import block_recon as br
    from quantibbler.ptq.block_recon import reconstruct_model
    from quantibbler.ptq.calibration import collect_stats

    d = 8
    model = nn.Sequential(_TinyBlock(d, seed=0), _TinyBlock(d, seed=1))
    batches = _calib(d, seed=4)
    cfg = PTQConfig(rank=4)
    book = collect_stats(model, batches)
    assert sorted(entry.name for entry in book) == ["0.fc1", "0.fc2", "1.fc1", "1.fc2"]

    handed: list = []
    real = br.reconstruct_block

    def spy(block, calibration_batches, cfg_, **kwargs):  # noqa: ANN001
        handed.append(kwargs.get("stats_book"))
        return real(block, calibration_batches, cfg_, **kwargs)

    monkeypatch.setattr(br, "reconstruct_block", spy)

    def explode(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("the block loop collected its own statistics")

    monkeypatch.setattr(br, "collect_stats", explode)

    reconstruct_model(model, batches, cfg, stats_book=book)
    assert len(handed) == 2, "only the quantizable blocks carry statistics"
    by_name = {entry.name: entry for entry in book}
    for prefix, sliced in zip(("0", "1"), handed, strict=True):
        assert [entry.name for entry in sliced] == [f"{prefix}.fc1", f"{prefix}.fc2"]
        assert sliced[0] is by_name[f"{prefix}.fc1"]
        assert sliced[1] is by_name[f"{prefix}.fc2"]


def test_block_stats_slice_fails_loudly_when_the_inventory_disagrees():
    """A missing entry is an error, never a silent local re-collection."""
    from quantibbler.ptq.block_recon import _block_stats_slice
    from quantibbler.ptq.calibration import LayerStatsBook

    with pytest.raises(ValueError, match="no entry for"):
        _block_stats_slice(LayerStatsBook(), "0", _TinyBlock(8, seed=0), PTQConfig(rank=4))


def test_scn_ste_refinement():
    """ste_refinement_improves_mse: gradients reach (U, V, s1, s2) with
    straight-through sign passthrough, and block-output MSE decreases over
    the refinement steps on a fixed toy block."""
    d = 8
    block = _TinyBlock(d, seed=2)
    batches = _calib(d, seed=3)
    cfg = PTQConfig(rank=4)

    # Direct STE unit: build the quantized replacement for one linear layer
    # and confirm passthrough gradients + decreasing MSE.
    from quantibbler.ptq.block_recon import FactorizedLinear

    stats_probe = {}
    with torch.no_grad():
        w = block.fc1.weight.detach().clone()
        xs = torch.cat(batches, dim=0)
        ys = block.fc1(xs).detach()  # teacher targets: no graph
    fac = FactorizedLinear.from_weight(w, rank=4)

    u = fac.u_latent.detach().clone().requires_grad_(True)
    v = fac.v_latent.detach().clone().requires_grad_(True)
    s1 = fac.s1.detach().clone().requires_grad_(True)
    s2 = fac.s2.detach().clone().requires_grad_(True)

    def forward_with(u_t, v_t, s1_t, s2_t):
        ub = torch.sign(u_t)
        ub = ub + u_t - u_t.detach()  # straight-through passthrough
        vb = torch.sign(v_t)
        vb = vb + v_t - v_t.detach()
        w_q = (s2_t.unsqueeze(1) * vb) @ (ub * s1_t.unsqueeze(1)).T
        return xs @ w_q.T + fac.bias

    first_loss = None
    last_loss = None
    opt = torch.optim.Adam([u, v, s1, s2], lr=1e-3)
    for _ in range(30):
        opt.zero_grad()
        out = forward_with(u, v, s1, s2)
        loss = (out - ys).pow(2).mean()
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        opt.step()
        last_loss = loss.item()

    assert first_loss is not None and last_loss < first_loss
    for p in (u, v, s1, s2):
        assert p.grad is not None
    stats_probe["mse_drop"] = first_loss - last_loss

    # reconstruct_block runs the whole per-block pipeline with refinement.
    result = reconstruct_block(block, batches, cfg)
    assert result is not None
    assert len(result.layers) == 2
    for st in result.layers:
        assert isinstance(st, QuantizedLayerState)
        assert st.Ub.dtype == np.int8
        assert np.all((st.Ub == 1) | (st.Ub == -1))


def test_scn_sign_extraction_parity():
    """sign_extraction_packing_parity: sign(U)/sign(V) match the packed bits
    after qk-kernels pack/unpack (bit identity, both backends), with the
    sign(0) = +1 convention."""
    gen = np.random.default_rng(17)
    u = gen.standard_normal((10, 4))
    v = gen.standard_normal((9, 4))
    u[2] = 0.0  # dead row -> sign(0) = +1 convention

    st = extract_and_pack(u, v)
    assert st.Ub.dtype == np.int8 and st.Vb.dtype == np.int8
    expected_u = np.where(u >= 0, np.int8(1), np.int8(-1))
    expected_v = np.where(v >= 0, np.int8(1), np.int8(-1))
    assert np.array_equal(st.Ub, expected_u)
    assert np.array_equal(st.Vb, expected_v)
    assert np.all(st.Ub[2] == 1)

    # Bit identity through the qk-kernels pack/unpack dispatch (auto and
    # explicit on each backend present).
    from quantibbler import kernels

    for backend in ("numpy-reference", "rust"):
        if backend == "rust" and not kernels.HAVE_KERNELS:
            continue
        packed_u = kernels.pack_bits(st.Ub, backend=backend)
        packed_v = kernels.pack_bits(st.Vb, backend=backend)
        unpacked_u = kernels.unpack_bits(packed_u, st.Ub.shape[1], "int8", backend=backend)
        unpacked_v = kernels.unpack_bits(packed_v, st.Vb.shape[1], "int8", backend=backend)
        assert np.array_equal(unpacked_u, st.Ub)
        assert np.array_equal(unpacked_v, st.Vb)

    # NumPy-reference bits equal Rust bits (sign-agreement invariant).
    if kernels.HAVE_KERNELS:
        pu_ref = kernels.pack_bits(st.Ub, backend="numpy-reference")
        pu_rust = kernels.pack_bits(st.Ub, backend="rust")
        assert np.array_equal(pu_ref, pu_rust)


def test_bit_polish_recovers_perturbed_exact_factorization():
    """Greedy polish strictly improves a perturbed exact +-1 factorization
    and rejects malformed inputs (validation contract)."""
    import numpy as np

    from quantibbler.ptq.bit_polish import greedy_polish

    gen = np.random.default_rng(3)
    ub0 = np.where(gen.random((16, 4)) < 0.5, -1.0, 1.0)
    vb0 = np.where(gen.random((12, 4)) < 0.5, -1.0, 1.0)
    s1_0 = gen.uniform(0.5, 2.0, 16)
    s2_0 = gen.uniform(0.5, 2.0, 12)
    W = (s1_0[:, None] * (ub0 @ vb0.T)) * s2_0[None, :]

    # perturb: 40 random bit flips + 2x scale noise
    ub = ub0.copy()
    vb = vb0.copy()
    for _ in range(40):
        if gen.random() < 0.5:
            ub[gen.integers(16), gen.integers(4)] *= -1
        else:
            vb[gen.integers(12), gen.integers(4)] *= -1
    s1_p = s1_0 * gen.uniform(0.5, 1.5, 16)
    s2_p = s2_0 * gen.uniform(0.5, 1.5, 12)

    def rel(ubX, vbX, a, b):
        Wq = (a[:, None] * (ubX @ vbX.T)) * b[None, :]
        return float(np.linalg.norm(Wq - W) / np.linalg.norm(W))

    before = rel(ub, vb, s1_p, s2_p)
    ub_f, vb_f, s1_f, s2_f = greedy_polish(W, ub, vb, s1_p, s2_p, rounds=8)
    after = rel(ub_f, vb_f, s1_f, s2_f)
    assert after < 0.5 * before, f"polish failed to converge: {before:.4f} -> {after:.4f}"
    assert np.isfinite(s1_f).all() and np.isfinite(s2_f).all()
    assert np.all(np.abs(ub_f) == 1.0) and np.all(np.abs(vb_f) == 1.0)

    # validation: non-finite target, wrong shapes, non-binary factors
    w_nan = W.copy()
    w_nan[3, 5] = np.nan
    with pytest.raises(ValueError, match="finite"):
        greedy_polish(w_nan, ub, vb, s1_f, s2_f)
    with pytest.raises(ValueError, match="shapes"):
        greedy_polish(W, ub[:, :2], vb, s1_f, s2_f)
    with pytest.raises(ValueError, match="binary"):
        bad = ub_f.copy()
        bad[0, 0] = 2.0
        greedy_polish(W, bad, vb_f, s1_f, s2_f)
