"""Tasks 6.1-6.3: public API entry point, qpk1 round trip, float32 opt-in
(scenarios caller_supplied_model_and_data, qpk1_round_trip_byte_exact,
rust_matches_reference_probes, float32_opt_in)."""

import ast
from pathlib import Path

import numpy as np
import torch
from torch import nn

from quantibbler.ptq.api import QuantizedModel, quantize_model
from quantibbler.ptq.config import PTQConfig


class _TinyApiModel(nn.Module):
    """Caller-constructed tiny model: embedding + 1 block + fp head (A-9)."""

    def __init__(self, d: int = 8, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(32, d, dtype=torch.float64)
        self.lin1 = nn.Linear(d, d, dtype=torch.float64)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(d, d, dtype=torch.float64)
        self.head = nn.Linear(d, 10, dtype=torch.float64)
        with torch.no_grad():
            g = torch.Generator().manual_seed(seed + 1)
            a = torch.randn(d, 4, generator=g, dtype=torch.float64)
            b = torch.randn(4, d, generator=g, dtype=torch.float64)
            self.lin1.weight.copy_(a @ b * 0.08)
            a2 = torch.randn(d, 4, generator=g, dtype=torch.float64)
            b2 = torch.randn(4, d, generator=g, dtype=torch.float64)
            self.lin2.weight.copy_(a2 @ b2 * 0.08)
            for m in (self.lin1, self.lin2, self.head):
                m.bias.zero_()
            self.embed.weight.copy_(torch.randn(32, d, generator=gen, dtype=torch.float64) * 0.5)
            self.head.weight.copy_(torch.randn(10, d, generator=gen, dtype=torch.float64) * 0.1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = x + self.lin2(self.act(self.lin1(x)))
        return self.head(x)


def _calib(seed: int = 5, vocab: int = 32) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab, (2, 8), generator=gen) for _ in range(2)]


def _gate_cfg() -> PTQConfig:
    """Test-speed stage overrides per A-10 (paper defaults stay in config)."""
    from quantibbler.ptq.config import StageHyper

    return PTQConfig(
        rank=4,
        stage_pre=StageHyper(1e-4, 4, 0),
        stage_latent=StageHyper(2e-3, 4, 3),
        stage_global=StageHyper(2e-2, 4, 4),
    )


def test_scn_entry_point_signature():
    """caller_supplied_model_and_data: caller-built module + iterable of
    batches -> QuantizedModel, no network code anywhere in quantibbler/ptq/."""
    model = _TinyApiModel(seed=0)
    cfg = _gate_cfg()
    qm = quantize_model(model, _calib(), cfg)
    assert isinstance(qm, QuantizedModel)
    assert {"lin1", "lin2"} <= set(qm.layers)
    # A-9 scope: embeddings and the head are never quantized.
    assert "head" not in qm.layers
    assert not any("embed" in name for name in qm.layers)

    # Import-site audit: no urllib/requests/huggingface/socket imports in
    # the quantibbler/ptq package (no-network rule).
    ptq_dir = Path(__file__).resolve().parents[2] / "src" / "quantibbler" / "ptq"
    banned = {"urllib", "requests", "huggingface", "socket", "http", "ftplib"}
    for path in ptq_dir.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in banned, f"{path.name} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                root = (node.module or "").split(".")[0]
                assert root not in banned, f"{path.name} imports from {node.module}"


def test_scn_qpk1_round_trip():
    """qpk1_round_trip_byte_exact + rust_matches_reference_probes: serialize
    -> reload is byte-exact on bits, bit-exact on float64 scales, and the
    reload-then-forward equals the serialize-time forward."""
    model = _TinyApiModel(seed=3)
    cfg = _gate_cfg()
    qm = quantize_model(model, _calib(seed=7), cfg)
    blobs = qm.serialize()
    probe = _calib(seed=9)[0]

    with torch.no_grad():
        before = model(probe)

    reloaded = QuantizedModel.deserialize(blobs, config=cfg)
    assert set(reloaded.layers) == set(qm.layers)

    # Byte-exact bits: reload the same streams again and compare arrays.
    for name, blob in blobs.items():
        again = QuantizedModel.deserialize({name: blob}, config=cfg)
        st0 = reloaded.layers[name]
        st1 = again.layers[name]
        assert np.array_equal(st0.Ub, st1.Ub)
        assert np.array_equal(st0.Vb, st1.Vb)
        # float64 scales preserved bit-for-bit through the round trip.
        assert st0.s1.tobytes() == st1.s1.tobytes()
        assert st0.s2.tobytes() == st1.s2.tobytes()

    # Serialize(reloaded) == original streams (byte-exact round trip, D4).
    blobs2 = reloaded.serialize()
    assert blobs2 == blobs

    # Reload-then-forward equals serialize-time forward. The reloaded
    # artifact writes its bits+scales back into a fresh host model.
    host2 = _TinyApiModel(seed=3)
    _apply_to_host(host2, reloaded)
    with torch.no_grad():
        after = host2(probe)
    assert torch.allclose(before, after, atol=1e-10)

    # Rust path matches the NumPy reference on the packed bits (sign
    # agreement) at 1e-12-tolerance float64 parity per config.yaml.
    from quantibbler import kernels

    st = next(iter(reloaded.layers.values()))
    pu_ref = kernels.pack_bits(st.Ub, backend="numpy-reference")
    if kernels.HAVE_KERNELS:
        pu_rust = kernels.pack_bits(st.Ub, backend="rust")
        assert np.array_equal(pu_ref, pu_rust)
        ur = kernels.unpack_bits(pu_rust, st.Ub.shape[1], "float64", backend="rust")
        un = kernels.unpack_bits(pu_ref, st.Ub.shape[1], "float64", backend="numpy-reference")
        assert np.allclose(ur, un, rtol=0, atol=0)  # bit identity
        assert ur.dtype == np.float64


def _apply_to_host(host: nn.Module, qm: QuantizedModel) -> None:
    """Write reloaded bits+scales back into a fresh host model's linears.

    The stored weight is the quantized map itself:
    ``W_q = (s1 ⊙ Ub) @ (s2 ⊙ Vb)ᵀ`` with shape (dout, din), matching
    nn.Linear's ``x @ W.T`` application (same convention as
    block_recon._set_linear and api.attach_to).
    """
    for name, module in host.named_modules():
        if name in qm.layers and isinstance(module, nn.Linear):
            st = qm.layers[name]
            w = (st.s1[:, None] * st.Ub.astype(np.float64)) @ (
                st.s2[:, None] * st.Vb.astype(np.float64)
            ).T
            with torch.no_grad():
                module.weight.copy_(torch.from_numpy(np.ascontiguousarray(w)))
                if module.bias is not None and st.bias is not None:
                    module.bias.copy_(torch.from_numpy(np.ascontiguousarray(st.bias)))


def test_scn_phase1_statistics_are_collected_once_over_the_pristine_model(monkeypatch):
    """`phase1_stats_from_pristine_model`: ONE pre-loop whole-model pass.

    The block loop consumes that pre-collected book. The per-block capture
    used to be the statistics source, and it read the compressed-prefix
    stream — line 9's TUNING input — instead of the paper's raw calibration
    set. `collect_stats` is booby-trapped here: any per-block call fails.
    """
    from quantibbler.ptq import block_recon as br

    model = _TinyApiModel(seed=11)
    batches = _calib(seed=12)
    record: list = []

    def recorder(event: str, **info) -> None:
        record.append((event, info))

    def explode(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError(
            "a per-block statistics pass ran: the block loop must consume the "
            "Phase-1 book, never re-collect from the compressed-prefix stream"
        )

    monkeypatch.setattr(br, "collect_stats", explode)

    qm = quantize_model(model, batches, _gate_cfg(), recorder=recorder)
    assert {"lin1", "lin2"} <= set(qm.layers)

    events = [event for event, _ in record]
    assert events[0] == "phase1_statistics", events[:3]
    info = record[0][1]
    # Every nn.Linear of the pristine model is in the book — including the
    # fp head, which the whole-model pass sees but never quantizes.
    assert info["num_linears"] == len(info["linears"]) == 3
    # The float64 reference path stays whole-batch (bit-identical to the
    # landed single-pass behavior); the streamed chunking is for real models.
    assert info["chunk_rows"] is None
    # The capture (lines 9-10) still happens, per block, AFTER the statistics.
    assert [e for e in events if e == "capture"]
    assert events.index("phase1_statistics") < events.index("capture")


def test_phase1_chunking_is_bounded_for_real_models():
    """bfloat16-class models stream ONE row per chunk — measured, not chosen.

    The Phase-1 pass runs the whole model, so it retains its chunk's N^2
    attention graph and the chunk size is the memory knob. Measured inside
    `quantibbler-ptq:latest` at the wrapper's 32g memcg on a single 8 x 2048
    batch through Qwen3-0.6B: `chunk_rows=4` was OOM-KILLED; `chunk_rows=1`
    completed at an 18.47 GB cgroup peak. The float64 executable-spec path
    stays whole-batch (bit-identical to the landed single-pass behavior).
    """
    from quantibbler.ptq import api as api_mod

    assert api_mod._phase1_chunk_rows(_TinyApiModel(seed=1).to(torch.bfloat16)) == 1
    assert api_mod._phase1_chunk_rows(_TinyApiModel(seed=1)) is None
    assert api_mod._PHASE1_CHUNK_ROWS == 1, (
        "the Phase-1 chunk size is a measured memory bound, not a preference"
    )


def test_scn_export_gate_refuses_nonfinite_scales():
    """r137-v4 regression: v4 shipped 72/196 layers with NaN scales to disk
    while the container exited 0. The serialize path is the last chokepoint
    every export and e2e run flows through; it must refuse hard."""
    model = _TinyApiModel(seed=3)
    cfg = _gate_cfg()
    qm = quantize_model(model, _calib(seed=7), cfg)
    name = next(iter(qm.layers))
    st = qm.layers[name]
    st.s2 = st.s2.copy()
    st.s2[0] = float("nan")

    import pytest

    from quantibbler.qpk1 import Qpk1Error

    with pytest.raises(Qpk1Error, match="non-finite scales"):
        qm.serialize()

    # inf is caught too; a finite artifact still serializes fine.
    st.s2[0] = float("inf")
    with pytest.raises(Qpk1Error, match="non-finite scales"):
        qm.serialize()
    st.s2[0] = 1.0
    blobs = qm.serialize()
    assert name in blobs


def test_scn_float32_opt_in():
    """float32_opt_in: default config runs float64; compute_dtype="float32"
    flips the solver entry dtype — opt-in observable via config only."""
    assert PTQConfig().np_compute_dtype == np.dtype(np.float64)
    opt_in = PTQConfig(compute_dtype="float32")
    assert opt_in.np_compute_dtype == np.dtype(np.float32)

    # Solver entry honors the dtype: LB-ADMM on a float32 config produces
    # float32 factors.
    from quantibbler.ptq.lb_admm import lb_admm

    gen = np.random.default_rng(0)
    w = (gen.standard_normal((8, 6)) @ gen.standard_normal((6, 8))).astype(np.float32)
    res = lb_admm(w, PTQConfig(rank=3, compute_dtype="float32"))
    assert res.U.dtype == np.float32 and res.V.dtype == np.float32
    res64 = lb_admm(w.astype(np.float64), PTQConfig(rank=3))
    assert res64.U.dtype == np.float64
