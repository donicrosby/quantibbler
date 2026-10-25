"""qk-bias-carry: the container carries the per-linear bias, and the tuning
stage's bias is what the artifact ships.

Three claims, one test each:

* ``tuned_bias_survives_factorization`` — the bias a tuning pass wrote to a
  remainder linear is the bias the factorized artifact carries. The stage is
  stubbed to a sentinel write, so the assertion is deterministic and does not
  depend on how far an optimizer happened to move anything.
* ``tunefp_record_carries_bias_delta_rel`` — the TuneFP row reports the bias
  movement separately from the aggregate weight figure, and the reported
  number is the one an independent recomputation from a held snapshot gives.
* ``packed_artifact_bias_round_trips`` — what the artifact carries is what
  the container's writer/reader pair returns.
"""

import numpy as np
import torch
from torch import nn

from quantibbler import qpk1
from quantibbler.ptq import _block_stage
from quantibbler.ptq.api import QuantizedModel
from quantibbler.ptq.block_recon import _tune_fp, reconstruct_block
from quantibbler.ptq.config import PTQConfig

BIAS_INIT = 0.25
SENTINEL = 7.5


class _BiasBlock(nn.Module):
    """Two float64 linears with a non-zero bias on each (pristine values kept
    distinguishable from any tuned write)."""

    def __init__(self, d: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.fc1 = nn.Linear(d, d, dtype=torch.float64)
        self.fc2 = nn.Linear(d, d, dtype=torch.float64)
        with torch.no_grad():
            for m in (self.fc1, self.fc2):
                m.weight.copy_(torch.randn(d, d, generator=gen, dtype=torch.float64) * 0.4)
                m.bias.fill_(BIAS_INIT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _calib(d: int, n: int = 4, seq: int = 8, seed: int = 1) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(n, seq, d, generator=gen, dtype=torch.float64) for _ in range(2)]


def _fixed_cot(module, batches, *, seed: int = 9):
    x = torch.cat(batches, dim=0) if len(batches) > 1 else batches[0]
    with torch.no_grad():
        probe = module(x)
    return torch.randn(
        probe.shape, generator=torch.Generator().manual_seed(seed), dtype=probe.dtype
    )


def test_tuned_bias_survives_factorization(monkeypatch):
    """The defect: a tuned compensation class computed and then discarded at
    factorization. The stage here is stubbed, so the ONLY bias any artifact
    can carry is the one the stub wrote."""
    d = 8
    block = _BiasBlock(d, seed=0)
    cfg = PTQConfig(solver_backend="numpy", rank=4)

    def _stub(
        block,
        x_b,
        y_b,
        cfg,
        batch_size,
        *,
        label="",
        tune_names=None,
        factorized_in_block=0,
        recorder=None,
        capture_index=None,
        teacher_fingerprint=None,
        teacher_source="",
        loss_weight=None,
        loss_weight_source="uniform",
        loss_weight_mean_before=None,
        loss_weight_mean_after=None,
    ):
        tuned = sorted(tune_names or [])
        for name, mod in block.named_modules():
            if name in tuned and getattr(mod, "bias", None) is not None:
                with torch.no_grad():
                    mod.bias.fill_(SENTINEL)
        return {"stage": "tune_fp", "steps": 0, "tuned": tuned}

    # `reconstruct_block` resolves `_tune_fp` in `_block_stage`'s namespace, so
    # the spy must be installed there (a patch on the `block_recon` façade is
    # never consulted and the stub would be silently inert).
    monkeypatch.setattr(_block_stage, "_tune_fp", _stub)
    result = reconstruct_block(block, _calib(d), cfg, cotangent=_fixed_cot(block, _calib(d)))

    by_name = {st.name: st for st in result.layers}
    # Target 1 onward was tuned before it was factorized: the sentinel is what
    # the artifact ships. (Pre-fix every artifact carried BIAS_INIT here.)
    assert by_name["fc2"].bias is not None
    np.testing.assert_allclose(by_name["fc2"].bias, np.full(d, SENTINEL))
    # Target 1 has no preceding tuning pass, so it keeps its pristine bias.
    np.testing.assert_allclose(by_name["fc1"].bias, np.full(d, BIAS_INIT))

    # The host module's bias agrees with the artifact's: the forward a caller
    # runs after reconstruction IS the artifact's forward.
    assert torch.allclose(block.fc2.bias, torch.full((d,), SENTINEL, dtype=torch.float64))


def test_tunefp_record_carries_bias_delta_rel():
    """The recorded bias movement is the independently recomputed one."""
    d = 6
    block = _BiasBlock(d, seed=3)
    batches = _calib(d)
    x_b = torch.cat(batches, dim=0)
    with torch.no_grad():
        pristine = block(x_b)
    # A target the pristine block does not already hit, so the loss has a real
    # gradient in both the weights and the biases.
    y_b = pristine * 1.2 + 0.1

    before = [m.bias.detach().clone() for m in (block.fc1, block.fc2)]
    cfg = PTQConfig(solver_backend="numpy", rank=3, cosine_schedule=False)
    rec = _tune_fp(
        block,
        x_b,
        y_b,
        cfg,
        cfg.stage_pre.batch_size,
        label="bias",
        tune_names=["fc1", "fc2"],
        factorized_in_block=1,
    )

    assert rec["steps"] > 0, "the stage must actually run for this measurement"
    assert rec["bias_delta_rel"] is not None
    assert rec["weight_delta_rel"] is not None

    num = den = 0.0
    for m, b0 in zip((block.fc1, block.fc2), before, strict=True):
        num += float((m.bias.detach() - b0).pow(2).sum())
        den += float(b0.pow(2).sum())
    expected = float(np.sqrt(num) / np.sqrt(den))
    assert rec["bias_delta_rel"] == expected
    assert rec["bias_delta_rel"] > 0.0

    # A block whose linears carry no bias has nothing to report: `None`, never
    # a confident 0.0.
    bare = _BiasBlock(d, seed=4)
    for m in (bare.fc1, bare.fc2):
        m.bias = None
    rec_bare = _tune_fp(
        bare,
        x_b,
        y_b,
        cfg,
        cfg.stage_pre.batch_size,
        label="bare",
        tune_names=["fc1", "fc2"],
        factorized_in_block=1,
    )
    assert rec_bare["bias_delta_rel"] is None


def test_packed_artifact_bias_round_trips(monkeypatch):
    """The artifact's bias is a container field: serialize -> deserialize
    returns it, byte-for-byte reproducible."""
    d = 8
    block = _BiasBlock(d, seed=1)
    cfg = PTQConfig(solver_backend="numpy", rank=4)

    def _stub(block, x_b, y_b, cfg, batch_size, *, label="", tune_names=None, **kwargs):
        for name, mod in block.named_modules():
            if name in set(tune_names or []) and getattr(mod, "bias", None) is not None:
                with torch.no_grad():
                    mod.bias.fill_(SENTINEL)
        return {"stage": "tune_fp", "steps": 0}

    monkeypatch.setattr(_block_stage, "_tune_fp", _stub)
    result = reconstruct_block(block, _calib(d), cfg, cotangent=_fixed_cot(block, _calib(d)))
    layers = {st.name: st for st in result.layers}

    model = QuantizedModel(layers, cfg)
    blobs = model.serialize()
    header = blobs["fc2"]
    assert header[:4] == b"QPK1"
    assert int.from_bytes(header[4:8], "little") == qpk1.REPRESENTATION_VERSION

    reloaded = QuantizedModel.deserialize(blobs, cfg)
    for name, st in layers.items():
        assert reloaded.layers[name].bias is not None
        np.testing.assert_allclose(reloaded.layers[name].bias, st.bias)
        np.testing.assert_array_equal(reloaded.layers[name].Ub, st.Ub)
        np.testing.assert_allclose(reloaded.layers[name].s1, st.s1)
    np.testing.assert_allclose(reloaded.layers["fc2"].bias, np.full(d, SENTINEL))
    # Byte-identical re-serialization: the bias rides the same round trip the
    # factors and scales do.
    assert reloaded.serialize() == blobs
