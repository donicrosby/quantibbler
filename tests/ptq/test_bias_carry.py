"""qk-bias-carry: the container carries the per-linear bias, and the tuning
stage's bias is what the artifact ships.

Five claims, one test each:

* ``tuned_bias_survives_factorization`` — the bias a tuning pass wrote to a
  remainder linear is the bias the factorized artifact carries. The stage is
  stubbed to a sentinel write, so the assertion is deterministic and does not
  depend on how far an optimizer happened to move anything.
* ``tunefp_record_carries_bias_delta_rel`` — the TuneFP row reports the bias
  movement separately from the aggregate weight figure, and the reported
  number is the one an independent recomputation from a held snapshot gives.
* ``packed_artifact_bias_round_trips`` — what the artifact carries is what
  the container's writer/reader pair returns.
* ``bias_delta_rel_none_is_disambiguated_by_bias_tunables`` — the TuneFP row
  names WHICH of the two causes left ``bias_delta_rel`` at ``None``: a
  bias-free tunable set (count 0 — design D12's legit-zero, "no value, never a
  confident 0.0") versus a stage that stopped before it could measure
  (count > 0). Without the count the two are one bare ``None``.
* ``biasless_block_packs_zero_bias_v2`` — the outcome the v6b artifact
  actually shipped (a bias-free model packs 196/196 exactly-zero biases) is
  pinned at the pack boundary: ``bias=False`` linears through
  ``reconstruct_block`` -> ``serialize`` carry a PRESENT, exactly-zero v2 bias,
  not an omitted field and not a confident non-zero.
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


class _BiasFreeBlock(nn.Module):
    """Two float64 linears with NO bias at all — the Qwen3-class shape the
    v6b artifact came from, where every projection is ``nn.Linear(bias=False)``.

    ``nn.Linear`` defaults to ``bias=True``, so the absence is stated, not
    inherited: this is the fixture the bias-carry suite never had.
    """

    def __init__(self, d: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.fc1 = nn.Linear(d, d, bias=False, dtype=torch.float64)
        self.fc2 = nn.Linear(d, d, bias=False, dtype=torch.float64)
        with torch.no_grad():
            for m in (self.fc1, self.fc2):
                m.weight.copy_(torch.randn(d, d, generator=gen, dtype=torch.float64) * 0.4)

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
    cfg = PTQConfig(solver_backend="numpy", compute_dtype="float64", rank=4)

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
    cfg = PTQConfig(solver_backend="numpy", compute_dtype="float64", rank=3, cosine_schedule=False)
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
    cfg = PTQConfig(solver_backend="numpy", compute_dtype="float64", rank=4)

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


def test_bias_delta_rel_none_is_disambiguated_by_bias_tunables():
    """The TuneFP row names WHICH cause left ``bias_delta_rel`` at ``None``.

    The overload the record fixed: a bare ``None`` meant both "the model has no
    bias to move" (a bias-free architecture — the v6b artifact's Qwen3-0.6B,
    all 196 linears ``nn.Linear(bias=False)``) and "the stage never ran far
    enough to have an answer". Only the design doc told them apart; the
    `bias_tunables` count makes it readable off the row itself.
    """
    d = 6
    cfg = PTQConfig(solver_backend="numpy", compute_dtype="float64", rank=3, cosine_schedule=False)
    x_b = torch.cat(_calib(d), dim=0)

    # A bias-bearing tunable set: the stage ran AND measured a bias movement.
    biased = _BiasBlock(d, seed=3)
    with torch.no_grad():
        y_biased = biased(x_b) * 1.2 + 0.1
    rec = _tune_fp(
        biased,
        x_b,
        y_biased,
        cfg,
        cfg.stage_pre.batch_size,
        label="biased",
        tune_names=["fc1", "fc2"],
        factorized_in_block=1,
    )
    assert rec["steps"] > 0, "the stage must actually run for this measurement"
    assert rec["bias_tunables"] == 2, "two bias-bearing tunables were in the set"
    assert rec["bias_delta_rel"] is not None, "a bias-bearing stage measures its bias"

    # A bias-free model: no bias compartment to move. `None` is the honest
    # figure (design D12: "no value, never a confident 0.0"), and the count 0
    # says WHY — this is the row that was indistinguishable from an unrun one.
    bare = _BiasFreeBlock(d, seed=4)
    with torch.no_grad():
        y_bare = bare(x_b) * 1.2 + 0.1
    rec_bare = _tune_fp(
        bare,
        x_b,
        y_bare,
        cfg,
        cfg.stage_pre.batch_size,
        label="bare",
        tune_names=["fc1", "fc2"],
        factorized_in_block=1,
    )
    assert rec_bare["steps"] > 0, "the bias-free stage still runs over the weights"
    assert rec_bare["bias_tunables"] == 0, "a bias-free model has no bias tunables"
    assert rec_bare["bias_delta_rel"] is None

    # The OTHER cause of a `None` figure, told apart by the SAME count: a stage
    # that never ran far enough to measure (here forced vacuous) over a
    # BIAS-BEARING set. On the old record this row was identical to the
    # bias-free one; now the count reads "there is a bias and it was not
    # measured" (2), not "there is no bias" (0).
    stalled = _BiasBlock(d, seed=5)
    rec_stalled = _tune_fp(
        stalled,
        x_b,
        torch.zeros_like(x_b),
        cfg,
        cfg.stage_pre.batch_size,
        label="stalled",
        tune_names=["fc1", "fc2"],
        factorized_in_block=0,  # forces the vacuous-teacher early return
    )
    assert rec_stalled["steps"] == 0, "the vacuous early return ran no steps"
    assert rec_stalled["bias_delta_rel"] is None
    assert rec_stalled["bias_tunables"] == 2, (
        "a bias-bearing set that never measured still declares its bias tunables; "
        "this is the row the old bare None could not tell from the bias-free one"
    )


def test_biasless_block_packs_zero_bias_v2():
    """The legit-zero path, pinned at the pack boundary.

    No fixture in this suite ever exercised an architecturally bias-free model,
    so the outcome the v6b artifact shipped (196/196 exactly-zero biases) was
    never asserted anywhere. This runs ``bias=False`` linears through
    ``reconstruct_block`` -> ``serialize`` and asserts the v2 wire carries a
    PRESENT, exactly-zero bias for every layer — not an omitted field, not a
    confident non-zero.
    """
    d = 8
    block = _BiasFreeBlock(d, seed=7)
    cfg = PTQConfig(solver_backend="numpy", compute_dtype="float64", rank=4)
    result = reconstruct_block(block, _calib(d), cfg, cotangent=_fixed_cot(block, _calib(d)))

    layers = {st.name: st for st in result.layers}
    assert set(layers) == {"fc1", "fc2"}
    for name, st in layers.items():
        # The field is PRESENT (v2 always carries it) and exactly zero: the
        # bias-free model's absence of a bias compartment, faithfully carried.
        assert st.bias is not None, f"{name}: the v2 field must be present, not omitted"
        np.testing.assert_array_equal(st.bias, np.zeros(d))
        assert not np.any(st.bias), f"{name}: a bias-free linear must pack exactly zero bias"

    model = QuantizedModel(layers, cfg)
    blobs = model.serialize()
    header = blobs["fc2"]
    assert header[:4] == b"QPK1"
    assert int.from_bytes(header[4:8], "little") == qpk1.REPRESENTATION_VERSION
    assert qpk1.REPRESENTATION_VERSION == 2

    reloaded = QuantizedModel.deserialize(blobs, cfg)
    for name in layers:
        got = reloaded.layers[name].bias
        assert got is not None, f"{name}: a v2 stream reads back WITH its bias field"
        np.testing.assert_array_equal(got, np.zeros(d))
    # The zeros ride the same wire any bias does: byte-identical round trip.
    assert reloaded.serialize() == blobs
