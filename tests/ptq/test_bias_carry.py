"""qk-bias-carry: the container carries the per-linear bias, and the tuning
stage's bias is what the artifact ships.

Eight claims, one test each:

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
* ``zero_bias_baseline_records_no_value`` — an all-zero bias baseline has no
  denominator, so the row carries ``None`` and never a confident ``0.0``:
  pinned for both degenerate arms (bias tuned off zero beside the zero
  baseline, and nothing moved beside it).
* ``biasless_block_packs_zero_bias_v2`` — the outcome the v6b artifact
  actually shipped (a bias-free model packs 196/196 exactly-zero biases) is
  pinned at the pack boundary: ``bias=False`` linears through
  ``reconstruct_block`` -> ``serialize`` carry a PRESENT, exactly-zero v2 bias,
  not an omitted field and not a confident non-zero.
* ``zero_weight_baseline_records_no_value`` — the weight arm of the same rule:
  a stage whose tuned parameters are ALL exactly 0.0 at entry has no
  denominator, so the row carries ``None`` and never a confident ``0.0``;
  pinned for both degenerate arms (weights tuned off zero beside the zero
  baseline, and nothing moved beside it), with ``inert`` left ``None`` beside a
  relative figure that does not exist.
* ``nonzero_baseline_records_zero_when_nothing_moves`` — the discriminator the
  ``None`` rule must not swallow: a NON-zero baseline with nothing moved still
  records a measured ``0.0``, for both the weight and the bias figure.
"""

import numpy as np
import torch
from torch import nn

from quantibbler import qpk1
from quantibbler.ptq import _block_stage
from quantibbler.ptq.api import QuantizedModel
from quantibbler.ptq.block_recon import _tune_fp, reconstruct_block
from quantibbler.ptq.config import PTQConfig, StageHyper

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


class _ZeroInitBiasBlock(nn.Module):
    """Two float64 linears whose biases are initialized to EXACTLY 0.0 — the
    zero-init bias-projection shape ``_BiasBlock`` cannot express (it fills
    every bias with ``BIAS_INIT``).

    ``freeze_bias`` holds the biases out of the optimizer so a stage can run
    over a bias-bearing set and leave the zero baseline exactly where it is:
    the ``bnum == 0`` arm beside an all-zero denominator, told apart from the
    arm where the stage tuned the bias off zero.
    """

    def __init__(self, d: int, seed: int, *, freeze_bias: bool = False) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.fc1 = nn.Linear(d, d, dtype=torch.float64)
        self.fc2 = nn.Linear(d, d, dtype=torch.float64)
        with torch.no_grad():
            for m in (self.fc1, self.fc2):
                m.weight.copy_(torch.randn(d, d, generator=gen, dtype=torch.float64) * 0.4)
                m.bias.zero_()
                if freeze_bias:
                    m.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class _ZeroWeightLinearBlock(nn.Module):
    """One float64 linear whose weight is initialized to EXACTLY 0.0, with no
    bias — the zero-init output-projection shape (residual-block output layers,
    adapter/LoRA B matrices, gated units) that ``_BiasBlock`` expresses only for
    biases.

    With no bias the tuned set IS the weight, so the aggregate entry baseline is
    exactly zero: the ``den == 0`` arm of the weight figure. ``out = W x`` with
    ``W == 0`` and a non-zero target still has a non-zero gradient in ``W``, so
    the stage runs and moves the weight off zero — the arm the old record
    reported as ``weight_delta_rel: 0.0``.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d, d, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.fc.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


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


def test_bias_delta_rel_none_is_disambiguated_by_bias_tunables():
    """The TuneFP row names WHICH cause left ``bias_delta_rel`` at ``None``.

    The overload the record fixed: a bare ``None`` meant both "the model has no
    bias to move" (a bias-free architecture — the v6b artifact's Qwen3-0.6B,
    all 196 linears ``nn.Linear(bias=False)``) and "the stage never ran far
    enough to have an answer". Only the design doc told them apart; the
    `bias_tunables` count makes it readable off the row itself.
    """
    d = 6
    cfg = PTQConfig(solver_backend="numpy", rank=3, cosine_schedule=False)
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
    cfg = PTQConfig(solver_backend="numpy", rank=4)
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


def test_zero_bias_baseline_records_no_value():
    """An all-zero bias baseline is no denominator: ``None``, never a 0.0.

    The degenerate arm the record got wrong: when every selected bias vector is
    exactly 0.0 at stage start, ``bden == 0.0`` and the relative movement is
    UNDEFINED — yet the tail recorded ``0.0``, i.e. "the bias never moved", even
    when the stage had tuned the bias OFF zero. Design D12's rule ("no value,
    never a confident 0.0") leaves the field at ``None``; the bias-tunable count
    that sits beside it keeps the row self-describing (bias compartment
    present, figure not available). Both degenerate arms are pinned here, each
    against movement recomputed from a held snapshot:

    * ``bnum > 0`` — the stage tuned the bias off zero. The old row read
      ``0.0``: the exact lie the defect is about.
    * ``bnum == 0`` — nothing moved either. A zero baseline is no denominator
      whether or not anything moved, so this is still no value, not a
      legitimate 0.0.
    """
    d = 6
    cfg = PTQConfig(solver_backend="numpy", rank=3, cosine_schedule=False)
    x_b = torch.cat(_calib(d), dim=0)

    def _run(block: nn.Module) -> tuple[dict, float, float]:
        with torch.no_grad():
            y_b = block(x_b) * 1.2 + 0.1
        before = [m.bias.detach().clone() for m in (block.fc1, block.fc2)]
        rec = _tune_fp(
            block,
            x_b,
            y_b,
            cfg,
            cfg.stage_pre.batch_size,
            label="zero-bias",
            tune_names=["fc1", "fc2"],
            factorized_in_block=1,
        )
        bnum = bden = 0.0
        with torch.no_grad():
            for m, b0 in zip((block.fc1, block.fc2), before, strict=True):
                bnum += float((m.bias.detach() - b0).pow(2).sum())
                bden += float(b0.pow(2).sum())
        return rec, bnum, bden

    # (b) The bias was tuned off a zero baseline: nonzero movement, undefined
    # relative figure. The recorded 0.0 said "never moved" about a moved bias.
    rec_moved, bnum_moved, bden_moved = _run(_ZeroInitBiasBlock(d, seed=11))
    assert rec_moved["steps"] > 0, "the stage must run for this measurement"
    assert rec_moved["bias_tunables"] == 2, "the row still declares its bias compartment"
    assert bden_moved == 0.0, "the baseline is exactly zero — no denominator"
    assert bnum_moved > 0.0, "the stage did move the bias off zero"
    assert rec_moved["bias_delta_rel"] is None

    # (a) The same zero baseline with nothing moved: still no value (a zero
    # denominator cannot yield a ratio), and the row says so rather than
    # claiming a measured 0.0. The weights moved, so the stage demonstrably ran.
    rec_still, bnum_still, bden_still = _run(_ZeroInitBiasBlock(d, seed=12, freeze_bias=True))
    assert rec_still["steps"] > 0, "the stage must run for this measurement"
    assert rec_still["weight_delta_rel"] > 0.0, "the stage ran and moved its weights"
    assert rec_still["bias_tunables"] == 2, "the row still declares its bias compartment"
    assert bden_still == 0.0, "the baseline is exactly zero — no denominator"
    assert bnum_still == 0.0, "the held zero baseline did not move"
    assert rec_still["bias_delta_rel"] is None


def test_zero_weight_baseline_records_no_value():
    """An all-zero weight baseline is no denominator: ``None``, never a 0.0.

    The weight arm of the same defect the bias figure already fixed (design
    D12: "no value, never a confident 0.0"). ``weight_delta_rel``'s ``den`` is
    the entry snapshot's squared norm over EVERY tuned parameter, so a stage
    whose selected linears are all exactly 0.0 at entry — a zero-init output
    projection, a LoRA B matrix, a gated unit — has no denominator at all, and
    the tail used to record ``0.0``: "the weights never moved", even when the
    stage had tuned them OFF zero. Both degenerate arms are pinned here, each
    against movement recomputed from a held snapshot:

    * ``num > 0`` — the stage tuned the weights off zero, and the old row read
      ``0.0``: the exact lie the defect is about.
    * ``num == 0`` — nothing moved either. A zero baseline is no denominator
      whether or not anything moved, so this is still no value, never a
      legitimate 0.0.

    A zero baseline also leaves inertness undecidable — ``inert`` is compared
    against the relative movement — so ``inert`` stays ``None`` beside it
    rather than claiming a verdict about a figure that does not exist. (The
    zero-baseline arm cannot be produced by freezing the parameters: a frozen
    tunable set has an exactly-zero first-step gradient, so the vacuous-teacher
    guard returns before the tail is reached. The held arm therefore runs its
    steps with an lr small enough that the step's update does not register.)
    """
    d = 6
    x_b = torch.cat(_calib(d), dim=0)

    def _run(*, lr: float) -> tuple[dict, float, float]:
        block = _ZeroWeightLinearBlock(d)
        with torch.no_grad():
            # A non-zero target: the loss gradient in the zero weight is real.
            y_b = block(x_b) * 1.2 + 0.1
        cfg = PTQConfig(
            solver_backend="numpy",
            rank=3,
            cosine_schedule=False,
            stage_pre=StageHyper(lr, 4, 8),
        )
        before = [p.detach().clone() for p in block.parameters()]
        rec = _tune_fp(
            block,
            x_b,
            y_b,
            cfg,
            cfg.stage_pre.batch_size,
            label="zero-weight",
            tune_names=["fc"],
            factorized_in_block=1,
        )
        num = den = 0.0
        with torch.no_grad():
            for p, p0 in zip(block.parameters(), before, strict=True):
                num += float((p.detach() - p0).pow(2).sum())
                den += float(p0.pow(2).sum())
        return rec, num, den

    # (b) The weights were tuned off a zero baseline: nonzero movement,
    # undefined relative figure. The recorded 0.0 said "never moved" about a
    # weight the stage demonstrably moved.
    rec_moved, num_moved, den_moved = _run(lr=1e-4)
    assert rec_moved["steps"] > 0, "the stage must run for this measurement"
    assert den_moved == 0.0, "the baseline is exactly zero — no denominator"
    assert num_moved > 0.0, "the stage did move the weight off zero"
    assert rec_moved["weight_delta_rel"] is None
    assert rec_moved["inert"] is None, "no relative figure, so no inertness verdict either"

    # (a) The same zero baseline with nothing moved: still no value (a zero
    # denominator cannot yield a ratio), and the row says so rather than
    # claiming a measured 0.0. `steps > 0` distinguishes this from the
    # never-ran arm the same `None` also covers.
    rec_still, num_still, den_still = _run(lr=1e-300)
    assert rec_still["steps"] > 0, "the stage runs its steps; nothing measurable moved"
    assert den_still == 0.0, "the baseline is exactly zero — no denominator"
    assert num_still == 0.0, "the held zero baseline did not move measurably"
    assert rec_still["weight_delta_rel"] is None
    assert rec_still["inert"] is None


def test_nonzero_baseline_records_zero_when_nothing_moves():
    """The discriminator the ``None`` rule must not swallow: 0.0 stays 0.0.

    A NON-zero baseline with no measurable movement is a legitimate, measured
    ``0.0`` — the row the ``None`` rule above must never swallow. Pinned for
    both figures in one run: the aggregate ``weight_delta_rel`` and the
    ``bias_delta_rel`` beside it, over linears that carry a non-zero weight and
    a non-zero bias.

    The movement here is zero because the per-step update rounds back on the
    parameter dtype (``lr`` small enough that ``p - update == p`` bitwise), not
    because the parameters are frozen: a frozen tunable set has a zero
    first-step gradient and returns via the vacuous-teacher guard with no
    figure at all, which is a different arm. What this row pins is the
    REACHABLE legit-zero: the stage ran, moved nothing, and the record says
    ``0.0`` — never ``None``, which is reserved for no denominator.
    """
    d = 6
    block = _BiasBlock(d, seed=13)
    x_b = torch.cat(_calib(d), dim=0)
    with torch.no_grad():
        y_b = block(x_b) * 1.2 + 0.1
    cfg = PTQConfig(
        solver_backend="numpy",
        rank=3,
        cosine_schedule=False,
        stage_pre=StageHyper(1e-300, 4, 8),
    )
    before = [p.detach().clone() for p in block.parameters()]
    bias_before = [m.bias.detach().clone() for m in (block.fc1, block.fc2)]
    rec = _tune_fp(
        block,
        x_b,
        y_b,
        cfg,
        cfg.stage_pre.batch_size,
        label="legit-zero",
        tune_names=["fc1", "fc2"],
        factorized_in_block=1,
    )

    num = den = 0.0
    with torch.no_grad():
        for p, p0 in zip(block.parameters(), before, strict=True):
            num += float((p.detach() - p0).pow(2).sum())
            den += float(p0.pow(2).sum())
    bnum = bden = 0.0
    with torch.no_grad():
        for m, b0 in zip((block.fc1, block.fc2), bias_before, strict=True):
            bnum += float((m.bias.detach() - b0).pow(2).sum())
            bden += float(b0.pow(2).sum())

    assert rec["steps"] > 0, "the stage must run for this measurement"
    assert den > 0.0 and num == 0.0, "a real baseline, and nothing moved"
    assert rec["weight_delta_rel"] == 0.0, "a measured non-movement is 0.0, not None"
    assert bden > 0.0 and bnum == 0.0, "a real bias baseline, and nothing moved"
    assert rec["bias_delta_rel"] == 0.0, "the bias figure's legit-zero is 0.0 too"
    assert rec["inert"] is True, (
        "steps ran and the relative movement is below the parameter dtype's resolution: "
        "the declared inert regime"
    )
