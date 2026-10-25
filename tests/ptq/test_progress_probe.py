"""Progress-probe contract.

The per-block observer: ``PTQConfig.progress_probe`` is invoked by the
sequential block reconstruction after each block is written back into the
host, with ``(model, depth, name)``. It may raise ``PrefixProbeAbort`` to
halt the run; an unset probe changes nothing.

Scenarios: ``probe_invoked_each_block`` / ``probe_absent_no_change`` /
``probe_abort_propagates``, plus the fingerprint-exclusion consequence
recorded in the change's design.md.

Torch tier: this exercises the real pipeline, so the tier conftest ignores
the module cleanly when torch/transformers are absent.
"""

import dataclasses
import inspect

import pytest
import torch
from test_api import _calib, _gate_cfg, _TinyApiModel

import quantibbler.ptq.block_recon as block_recon
from quantibbler.ptq.api import quantize_model
from quantibbler.ptq.checkpoints import config_fingerprint
from quantibbler.ptq.config import PrefixProbeAbort, PTQConfig
from quantibbler.qk_report import COLLAPSE_LINE


def test_scn_probe_invoked_each_block():
    """Once per block, in construction order, on the LIVE host, post write-back."""
    model = _TinyApiModel(seed=3)
    pristine = _TinyApiModel(seed=3)
    seen = []

    def probe(host, depth, name):
        seen.append((host, depth, name))

    cfg = dataclasses.replace(_gate_cfg(), progress_probe=probe)
    quantize_model(model, _calib(), cfg)

    assert [(depth, name) for _, depth, name in seen] == [(1, "lin1"), (2, "lin2")]

    # The probe sees the host object itself, not a copy...
    assert all(host is model for host, _, _ in seen)

    # ...and by the time it fires, that block's quantized weights are already
    # written back (the prefix is live, which is the whole point).
    for host, depth, name in seen:
        assert not torch.equal(getattr(host, name).weight, getattr(pristine, name).weight), (
            f"block {name!r} must be quantized when the probe fires at depth {depth}"
        )


def test_scn_probe_absent_no_change():
    """An unset probe is byte-identical to an inert one, and runs unchanged."""
    base = _gate_cfg()
    inert = dataclasses.replace(base, progress_probe=lambda *a: None)

    assert config_fingerprint(base) == config_fingerprint(inert)

    torch.manual_seed(0)
    without = quantize_model(_TinyApiModel(seed=3), _calib(), base).serialize()
    torch.manual_seed(0)
    with_inert = quantize_model(_TinyApiModel(seed=3), _calib(), inert).serialize()
    assert without == with_inert, "an observer must not change the artifact"


def test_scn_probe_abort_propagates():
    """The abort surfaces unchanged and no later block is captured or solved."""
    fired = []

    def probe(_host, depth, name):
        fired.append((depth, name))
        if depth == 1:
            raise PrefixProbeAbort(
                block=name,
                depth=depth,
                prefix_ppl=1.2e6,
                fp_prefix_ppl=2.4,
                ratio=5.0e5,
            )

    cfg = dataclasses.replace(_gate_cfg(), progress_probe=probe)
    with pytest.raises(PrefixProbeAbort) as excinfo:
        quantize_model(_TinyApiModel(seed=3), _calib(), cfg)

    assert fired == [(1, "lin1")], "a later block must not be reached after an abort"
    err = excinfo.value
    assert (err.block, err.depth) == ("lin1", 1)
    assert (err.prefix_ppl, err.fp_prefix_ppl, err.ratio) == (1.2e6, 2.4, 5.0e5)
    assert err.line == COLLAPSE_LINE
    assert "lin1" in str(err)


def test_probe_line_is_not_an_override_parameter():
    """The abort line is the shared constant; there is no argument for it."""
    assert "line" not in inspect.signature(PrefixProbeAbort.__init__).parameters
    err = PrefixProbeAbort(block="b", depth=2, prefix_ppl=1.0, fp_prefix_ppl=1.0, ratio=1.0)
    assert err.line == COLLAPSE_LINE


def test_probe_must_be_callable():
    with pytest.raises(TypeError, match="progress_probe must be callable"):
        PTQConfig(solver_backend="numpy", progress_probe=17)


def test_probe_field_stays_out_of_config_equality():
    """An observer is not part of the config's identity."""
    base = _gate_cfg()
    other = dataclasses.replace(base, progress_probe=lambda *a: None)
    assert base == other


def test_probe_can_be_added_on_resume_without_refusing(tmp_path, monkeypatch):
    """Fingerprint exclusion, end to end: probe on, resume still allowed."""
    real = block_recon.reconstruct_block
    calls = {"n": 0}

    def crashing(*a, **kw):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("simulated mid-solve crash")
        return real(*a, **kw)

    ckpt_dir = tmp_path / "ckpts"
    base = _gate_cfg()
    monkeypatch.setattr(block_recon, "reconstruct_block", crashing)
    with pytest.raises(RuntimeError, match="simulated mid-solve crash"):
        quantize_model(_TinyApiModel(seed=3), _calib(), base, checkpoint_dir=ckpt_dir)
    monkeypatch.undo()

    fired = []
    probing = dataclasses.replace(
        base, progress_probe=lambda host, depth, name: fired.append((depth, name))
    )
    # No fingerprint mismatch despite the added probe.
    quantize_model(_TinyApiModel(seed=3), _calib(), probing, checkpoint_dir=ckpt_dir)

    # Both boundaries are probed: a restored block is still written back into
    # the host and the prefix still advances through it, so the probe observes
    # the same live-prefix boundary a fresh run's would. (The "remaining blocks
    # only" rule is about *solve* work, not about probe coverage.)
    assert fired == [(1, "lin1"), (2, "lin2")]
