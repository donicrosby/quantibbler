"""The resume fingerprint binds the model identity.

Scenarios under test: ``model_identity_binds_the_resume``,
``model_identity_mismatch_names_the_divergence``,
``legacy_fingerprint_without_model_identity_is_refused``,
``driver_default_checkpoint_dir_is_per_run``.

The measured defect:
the fingerprint binds PTQConfig + version + git describe and NOTHING about the
model, so ``check_fingerprint`` PASSED for a second model against the first
model's directory. End to end that is a silent chimera — model B
(``embed+lin1+act+head``) resumed model A's directory (``embed+lin1+lin2+head``)
with the solver rigged to raise if called, completed, and printed
``[checkpoint] block 1 (lin1): restored from checkpoint, skipping solve`` with
solver calls ``0``. A different-shape pair fails instead inside
``apply_layer_states`` on an anonymous tensor-size error, which names no cause.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from test_api import _calib, _gate_cfg, _TinyApiModel, _TinyTrunkView
from torch import nn

import quantibbler.ptq.block_recon as br
from quantibbler.ptq.api import quantize_model
from quantibbler.ptq.block_recon import QuantizedLayerState
from quantibbler.ptq.checkpoints import (
    check_fingerprint,
    completed_blocks,
    config_fingerprint,
    save_block_checkpoint,
)

REPO = Path(__file__).resolve().parents[2]


class _SiblingBlock(nn.Module):
    """embed + lin1 + act + fp head — lin1 keeps A's name and (8, 8); no lin2.

    The same-name/same-shape overlap is the point: block index 1 is a
    shape-compatible restore target, which is exactly why the pre-fix resume
    could complete silently instead of crashing.
    """

    def __init__(self, d: int = 8, seed: int = 9) -> None:
        super().__init__()
        ref = _TinyApiModel(d=d, seed=seed)
        self.embed = ref.embed
        self.lin1 = ref.lin1
        self.act = ref.act
        self.head = ref.head

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = x + self.lin1(x)
        return self.head(x)

    @property
    def trunk(self) -> _TinyTrunkView:
        """Same handle shape ``_TinyApiModel`` exposes (whole-model detection)."""
        return _TinyTrunkView(self.embed, self.lin1, self.act, self.lin1)


def _seed_one_block(ckpt_dir: Path, cfg, batches, model) -> None:
    """Seed a checkpoint directory with block 1 (lin1) plus its fingerprint."""
    real = br.reconstruct_block
    calls = {"n": 0}

    def crashing(*a, **kw):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("stop after first block")
        return real(*a, **kw)

    br.reconstruct_block = crashing
    try:
        quantize_model(model, batches, cfg, checkpoint_dir=ckpt_dir)
    except RuntimeError as exc:
        assert "stop after first block" in str(exc), exc
    finally:
        br.reconstruct_block = real
    assert completed_blocks(ckpt_dir) == [1], completed_blocks(ckpt_dir)


def _no_solver(*a, **kw):
    raise AssertionError(
        "the solver ran on a resume that must be refused by the fingerprint: "
        "the refusal did not happen where it is supposed to happen"
    )


def test_scn_model_identity_binds_the_resume(tmp_path, monkeypatch):
    """Model B may not resume model A's archives under an identical PTQConfig."""
    torch.manual_seed(0)
    batches = _calib(seed=7)
    cfg = _gate_cfg()
    ckpt_dir = tmp_path / "ck"
    _seed_one_block(ckpt_dir, cfg, batches, _TinyApiModel(d=8, seed=3))

    model_b = _SiblingBlock(d=8, seed=9)
    assert config_fingerprint(cfg) == config_fingerprint(cfg)  # same config, still

    with pytest.raises(RuntimeError, match="model identity"):
        check_fingerprint(ckpt_dir, cfg, model_b)

    # End to end: refused BEFORE any block is restored or solved.
    monkeypatch.setattr(br, "reconstruct_block", _no_solver)
    with pytest.raises(RuntimeError, match="model identity"):
        quantize_model(model_b, batches, cfg, checkpoint_dir=ckpt_dir)


def test_scn_model_identity_mismatch_names_the_divergence(tmp_path):
    """The refusal has to be diagnosable: it names the diverged module(s)."""
    torch.manual_seed(0)
    batches = _calib(seed=7)
    cfg = _gate_cfg()
    ckpt_dir = tmp_path / "ck"
    _seed_one_block(ckpt_dir, cfg, batches, _TinyApiModel(d=8, seed=3))

    # (a) per-module shape divergence
    with pytest.raises(RuntimeError) as excinfo:
        check_fingerprint(ckpt_dir, cfg, _TinyApiModel(d=6, seed=3))
    msg = str(excinfo.value)
    assert "model identity" in msg, msg
    for name in ("lin1", "lin2"):
        assert name in msg, f"{name} not named in the refusal: {msg}"
    assert "(8, 8, float64)" in msg, f"the recorded shape is unnamed: {msg}"
    assert "(6, 6, float64)" in msg, f"the live shape is unnamed: {msg}"

    # (b) membership divergence — a module the archive's model had and this one
    # does not (the silent-chimera shape: the overlapping block restores cleanly)
    with pytest.raises(RuntimeError) as excinfo:
        check_fingerprint(ckpt_dir, cfg, _SiblingBlock(d=8, seed=9))
    msg = str(excinfo.value)
    assert "lin2" in msg, f"the missing module is unnamed: {msg}"

    # (c) dtype divergence — same inventory, same shapes, different weight dtype
    with pytest.raises(RuntimeError) as excinfo:
        check_fingerprint(ckpt_dir, cfg, _TinyApiModel(d=8, seed=3).to(torch.float32))
    msg = str(excinfo.value)
    assert "float32" in msg and "float64" in msg, f"the dtype is unnamed: {msg}"


def test_scn_legacy_fingerprint_without_model_identity_is_refused(tmp_path):
    """A directory written before this binding cannot be verified: refuse."""
    cfg = _gate_cfg()
    ckpt_dir = tmp_path / "legacy"
    ckpt_dir.mkdir()
    # A config-matching legacy record: the ONLY divergence is the missing
    # identity component, so a config-digest refusal here would be the wrong
    # verdict and this test is what tells them apart.
    (ckpt_dir / "fingerprint.json").write_text(
        json.dumps({"fingerprint": config_fingerprint(cfg)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    gen = np.random.default_rng(0)
    ub = np.where(gen.random((8, 2)) < 0.5, -1, 1).astype(np.int8)
    vb = np.where(gen.random((8, 2)) < 0.5, -1, 1).astype(np.int8)
    save_block_checkpoint(
        ckpt_dir,
        1,
        [QuantizedLayerState("lin1", ub, vb, gen.random(8) + 0.5, gen.random(8) + 0.5, None)],
    )
    assert completed_blocks(ckpt_dir) == [1]

    with pytest.raises(RuntimeError) as excinfo:
        check_fingerprint(ckpt_dir, cfg, _TinyApiModel(d=8, seed=3))
    msg = str(excinfo.value)
    assert "fingerprint mismatch" in msg, msg
    assert "no model-identity component" in msg, msg
    assert "before" in msg, f"the refusal does not say why it cannot be trusted: {msg}"


def test_scn_driver_default_checkpoint_dir_is_per_run(tmp_path):
    """The driver's default must not be one shared, stable, mutable directory.

    A shared default is the amplifier of the defect above: two models launched
    with no flags collide in `artifacts/ptq-checkpoints` without anyone choosing
    to share it. The default is therefore a FRESH per-run directory, and naming
    a directory is how a launch opts into resumption.
    """
    import importlib.util

    driver_path = REPO / "tools" / "quantize_hf_model.py"
    spec = importlib.util.spec_from_file_location("qhm_model_identity_under_test", driver_path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    import shutil

    first = driver._checkpoint_launcher_dir(None)
    second = driver._checkpoint_launcher_dir(None)
    try:
        assert first != second, f"two defaulted launches share {first}"
        assert first.is_dir() and second.is_dir()
        assert first != REPO / "artifacts" / "ptq-checkpoints", "the shared stable default is back"
        assert REPO / "artifacts" not in first.parents, (
            f"the per-run default must not land in the repo's artifacts tree: {first}"
        )
    finally:
        shutil.rmtree(first, ignore_errors=True)
        shutil.rmtree(second, ignore_errors=True)

    explicit = tmp_path / "resume-here"
    assert driver._checkpoint_launcher_dir(explicit) == explicit, (
        "an explicitly named directory is the resumable path and must be used as given"
    )

    # argparse defaults are unreachable without running main(); pin the source,
    # the same way the probe's positional signature is pinned.
    source = driver_path.read_text(encoding="utf-8")
    block = re.search(r'"--checkpoint-dir",(.*?)\)\n', source, re.DOTALL)
    assert block is not None, "the --checkpoint-dir option is gone"
    assert re.search(r"default=None\b", block.group(1)), (
        "the --checkpoint-dir default must be None (per-run), not a shared stable "
        f"path: {block.group(1)!r}"
    )
