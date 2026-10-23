"""Solve-checkpoint resume: bit-identity and stale-state protection.

The contract (checkpoints.py module docstring): a resumed run's
serialized artifact is BYTE-IDENTICAL to a never-interrupted run's —
the solve is deterministic, resume is exact recovery, not approximation.
Resume must refuse a checkpoint written by a different PTQConfig or a
missing fingerprint (v6's lesson: stale state must crash, never blend).
"""

import dataclasses

import pytest
import torch
from test_api import _calib, _TinyApiModel

import quantibbler.ptq.block_recon as block_recon
from quantibbler.ptq.api import quantize_model
from quantibbler.ptq.checkpoints import completed_blocks
from quantibbler.ptq.config import PTQConfig, StageHyper


def _cfg() -> PTQConfig:
    return PTQConfig(
        solver_backend="numpy",
        rank=4,
        stage_pre=StageHyper(1e-4, 4, 0),
        stage_latent=StageHyper(2e-3, 4, 3),
        stage_global=StageHyper(2e-2, 4, 4),
    )


def _crash_after_first_block(monkeypatch):
    """Patch reconstruct_block to solve the 1st block, then die on the 2nd
    call — a faithful mid-solve crash: block 0 is checkpointed (save happens
    after the solve returns), block 1 never runs."""
    calls = {"n": 0}
    real = block_recon.reconstruct_block

    def crashing(*a, **kw):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("simulated mid-solve crash")
        return real(*a, **kw)

    monkeypatch.setattr(block_recon, "reconstruct_block", crashing)
    return calls


def test_scn_checkpoint_resume_bit_identical(tmp_path, monkeypatch):
    torch.manual_seed(0)
    batches = _calib(seed=7)
    cfg = _cfg()
    ckpt_dir = tmp_path / "ckpts"

    # The reference: one uninterrupted run.
    qm_ref = quantize_model(_TinyApiModel(seed=3), batches, cfg)
    blobs_ref = qm_ref.serialize()

    # The interrupted run: crash after block 0's solve.
    _crash_after_first_block(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated mid-solve crash"):
        quantize_model(_TinyApiModel(seed=3), batches, cfg, checkpoint_dir=ckpt_dir)
    monkeypatch.undo()

    # Exactly one block survived, atomically (no partial writes visible).
    # Index = enumerate position among ALL children (embed=0, lin1=1,
    # lin2=3): gappy but deterministic per model structure, which is what
    # resume requires.
    assert completed_blocks(ckpt_dir) == [1]

    # Resume on a fresh host: block 0 restored from disk, block 1 solved,
    # then KD + cast + export as usual.
    qm_resumed = quantize_model(_TinyApiModel(seed=3), batches, cfg, checkpoint_dir=ckpt_dir)
    blobs_resumed = qm_resumed.serialize()

    assert blobs_resumed == blobs_ref, "resumed artifact differs byte-wise"


def test_scn_checkpoint_resume_restores_skipped_solve(tmp_path, monkeypatch):
    """The resume path must actually SKIP the solve: with reconstruct_block
    rigged to ALWAYS fail, a resumed run still completes using checkpoints
    for every block."""
    torch.manual_seed(0)
    batches = _calib(seed=7)
    cfg = _cfg()
    ckpt_dir = tmp_path / "ckpts2"

    # Reference run WITH checkpoints (same code path, no interruption).
    qm_ref = quantize_model(_TinyApiModel(seed=3), batches, cfg, checkpoint_dir=ckpt_dir)
    blobs_ref = qm_ref.serialize()

    # A resumed run where the solver is unusable: everything must come
    # from disk. (Proves restore, not re-solve, feeds the artifact.)
    def exploding(*a, **kw):
        raise RuntimeError("solver must not be called on a fully-checkpointed resume")

    monkeypatch.setattr(block_recon, "reconstruct_block", exploding)
    qm_restored = quantize_model(_TinyApiModel(seed=3), batches, cfg, checkpoint_dir=ckpt_dir)
    monkeypatch.undo()
    assert qm_restored.serialize() == blobs_ref


def test_scn_checkpoint_fingerprint_refuses_stale_config(tmp_path):
    torch.manual_seed(0)
    batches = _calib(seed=7)
    cfg = _cfg()
    ckpt_dir = tmp_path / "ckpts3"

    # Seed a real, fingerprinted single-block checkpoint dir.
    _seed_one_block_checkpoint(ckpt_dir, cfg, batches)

    # Different config (rank bump) must refuse on fingerprint.
    stale = dataclasses.replace(cfg, rank=cfg.rank + 1)
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        quantize_model(_TinyApiModel(seed=3), batches, stale, checkpoint_dir=ckpt_dir)

    # Missing fingerprint must refuse too.
    (ckpt_dir / "fingerprint.json").unlink()
    with pytest.raises(RuntimeError, match="fingerprint"):
        quantize_model(_TinyApiModel(seed=3), batches, cfg, checkpoint_dir=ckpt_dir)


def _seed_one_block_checkpoint(ckpt_dir, cfg, batches):
    """Seed a checkpoint dir with one completed block + fingerprint, via a
    one-shot crash wrapper around reconstruct_block."""
    import quantibbler.ptq.block_recon as br

    calls = {"n": 0}
    real = br.reconstruct_block

    def crashing(*a, **kw):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("stop after first block")
        return real(*a, **kw)

    br.reconstruct_block = crashing
    try:
        br.reconstruct_model(_TinyApiModel(seed=3), batches, cfg, checkpoint_dir=ckpt_dir)
    except RuntimeError:
        pass
    finally:
        br.reconstruct_block = real
    assert completed_blocks(ckpt_dir) == [1]  # lin1 = child index 1 (embed=0)
