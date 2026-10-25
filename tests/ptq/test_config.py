"""Task 1.1: PTQConfig defaults (scenarios config_defaults_paper_values,
default_numerics_float64, float32_opt_in)."""

import dataclasses

import numpy as np
import pytest

from quantibbler.ptq.checkpoints import config_fingerprint
from quantibbler.ptq.config import PTQConfig, StageHyper


def test_scn_config_defaults():
    """A bare PTQConfig carries [NQ] App. C defaults and the qk-ptq design
    informed-assumption values (A-1..A-7)."""
    cfg = PTQConfig(
        solver_backend="numpy",
    )

    # Representation / global knobs.
    assert isinstance(cfg.rank, int) and cfg.rank >= 1
    assert cfg.seed == 0
    assert cfg.temperature == 1.0  # A-7
    # Robust diagonal (tau, gamma); gamma 0.2 per [NQ] Step 2-1.
    assert cfg.gamma == pytest.approx(0.2)
    assert cfg.tau is None or cfg.tau > 0
    # ADMM: K = 400 with linear penalty schedule ([NQ] App. C), endpoints A-2.
    assert cfg.admm_max_iter == 400
    # Lane 3 (normalized regime): the endpoints are recalibrated to linear ~0->1.
    # The old 1e-2 -> 1e-1 pair was characterised against the absolute-ridge
    # solver and does not carry over: with the scale-relative stabilizer
    # rho*mean(diag(X^T X)) + reg the data term is O(1) at the normalized design,
    # so rho ~ 1 sits at its own scale. rho_start = 0 is safe because the
    # stabilizer at rho = 0 is admm_reg alone, which keeps the subproblem SPD.
    assert cfg.rho_start == pytest.approx(0.0)
    assert cfg.rho_end == pytest.approx(1.0)
    assert cfg.rho_end > cfg.rho_start >= 0
    assert cfg.admm_reg == pytest.approx(3e-2)
    assert cfg.admm_lam == pytest.approx(1e-6)  # A-3
    assert cfg.admm_epsilon == pytest.approx(1e-7)  # A-4
    # Stage lr/batch per [NQ] App. C: 1e-4/bs4 (pre), 1e-5/bs1 (latent),
    # 1e-6/bs1 (global); 8 epochs each, cosine schedule.
    assert cfg.stage_pre == StageHyper(lr=1e-4, batch_size=4, epochs=8)
    assert cfg.stage_latent == StageHyper(lr=1e-5, batch_size=1, epochs=8)
    assert cfg.stage_global == StageHyper(lr=1e-6, batch_size=1, epochs=8)
    assert cfg.cosine_schedule is True


def test_scn_default_numerics_float64():
    """default_numerics_float64: float64 is the default compute dtype and the
    mapped numpy dtype follows it."""
    cfg = PTQConfig(
        solver_backend="numpy",
    )
    assert cfg.compute_dtype == "float64"
    assert cfg.np_compute_dtype == np.dtype(np.float64)


def test_scn_float32_opt_in():
    """float32_opt_in: float32 is reachable only through an explicit config
    setting; anything else is rejected loudly."""
    opt_in = PTQConfig(solver_backend="numpy", compute_dtype="float32")
    assert opt_in.compute_dtype == "float32"
    assert opt_in.np_compute_dtype == np.dtype(np.float32)
    with pytest.raises(ValueError):
        PTQConfig(solver_backend="numpy", compute_dtype="float16")
    with pytest.raises(ValueError):
        PTQConfig(solver_backend="numpy", compute_dtype="float64 ")
    # Serialization knob is likewise explicit and validated.
    assert (
        PTQConfig(
            solver_backend="numpy",
        ).scale_dtype
        == "float32"
    )
    assert PTQConfig(solver_backend="numpy", scale_dtype="float32").scale_dtype == "float32"
    with pytest.raises(ValueError):
        PTQConfig(solver_backend="numpy", scale_dtype="float64")


def test_scn_config_validation():
    """Out-of-range hyperparameters are construction errors, never silent."""
    with pytest.raises(ValueError):
        PTQConfig(solver_backend="numpy", rank=0)
    with pytest.raises(ValueError):
        PTQConfig(solver_backend="numpy", gamma=1.5)
    with pytest.raises(ValueError):
        PTQConfig(solver_backend="numpy", admm_max_iter=0)
    # rho_start = 0.0 is the recalibrated default (the admm_reg floor keeps the
    # subproblem SPD); a NEGATIVE endpoint is what must fail closed.
    with pytest.raises(ValueError):
        PTQConfig(solver_backend="numpy", rho_start=-1e-3)
    with pytest.raises(ValueError):
        PTQConfig(solver_backend="numpy", admm_reg=0.0)
    with pytest.raises(ValueError):
        PTQConfig(solver_backend="numpy", rho_end=1e-4, rho_start=1e-2)
    with pytest.raises(ValueError):
        PTQConfig(solver_backend="numpy", temperature=0.0)


def test_scn_loss_weight_knob_joins_checkpoint_fingerprint():
    """The weighting mode is solver state, so it must reach the fingerprint.

    Two configs differing ONLY in `block_loss_weight` must not share a digest: a
    checkpoint written under terminal-gradient weighting may not resume under a
    uniform-weighted run, and the resume refusal is the feature. `config_fingerprint`
    hashes `dataclasses.asdict(cfg)` (checkpoints.py), so the field should join by
    construction -- this pin holds that construction to account rather than
    assuming it.
    """
    assert PTQConfig(solver_backend="numpy").block_loss_weight == "terminal_grad", (
        "the paper-anchored mode is the default"
    )

    terminal = PTQConfig(solver_backend="numpy")
    uniform = dataclasses.replace(terminal, block_loss_weight="uniform")
    assert uniform.block_loss_weight == "uniform", "the knob must be a plain dataclass field"

    assert config_fingerprint(terminal) != config_fingerprint(uniform), (
        "configs differing only in the weighting mode must not share a fingerprint"
    )

    # The knob accepts exactly the two modes the spec names: an unknown mode is a
    # construction error, never a silent fallback to the default.
    with pytest.raises(ValueError):
        PTQConfig(solver_backend="numpy", block_loss_weight="rowwise")
