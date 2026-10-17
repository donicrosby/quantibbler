"""Task 1.1: PTQConfig defaults (scenarios config_defaults_paper_values,
default_numerics_float64, float32_opt_in)."""

import numpy as np
import pytest

from quantibbler.ptq.config import PTQConfig, StageHyper


def test_scn_config_defaults():
    """A bare PTQConfig carries [NQ] App. C defaults and the qk-ptq design
    informed-assumption values (A-1..A-7)."""
    cfg = PTQConfig()

    # Representation / global knobs.
    assert isinstance(cfg.rank, int) and cfg.rank >= 1
    assert cfg.seed == 0
    assert cfg.temperature == 1.0  # A-7
    # Robust diagonal (tau, gamma); gamma 0.2 per [NQ] Step 2-1.
    assert cfg.gamma == pytest.approx(0.2)
    assert cfg.tau is None or cfg.tau > 0
    # ADMM: K = 400 with linear penalty schedule ([NQ] App. C), endpoints A-2.
    assert cfg.admm_max_iter == 400
    assert cfg.rho_start == pytest.approx(1e-2)
    assert cfg.rho_end == pytest.approx(1e2)
    assert cfg.rho_end > cfg.rho_start > 0
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
    cfg = PTQConfig()
    assert cfg.compute_dtype == "float64"
    assert cfg.np_compute_dtype == np.dtype(np.float64)


def test_scn_float32_opt_in():
    """float32_opt_in: float32 is reachable only through an explicit config
    setting; anything else is rejected loudly."""
    opt_in = PTQConfig(compute_dtype="float32")
    assert opt_in.compute_dtype == "float32"
    assert opt_in.np_compute_dtype == np.dtype(np.float32)
    with pytest.raises(ValueError):
        PTQConfig(compute_dtype="float16")
    with pytest.raises(ValueError):
        PTQConfig(compute_dtype="float64 ")
    # Serialization knob is likewise explicit and validated.
    assert PTQConfig().scale_dtype == "float16"
    assert PTQConfig(scale_dtype="float32").scale_dtype == "float32"
    with pytest.raises(ValueError):
        PTQConfig(scale_dtype="float64")


def test_scn_config_validation():
    """Out-of-range hyperparameters are construction errors, never silent."""
    with pytest.raises(ValueError):
        PTQConfig(rank=0)
    with pytest.raises(ValueError):
        PTQConfig(gamma=1.5)
    with pytest.raises(ValueError):
        PTQConfig(admm_max_iter=0)
    with pytest.raises(ValueError):
        PTQConfig(rho_start=0.0)
    with pytest.raises(ValueError):
        PTQConfig(rho_end=1e-4, rho_start=1e-2)
    with pytest.raises(ValueError):
        PTQConfig(temperature=0.0)
