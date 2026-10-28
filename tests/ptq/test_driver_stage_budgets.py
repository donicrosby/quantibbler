"""qk-cli-sane: the driver's regime->stage-budget switch, as a unit.

Scenario (spec delta ``specs/qk-ptq``): ``paper_regime_stage_budgets_are_app_c``.

``quantized_config`` derives the stage budgets from the DECLARED config: a run
whose config matches the paper regime gets the [NQ] App. C triple verbatim, every
other run keeps the gate-shaped triple (which turns TuneFP into a single step).
This is one of the four runtime gates re-keyed off the config-match conjunction,
and it is the only one with no executing test on any host: the sibling pin
(``test_ste_per_target.py::test_scn_joint_budget_is_the_appendix_c_regime_once_per_block``)
asserts the arithmetic on its OWN ``StageHyper``, and the one e2e regime leg skips
without CUDA.

The switch reads the DECLARED ``device`` field, never the host's accelerator
state, so this leg is a pure unit test that runs on a CPU-only box — the third
test below pins exactly that (the host's accelerator predicate is poisoned).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from quantibbler.ptq.config import (
    PTQConfig,
    StageHyper,
    config_matches_paper_regime,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import quantize_hf_model as qhf  # noqa: E402

# [NQ] App. C, the regime's stage schedule (lr, batch_size, epochs).
_APP_C = (
    StageHyper(1e-4, 4, 8),
    StageHyper(1e-5, 1, 8),
    StageHyper(1e-6, 1, 8),
)
# The gate-shaped triple: every non-regime run's schedule.
_GATE_SHAPED = (
    StageHyper(1e-4, 4, 0),
    StageHyper(2e-3, 4, 3),
    StageHyper(1e-3, 4, 4),
)

# Every declared field but one at its paper-regime value.
_REGIME = {
    "seed": 0,
    "rank": 4,
    "compute_dtype": "bf16",
    "scale_dtype": "float32",
    "max_admm_iter": 400,
    "solver": "admm",
    "device": "cuda",
    "factor_mode": "binary",
    "ablation": "none",
    "layer_scope": "all",
}


def _config(**overrides: object) -> PTQConfig:
    fields = {**_REGIME, **overrides}
    return qhf.quantized_config(SimpleNamespace(**fields))


def _budgets(cfg: PTQConfig) -> tuple[StageHyper, StageHyper, StageHyper]:
    return (cfg.stage_pre, cfg.stage_latent, cfg.stage_global)


def test_scn_paper_regime_stage_budgets_are_app_c():
    """A config matching the regime gets the App. C triple, field for field."""
    cfg = _config()
    assert config_matches_paper_regime(cfg) is True, (
        "the fixture is not the paper regime, so it cannot pin the App. C arm"
    )
    assert _budgets(cfg) == _APP_C, _budgets(cfg)


def test_a_non_regime_run_keeps_the_gate_shaped_triple():
    """Every way of leaving the regime keeps the gate-shaped schedules.

    One deviation per declared field: the switch is a function of the
    conjunction, so a single non-paper value must be enough to keep the gate
    shape — and the sibling assertions prove the triple is not merely the
    untouched constructor default.
    """
    deviations = (
        {"solver": "svd"},
        {"compute_dtype": "float32"},
        {"factor_mode": "fp"},
        {"ablation": "untuned-signs"},
        {"layer_scope": "self_attn"},
        {"max_admm_iter": 2},
        {"device": "cpu"},
    )
    for deviation in deviations:
        cfg = _config(**deviation)
        assert config_matches_paper_regime(cfg) is False, deviation
        assert _budgets(cfg) == _GATE_SHAPED, (deviation, _budgets(cfg))


def test_the_switch_reads_the_declared_device_not_the_host(monkeypatch):  # noqa: ANN001
    """No accelerator is needed: the regime arm keys on the DECLARED device.

    Poison the host's accelerator predicate. A switch that resolved the device
    itself (instead of reading the declared ``device`` field) would raise here;
    this leg is what makes the unit test runnable on a CPU-only box.
    """

    def _boom(*args: object, **kwargs: object) -> bool:
        raise AssertionError(
            "the stage-budget switch consulted the host's accelerator state instead of "
            "the declared device field"
        )

    monkeypatch.setattr(torch.cuda, "is_available", _boom)
    cfg = _config(device="cuda")
    assert config_matches_paper_regime(cfg) is True
    assert _budgets(cfg) == _APP_C, _budgets(cfg)
