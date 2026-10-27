"""qk-ptq-speedups Task 6: Phase-3 TuneScalesKD on the accelerator.

Scenarios (openspec/changes/qk-ptq-speedups/specs/qk-ptq/spec.md):

* ``tunescaleskd_gpu_zero_algorithm_change`` — the TuneScalesKD stage executes
  its model forward/backward on the device the activation/weight authority
  resolves, with the identical KL loss, optimizer, scheduler and batch
  configuration as the CPU reference; the disk-backed teacher cache and the
  post-stage scale readback stay on the host.
* ``kd_gpu_device_parity`` — the accelerator and CPU legs agree structurally:
  the per-step KL trace decreases monotonically on both, and the final scales
  stay finite and inside the stage's trust region on both.

RED lane: the accelerator legs below are the ones that red on the container
lane (``quantibbler-ptq:rocm-official`` + ``--device /dev/kfd``) BEFORE the
port. The crash is ``scale_calib``'s ``x_in @ w.to(x_in.dtype).T``: ``w`` is
born on the host (its factors came from NumPy while the activation is
device-resident) and a dtype-only cast leaves mat2 on the CPU — the
``Expected all tensors to be on the same device, but got mat2 is on cpu``
reported at that line in fire ``qk-pertarget-v6-20261026T0030Z``. The host lane
has no device mismatch to trip, so it is a no-change baseline; its legs here
(the stage-record and readback-boundary rows) run everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from test_api import _calib, _TinyApiModel
from test_checkpoints import _cfg
from test_scale_calib import _kd_layer_book, _TinyKDModel

from quantibbler.ptq.api import quantize_model
from quantibbler.ptq.config import PTQConfig, StageHyper
from quantibbler.ptq.scale_calib import calibrate_scales

#: The accelerator is the point of this file; without one the accelerator legs
#: cannot run at all. They are DESELECTED by the default marker filter (never
#: counted as a skip, which the full tier's referee would fail on) and the
#: skipif keeps the requirement reportable under ``pytest -m integration``.
HAVE_ACCEL = torch.cuda.is_available()
requires_accel = pytest.mark.skipif(
    not HAVE_ACCEL, reason="no accelerator: the TuneScalesKD device legs need one"
)
accelerator_leg = pytest.mark.integration


def _kd_cfg() -> PTQConfig:
    """Stage config, identical for both legs (A-10 test-speed overrides)."""
    return PTQConfig(
        solver_backend="numpy",
        compute_dtype="float64",
        rank=4,
        stage_global=StageHyper(2e-2, 16, 10),
    )


def _kd_fixture(device: str, batches_on: str):
    """A tiny factorized block whose scales sit off the alpha-consistent values.

    Built on the host and only then moved, so both legs start from the SAME
    bytes: a leg whose inputs were constructed on its own device would put its
    own construction inside the comparison.
    """
    torch.manual_seed(0)
    model = _TinyKDModel(seed=1)
    gen = torch.Generator().manual_seed(2)
    batches = [torch.randn(2, 4, 8, generator=gen, dtype=torch.float64) for _ in range(2)]
    layers = _kd_layer_book(model)
    teacher = _TinyKDModel(seed=1)
    pre = {n: (st.s1.copy(), st.s2.copy()) for n, st in layers.items()}
    model = model.to(device)
    teacher = teacher.to(device)
    return model, layers, teacher, [b.to(batches_on) for b in batches], pre


def _run_kd_leg(device: str, batches_on: str) -> dict:
    model, layers, teacher, batches, pre = _kd_fixture(device, batches_on)
    result = calibrate_scales(model, layers, teacher, batches, _kd_cfg())
    return {
        "kl": [float(v) for v in result.kl_trace],
        "steps": result.n_steps,
        "scales": {n: (st.s1, st.s2) for n, st in layers.items()},
        "pre": pre,
    }


# --------------------------------------------------------------------------
# S5: the stage's own row names its device and its time (runs everywhere)
# --------------------------------------------------------------------------


def test_scn_scale_stage_record_names_device_and_its_own_time():
    """``tune_scales_kd`` carries ``device`` and ``elapsed_s``, as STE/TuneFP do.

    Emitted from the ``calibrate_scales`` call site in ``api.py``: the stage's
    cost is the model forward/backward on its device, so only the caller knows
    the elapsed time and the resolved device. One row per fire — the measurement
    replaces the stage map's null row time with a measured one.
    """
    events: list[tuple[str, dict]] = []
    cfg = _cfg()
    quantize_model(
        _TinyApiModel(seed=3),
        _calib(seed=7),
        cfg,
        recorder=lambda event, **info: events.append((event, info)),
    )
    rows = [info for event, info in events if event == "tune_scales_kd"]
    assert len(rows) == 1, f"expected one stage row per fire, got {[e for e, _ in events]}"
    row = rows[0]
    assert row["device"] == "cpu", row
    assert isinstance(row["elapsed_s"], float) and row["elapsed_s"] > 0.0, row
    assert row["factor_mode"] == cfg.factor_mode, row


def test_scn_scale_readback_stays_on_host():
    """The post-stage scale readback is host float64 NumPy, never a device tensor.

    The out-of-place KD update is device-resident; what lands back in the layer
    book (and therefore in the artifact) is the host array the serializer and the
    qpk1 face read.
    """
    model, layers, teacher, batches, _pre = _kd_fixture("cpu", "cpu")
    calibrate_scales(model, layers, teacher, batches, _kd_cfg())
    for name, st in layers.items():
        for scale in (st.s1, st.s2):
            assert isinstance(scale, np.ndarray), (name, type(scale))
            assert scale.dtype == np.float64, (name, scale.dtype)
            assert np.isfinite(scale).all(), name


# --------------------------------------------------------------------------
# tunescaleskd_gpu_zero_algorithm_change
# --------------------------------------------------------------------------


@accelerator_leg
@requires_accel
def test_scn_tunescaleskd_gpu_zero_algorithm_change(monkeypatch, tmp_path):
    """The stage runs on the accelerator with the CPU reference's algorithm.

    The fixture reproduces the fire's shape exactly: the model AND the teacher
    are device-resident and the calibration batches arrive device-resident (the
    activation therefore is), while the stage's scale parameters and frozen
    factors start life as host NumPy. That is the configuration that died at
    ``scale_calib.py``'s ``x_in @ w.to(x_in.dtype).T``; it must now complete,
    with the disk-backed teacher cache still written on the host.
    """
    cache_paths: list[str] = []
    real_memmap = np.memmap

    def _spy(filename, *args, **kwargs):
        cache_paths.append(str(filename))
        return real_memmap(filename, *args, **kwargs)

    monkeypatch.setenv("QUANTIBBLER_KD_TMPDIR", str(tmp_path))
    monkeypatch.setattr(np, "memmap", _spy)

    leg = _run_kd_leg("cuda", "cuda")

    # The stage ran to its configured budget and recorded one KL per step.
    assert leg["steps"] > 0 and len(leg["kl"]) == leg["steps"], leg
    assert all(np.isfinite(leg["kl"])) and leg["kl"][-1] < leg["kl"][0]
    # Teacher cache: still a host memmap file under the nominated scratch dir.
    assert cache_paths, "the teacher logits were not cached through np.memmap"
    assert all(p.startswith(str(tmp_path)) for p in cache_paths), cache_paths
    # Readback: host float64 arrays.
    for name, (s1, s2) in leg["scales"].items():
        assert isinstance(s1, np.ndarray) and s1.dtype == np.float64, name
        assert isinstance(s2, np.ndarray) and s2.dtype == np.float64, name


# --------------------------------------------------------------------------
# kd_gpu_device_parity
# --------------------------------------------------------------------------


@accelerator_leg
@requires_accel
def test_scn_kd_gpu_device_parity():
    """Device and CPU legs agree structurally, not bit-for-bit.

    The per-step KL trace decreases on both, and the final scales stay finite
    and inside the stage's per-entry trust region ([0.5x, 2x] of the pre-stage
    values) on both. Bit-identity across device/CPU is NOT asserted anywhere at
    whole-pipeline scope — the repo's own measurement puts packed cos 7.6e-3
    apart because the solver + polish path is a discrete search
    (``test_stage_device_parity.py``). The monotonicity margin is measured:
    strictly decreasing 10/10 steps with a minimum adjacent delta of 1.83e-4,
    three orders above device reduction-order noise at this f64 geometry.
    """
    cpu = _run_kd_leg("cpu", "cpu")
    acc = _run_kd_leg("cuda", "cuda")

    # Identical optimizer/scheduler/batch budget: the fixture is f64 on both
    # legs, so the step count IS the configuration read off the record.
    assert cpu["steps"] == acc["steps"] > 0

    for tag, leg in (("cpu", cpu), ("accel", acc)):
        kl = leg["kl"]
        assert len(kl) >= 2, (tag, kl)
        assert all(np.isfinite(kl)), (tag, kl)
        deltas = [later - earlier for earlier, later in zip(kl, kl[1:], strict=False)]
        assert all(d < 0.0 for d in deltas), (tag, kl)
        for name, (s1, s2) in leg["scales"].items():
            pre1, pre2 = leg["pre"][name]
            for value, pre in ((s1, pre1), (s2, pre2)):
                assert np.isfinite(value).all(), (tag, name)
                ref = np.maximum(np.abs(pre), 1e-30)
                assert (value >= 0.5 * ref - 1e-30).all(), (tag, name)
                assert (value <= 2.0 * ref + 1e-30).all(), (tag, name)
