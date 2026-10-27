"""qk-pipeline-dtype: the ACCELERATOR cells of the dtype x device matrix.

This is the GPU hardware lane's module (kanban card ``t_f8e3b1d8``). The CPU
cells of the same matrix, and the per-dtype tolerance classes they assert, live
in ``tests/ptq/test_pipeline_dtype_generic.py``; this file carries what cannot
run without an accelerator:

* ``admm_parity_per_dtype`` GPU legs — the three accelerator cells
  (bf16 / float32 / float64 x GPU) on the SAME fixture and the SAME reachable
  metric as the CPU cells, against the float64 NumPy reference.
* ``default_numerics_bf16`` — the bf16 DEFAULT end to end on the accelerator,
  on the tiny/toy model path (the spec's verification scale, not a fire).
* ``per_stage_device_from_module`` / ``per_stage_timing_recorded`` — the
  accelerator stages report ``cuda:0`` and their own measured wall seconds.
* the GPU memory cap (tasks.md 5.4, deferred here): the fraction cap, the
  override inputs, the unified-memory (APU) detection and the recorded
  ``max_memory_reserved()``.

WHY ``integration``: CI's full-python job asserts ZERO skips on the default
tier (``tools/ci/pytest_summary.py --expect-skipped 0``), and an accelerator-only
leg can only skip on a GPU-less runner. Carrying ``@pytest.mark.integration``
DESELECTS these legs from the default tier (``addopts = "-m 'not integration'"``)
instead of counting them as skips there, while ``pytest -m integration`` — the
GPU lane — runs them. The ``skipif`` stays so the requirement is still REPORTED,
with its reason, wherever the lane runs without an accelerator.

Lane invocation (host-side container discipline, breadcrumbed before launch):

    docker run --rm --memory 16g --memory-swap 16g \\
      --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" \\
      --network none -v <worktree>:/repo:ro -e PYTHONPATH=/repo/src -w /repo \\
      --entrypoint python quantibbler-ptq:rocm-official-v6 \\
      -m pytest -m integration tests/ptq/test_pipeline_dtype_gpu.py -v
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from quantibbler.ptq.config import PTQConfig
from quantibbler.ptq.lb_admm import lb_admm

#: The device the torch backend is told to execute on. The backend's device
#: comes from ``QUANTIBBLER_DEVICE`` (resolved at the solve site, never at config
#: construction), so every accelerator leg sets it explicitly instead of trusting
#: an unset env on a host that happens to have a GPU.
ACCEL_DEVICE = "cuda"

HAVE_ACCEL = torch.cuda.is_available()
requires_accel = pytest.mark.skipif(
    not HAVE_ACCEL,
    reason=(
        f"gpu cell: no accelerator on this host (torch {torch.__version__}); the "
        "accelerator cells run on the GPU hardware lane card t_f8e3b1d8 "
        "(quantibbler-ptq:rocm-official-v6, gfx1151, --device /dev/kfd --device /dev/dri)"
    ),
)
accelerator_leg = pytest.mark.integration

#: Per-dtype classes — the DECLARED format-ULP bounds (design.md "Per-dtype
#: tolerances"), identical to the CPU cells'. Never widened per device: a GPU
#: cell that needed a looser bound than its CPU sibling would be a finding, not
#: a knob.
_DTYPE_CELLS = (("bf16", 1e-2), ("float32", 1e-4), ("float64", 1e-12))


def _first_update_rel(arm, ref) -> float:
    """Relative difference of the FIRST ADMM update residual vs the reference.

    The reachable per-dtype metric: design.md C1 measured that the
    CONVERGED-output cosine measures basin selection (the SVID proxy is a
    sign-quantized rank-1 projection, so two backends select among near-equal
    discrete optima) and therefore cannot carry a per-dtype bound. The early
    trajectory is where the accumulation contract is observable, and it is what
    the CPU cells assert too.
    """
    ref_first = float(ref.trace.residuals[0])
    arm_first = float(arm.trace.residuals[0])
    return abs(arm_first - ref_first) / abs(ref_first)


@pytest.mark.parametrize("compute_dtype,class_bound", _DTYPE_CELLS)
@accelerator_leg
@requires_accel
def test_scn_admm_parity_gpu_cell(compute_dtype, class_bound, monkeypatch):
    """The GPU cell of ``admm_parity_per_dtype``: same fixture, same metric.

    The arm executes on the accelerator and the returned latents must be device
    tensors — an arm that silently ran on the host while the assertion compared
    numbers would be a green cell for a lane that never left CPU.
    """
    monkeypatch.setenv("QUANTIBBLER_DEVICE", ACCEL_DEVICE)
    rng = np.random.default_rng(11)
    w = rng.standard_normal((32, 24))
    budget = 900
    ref = lb_admm(
        w,
        PTQConfig(rank=4, solver_backend="numpy", compute_dtype="float64", admm_max_iter=budget),
    )
    assert ref.converged, "the float64 NumPy reference must converge within the cell budget"
    res = lb_admm(
        torch.as_tensor(w, device=ACCEL_DEVICE),
        PTQConfig(
            rank=4, solver_backend="torch", compute_dtype=compute_dtype, admm_max_iter=budget
        ),
    )
    assert res.converged, f"the {compute_dtype} accelerator arm did not converge within {budget}"
    assert res.U.device.type == "cuda", (
        f"{compute_dtype}: the arm did not execute on the accelerator"
    )
    assert str(res.U.dtype) == f"torch.{_torch_dtype_name(compute_dtype)}", res.U.dtype
    assert res.U.shape[0] == w.shape[0]

    rel = _first_update_rel(res, ref)
    assert rel <= class_bound, (
        f"{compute_dtype} accelerator cell: first-update residual differs from the float64 "
        f"reference by {rel:.3e}, above the declared class {class_bound:.0e}"
    )


def _torch_dtype_name(spec: str) -> str:
    from quantibbler.ptq.accumulation import TORCH_DTYPE_SPECS

    return TORCH_DTYPE_SPECS[spec]


@accelerator_leg
@requires_accel
def test_scn_default_numerics_bf16_gpu_end_to_end(monkeypatch):
    """``default_numerics_bf16`` on the accelerator: the tiny model path, no fire.

    Mirrors ``test_bf16_default_end_to_end_on_cpu`` but with the model and the
    calibration batches on the device and the torch backend resolving there; the
    stage records must name ``cuda:0`` for the accelerator stages, otherwise the
    "end-to-end on GPU" claim would rest on a device nobody checked.
    """
    from quantibbler.ptq.api import QuantizedModel, quantize_model
    from quantibbler.ptq.e2e import TinyGateModel, build_gate_batches, gate_config

    monkeypatch.setenv("QUANTIBBLER_DEVICE", ACCEL_DEVICE)
    cfg = dataclasses.replace(
        gate_config(seed=0), solver_backend="torch", compute_dtype="bf16", admm_max_iter=120
    )
    model = TinyGateModel(seed=cfg.seed).to(ACCEL_DEVICE)
    batches = [b.to(ACCEL_DEVICE) for b in build_gate_batches(seed=cfg.seed)]
    events: list[tuple[str, dict]] = []
    qm = quantize_model(model, batches, cfg, recorder=lambda e, **info: events.append((e, info)))

    blobs = qm.serialize()
    assert blobs, "bf16 GPU e2e produced no artifact blobs"
    assert qm.layers, "bf16 GPU e2e produced no layer states"
    assert all(st.bias is not None for st in qm.layers.values())

    by_event = dict(events)
    for stage in ("capture", "lb_admm", "sign_extract_pack", "ste", "tune_scales_kd"):
        assert stage in by_event, f"stage {stage!r} did not record a row on the accelerator"
    for stage in ("ste", "tune_scales_kd"):
        assert by_event[stage]["device"].startswith("cuda"), (stage, by_event[stage].get("device"))

    reloaded = QuantizedModel.deserialize(blobs, config=cfg)
    reloaded.attach_to(TinyGateModel(seed=cfg.seed).to(ACCEL_DEVICE))
    assert reloaded.effective_bpw() < 1.0


@accelerator_leg
@requires_accel
def test_scn_per_stage_timing_and_device_on_gpu(monkeypatch):
    """GPU per-stage timing rows + the device each stage actually ran on.

    Same block fixture as the CPU timing leg, moved to the accelerator: every
    stage that emits a timing row must carry a MEASURED wall time (a missing
    reading must never render as a zero-cost claim), and the accelerator stages
    must name the device the module's own parameters are on.
    """
    from test_block_recon import _calib, _fixed_cot, _TinyBlock

    from quantibbler import qk_report
    from quantibbler.ptq.block_recon import reconstruct_block

    monkeypatch.setenv("QUANTIBBLER_DEVICE", ACCEL_DEVICE)
    block = _TinyBlock(8, seed=3).to(ACCEL_DEVICE)
    batches = [b.to(ACCEL_DEVICE) for b in _calib(8)]
    events: list[tuple[str, dict]] = []
    reconstruct_block(
        block,
        batches,
        PTQConfig(solver_backend="torch", compute_dtype="bf16", rank=4, admm_max_iter=60),
        cotangent=_fixed_cot(block, batches),
        recorder=lambda e, **info: events.append((e, info)),
    )
    own = str(next(block.parameters()).device)
    assert own.startswith("cuda"), own
    by_event = dict(events)
    for stage in ("capture", "lb_admm", "sign_extract_pack"):
        assert stage in by_event, f"stage {stage!r} did not record a row"
        elapsed = by_event[stage].get("elapsed_s")
        assert isinstance(elapsed, (int, float)) and elapsed >= 0.0, (
            f"stage {stage!r} recorded no wall time (got {elapsed!r})"
        )
    ste = by_event["ste"]
    assert ste["device"] == own
    for name, per in ste["per_linear"].items():
        assert per["device"] == own, name
    mapped = qk_report.stage_map(events)
    for stage in ("capture", "lb_admm", "ste"):
        assert mapped[stage]["row_time_s"] is not None and mapped[stage]["row_time_s"] >= 0.0, stage


@accelerator_leg
@requires_accel
def test_scn_gpu_memory_cap_recorded():
    """tasks.md 5.4: the cap, its overrides, the APU bound, and the peak.

    The deferred clause was only observable with an accelerator: the cap is
    applied at startup, the APU (unified-memory) detection decides whether the
    absolute-GB bound is the one that protects co-tenants, and
    ``max_memory_reserved()`` is meaningless without a device to reserve on.
    """
    from quantibbler.good_guest import (
        DEFAULT_GPU_FRACTION,
        apply_gpu_memory_cap,
        is_unified_memory_device,
        plan_from_argv,
        resolve_memory_fraction,
    )

    assert HAVE_ACCEL
    props = torch.cuda.get_device_properties(0)
    total = int(props.total_memory)
    assert total > 0

    plan = plan_from_argv([])
    applied = apply_gpu_memory_cap(plan)
    assert applied is not None and 0.0 < applied <= DEFAULT_GPU_FRACTION, applied

    # The absolute-GB override is TIGHTER than the bare fraction on a unified
    # device: that narrowing is the whole reason the APU clause exists.
    gb = 8.0
    narrowed = resolve_memory_fraction(DEFAULT_GPU_FRACTION, gb, total)
    assert narrowed < DEFAULT_GPU_FRACTION, (narrowed, DEFAULT_GPU_FRACTION)

    # Recorded, not assumed: the arch decides (gfx11* is an APU here).
    unified = is_unified_memory_device()
    assert isinstance(unified, bool)

    torch.cuda.reset_peak_memory_stats()
    hold = torch.zeros(1 << 22, device=ACCEL_DEVICE)  # noqa: F841 - keeps the block live
    torch.cuda.synchronize()
    peak = int(torch.cuda.max_memory_reserved())
    assert peak > 0, "max_memory_reserved() recorded nothing under a live allocation"
    assert peak <= applied * total, (peak, applied, total)
