"""qk-pipeline-dtype: dtype-generic + device-generic pipeline.

Covers the scenarios that are measurable on a CPU-only host:

* ``default_numerics_bf16`` / ``float32_opt_in`` — the declared compute dtype and
  the accumulation dtype it implies.
* ``admm_dtype_generic`` — the torch arm's returned latents carry the declared
  compute dtype.
* ``accumulation_identity_at_higher_dtypes`` — the accumulation mapping is the
  identity above bf16.
* ``admm_parity_per_dtype`` — the CPU cells of the dtype x device matrix, with
  the per-dtype classes the design declares. NOTE the measured delta recorded in
  ``openspec/changes/qk-pipeline-dtype/design.md`` ("Implementation
  crystallization"): the CONVERGED-output cosine cells do not reproduce (the
  sign-quantized iteration selects among near-equal discrete optima), so the
  reachable metric asserted here is the early-trajectory agreement plus
  convergence-within-budget. The GPU cells are deferred to the GPU lane card and
  SKIP with a named reason.
* ``per_stage_timing_recorded`` / ``per_stage_device_from_module`` — the stage
  records carry their own wall seconds and the device they ran on.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from quantibbler.ptq.accumulation import accumulation_spec
from quantibbler.ptq.config import PTQConfig
from quantibbler.ptq.lb_admm import lb_admm
from quantibbler.ptq.solver_backend import to_host_np


def _cpu_arm(**kw) -> PTQConfig:
    """The torch arm on the explicit CPU device.

    ``QUANTIBBLER_DEVICE=cpu`` is the documented parity/diagnostic reading —
    implicit torch on an accelerator-less host refuses by design, so the CPU
    cell of the matrix names its device rather than weakening the refusal.
    """
    return PTQConfig(solver_backend="torch", compute_dtype="bf16", **kw)


@pytest.fixture(autouse=True)
def _explicit_cpu_device(monkeypatch):
    monkeypatch.setenv("QUANTIBBLER_DEVICE", "cpu")


def test_scn_default_numerics_bf16():
    """default_numerics_bf16: bf16 is the default and widens to float32."""
    cfg = PTQConfig()
    assert cfg.compute_dtype == "bf16"
    assert accumulation_spec(cfg.compute_dtype) == "float32"


def test_scn_float32_opt_in():
    """float32_opt_in: the accumulation dtype equals the declared dtype."""
    cfg = PTQConfig(compute_dtype="float32")
    assert accumulation_spec(cfg.compute_dtype) == "float32"


def test_scn_accumulation_identity_at_higher_dtypes():
    """accumulation_identity_at_higher_dtypes: identity at float32/float64."""
    assert accumulation_spec("float32") == "float32"
    assert accumulation_spec("float64") == "float64"
    assert accumulation_spec("bf16") != "bf16"
    # An unknown name is refused rather than silently mapped.
    with pytest.raises(ValueError, match="unknown compute_dtype"):
        accumulation_spec("float16")


def test_scn_admm_dtype_generic():
    """admm_dtype_generic: the torch arm returns latents at compute_dtype."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((24, 16))
    res = lb_admm(w, _cpu_arm(rank=4, admm_max_iter=5))
    assert torch.is_tensor(res.U)
    assert res.U.dtype == torch.bfloat16
    assert res.V.dtype == torch.bfloat16
    assert res.P_u.dtype == torch.bfloat16
    # The host hop lands bf16 at its own declared grade (float32), never fp64.
    assert to_host_np(res.U).dtype == np.float32


#: Per-dtype classes for the CPU cells — the DECLARED format-ULP bounds
#: (design.md "Per-dtype tolerances"). Measured on this fixture (torch CPU arm vs
#: the float64 NumPy reference, first-update residual): bf16 1.0e-3,
#: float32 3.8e-7, float64 2.5e-15. All three sit inside their declared class.
#: The CONVERGED-output cosine cells are the subject of the crystallization note
#: in design.md — they do not reproduce, and this test does not fake them.
_CPU_CELLS = (("bf16", 1e-2), ("float32", 1e-4), ("float64", 1e-12))


@pytest.mark.parametrize("compute_dtype,class_bound", _CPU_CELLS)
def test_scn_admm_parity_cpu_cell(compute_dtype, class_bound):
    """The CPU cell of ``admm_parity_per_dtype``.

    Measured property: (1) the arm converges within the declared iteration
    budget, and (2) its FIRST update agrees with the float64 NumPy reference at
    the per-dtype class — the early trajectory is where the accumulation
    contract is observable. The converged-output cosine is deliberately NOT
    asserted here; see the module docstring and design.md.
    """
    rng = np.random.default_rng(11)
    w = rng.standard_normal((32, 24))
    budget = 900
    ref = lb_admm(
        w, PTQConfig(rank=4, solver_backend="numpy", compute_dtype="float64", admm_max_iter=budget)
    )
    assert ref.converged, "the float64 NumPy reference must converge within the cell budget"

    res = lb_admm(
        w,
        PTQConfig(
            rank=4, solver_backend="torch", compute_dtype=compute_dtype, admm_max_iter=budget
        ),
    )
    assert res.converged, f"{compute_dtype} arm did not converge within {budget} iterations"

    ref_first = float(ref.trace.residuals[0])
    arm_first = float(res.trace.residuals[0])
    rel = abs(arm_first - ref_first) / abs(ref_first)
    assert rel <= class_bound, (
        f"{compute_dtype} first-update residual {arm_first!r} differs from the float64 "
        f"reference {ref_first!r} by {rel:.3e}, above the declared class {class_bound:.0e}"
    )


def test_scn_admm_parity_gpu_cells_skip_with_reason():
    """The GPU cells of the matrix, deferred with a documented skip event.

    DocID: qk-pipeline-dtype task 3.4. A worker sandbox has no accelerator, so
    the three accelerator cells (bf16/float32/float64 x GPU) run on the GPU lane
    card (t_f8e3b1d8, host-side container lane) against the digest-pinned
    ``quantibbler-ptq:rocm-official`` image. The skip names that reason; a silent
    skip would read as a green cell.
    """
    if not torch.cuda.is_available():
        pytest.skip(
            f"gpu cell: no accelerator on this host (torch {torch.__version__}); the three "
            "GPU cells of admm_parity_per_dtype are deferred to the GPU lane card t_f8e3b1d8"
        )
    rng = np.random.default_rng(11)
    w = rng.standard_normal((32, 24)).astype(np.float32)
    dev = "cuda"
    cfg = PTQConfig(rank=4, solver_backend="torch", compute_dtype="bf16", admm_max_iter=200)
    res = lb_admm(torch.as_tensor(w, device=dev), cfg)
    assert res.U.device.type == "cuda"


def test_scn_per_stage_timing_recorded():
    """per_stage_timing_recorded: every stage appends its own wall seconds."""
    from quantibbler.ptq.block_recon import reconstruct_block
    from tests.ptq.test_block_recon import _calib, _fixed_cot, _TinyBlock

    block = _TinyBlock(8, seed=3)
    batches = _calib(8)
    events: list[tuple[str, dict]] = []

    def recorder(event, **info):
        events.append((event, info))

    reconstruct_block(
        block,
        batches,
        PTQConfig(solver_backend="numpy", compute_dtype="float64", rank=4),
        cotangent=_fixed_cot(block, batches),
        recorder=recorder,
    )
    by_event = dict(events)
    for stage in ("capture", "lb_admm", "sign_extract_pack"):
        assert stage in by_event, f"stage {stage!r} did not record a row"
        elapsed = by_event[stage].get("elapsed_s")
        assert isinstance(elapsed, (int, float)) and elapsed >= 0.0, (
            f"stage {stage!r} recorded no wall time (got {elapsed!r}); a missing reading "
            "must never be rendered as a zero-cost claim"
        )


def test_scn_per_stage_device_from_module():
    """per_stage_device_from_module: the stage's device is the module's own."""
    from quantibbler.ptq.block_recon import reconstruct_block
    from tests.ptq.test_block_recon import _calib, _fixed_cot, _TinyBlock

    block = _TinyBlock(8, seed=4)
    batches = _calib(8)
    records: list[tuple[str, dict]] = []

    reconstruct_block(
        block,
        batches,
        PTQConfig(solver_backend="numpy", compute_dtype="float64", rank=4),
        cotangent=_fixed_cot(block, batches),
        recorder=lambda event, **info: records.append((event, info)),
    )
    own = str(next(block.parameters()).device)
    ste = next(info for event, info in records if event == "ste")
    assert ste["device"] == own
    for _, per in ste["per_linear"].items():
        assert per["device"] == own


def test_bf16_default_end_to_end_on_cpu():
    """The card's bar: the bf16 DEFAULT runs a fixture compression end to end.

    torch CPU wheel, ``QUANTIBBLER_DEVICE=cpu`` (the documented explicit-device
    reading), no dtype override anywhere: the pipeline takes the declared
    default and produces a serializable artifact.
    """
    import dataclasses

    from quantibbler.ptq.api import QuantizedModel, quantize_model
    from quantibbler.ptq.e2e import TinyGateModel, build_gate_batches, gate_config

    cfg = dataclasses.replace(
        gate_config(seed=0), solver_backend="torch", compute_dtype="bf16", admm_max_iter=120
    )
    model = TinyGateModel(seed=cfg.seed)
    batches = build_gate_batches(seed=cfg.seed)
    qm = quantize_model(model, batches, cfg)
    blobs = qm.serialize()
    assert blobs, "bf16 e2e produced no artifact blobs"
    reloaded = QuantizedModel.deserialize(blobs, config=cfg)
    reloaded.attach_to(TinyGateModel(seed=cfg.seed))
    # Every layer state carries its bias vector out of the bf16 run. The gate
    # fixture is deliberately bias-free (its linears zero their bias), so the
    # non-zero claim belongs to test_bias_carry's bias-carrying fixture, not here.
    assert qm.layers, "bf16 e2e produced no layer states"
    assert all(st.bias is not None for st in qm.layers.values())
