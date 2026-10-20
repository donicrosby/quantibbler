"""End-to-end quality verdict for real-model production runs.

Change: ``qk-e2e-quality-gate``. This module is deliberately TORCH-FREE and
lives in the compressor package (not in ``tools/quantize_hf_model.py``) so
the verdict and sanitizer get real suite coverage in a lean environment —
the driver imports torch at module top, so helpers kept there would silently
skip. The sanitizer previously lived only in a REPL transcript; it must not
again.

The band
--------

Both edges are DERIVED from the pinned papers' own measurements, not chosen.
Anchors are [LB2] Sec. 5.2 Table 1 (Llama-3-8B, WikiText-2): STBLLM's
documented method collapse at 0.55 bpp (PPL 241.96) against the FP16
reference (PPL 6.10), and the functional envelope at the extreme end of the
budget range, LittleBit-2 at 0.1 bpp (PPL 23.74) against the same 6.10.

    COLLAPSE_LINE = ceil(241.96 / 6.10) = 40
    OK_LINE       = ceil(23.74 / 6.10)  = 4

Reaching the papers' own documented collapse for a method in this regime is
a failed run. The historical best ratio ever measured in this repo is 943.49
(grep ``artifacts/*/quantize_report.json``) — ~24x past the collapse line.
Under this gate every historical run classifies ``failed``; that is the
point.

Calibration-rule debt (declared in the change's design.md): the repo's
gate-calibration rule wants the band set from production healthy extremes,
and none exist — every production extreme is a failure. The band is
paper-anchored instead, and the owning change makes recalibration from the
first ``ok`` artifact a mandatory spec update.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

__all__ = [
    "COLLAPSE_LINE",
    "OK_LINE",
    "FAILED",
    "DEGRADED",
    "OK",
    "verdict",
    "exit_code",
    "json_safe",
    "dump_report",
    "provenance",
]

# [LB2] Sec. 5.2 Table 1 anchors (Llama-3-8B, WikiText-2).
_STBLLM_COLLAPSE_PPL = 241.96  # 0.55 bpp — documented method collapse
_LITTLEBIT2_FUNCTIONAL_PPL = 23.74  # 0.1 bpp — functional envelope
_FP16_REFERENCE_PPL = 6.10  # the FP16 baseline both are measured against

# Derived, never hand-tuned: ceil(241.96 / 6.10) and ceil(23.74 / 6.10).
# tests/test_qk_report.py asserts the arithmetic of both edges.
COLLAPSE_LINE: float = float(math.ceil(_STBLLM_COLLAPSE_PPL / _FP16_REFERENCE_PPL))
OK_LINE: float = float(math.ceil(_LITTLEBIT2_FUNCTIONAL_PPL / _FP16_REFERENCE_PPL))

FAILED = "failed"
DEGRADED = "degraded"
OK = "ok"


def verdict(ratio: Any) -> str:
    """Classify an end-to-end perplexity ratio into the three-value band.

    ``ok`` for ``ratio <= OK_LINE``, ``degraded`` for
    ``OK_LINE < ratio <= COLLAPSE_LINE``, ``failed`` above ``COLLAPSE_LINE``.

    The missing / non-finite check runs FIRST and is the only thing that can
    catch a NaN ratio: every comparison against NaN is False, so a bare
    ``ratio > COLLAPSE_LINE`` gate silently passes the exact shape that
    r137-v3/v4 wrote into their reports. A missing (``None``) or non-finite
    ratio is a measurement or numerics defect in the artifact path — there is
    no legitimate successful run that produces one.
    """
    try:
        finite = ratio is not None and math.isfinite(ratio)
    except TypeError:
        # Non-numeric junk is a missing measurement, not a good one.
        finite = False
    if not finite:
        return FAILED
    value = float(ratio)
    if value > COLLAPSE_LINE:
        return FAILED
    if value > OK_LINE:
        return DEGRADED
    return OK


def exit_code(status: str) -> int:
    """Process exit code for a verdict: 0 for ``ok``/``degraded``, else 1.

    The artifact and report are written regardless of verdict — the gate
    fails the *process*, never the forensic evidence.
    """
    return 0 if status in (OK, DEGRADED) else 1


def json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats with ``None``.

    Numpy scalars are converted first, through ``np.generic.item()``: an
    ``np.float32(np.inf)`` is not a Python ``float``, so it survived an
    ``isinstance(obj, float)`` check, reached ``json.dumps(...,
    allow_nan=False)`` and raised ``TypeError`` — no report at all on exactly
    the weirdest runs, which are the ones forensics exists for. ``np.int64``
    and friends are covered by the same conversion, since they are not
    JSON-serializable either.

    A bare ``NaN`` in the report is (a) not valid JSON — strict parsers
    reject it, and Python's own ``allow_nan=False`` refuses to emit it — and
    (b) silently defeats any downstream ``value > MAX`` check. Non-finite
    measurements are recorded as ``null`` instead, paired with a
    ``status: failed`` verdict as the durable trace.
    """
    if isinstance(obj, np.generic):  # numpy scalar -> Python scalar
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def dump_report(report: dict) -> tuple[str, bool]:
    """Serialize a report to strict JSON. Always returns a payload.

    Returns ``(payload, faithful)``. ``faithful`` is ``False`` only when the
    report could not be serialized as measured — in which case ``payload`` is
    the minimal failed report instead of nothing at all. A value the
    sanitizer did not recognize must not cost the verdict: the caller is
    required to fail the process whenever ``faithful`` is ``False``, so the
    durable trace and the exit code can never disagree.
    """
    try:
        return json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n", True
    except (TypeError, ValueError) as exc:
        fallback = {
            "status": FAILED,
            "report_error": f"{type(exc).__name__}: {exc}",
        }
        return json.dumps(fallback, indent=2, allow_nan=False) + "\n", False


def provenance(solver: str, scale_dtype: str) -> dict:
    """Report provenance fields derived from the ACTUAL run arguments.

    The driver used to hardcode ``regime.artifact`` as
    ``"qpk1: +-1 bitplanes + float16 scales"`` while ``--scale-dtype`` had
    already defaulted to float32, so every forensic report lied about its own
    artifact, and ``solver`` (the only field distinguishing the r512-svd line
    from the r137 line) was never recorded at all. These strings are emitted
    from the arguments rather than from a constant, so a report cannot
    disagree with the run it describes.
    """
    return {
        "solver": solver,
        "scale_dtype": scale_dtype,
        "regime_artifact": f"qpk1: +-1 bitplanes + {scale_dtype} scales",
        "dtype_note": (f"solver {solver} (float64 compute) / scales {scale_dtype} / artifact qpk1"),
    }
