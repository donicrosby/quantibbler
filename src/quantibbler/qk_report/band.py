"""The ok/degraded/failed band: the paper-derived lines and the strict JSON serializer."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

# [LB2] Sec. 5.2 Table 1 anchors (Llama-3-8B, WikiText-2).
_STBLLM_COLLAPSE_PPL = 241.96  # 0.55 bpp — documented method collapse
_LITTLEBIT2_FUNCTIONAL_PPL = 23.74  # 0.1 bpp — functional envelope
_FP16_REFERENCE_PPL = 6.10  # the FP16 baseline both are measured against

# Derived, never hand-tuned: ceil(241.96 / 6.10) and ceil(23.74 / 6.10).
# tests/test_qk_report.py asserts the arithmetic of both edges.
COLLAPSE_LINE: float = float(math.ceil(_STBLLM_COLLAPSE_PPL / _FP16_REFERENCE_PPL))
OK_LINE: float = float(math.ceil(_LITTLEBIT2_FUNCTIONAL_PPL / _FP16_REFERENCE_PPL))

# Minimum relative reduction of a stage's own objective for the stage to count
# as having done work: a record whose ``residual_after`` is not at least this
# much below its ``residual_before`` is an inert stage, however many steps it
# booked. One constant, shared by the report gate and the stage tests — a
# second threshold anywhere would be a hand-tuned re-pin (the same rule
# ``COLLAPSE_LINE`` already carries).
#
# The measured pair this exists to reject: a recorded pair
# ``1.9e-04 -> 1.9e-04`` on a full TuneFP budget — display-identical, i.e. no
# measurable work — and every gate in the pipeline passed it. Real runs clear
# it by two orders of magnitude (a run's TuneFP moved ``0.3184 -> 0.1973``,
# 38%).
STEPS_EFFECT_EPS: float = 1e-3

FAILED = "failed"
DEGRADED = "degraded"
OK = "ok"


def verdict(ratio: Any) -> str:
    """Classify an end-to-end perplexity ratio into the three-value band.

    ``ok`` for ``ratio <= OK_LINE``, ``degraded`` for
    ``OK_LINE < ratio <= COLLAPSE_LINE``, ``failed`` above ``COLLAPSE_LINE``.

    The missing / non-finite check runs FIRST and is the only thing that can
    catch a NaN ratio: every comparison against NaN is False, so a bare
    ``ratio > COLLAPSE_LINE`` gate silently passes the exact shape real
    collapsed runs produce. A missing (``None``) or non-finite
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


def _as_finite_float(value: Any) -> float | None:
    """Coerce to a finite Python float, or ``None`` if that is not possible."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
