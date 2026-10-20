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
    "MIN_PROBE_DEPTH",
    "FAILED",
    "DEGRADED",
    "OK",
    "verdict",
    "exit_code",
    "json_safe",
    "dump_report",
    "probe_ratio",
    "probe_aborts",
    "abort_report",
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


def _as_finite_float(value: Any) -> float | None:
    """Coerce to a finite Python float, or ``None`` if that is not possible."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


#: Prefix probes begin no earlier than this depth. Block-0-alone prefixes are
#: the noisiest, and block-0 damage surfaces amplified at depth >= 2 anyway
#: (the forensics' compounding pattern).
MIN_PROBE_DEPTH = 2


def probe_ratio(prefix_ppl: Any, fp_prefix_ppl: Any) -> float | None:
    """Quantized-prefix / fp-prefix ratio at the SAME depth.

    Dividing by the fp prefix at the same depth is the whole trick: it divides
    out the depth-dependent PPL floor, so the ratio isolates the damage
    quantization inflicted rather than the fact that a 2-block prefix predicts
    worse than a 28-block one.

    ``None`` when either side is missing, non-finite, or the fp baseline is
    zero. A perplexity of zero is impossible and a non-finite one is a
    numerics defect, so neither is a ratio — it is a defective measurement,
    and the caller fails closed on it.
    """
    prefix = _as_finite_float(prefix_ppl)
    baseline = _as_finite_float(fp_prefix_ppl)
    if prefix is None or baseline is None or baseline == 0.0:
        return None
    return prefix / baseline


def probe_aborts(depth: int, prefix_ppl: Any, fp_prefix_ppl: Any) -> bool:
    """Whether the progress probe must halt the run at this block boundary.

    Runs the SAME comparator as the end-of-run gate — ``verdict()`` over the
    shared ``COLLAPSE_LINE`` — with the missing/non-finite check first, so
    "collapsed" means the same thing at depth 3 as at depth 28 and there is
    no second copy of the line to drift.

    Kill-only by construction: it returns ``True`` only for the gate's
    ``failed`` verdict. A probe can never report success. A 2-batch prefix
    supports *death* (every dead run measured 20x+ past the line, far outside
    slice noise) but it can never support *success* — a lucky prefix ratio
    below ``OK_LINE`` on a model that is degraded deeper is exactly the case
    this asymmetry forbids acting on. ``ok`` and ``degraded`` come only from
    the full held-out measurement.

    A defective measurement (missing, non-finite, or a zero baseline) fails
    closed: a false positive costs a re-run, a false negative costs the
    hour-plus run this probe exists to kill.
    """
    if depth is None or depth < MIN_PROBE_DEPTH:
        return False
    ratio = probe_ratio(prefix_ppl, fp_prefix_ppl)
    if ratio is None:
        return True
    return verdict(ratio) == FAILED


#: End-of-run numerics that do not exist when the probe aborted the run before
#: the end-of-run gate measured anything. On an abort report they are written as
#: JSON ``null`` — never omitted, never invented — so both report paths carry an
#: identical key set and a strict consumer cannot KeyError on either shape.
_ABORT_UNMEASURED = ("ppl_quant_native", "ppl_ratio", "effective_bpw", "qpk1_layer_bytes")


def abort_report(exc: Any, identity_fields: dict, history: list) -> dict:
    """The report for a probe-aborted run — built without torch, weights, or IO.

    This is the catch→report→status construction, kept here (not inline in the
    torch-importing driver) so it has repeatable suite coverage in a lean
    environment. The caller writes the result through ``dump_report`` on the
    same path as every other report, so the abort report is sanitized and
    strict-parse-clean exactly like a completion report.

    ``exc`` is duck-typed on purpose: the abort exception lives in
    ``quantibbler.ptq.config``, which imports this module for ``COLLAPSE_LINE``,
    so importing it back would be a cycle. Its attributes are read directly,
    with no ``getattr`` defaults, so a wrong object fails loudly rather than
    producing a report full of nulls.

    ``identity_fields`` is everything the driver knows at the moment of the
    abort — model identity, regime, the ``probe`` block without its history,
    wall time. ``history`` is the probe trajectory so far: an aborted run keeps
    it, because the trajectory that ended in a collapse is the autopsy nobody
    otherwise has. The unmeasured end-of-run numerics are forced to ``None``
    and the status to ``FAILED`` here rather than at the call site, so "an
    aborted run is never reported healthy" is structural instead of copied by
    hand — a caller that smuggles ``status`` or ``artifact_exported`` into the
    identity fields cannot overwrite either one.
    """
    report = dict(identity_fields)
    probe = dict(report.get("probe") or {})
    probe["history"] = [dict(entry) for entry in history]
    report["probe"] = probe
    report["probe_abort"] = {
        "block": exc.block,
        "depth": exc.depth,
        "prefix_ppl": exc.prefix_ppl,
        "fp_prefix_ppl": exc.fp_prefix_ppl,
        "ratio": exc.ratio,
        "line": exc.line,
    }
    report["artifact_exported"] = False
    report["status"] = FAILED
    for key in _ABORT_UNMEASURED:
        report[key] = None
    return report


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
