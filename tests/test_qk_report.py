"""Suite coverage for the end-to-end quality gate — change qk-e2e-quality-gate.

The logic under test lives in ``quantibbler.qk_report``, deliberately OUTSIDE
the torch-importing driver (``tools/quantize_hf_model.py``), so these tests
execute with no skips in a lean environment. A verdict/sanitizer test that
silently skips is the exact failure mode this split exists to prevent: the
sanitizer previously lived only in a REPL transcript.

Surface 1 of 6 (split out of a single 2.4k-line module; every test body is
verbatim and the docstrings keep their scenario IDs): the ok/degraded/failed
band derived from the pinned paper anchors ([LB2] Sec. 5.2 Table 1), the
archived production ratio history, ``provenance()`` /
``json_safe()`` / ``dump_report()``, and the numpy-scalar and non-finite paths
the sanitizer exists for.

Scenario IDs from ``openspec/changes/qk-e2e-quality-gate/specs/qk-ptq/spec.md``.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from quantibbler.qk_report import (
    COLLAPSE_LINE,
    OK_LINE,
    dump_report,
    exit_code,
    json_safe,
    provenance,
    verdict,
)

# Anchors: [LB2] Sec. 5.2 Table 1, Llama-3-8B / WikiText-2.
_STBLLM_COLLAPSE_PPL = 241.96  # STBLLM at 0.55 bpp — documented method collapse
_LITTLEBIT2_FUNCTIONAL_PPL = 23.74  # LittleBit-2 at 0.1 bpp — functional envelope
_FP16_REFERENCE_PPL = 6.10


def _reject_constant(name: str):
    """``parse_constant`` hook that rejects NaN/Infinity/-Infinity tokens."""
    raise ValueError(f"non-finite JSON constant: {name}")


# --------------------------------------------------------------------------
# verdict_* — the band
# --------------------------------------------------------------------------


def test_verdict_edge_arithmetic_pinned():
    """Both edges are DERIVED from the pinned paper table, not chosen."""
    assert math.ceil(_STBLLM_COLLAPSE_PPL / _FP16_REFERENCE_PPL) == 40
    assert math.ceil(_LITTLEBIT2_FUNCTIONAL_PPL / _FP16_REFERENCE_PPL) == 4
    assert float(math.ceil(_STBLLM_COLLAPSE_PPL / _FP16_REFERENCE_PPL)) == COLLAPSE_LINE
    assert float(math.ceil(_LITTLEBIT2_FUNCTIONAL_PPL / _FP16_REFERENCE_PPL)) == OK_LINE
    assert (OK_LINE, COLLAPSE_LINE) == (4.0, 40.0)


def test_verdict_ok_within_functional_envelope():
    for ratio in (0.5, 1.0, 2.5, 3.89, OK_LINE):
        assert verdict(ratio) == "ok", ratio
        assert exit_code(verdict(ratio)) == 0


def test_verdict_degraded_between_lines():
    for ratio in (4.0001, 12.0, 24.0, COLLAPSE_LINE):
        assert verdict(ratio) == "degraded", ratio
        assert exit_code(verdict(ratio)) == 0


def test_verdict_boundary_semantics():
    """`<=` at both edges: the lines themselves are the last admitted values."""
    assert verdict(OK_LINE) == "ok"
    assert verdict(OK_LINE + 1e-4) == "degraded"
    assert verdict(COLLAPSE_LINE) == "degraded"
    assert verdict(COLLAPSE_LINE + 1e-4) == "failed"


def test_verdict_failed_above_collapse_line():
    for ratio in (40.0001, 943.49, 5517.8, 463742.5, 1156577.1):
        assert verdict(ratio) == "failed", ratio
        assert exit_code(verdict(ratio)) != 0


def test_verdict_missing_or_nonfinite_failed():
    for ratio in (None, float("nan"), float("inf"), float("-inf")):
        assert verdict(ratio) == "failed", ratio
        assert exit_code(verdict(ratio)) != 0


def test_verdict_nan_cannot_slip_through_a_threshold_compare():
    """The trap that let r137-v3/v4 through: every NaN comparison is False."""
    nan = float("nan")
    assert (nan > COLLAPSE_LINE) is False
    assert (nan <= COLLAPSE_LINE) is False
    assert (nan <= OK_LINE) is False
    # Only the explicit missing/non-finite check catches it.
    assert verdict(nan) == "failed"


def test_verdict_helpers_torch_free_tested():
    """Import and run in a process that has never seen torch."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import quantibbler.qk_report as m; "
            "print(int('torch' in sys.modules), m.verdict(1.0), m.verdict(1e7), "
            "m.probe_aborts(2, 1e4, 1.0), m.probe_aborts(2, 1.0, 1.0))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.split() == ["0", "ok", "failed", "True", "False"], proc.stdout


# --------------------------------------------------------------------------
# ratio_* — the recorded history
# --------------------------------------------------------------------------


def test_ratio_history_all_refused():
    """Every archived production shape. The gate refuses all of them."""
    historical = {
        "r137_v0": 1156577.1,
        "r137_v2_best_ever": 943.49,
        "r137_v3": float("nan"),
        "r137_v4": float("nan"),
        "r137_v7": 5517.8,
        "r137_v8": 463742.5,
        "r512_svd": 524894.8,
        "r8_line": 528419.4,
    }
    assert len(historical) == 8
    verdicts = {name: verdict(r) for name, r in historical.items()}
    assert set(verdicts.values()) == {"failed"}, verdicts
    assert all(exit_code(v) != 0 for v in verdicts.values())


def test_ratio_yields_the_declared_verdict_for_each_band():
    assert [verdict(r) for r in (1.0, 10.0, 100.0)] == ["ok", "degraded", "failed"]


# --------------------------------------------------------------------------
# report_* — provenance and strict JSON
# --------------------------------------------------------------------------


def test_report_mirrors_run_args():
    """`--solver svd --scale-dtype float32` must never name float16."""
    p = provenance("svd", "float32")
    assert p["solver"] == "svd"
    assert p["scale_dtype"] == "float32"
    assert "float32" in p["regime_artifact"]
    assert "float16" not in p["regime_artifact"]
    assert "svd" in p["dtype_note"]
    assert "float32" in p["dtype_note"]
    assert "float16" not in p["dtype_note"]


def test_report_fp16_opt_in_named():
    """The float16 opt-in is recorded FROM the arguments, not a constant."""
    p = provenance("admm", "float16")
    assert "float16" in p["regime_artifact"]
    assert "float16" in p["dtype_note"]
    assert p["solver"] == "admm"


def test_report_nonfinite_as_null():
    report = {
        "ppl_baseline_bf16": 28.66960268125778,
        "ppl_quant_native": float("nan"),
        "ppl_ratio": float("nan"),
        "nested": {"x": [1.0, float("inf")]},
    }
    # A bare NaN really is unparseable-by-strict-parsers JSON.
    with pytest.raises(ValueError):
        json.dumps(report, allow_nan=False)

    safe = json_safe(report)
    assert safe["ppl_quant_native"] is None
    assert safe["ppl_ratio"] is None
    assert safe["nested"]["x"] == [1.0, None]

    text = json.dumps(safe, indent=2, allow_nan=False)
    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text, parse_constant=_reject_constant)["ppl_ratio"] is None


def test_report_status_is_failed_for_nonfinite_and_ok_for_healthy():
    """status comes from the same verdict function the exit code uses."""
    dead = {"ppl_ratio": float("nan")}
    healthy = {"ppl_ratio": 2.5}
    assert verdict(dead["ppl_ratio"]) == "failed"
    assert json_safe(dead)["ppl_ratio"] is None
    assert verdict(healthy["ppl_ratio"]) == "ok"

    dead_text = json.dumps(json_safe(dead), allow_nan=False)
    healthy_text = json.dumps(json_safe(healthy), allow_nan=False)
    assert json.loads(dead_text, parse_constant=_reject_constant)["ppl_ratio"] is None
    assert json.loads(healthy_text, parse_constant=_reject_constant) == healthy


def test_report_inf_overflow_class_is_failed():
    """The fp16-overflow class (scales -> inf) is a failure, not a pass."""
    safe = json_safe({"ppl_ratio": float("inf")})
    assert safe["ppl_ratio"] is None
    assert verdict(float("inf")) == "failed"


# --------------------------------------------------------------------------
# verdict_* — numpy scalars and junk input
# --------------------------------------------------------------------------


def test_verdict_accepts_numpy_scalars():
    """The driver's numbers come off numpy arrays; they must classify."""
    assert verdict(np.float32(1.0)) == "ok"
    assert verdict(np.float64(2.5)) == "ok"
    assert verdict(np.float64(1e7)) == "failed"
    assert verdict(np.float32("nan")) == "failed"
    assert verdict(np.float32("inf")) == "failed"


def test_exit_code_fails_closed_on_junk_status():
    """Only the two passing verdicts exit 0; anything else fails closed."""
    for junk in ("OK", "Degraded", "", None, "done"):
        assert exit_code(junk) != 0, junk
    assert exit_code("ok") == 0
    assert exit_code("degraded") == 0


# --------------------------------------------------------------------------
# report_* — numpy scalars, and a report that must always exist
# --------------------------------------------------------------------------


def test_report_numpy_scalars_are_sanitized_and_serializable():
    """np.float32(inf) is not a Python float.

    It survived the old ``isinstance(obj, float)`` check, reached
    ``json.dumps(..., allow_nan=False)`` and raised — costing the report
    entirely on exactly the weirdest runs.
    """
    with pytest.raises(TypeError):
        json.dumps({"x": np.float32("inf")}, allow_nan=False)

    assert json_safe({"x": np.float32("inf")}) == {"x": None}
    assert json_safe({"n": np.int64(3), "b": np.bool_(True), "f": np.float32(1.5)}) == {
        "n": 3,
        "b": True,
        "f": 1.5,
    }

    payload, faithful = dump_report({"ppl_ratio": np.float32("inf"), "status": "failed"})
    assert faithful is True
    parsed = json.loads(payload, parse_constant=_reject_constant)
    assert parsed["ppl_ratio"] is None


def test_report_write_crash_fallback_keeps_a_verdict():
    """An unrecognized value costs measurements, never the verdict."""
    payload, faithful = dump_report({"ppl_ratio": 1.0, "weird": object()})
    assert faithful is False
    parsed = json.loads(payload, parse_constant=_reject_constant)
    assert parsed["status"] == "failed"
    assert "report_error" in parsed


def test_report_dump_report_is_faithful_for_healthy_reports():
    payload, faithful = dump_report({"status": "ok", "ppl_ratio": 2.5})
    assert faithful is True
    assert json.loads(payload, parse_constant=_reject_constant)["status"] == "ok"
