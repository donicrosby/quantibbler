"""Suite coverage for the end-to-end quality gate — change qk-e2e-quality-gate.

The logic under test lives in ``quantibbler.qk_report``, deliberately OUTSIDE
the torch-importing driver (``tools/quantize_hf_model.py``), so these tests
execute with no skips in a lean environment. A verdict/sanitizer test that
silently skips is the exact failure mode this split exists to prevent: the
sanitizer previously lived only in a REPL transcript.

Scenario IDs from ``openspec/changes/qk-e2e-quality-gate/specs/qk-ptq/spec.md``.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from quantibbler import qk_report
from quantibbler.qk_report import (
    COLLAPSE_LINE,
    FAILED,
    MIN_PROBE_DEPTH,
    OK,
    OK_LINE,
    abort_report,
    dump_report,
    exit_code,
    json_safe,
    probe_aborts,
    probe_ratio,
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


# --------------------------------------------------------------------------
# report_* — the driver tripwire
#
# ``tools/quantize_hf_model.py`` imports torch at module top, so it cannot be
# imported in a lean env; the wiring is pinned by parsing it. The tripwire is
# itself tested against poisoned drivers — a guard nobody has shown rejecting
# anything is not a guard.
# --------------------------------------------------------------------------

_DRIVER = pathlib.Path(__file__).resolve().parents[1] / "tools" / "quantize_hf_model.py"
#: Source of the module that owns `PrefixProbeAbort`. Read as text, never
#: imported: `quantibbler.ptq.__init__` pulls `.api`, which imports torch.
_PTQ_CONFIG = (
    pathlib.Path(__file__).resolve().parents[1] / "src" / "quantibbler" / "ptq" / "config.py"
)
_PROTECTED_NAMES = {
    "verdict",
    "verdict_exit_code",
    "json_safe",
    "dump_report",
    "provenance",
    "exit_code",
    "FAILED",
}
_EXIT_FUNCS = {"exit", "quit", "_exit", "abort"}
_REPORT_MUTATORS = {"update", "setdefault", "pop", "popitem", "clear", "__setitem__"}


def _rooted_in_report(node: ast.AST) -> bool:
    """Whether a target/subject expression is rooted at the name ``report``.

    Walks through any nesting: ``report``, ``report["status"]``,
    ``report["probe_abort"]["line"]``, ``report.update``. Destructuring targets
    (``a, report = ...``) count too: any element rooted at ``report`` makes the
    whole target a write into it.
    """
    while True:
        if isinstance(node, ast.Name):
            return node.id == "report"
        if isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
            continue
        if isinstance(node, ast.Call):
            node = node.func
            continue
        if isinstance(node, (ast.Tuple, ast.List)):
            return any(_rooted_in_report(element) for element in node.elts)
        return False


def _binds_report_name(node: ast.AST) -> bool:
    """Whether a statement binds the NAME ``report`` (as opposed to writing into it).

    Walks every name-binding shape Python has — ``Assign``, ``AnnAssign`` *with*
    a value (an annotation without one is a bare declaration and binds nothing),
    ``AugAssign``, ``NamedExpr``, ``for`` targets, ``with ... as`` targets, and
    destructuring patterns containing the name. This predicate and the write
    sweep below are the two halves of one rule, and they must walk the same node
    classes: an Assign-only count beside a wider sweep is exactly how
    ``report: dict = {"status": "ok"}`` slipped between them (bug-hunter's
    finding), after the walrus had slipped between the previous pair.
    """
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        targets = [item.optional_vars for item in node.items if item.optional_vars]
    elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr, ast.For, ast.AsyncFor)):
        # An annotation with no value is a bare declaration: it binds nothing.
        if isinstance(node, ast.AnnAssign) and node.value is None:
            return False
        targets = [node.target]
    else:
        return False
    for target in targets:
        if isinstance(target, ast.Name) and target.id == "report":
            return True
        if isinstance(target, (ast.Tuple, ast.List)) and _rooted_in_report(target):
            return True
    return False


def _driver_wiring_findings(src: str) -> list[str]:
    """Every way ``src`` breaks the driver contract. Empty list == clean."""
    findings: list[str] = []
    tree = ast.parse(src)

    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "quantibbler.qk_report":
            imported |= {a.asname or a.name for a in node.names}
    for needed in ("verdict", "verdict_exit_code", "dump_report"):
        if needed not in imported:
            findings.append(f"missing `{needed}` import from quantibbler.qk_report")

    # A driver-local definition of the gate is a shadow: the source still
    # looks wired while the runtime gate is whatever the local def says.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _PROTECTED_NAMES:
            findings.append(
                f"line {node.lineno}: driver-local def {node.name}() shadows a gate import"
            )
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) in _PROTECTED_NAMES for t in node.targets
        ):
            findings.append(f"line {node.lineno}: module-level assignment shadows a gate import")

    main = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"),
        None,
    )
    if main is None:
        findings.append("main() is missing")
        return findings

    # No exit path other than the single verdict return.
    for node in ast.walk(main):
        if isinstance(node, ast.Raise):
            exc = ast.unparse(node.exc) if node.exc else ""
            if not exc or "SystemExit" in exc:
                findings.append(
                    f"line {node.lineno}: `raise {exc or '...'}` exits main() outside the verdict"
                )
        elif isinstance(node, ast.Call):
            fname = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if fname in _EXIT_FUNCS:
                findings.append(f"line {node.lineno}: {fname}() bypasses the verdict exit")

    # Nested shadows: the same bypass one scope deeper. `verdict = lambda r:
    # "ok"` inside main() is invisible to the module-level pass (a nested
    # assignment is not a tree-level def) and to the return-count pin (a
    # lambda body is not an ast.Return). Named residual from bug-hunter's
    # mutation battery, closed here.
    for node in ast.walk(main):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name in _PROTECTED_NAMES
        ):
            findings.append(f"line {node.lineno}: nested def {node.name}() shadows a gate import")
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) in _PROTECTED_NAMES:
                    findings.append(
                        f"line {node.lineno}: nested assignment to {target.id} "
                        "shadows a gate import"
                    )

    returns = [n for n in ast.walk(main) if isinstance(n, ast.Return)]
    if len(returns) != 1:
        findings.append(f"expected exactly one return in main(), found {len(returns)}")
        return findings

    # The verdict feeds both the report and the exit code. The second
    # assignment is the serialization-failure fallback (S4): a report that
    # could not be written as measured must force the process to fail too.
    status_assigns = [
        ast.unparse(n.value)
        for n in ast.walk(main)
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "status" for t in n.targets)
    ]
    if status_assigns != ["verdict(ppl_ratio)", "FAILED"]:
        findings.append(
            f"status assignments are {status_assigns}, expected ['verdict(ppl_ratio)', 'FAILED']"
        )

    ret = returns[0]
    if ast.unparse(ret.value) != "verdict_exit_code(status)":
        findings.append(
            f"line {ret.lineno}: return is `{ast.unparse(ret.value)}`, not the verdict code"
        )

    # Forensic survival: the export and the report write must both happen
    # BEFORE the process can exit. A hoisted return keeps the shape green
    # while silently dropping evidence on every failed run.
    for label, match in (
        (
            "export_model_directory(...)",
            lambda c: getattr(c.func, "id", None) == "export_model_directory",
        ),
        (
            "the report write_text(...)",
            lambda c: getattr(c.func, "attr", None) == "write_text",
        ),
    ):
        lines = [c.lineno for c in ast.walk(main) if isinstance(c, ast.Call) and match(c)]
        if not lines:
            findings.append(f"{label} call not found in main()")
        elif max(lines) >= ret.lineno:
            findings.append(
                f"the single return at line {ret.lineno} precedes {label} (line {max(lines)})"
            )

    # The probe abort path (change qk-fail-fast-loop). An aborted run must
    # still write a failed report and exit nonzero, through the SAME single
    # verdict return: never returned from (a collapse would exit 0), never
    # re-raised (the report would never be written), and never widened past
    # PrefixProbeAbort (a crash would be labeled a measured collapse).
    for node in ast.walk(main):
        if isinstance(node, ast.Try):
            for wide in node.handlers:
                caught = ast.unparse(wide.type) if wide.type is not None else "bare except"
                if caught in ("Exception", "BaseException", "bare except"):
                    findings.append(
                        f"line {wide.lineno}: `{caught}` in main() swallows failures "
                        f"into the verdict path instead of letting them surface"
                    )
    probe_handlers = [
        h
        for node in ast.walk(main)
        if isinstance(node, ast.Try)
        for h in node.handlers
        if h.type is not None and "PrefixProbeAbort" in ast.unparse(h.type)
    ]
    if len(probe_handlers) != 1:
        findings.append(
            f"expected exactly one `except PrefixProbeAbort` handler in main(), "
            f"found {len(probe_handlers)}"
        )
    else:
        handler = probe_handlers[0]
        if ast.unparse(handler.type) != "PrefixProbeAbort":
            findings.append(
                f"line {handler.lineno}: the abort handler catches "
                f"`{ast.unparse(handler.type)}`, not PrefixProbeAbort alone"
            )
        for node in ast.walk(handler):
            if isinstance(node, ast.Return):
                findings.append(
                    f"line {node.lineno}: the abort handler returns directly — an "
                    f"aborted run must fall through to the verdict exit, never exit 0"
                )
            if isinstance(node, ast.Raise):
                findings.append(
                    f"line {node.lineno}: the abort handler raises — the abort must "
                    f"become a failed report, not an uncaught exception"
                )
        recorded = [
            getattr(target, "id", None)
            for stmt in handler.body
            if isinstance(stmt, ast.Assign)
            for target in stmt.targets
        ]
        if "aborted" not in recorded:
            findings.append(
                "the abort handler does not record the abort (`aborted = ...`), so the "
                "exit-code fallback cannot see it"
            )
        built = [
            stmt
            for stmt in handler.body
            if isinstance(stmt, ast.Assign)
            and any(getattr(t, "id", None) == "report" for t in stmt.targets)
            and isinstance(stmt.value, ast.Call)
            and getattr(stmt.value.func, "id", None) == "abort_report"
        ]
        if len(built) != 1:
            findings.append(
                f"the abort handler binds `report` to abort_report(...) {len(built)} "
                f"times, expected exactly once — the catch→report→status construction "
                f"belongs in the lean-tested builder, not inline in a torch-importing "
                f"main()"
            )

    # `report` is write-only-at-binding: exactly two statements establish the
    # name in main() (the completion literal and `abort_report(...)`), and ZERO
    # writes of any other AST shape rooted at it. The two halves of that rule
    # are `_binds_report_name` (name bindings, every shape) and the sweep below
    # (writes into the value: `report[...] = ` at any nesting, attribute writes,
    # deletes, destructuring targets, and the mutating calls `update`,
    # `setdefault`, `pop`, `clear`). They walk the same node classes on purpose:
    # twice now a shape has slipped between an Assign-only count and a wider
    # sweep (the walrus, then `report: dict = {...}`).
    # Named residual no source pin can see: a callee that mutates the dict
    # handed to it (`def _fix(r): r.update(...)`) — that class belongs to the
    # runtime abort leg (status failed, exit 1, the full key set on a real run),
    # not here. Layers, not an ever-growing case list.
    bindings = [node for node in ast.walk(main) if _binds_report_name(node)]
    if len(bindings) != 2:
        lines = sorted({node.lineno for node in bindings})
        findings.append(
            f"main() establishes the `report` name {len(bindings)} times (lines {lines}), "
            f"expected exactly 2 (the completion literal and the abort_report(...) call)"
        )
    for node in ast.walk(main):
        written: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            written = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            written = [node.target]
        elif isinstance(node, ast.Delete):
            written = list(node.targets)
        for target in written:
            if not isinstance(target, ast.Name) and _rooted_in_report(target):
                findings.append(
                    f"line {node.lineno}: write into the built report "
                    f"(`{ast.unparse(target)}`) — report is write-only at binding"
                )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _REPORT_MUTATORS
            and _rooted_in_report(node.func.value)
        ):
            findings.append(
                f"line {node.lineno}: `report.{node.func.attr}(...)` mutates the built "
                f"report — report is write-only at binding"
            )

    # ... and the process verdict must see the abort too, not only the
    # serialization failure: an aborted run that exits 0 is the whole bug.
    fail_over = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.If)
        and any(
            isinstance(stmt, ast.Assign)
            and any(getattr(t, "id", None) == "status" for t in stmt.targets)
            and ast.unparse(stmt.value) == "FAILED"
            for stmt in node.body
        )
    ]
    if fail_over and "aborted" not in ast.unparse(fail_over[0].test):
        findings.append(
            "the FAILED exit-code fallback does not consider the probe abort — an "
            "aborted run would exit 0"
        )
    return findings


def test_report_driver_is_wired_to_the_gate():
    """The driver is wired to this module, in shape AND in order."""
    src = _DRIVER.read_text()
    findings = _driver_wiring_findings(src)
    assert findings == [], findings

    # The retired helpers and the hardcoded provenance literals stay retired.
    assert "_json_safe" not in src and "_report_status" not in src
    assert "qpk1: +-1 bitplanes + float16 scales" not in src
    assert "scales fp16" not in src
    # Provenance is emitted from the run's own arguments.
    assert "provenance(args.solver, args.scale_dtype)" in src


def test_driver_probe_knobs_are_parser_owned_and_threshold_free():
    """`probe_knob_defaults_and_validation` + `probe_threshold_not_overrideable`,
    in the lean tier: the two specced knobs are argparse-owned with the specced
    defaults behind a parse-time positive-int validator, and nothing — no flag,
    no argument at the call site — re-pins the derived abort line. The runtime
    half (a non-positive value really is a parse error, exit 2) belongs to the
    torch-tier driver test; this half runs without torch.
    """
    main = [
        node
        for node in ast.parse(_DRIVER_SRC).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    ][0]
    flags: dict[str, ast.Call] = {}
    for node in ast.walk(main):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            flags[node.args[0].value] = node
    for flag, default in (("--probe-interval", 1), ("--probe-batches", 2)):
        call = flags.get(flag)
        assert call is not None, f"{flag} is not a driver flag: {sorted(flags)}"
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        assert ast.unparse(kwargs["type"]) == "_positive_int", flag
        assert ast.unparse(kwargs["default"]) == str(default), flag
    banned = ("probe-line", "probe-limit", "probe-threshold", "collapse", "abort-line")
    offenders = [flag for flag in flags if any(word in flag for word in banned)]
    assert offenders == [], f"a flag re-pins the abort line: {offenders}"

    probe_calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "probe_aborts"
    ]
    assert len(probe_calls) == 1, probe_calls
    assert [ast.unparse(arg) for arg in probe_calls[0].args] == [
        "depth",
        "prefix_ppl",
        "fp_probe_ppl",
    ]
    assert probe_calls[0].keywords == [], "the call site passes a threshold override"
    # The pipeline receives exactly one probe field, and it is this closure.
    assert "quantized_config(args, progress_probe=_probe)" in _DRIVER_SRC


_DRIVER_SRC = _DRIVER.read_text()
_PROVENANCE_ANCHOR = "    prov = provenance(args.solver, args.scale_dtype)\n"
_RETURN_LINE = "    return verdict_exit_code(status)\n"
_ABORT_ANCHOR = "    except PrefixProbeAbort as exc:\n"
#: The end of the abort branch's builder call: unique because the completion
#: report spells its history line differently.
_ABORT_BUILT_ANCHOR = "            probe_history,\n        )\n"


def _poison(kind: str) -> str:
    """Apply one of sec-eng's demonstrated bypasses to the real driver source."""
    src = _DRIVER_SRC
    if kind == "hidden_system_exit":
        return src.replace(_PROVENANCE_ANCHOR, "    raise SystemExit(0)\n" + _PROVENANCE_ANCHOR, 1)
    if kind == "sys_exit_call":
        return src.replace(_PROVENANCE_ANCHOR, "    sys.exit(0)\n" + _PROVENANCE_ANCHOR, 1)
    if kind == "shadowed_verdict_def":
        return src + '\n\ndef verdict(_ratio):\n    return "ok"\n'
    if kind == "return_hoisted_above_export":
        return src.replace(_RETURN_LINE, "", 1).replace(
            _PROVENANCE_ANCHOR, _RETURN_LINE + _PROVENANCE_ANCHOR, 1
        )
    if kind == "status_pinned_ok":
        return src.replace("    status = verdict(ppl_ratio)\n", '    status = "ok"\n', 1)
    if kind == "nested_lambda_shadow":
        return src.replace(
            _PROVENANCE_ANCHOR,
            '    verdict = lambda _ratio: "ok"\n' + _PROVENANCE_ANCHOR,
            1,
        )
    if kind == "abort_return_from_except":
        return src.replace(_ABORT_ANCHOR, _ABORT_ANCHOR + "        return 0\n", 1)
    if kind == "abort_reraise_after_record":
        return src.replace(_ABORT_ANCHOR, _ABORT_ANCHOR + "        raise\n", 1)
    if kind == "abort_widened_to_exception":
        return src.replace(_ABORT_ANCHOR, "    except Exception as exc:\n", 1)
    if kind == "abort_swallowed":
        return src.replace(
            _ABORT_ANCHOR + "        aborted = exc\n",
            _ABORT_ANCHOR + "        pass\n",
            1,
        )
    if kind == "abort_bare_except":
        return src.replace(_ABORT_ANCHOR, "    except:\n", 1)
    if kind == "abort_status_relabelled":
        return src.replace(
            _ABORT_ANCHOR + "        aborted = exc\n",
            _ABORT_ANCHOR
            + '        aborted = exc\n        report = {}\n        report["status"] = "ok"\n',
            1,
        )
    if kind == "abort_report_updated":
        return src.replace(
            _ABORT_BUILT_ANCHOR,
            _ABORT_BUILT_ANCHOR + '        report.update({"status": "ok"})\n',
            1,
        )
    if kind == "abort_line_rewritten":
        return src.replace(
            _ABORT_BUILT_ANCHOR,
            _ABORT_BUILT_ANCHOR + '        report["probe_abort"]["line"] = 1.0\n',
            1,
        )
    if kind == "abort_walrus_rebind":
        return src.replace(
            _ABORT_BUILT_ANCHOR,
            _ABORT_BUILT_ANCHOR + "        (report := abort_report(exc, {}, probe_history))\n",
            1,
        )
    if kind == "abort_annassign_rebind":
        return src.replace(
            _ABORT_BUILT_ANCHOR,
            _ABORT_BUILT_ANCHOR + '        report: dict = {"status": "ok"}\n',
            1,
        )
    if kind == "abort_ignored_by_exit_code":
        return src.replace(
            "    if not faithful or aborted is not None:\n", "    if not faithful:\n", 1
        )
    raise AssertionError(f"unknown poison {kind}")


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("hidden_system_exit", "SystemExit"),
        ("sys_exit_call", "bypasses the verdict exit"),
        ("shadowed_verdict_def", "shadows a gate import"),
        ("return_hoisted_above_export", "precedes"),
        ("status_pinned_ok", "status assignments are"),
        ("nested_lambda_shadow", "nested assignment to verdict"),
        ("abort_return_from_except", "exactly one return"),
        ("abort_reraise_after_record", "the abort handler raises"),
        ("abort_widened_to_exception", "swallows failures"),
        ("abort_bare_except", "swallows failures"),
        ("abort_swallowed", "does not record the abort"),
        ("abort_status_relabelled", "write into the built report"),
        ("abort_report_updated", "mutates the built report"),
        ("abort_line_rewritten", "write into the built report"),
        ("abort_walrus_rebind", "establishes the `report` name"),
        ("abort_annassign_rebind", "establishes the `report` name"),
        ("abort_ignored_by_exit_code", "does not consider the probe abort"),
    ],
)
def test_report_driver_tripwire_rejects_poisoned_drivers(kind, expected):
    """Each demonstrated bypass is caught, against the real source."""
    poisoned = _poison(kind)
    assert poisoned != _DRIVER_SRC, "mutation did not apply"
    findings = _driver_wiring_findings(poisoned)
    assert findings, f"tripwire accepted the `{kind}` bypass"
    assert any(expected in f for f in findings), (kind, findings)


def test_report_writeonly_pin_covers_every_binding_shape():
    """`report` is frozen by NAME, so every way of rebinding it is a finding.

    Two rounds of the same finding, closed here shape by shape. sec-eng's walrus
    (`(report := ...)` — the natural refactor of the except path) and
    bug-hunter's annotated rebind (`report: dict = {...}`) both slipped between
    an Assign-only binding count and a wider write sweep; the count and the sweep
    now derive from one predicate, ``_binds_report_name``, so a shape has to pass
    *both* halves to escape. `for`, `with ... as` and `+=` are covered by the
    same predicate; a destructuring target is a write into the value.
    """
    mutations = {
        "walrus": "        (report := abort_report(exc, {}, probe_history))\n",
        "annotated": '        report: dict = {"status": "ok"}\n',
        "augassign": '        report += {"status": "ok"}\n',
        "for": "        for report in ():\n            break\n",
        "with": "        with open('/dev/null') as report:\n            pass\n",
    }
    for name, statement in mutations.items():
        poisoned = _DRIVER_SRC.replace(_ABORT_BUILT_ANCHOR, _ABORT_BUILT_ANCHOR + statement, 1)
        assert poisoned != _DRIVER_SRC, name
        findings = _driver_wiring_findings(poisoned)
        assert findings, f"the pin accepted the `{name}` binding shape"
        assert any("establishes the `report` name" in finding for finding in findings), (
            name,
            findings,
        )
    # ... and a destructuring target, which binds the value rather than the name.
    destructured = _DRIVER_SRC.replace(
        "    payload, faithful = dump_report(report)\n",
        "    report, faithful = report, True\n",
        1,
    )
    assert destructured != _DRIVER_SRC
    findings = _driver_wiring_findings(destructured)
    assert any("write into the built report" in finding for finding in findings), findings
    # The replace form: annotate the abort binding itself rather than adding a
    # third. The name count still reads two, so this is caught by the
    # builder-binding check instead — the two halves together leave no hole.
    annotated_binding = _DRIVER_SRC.replace(
        "        report = abort_report(\n",
        "        report: dict = abort_report(\n",
        1,
    )
    assert annotated_binding != _DRIVER_SRC
    findings = _driver_wiring_findings(annotated_binding)
    assert any("abort_report(...)" in finding for finding in findings), findings


# --------------------------------------------------------------------------
# probe_* — the fail-fast progress probe (change qk-fail-fast-loop)
#
# Kill-only by construction: a 2-batch prefix supports *death* (every dead
# run sat 20x+ past the line) and can never support *success*. Only the
# gate's `failed` verdict may halt a run, so nobody can later "optimize" the
# full held-out measurement away because a prefix already looked fine.
# --------------------------------------------------------------------------


def test_probe_comparator_reuses_gate_band():
    """The probe decision IS the shared band, not a second threshold."""
    assert probe_aborts(2, COLLAPSE_LINE * 2.0, 1.0) is True
    assert probe_aborts(2, COLLAPSE_LINE * 1.0001, 1.0) is True
    assert probe_aborts(2, COLLAPSE_LINE, 1.0) is False  # the line is degraded
    assert probe_aborts(2, OK_LINE, 1.0) is False
    assert probe_aborts(2, 1.0, 1.0) is False


def test_probe_threshold_is_the_shared_constant_not_a_copy():
    """No second literal 40 lives in qk_report (bug-hunter's check (c))."""
    tree = ast.parse(pathlib.Path(qk_report.__file__).read_text())
    numeric = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
    }
    assert 40 not in numeric
    assert 40.0 not in numeric
    # ...and the module really does carry the derived constant.
    assert COLLAPSE_LINE == 40.0


def test_probe_min_depth_two():
    """Probes begin no earlier than depth 2; block-0-alone is the noisiest."""
    assert MIN_PROBE_DEPTH == 2
    for depth in (0, 1):
        assert probe_aborts(depth, float("inf"), 1.0) is False, depth
    for depth in (2, 3, 28):
        assert probe_aborts(depth, COLLAPSE_LINE * 2.0, 1.0) is True, depth


def test_probe_boundary_semantics():
    assert probe_aborts(2, COLLAPSE_LINE, 1.0) is False
    assert probe_aborts(2, COLLAPSE_LINE + 1e-4, 1.0) is True
    assert probe_aborts(2, 4.0, 1.0) is False  # ok
    assert probe_aborts(2, 12.0, 1.0) is False  # degraded — alive, not working


def test_probe_nan_check_runs_first():
    """`nan > COLLAPSE_LINE` is False, so a bare comparison passes a dead run."""
    assert (float("nan") > COLLAPSE_LINE) is False
    assert probe_aborts(2, float("nan"), 1.0) is True
    assert probe_aborts(2, float("inf"), 1.0) is True
    assert probe_aborts(2, 1.0, float("nan")) is True
    assert probe_aborts(2, 1.0, 0.0) is True  # defective baseline: fail closed
    assert probe_aborts(2, None, 1.0) is True
    assert probe_aborts(2, 1.0, None) is True


def test_probe_is_kill_only_and_never_passes_early():
    """No probe result can report success — only the full gate can."""
    for ratio in (0.5, 1.0, 3.99, OK_LINE, OK_LINE + 1e-4, 10.0, 39.999):
        assert probe_aborts(2, ratio, 1.0) is False, ratio


def test_probe_accepts_numpy_scalars():
    assert probe_aborts(2, np.float32(1000.0), np.float32(1.0)) is True
    assert probe_aborts(2, np.float64(1.0), np.float64(1.0)) is False


def test_probe_ratio_helper_isolates_quantization_damage():
    """The depth-ratio divides out the depth-dependent PPL floor."""
    assert probe_ratio(80.0, 2.0) == 40.0
    assert probe_ratio(2.0, 2.0) == 1.0
    for dead in (
        (float("inf"), 2.0),
        (2.0, float("nan")),
        (None, 2.0),
        (2.0, None),
        (2.0, 0.0),  # a zero fp baseline is impossible: defective measurement
        ("nope", 2.0),
    ):
        assert probe_ratio(*dead) is None, dead


# --------------------------------------------------------------------------
# abort_report — the catch→report→status construction, lean-tested.
# Scenarios `abort_report_construction_lean_tested` and
# `report_single_shape_both_paths`. The driver's `except PrefixProbeAbort`
# binds `report` to this builder, so the wiring that delivers the fail-fast
# ask has repeatable proof with no weights, no torch, and no CI.
# --------------------------------------------------------------------------


def _abort_exc(**overrides):
    """A stand-in carrying exactly the production abort type's fields.

    Deliberately NOT the production class. Importing
    ``quantibbler.ptq.config`` executes ``quantibbler/ptq/__init__.py`` →
    ``.api`` → ``import torch``, which is how the first version of this test
    passed as "lean" only in an environment that already had torch synced: in a
    real lean env all four builder tests failed at the import. ``abort_report``
    is duck-typed by design (the exception cannot be imported into
    ``qk_report`` without a cycle), so a stand-in exercises the actual contract
    — and pinning it HERE, locally, means a future "fix" that re-couples the
    lean tier to torch through the back door dies at collection instead of
    silently regressing. Both halves of the contract are checked against source
    in ``test_abort_report_contract_matches_the_production_abort_type``.
    """
    fields = {
        "block": "layers.1",
        "depth": 2,
        "prefix_ppl": 2906604.684589879,
        "fp_prefix_ppl": 83.08454639156159,
        "ratio": 34983.69806211141,
        "line": COLLAPSE_LINE,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_abort_report_contract_matches_the_production_abort_type():
    """The duck-typed attribute set IS the contract — pinned from source.

    Read both sides without importing the owning module: the builder's
    ``exc.<field>`` reads out of ``qk_report``'s own source, and
    ``PrefixProbeAbort``'s declared fields out of ``ptq/config.py``. Importing
    the type would drag torch in through ``ptq/__init__`` and defeat the tier.
    """
    builder_source = inspect.getsource(qk_report.abort_report)
    read = {
        node.attr
        for node in ast.walk(ast.parse(builder_source))
        if isinstance(node, ast.Attribute) and getattr(node.value, "id", None) == "exc"
    }
    abort_class = next(
        node
        for node in ast.parse(_PTQ_CONFIG.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "PrefixProbeAbort"
    )
    init = next(
        node
        for node in abort_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    declared = {arg.arg for arg in (*init.args.args, *init.args.kwonlyargs)} - {"self"}
    assert declared, "PrefixProbeAbort declares no fields"
    assert declared <= read, (
        f"the abort type carries fields the abort report drops: {sorted(declared - read)}"
    )
    # `line` is set by the type itself rather than passed in; the report still
    # has to carry it, because it is what the run compared against.
    assert "line" in read
    # The stand-in the builder tests use must supply every field the type does.
    assert declared <= set(vars(_abort_exc()))


#: The identity fields the driver hands the builder on a real abort (values from
#: the recorded Qwen3-0.6B abort run).
_ABORT_IDENTITY = {
    "model": "Qwen/Qwen3-0.6B",
    "rank": 1,
    "solver": "svd",
    "scale_dtype": "float32",
    "seq_len": 64,
    "calibration_rows": 8,
    "eval_rows": 16,
    "max_admm_iter": 1,
    "seed": 0,
    "ppl_baseline_bf16": 75.90947478697456,
    "regime": {"artifact": "qpk1: +-1 bitplanes + float32 scales"},
    "probe": {"interval": 1, "batches": 1, "fp_prefix_ppl": 83.08454639156159},
    "wall_seconds": 69.6,
}
_ABORT_HISTORY = [
    {
        "block": "layers.1",
        "depth": 2,
        "prefix_ppl": 2906604.684589879,
        "fp_prefix_ppl": 83.08454639156159,
        "ratio": 34983.69806211141,
    }
]


def _completion_report_keys() -> set[str]:
    """The keys the driver's completion report literal declares, from source."""
    main = [
        node
        for node in ast.parse(_DRIVER_SRC).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    ][0]
    for node in ast.walk(main):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Dict)
            and any(getattr(target, "id", None) == "report" for target in node.targets)
        ):
            return {key.value for key in node.value.keys if isinstance(key, ast.Constant)}
    raise AssertionError("the completion report dict literal is missing from main()")


def test_abort_report_is_failed_and_carries_the_abort_fields():
    report = abort_report(_abort_exc(), _ABORT_IDENTITY, _ABORT_HISTORY)
    assert report["status"] == FAILED
    assert report["artifact_exported"] is False
    assert report["probe_abort"] == {
        "block": "layers.1",
        "depth": 2,
        "prefix_ppl": 2906604.684589879,
        "fp_prefix_ppl": 83.08454639156159,
        "ratio": 34983.69806211141,
        "line": COLLAPSE_LINE,
    }
    # The trajectory that ended in the collapse survives into the report.
    assert report["probe"]["history"] == _ABORT_HISTORY
    assert report["probe"]["fp_prefix_ppl"] == 83.08454639156159
    assert report["probe"]["interval"] == 1


def test_abort_report_forces_unmeasured_numerics_and_the_failed_status():
    """A caller cannot smuggle a measured number or a healthy status in."""
    identity = dict(
        _ABORT_IDENTITY,
        status=OK,
        artifact_exported=True,
        ppl_quant_native=1.0,
        ppl_ratio=0.5,
        effective_bpw=0.7,
        qpk1_layer_bytes=42,
    )
    report = abort_report(_abort_exc(), identity, _ABORT_HISTORY)
    assert report["status"] == FAILED
    assert report["artifact_exported"] is False
    for key in ("ppl_quant_native", "ppl_ratio", "effective_bpw", "qpk1_layer_bytes"):
        assert report[key] is None, key
    # The builder copies rather than mutating the caller's fields.
    assert identity["status"] == OK
    assert identity["artifact_exported"] is True


def test_abort_report_has_the_completion_reports_key_set():
    """`report_single_shape_both_paths`: identical key set, measured or not."""
    completion_keys = _completion_report_keys()
    report = abort_report(_abort_exc(), _ABORT_IDENTITY, _ABORT_HISTORY)
    assert set(report) == completion_keys, (
        sorted(set(report) - completion_keys),
        sorted(completion_keys - set(report)),
    )
    assert "artifact_exported" in completion_keys
    assert {"probe_abort", "probe"} <= completion_keys


def test_abort_report_dumps_strict_and_writes_non_finite_as_null():
    """The abort report goes through the same sanitizer as every other report."""
    exc = _abort_exc(prefix_ppl=float("inf"), ratio=None)
    payload, faithful = dump_report(abort_report(exc, _ABORT_IDENTITY, _ABORT_HISTORY))
    assert faithful is True
    parsed = json.loads(payload, parse_constant=_reject_constant)
    assert parsed["status"] == "failed"
    assert parsed["artifact_exported"] is False
    assert parsed["probe_abort"]["prefix_ppl"] is None
    assert parsed["probe_abort"]["ratio"] is None
    assert parsed["probe_abort"]["block"] == "layers.1"
    assert parsed["probe_abort"]["line"] == COLLAPSE_LINE
    assert parsed["probe"]["history"][0]["ratio"] == 34983.69806211141
