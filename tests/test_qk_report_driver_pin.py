"""The driver tripwire: ``tools/quantize_hf_model.py``'s wiring, pinned structurally.

``tools/quantize_hf_model.py`` imports torch at module top, so it cannot be
imported in a lean env and the wiring is pinned by parsing it instead. What is
pinned, each contract beside the behavior it names: the gate wiring (one return
through ``verdict_exit_code``, the export and report write before it, exactly
one handler per abort/failure type, ``report`` write-only at binding), the KD
scratch-dir default surviving a host where ``/opt/data/tmp`` is absent, the
probe knobs being parser-owned and threshold-free, and the launch line's
identity fields in the pinned order. The tripwire is itself tested
against poisoned drivers — a guard nobody has shown rejecting anything is not a
guard. The runtime half of these contracts lives in ``tests/ptq/test_driver_*``.

Surface 2 of 6 (split out of a single 2.4k-line module; every test body is
verbatim and the docstrings keep their scenario IDs).
"""

from __future__ import annotations

import ast
import pathlib

import pytest

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


#: The launch line's identity fields, in fixed order: the epoch crumb field
#: leads the line, then pid, argv, rev, out, and the launch-ownership pair --
#: owner, card. `card` is the literal `unset` when the run has no board card, so
#: the MARKER is always present; only the value varies.
_LAUNCH_LINE_FIELDS = (" pid=", " argv=", " rev=", " out=", " owner=", " card=")


def _crumb_literal_text(node: ast.AST) -> str:
    """The literal characters of a crumb argument, with f-string holes dropped.

    ``f"a={x} b={y}"`` is ONE ``ast.JoinedStr`` whose constant parts are ``"a="``
    and ``" b="``; joining those recovers the field markers a structural pin
    reads, without evaluating anything.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value
            for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
    return ""


def _launch_line_findings(main: ast.AST) -> list[str]:
    """The launch line must carry every identity field, in the pinned order.

    The trail's FIRST line is the corpse's self-attribution (change
    qk-launch-attribution): an edit that drops ``rev``, ``out``, ``pid`` or
    ``argv`` — or reorders them — leaves the survivor unable to name its own
    launch, which is exactly the attribution failure that change exists to
    close. Pinned structurally off the crumb f-string's own field markers, so
    renaming the locals cannot evade it and a dropped field cannot hide behind a
    green shape.
    """
    crumbs = [
        _crumb_literal_text(arg)
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_bread"
        for arg in node.args
    ]
    launch = [text for text in crumbs if "start pid=" in text]
    if len(launch) != 1:
        return [f"expected exactly one launch line (`start pid=…`) in main(), found {len(launch)}"]
    text = launch[0]
    order = [text.find(field) for field in _LAUNCH_LINE_FIELDS]
    if -1 in order:
        missing = [f for f, at in zip(_LAUNCH_LINE_FIELDS, order, strict=True) if at == -1]
        return [f"launch line is missing identity field(s){missing}: {text!r}"]
    if order != sorted(order):
        return [
            f"launch identity fields are out of the pinned order {_LAUNCH_LINE_FIELDS}: {text!r}"
        ]
    return []


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

    # The probe abort path and the fail-closed stage gate share one discipline.
    # An aborted or
    # stage-failed run must still write a failed report and exit nonzero,
    # through the SAME single verdict return: never returned from (a collapse
    # would exit 0), never re-raised (the report would never be written), and
    # never widened past its own exception type (a crash would be labeled a
    # measured collapse).
    for node in ast.walk(main):
        if isinstance(node, ast.Try):
            for wide in node.handlers:
                caught = ast.unparse(wide.type) if wide.type is not None else "bare except"
                if caught in ("Exception", "BaseException", "bare except"):
                    findings.append(
                        f"line {wide.lineno}: `{caught}` in main() swallows failures "
                        f"into the verdict path instead of letting them surface"
                    )
    handler_specs = (
        (
            "PrefixProbeAbort",
            "aborted",
            "abort_report",
            "the abort handler",
            "the abort handler does not record the abort (`aborted = ...`), so the "
            "exit-code fallback cannot see it",
        ),
        (
            "VacuousTeacherError",
            "stage_failed",
            "stage_failure_report",
            "the stage-gate handler",
            "the stage-gate handler does not record the failure (`stage_failed = ...`), "
            "so the exit-code fallback cannot see it",
        ),
    )
    for exc_name, flag, builder, label, missing_msg in handler_specs:
        handlers = [
            h
            for node in ast.walk(main)
            if isinstance(node, ast.Try)
            for h in node.handlers
            if h.type is not None and exc_name in ast.unparse(h.type)
        ]
        if len(handlers) != 1:
            findings.append(
                f"expected exactly one `except {exc_name}` handler in main(), found {len(handlers)}"
            )
            continue
        handler = handlers[0]
        if ast.unparse(handler.type) != exc_name:
            findings.append(
                f"line {handler.lineno}: the handler catches "
                f"`{ast.unparse(handler.type)}`, not {exc_name} alone"
            )
        for node in ast.walk(handler):
            if isinstance(node, ast.Return):
                findings.append(
                    f"line {node.lineno}: {label} returns directly — a run that "
                    f"tripped it must fall through to the verdict exit, never exit 0"
                )
            if isinstance(node, ast.Raise):
                findings.append(
                    f"line {node.lineno}: {label} raises — the failure must "
                    f"become a failed report, not an uncaught exception"
                )
        recorded = [
            getattr(target, "id", None)
            for stmt in handler.body
            if isinstance(stmt, ast.Assign)
            for target in stmt.targets
        ]
        if flag not in recorded:
            findings.append(missing_msg)
        built = [
            stmt
            for stmt in handler.body
            if isinstance(stmt, ast.Assign)
            and any(getattr(t, "id", None) == "report" for t in stmt.targets)
            and isinstance(stmt.value, ast.Call)
            and getattr(stmt.value.func, "id", None) == builder
        ]
        if len(built) != 1:
            findings.append(
                f"{label} binds `report` to {builder}(...) {len(built)} "
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
    if len(bindings) != 3:
        lines = sorted({node.lineno for node in bindings})
        findings.append(
            f"main() establishes the `report` name {len(bindings)} times (lines {lines}), "
            f"expected exactly 3 (the completion literal, abort_report(...) for the probe "
            f"abort, and stage_failure_report(...) for the fail-closed stage gate)"
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
    if fail_over:
        fallback_test = ast.unparse(fail_over[0].test)
        if "aborted" not in fallback_test:
            findings.append(
                "the FAILED exit-code fallback does not consider the probe abort — an "
                "aborted run would exit 0"
            )
        if "stage_failed" not in fallback_test:
            findings.append(
                "the FAILED exit-code fallback does not consider the fail-closed stage "
                "gate — a stage-failed run would exit 0"
            )

    # The launch line IS the corpse's self-attribution (change
    # qk-launch-attribution): its construction is pinned beside the behavior it
    # names, so a later edit that drops a field reds HERE rather than shipping.
    findings.extend(_launch_line_findings(main))
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
    assert "provenance(" in src and "rho_end_source=_rho_end_source" in src


#: The default expression, anchored where it now lives: the driver names the
#: scratch root ONCE (`_scratch_root()`) so the KD
#: teacher-logit cache and the per-run checkpoint directory cannot drift apart,
#: and the `setdefault` call is a bare call to that helper. The mutations below
#: rewrite THIS expression inside the helper, which is what keeps them applied to
#: the real driver source instead of to a copy the pin no longer reads.
_SCRATCH_DEFAULT = "str(_SANDBOX_SCRATCH) if _SANDBOX_SCRATCH.is_dir() else tempfile.gettempdir()"


def _resolve_scratch_helper(tree: ast.Module, expression: ast.expr) -> ast.expr:
    """Follow a module-level helper call to the expression that helper returns.

    The structural checks below are about the DEFAULT's shape, not about the
    call's: while the default was written inline at `setdefault`, the two were the
    same node. Once the driver extracted it into a single named helper, a pin that
    read only the call would judge `_scratch_root()` — a bare Call — and report
    "no platform-scratch fallback" / "not existence-probed" for the CALL rather
    than for the default, which is vacuously true of any helper name and would
    pass three of the mutations below for the wrong reason. So the reference is
    resolved to the helper's return expression, and the checks run there.

    Bounded and fail-closed: only a module-level FunctionDef with exactly one
    Return is followed, a chain of at most four links; anything else is returned
    unchanged so the callers' own checks see (and report) it.
    """
    for _ in range(4):
        if not (isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name)):
            break
        funcs = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == expression.func.id
        ]
        if len(funcs) != 1:
            break
        returns = [
            node for node in ast.walk(funcs[0]) if isinstance(node, ast.Return) and node.value
        ]
        if len(returns) != 1:
            break
        expression = returns[0].value
    return expression


def _scratch_dir_findings(src: str) -> list[str]:
    """Findings for a scratch-dir default that cannot survive a foreign host.

    The default is load-bearing: `scale_calib` hands `QUANTIBBLER_KD_TMPDIR` to
    `tempfile.TemporaryDirectory`, which does NOT create parents, so a default
    naming a directory that only exists on one machine is a guaranteed crash
    everywhere else.

    The checks are STRUCTURAL, not token greps — an `ast.IfExp` whose condition is
    a bare `.is_dir()` call (never negated) and whose else-arm is exactly the
    platform scratch dir — because two natural one-token drifts keep every
    substring in place while destroying the invariant: `if not …is_dir()` swaps
    which host crashes, and namespacing the fallback
    (`Path(gettempdir()) / "quantibbler-kd"`) reintroduces the same missing-parent
    `FileNotFoundError` because the platform dir is the only thing guaranteed to
    exist. Over-strict is the correct failure direction: a deliberate fallback
    change forces a deliberate pin revisit.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:  # pragma: no cover - a broken driver is a bigger finding
        return [f"unparseable driver source: {exc}"]
    # `_SANDBOX_SCRATCH = Path("/opt/data/tmp")` — the literal may live in a
    # module-level constant instead of inline in the default.
    path_constants = {
        node.targets[0].id: node.value.args[0].value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "Path"
        and node.value.args
        and isinstance(node.value.args[0], ast.Constant)
        and isinstance(node.value.args[0].value, str)
    }
    defaults = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "setdefault"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "QUANTIBBLER_KD_TMPDIR"
    ]
    if len(defaults) != 1:
        return [f"expected exactly one QUANTIBBLER_KD_TMPDIR default, found {len(defaults)}"]
    expression = _resolve_scratch_helper(tree, defaults[0].args[1])
    default = ast.unparse(expression)
    referenced = {node.id for node in ast.walk(expression) if isinstance(node, ast.Name)}
    sandbox = "/opt/data/tmp"
    findings = []
    if isinstance(expression, ast.IfExp):
        test = expression.test
        if not (
            isinstance(test, ast.Call)
            and isinstance(test.func, ast.Attribute)
            and test.func.attr == "is_dir"
        ):
            findings.append(
                "the probe must be the true condition of the branch — a bare "
                f".is_dir() call, never negated or wrapped: {ast.unparse(test)}"
            )
        fallback = ast.unparse(expression.orelse)
        if fallback != "tempfile.gettempdir()":
            findings.append(
                "the fallback must be exactly the platform scratch dir, not path "
                f"arithmetic on it: {fallback}"
            )
    else:
        if "tempfile.gettempdir()" not in default:
            findings.append(f"no platform-scratch fallback in the default: {default}")
        if "is_dir()" not in default:
            findings.append(f"the sandbox path is not existence-probed: {default}")
    if sandbox not in default and not any(path_constants.get(n) == sandbox for n in referenced):
        findings.append(f"the sandbox scratch preference is gone: {default}")
    return findings


def test_driver_scratch_dir_default_is_not_a_sandbox_only_path():
    """The KD scratch default must survive a host where `/opt/data/tmp` is absent.

    It was a bare `/opt/data/tmp` — a path that exists in this sandbox and on no
    CI runner — and that is what turned the first full-python CI run red while
    every sandbox rehearsal stayed green (206/0): the run died at KD calibration
    with `FileNotFoundError` from `tempfile.TemporaryDirectory`. The default is
    now an existence-probed sandbox preference with a platform-scratch fallback.
    """
    findings = _scratch_dir_findings(_DRIVER.read_text())
    assert findings == [], findings


def _flip_the_probe(src: str) -> str:
    """One stray `not`: every token survives, the invariant inverts."""
    return src.replace(
        _SCRATCH_DEFAULT,
        "str(_SANDBOX_SCRATCH) if not _SANDBOX_SCRATCH.is_dir() else tempfile.gettempdir()",
        1,
    )


def _namespace_the_fallback(src: str) -> str:
    """One nested path: the platform scratch dir stops being guaranteed to exist."""
    return src.replace(
        _SCRATCH_DEFAULT,
        "str(_SANDBOX_SCRATCH) if _SANDBOX_SCRATCH.is_dir() else "
        'str(Path(tempfile.gettempdir()) / "quantibbler-kd")',
        1,
    )


def _a_bare_literal(src: str) -> str:
    return src.replace(_SCRATCH_DEFAULT, '"/opt/data/tmp"', 1)


def _drop_the_probe(src: str) -> str:
    return src.replace(_SCRATCH_DEFAULT, "str(_SANDBOX_SCRATCH)", 1)


def _drop_the_fallback(src: str) -> str:
    return src.replace(_SCRATCH_DEFAULT, "tempfile.gettempdir()", 1)


_DEFAULT_CALL_HEAD = 'os.environ.setdefault("QUANTIBBLER_KD_TMPDIR",'


def _a_second_default(src: str) -> str:
    return src.replace(
        _DEFAULT_CALL_HEAD,
        'os.environ.setdefault("QUANTIBBLER_KD_TMPDIR", tempfile.gettempdir())\n'
        + _DEFAULT_CALL_HEAD,
        1,
    )


@pytest.mark.parametrize(
    ("mutate", "expected", "escapes_token_grep"),
    [
        pytest.param(
            _flip_the_probe,
            "probe must be the true condition of the branch",
            True,
            id="probe-polarity-flip",
        ),
        pytest.param(
            _namespace_the_fallback,
            "fallback must be exactly the platform scratch",
            True,
            id="nested-fallback",
        ),
        pytest.param(_a_bare_literal, "no platform-scratch fallback", False, id="bare-literal"),
        pytest.param(_drop_the_probe, "not existence-probed", False, id="no-probe"),
        pytest.param(
            _drop_the_fallback, "sandbox scratch preference is gone", False, id="no-fallback"
        ),
        pytest.param(_a_second_default, "exactly one", False, id="two-defaults"),
    ],
)
def test_scratch_dir_pin_rejects_every_sandbox_only_shape(mutate, expected, escapes_token_grep):  # noqa: ANN001
    """Mutations are applied to the REAL driver source, not a synthetic snippet.

    A synthetic snippet would be rejected by the undefined-constant check rather
    than by the shape it is meant to probe, hiding the very escape it exists to
    catch — which is how the polarity flip and the nested fallback survived the
    first version of this pin.
    """
    src = _DRIVER.read_text()
    mutated = mutate(src)
    assert mutated != src, "the mutation is a no-op: the driver drifted out from under it"
    if escapes_token_grep:
        # Precisely the shapes that beat a substring pin: the `is_dir()` token, the
        # platform-scratch token and the sandbox path are ALL still present, and the
        # default is still wrong on whichever host the grep happens to run on.
        for token in ("is_dir()", "tempfile.gettempdir()", "/opt/data/tmp"):
            assert token in mutated, token
    findings = _scratch_dir_findings(mutated)
    assert any(expected in finding for finding in findings), (expected, findings)


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
_PROVENANCE_ANCHOR = "    prov = provenance(\n"
_RETURN_LINE = "    return verdict_exit_code(status)\n"
_ABORT_ANCHOR = "    except PrefixProbeAbort as exc:\n"
#: The end of the abort branch's builder call: unique because the completion
#: report spells its history line differently.
_ABORT_BUILT_ANCHOR = "            probe_history,\n        )\n"
#: The fail-closed stage-gate handler's first line: unique in the driver.
_VACUOUS_ANCHOR = "    except VacuousTeacherError as exc:\n"
#: The launch line's f-string HOLES — every one of them a field read off the
#: launch record. Anchoring on the holes rather than on a whole line keeps these
#: poisons valid under `ruff format`: a reformat may split the implicit
#: concatenation BETWEEN two holes, but never inside one.
_LAUNCH_REV_HOLE = "rev={launch.rev}"
_LAUNCH_OUT_HOLE = "out={launch.out}"
#: The launch-ownership hole: a driver that starts fires naming no owner is the
#: defect these legs pin.
_LAUNCH_OWNER_HOLE = "owner={launch.owner}"


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
            "    if not faithful or aborted is not None or stage_failed is not None:\n",
            "    if not faithful:\n",
            1,
        )
    if kind == "stage_gate_ignored_by_exit_code":
        return src.replace(
            "    if not faithful or aborted is not None or stage_failed is not None:\n",
            "    if not faithful or aborted is not None:\n",
            1,
        )
    if kind == "stage_gate_returned_from":
        return src.replace(
            _VACUOUS_ANCHOR,
            _VACUOUS_ANCHOR + "        return 1\n",
            1,
        )
    # The launch line's identity fields: each of these leaves the run starting
    # fine and the trail looking shaped, while the corpse can no longer name its
    # own launch.
    if kind == "launch_line_dropped_rev":
        return src.replace(_LAUNCH_REV_HOLE, "", 1)
    if kind == "launch_line_dropped_out":
        return src.replace(_LAUNCH_OUT_HOLE, "", 1)
    if kind == "launch_line_dropped_owner":
        return src.replace(_LAUNCH_OWNER_HOLE, "", 1)
    if kind == "launch_line_fields_reordered":
        # Swap the two holes through sentinels: the fix is to move the fields,
        # not to respell the literal text around them, so this stays honest if
        # the f-string is ever re-wrapped.
        swapped = src.replace(_LAUNCH_REV_HOLE, "@@REV@@", 1).replace(
            _LAUNCH_OUT_HOLE, "@@OUT@@", 1
        )
        return swapped.replace("@@REV@@", _LAUNCH_OUT_HOLE, 1).replace(
            "@@OUT@@", _LAUNCH_REV_HOLE, 1
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
        ("stage_gate_ignored_by_exit_code", "does not consider the fail-closed stage gate"),
        ("stage_gate_returned_from", "exactly one return"),
        ("launch_line_dropped_rev", "missing identity field"),
        ("launch_line_dropped_out", "missing identity field"),
        ("launch_line_dropped_owner", "missing identity field"),
        ("launch_line_fields_reordered", "out of the pinned order"),
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
