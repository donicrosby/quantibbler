"""The catch→report→status construction, lean-tested (``abort_report``).

Split out of ``tests/test_qk_report.py`` (surface 4 of 6); every test body is
verbatim, and the docstrings keep their scenario IDs. Scenarios
`abort_report_construction_lean_tested` and `report_single_shape_both_paths`.
The driver's `except PrefixProbeAbort` binds `report` to this builder, so the
wiring that delivers the fail-fast ask has repeatable proof with no weights, no
torch, and no CI.
"""

from __future__ import annotations

import ast
import inspect
import json
from types import SimpleNamespace

from test_qk_report import _reject_constant
from test_qk_report_driver_pin import _DRIVER_SRC, _PTQ_CONFIG

from quantibbler import qk_report
from quantibbler.qk_report import (
    COLLAPSE_LINE,
    EVAL_EVALUATOR,
    FAILED,
    OK,
    abort_report,
    dump_report,
    evaluation_record,
)

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


#: The identity fields the driver hands the builder on a real abort. The VALUES
#: are the recorded Qwen3-0.6B abort run; the KEYS are NOT written here — they
#: come from the driver's abort payload literal via `_abort_payload_keys()`, so a
#: field added to the driver's abort path cannot be silently left out of the
#: fixture that mirrors it. This is the same structural move
#: `_completion_report_keys()` makes for the completion literal. The value table
#: stays hand-pinned on purpose: the four unmeasured nulls and the
#: status/artifact_exported/probe_abort contract belong to `abort_report`
#: (asserted below), not to this mirror of the driver's inputs.
_ABORT_IDENTITY_VALUES: dict[str, object] = {
    "model": "Qwen/Qwen3-0.6B",
    "rank": 1,
    "solver": "svd",
    "scale_dtype": "float32",
    "seq_len": 64,
    "tune_seq_len": 64,
    "calibration_rows": 8,
    "eval_rows": 16,
    "max_admm_iter": 1,
    "seed": 0,
    # Ablation provenance: the recorded abort run
    # used the defaults, and the knobs must travel on the abort path too so a
    # measurement can never be mistaken for a landing.
    "factor_mode": "binary",
    "ablation": "none",
    "layer_scope": "all",
    # Paper-faithful claim + the bounded discrete flip pass.
    # Both are solver state and both must travel on every report path.
    "paper_faithful": False,
    "ste_flip_pass_budget": 0,
    "ppl_baseline_bf16": 75.90947478697456,
    "regime": {"artifact": "qpk1: +-1 bitplanes + float32 scales"},
    "probe": {"interval": 1, "batches": 1, "fp_prefix_ppl": 83.08454639156159},
    # The stage record + Algorithm 1 map: every report
    # path carries them, so an aborted or stage-failed run is as auditable as
    # a completed one.
    "stages": [],
    "stage_map": {},
    "runtime_deviations": [],
    # Evaluation verdict: the pedestal's coverage is
    # recorded beside the method verdict on every path, so an aborted run still
    # proves what it measured. Built by the production helper so this fixture
    # cannot drift from the record the driver emits.
    "evaluation": evaluation_record(
        evaluator=EVAL_EVALUATOR,
        source_tokens=298_938,
        seq_len=2048,
        selected_blocks=144,
        batch=8,
        selection_seed=1,
        selection_fingerprint="0123456789abcdef",
    ),
    "wall_seconds": 69.6,
}


def _abort_payload_keys(src: str | None = None) -> set[str]:
    """The keys the driver's abort payload carries, from source.

    `main()` hands `abort_report(...)` a dict that SPREADS the shared
    `_identity()` helper and adds the probe block; those keys together are the
    identity fields a real abort carries. Deriving them here is what makes the
    fixture structural: the completion and abort report shapes are compared as
    key sets, so the set standing in for the abort path must be read from the
    driver's payload rather than remembered beside it. `src` is overridable so a
    test can mutate the source and watch the derivation move (non-vacuity).

    Re-homed when the property moved layer: the abort identity fields used to
    be one flat literal at the call site, and the fail-closed stage-gate report
    (qk-fix-tunefp-teacher) needs the identical set. They now live in the
    shared `_identity()` helper, so the derivation follows the helper AND
    asserts the call site really spreads it instead of re-inlining a second
    copy that could drift.
    """
    text = _DRIVER_SRC if src is None else src
    tree = ast.parse(text)
    identity_keys: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_run_identity":
            for inner in ast.walk(node):
                if isinstance(inner, ast.Return) and isinstance(inner.value, ast.Dict):
                    identity_keys = {
                        key.value for key in inner.value.keys if isinstance(key, ast.Constant)
                    }
    assert identity_keys, "the shared _run_identity() payload literal is missing from the driver"
    main = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    ][0]
    for node in ast.walk(main):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "abort_report"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Dict)
        ):
            explicit = {key.value for key in node.args[1].keys if isinstance(key, ast.Constant)}
            assert any(key is None for key in node.args[1].keys), (
                "the abort payload no longer SPREADS the shared identity helper"
            )
            return identity_keys | explicit
    raise AssertionError("the abort payload dict literal is missing from main()")


def _abort_identity() -> dict[str, object]:
    """The abort fixture: driver-derived KEYS over hand-pinned VALUES."""
    keys = _abort_payload_keys()
    missing = keys - set(_ABORT_IDENTITY_VALUES)
    extra = set(_ABORT_IDENTITY_VALUES) - keys
    assert not missing and not extra, (
        "the abort fixture values no longer cover the driver's abort payload "
        f"keys: missing {sorted(missing)}, extra {sorted(extra)}"
    )
    return {key: _ABORT_IDENTITY_VALUES[key] for key in sorted(keys)}


_ABORT_IDENTITY = _abort_identity()
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


def test_abort_fixture_keys_are_derived_from_the_driver_not_remembered():
    """The abort fixture's key set is READ from the driver's abort payload.

    The completion literal is already pinned by derivation
    (`_completion_report_keys()`); the abort fixture used to mirror it by hand,
    so a key added to the driver's abort payload reddened the guard only until
    someone remembered to edit the literal too. The derivation closes that gap.
    Non-vacuity: a field added to the driver's payload grows the derived set — a
    fixture that merely agreed on a frozen list could not do this.
    """
    keys = _abort_payload_keys()
    assert set(_ABORT_IDENTITY) == keys
    # The field this cycle added arrives through the derivation, not a hand edit.
    assert "tune_seq_len" in keys
    # The unmeasured nulls belong to `abort_report`, not to the driver's payload:
    # they are the contract asserted elsewhere in this file, not inputs.
    assert "ppl_quant_native" not in keys
    # A payload that gains a field yields a fixture that gains it. The tail is
    # unique to the abort literal (16-space keys, 12-space `},`); the completion
    # literal closes at 8 spaces and must not match.
    tail = '"wall_seconds": round(time.time() - t0, 1),\n            },'
    assert _DRIVER_SRC.count(tail) == 1, "the abort payload literal moved; re-anchor"
    mutated = _DRIVER_SRC.replace(
        tail,
        '"wall_seconds": round(time.time() - t0, 1),\n'
        '                "bogus_probe_field": 1,\n            },',
        1,
    )
    assert mutated != _DRIVER_SRC
    assert "bogus_probe_field" in _abort_payload_keys(mutated)


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
