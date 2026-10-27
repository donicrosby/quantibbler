"""The resume record: a reader-facing restart declaration, never a gate.

Scenarios (spec delta ``specs/qk-ptq``): ``resume_record_on_resumed_run``,
``resume_record_fresh_run_consistent_shape``,
``resume_identity_is_evidence_not_an_assertion``,
``resume_record_is_a_record_not_a_gate`` and ``resume_record_on_failure_paths``.

The builder lives in ``quantibbler.qk_report``, deliberately OUTSIDE the
torch-importing driver, so these execute with no skips in a lean environment —
the same split that gives the verdict and the abort builder repeatable
coverage. The runtime half lives beside the driver's other real-run tests
(``tests/ptq/test_resume_report.py``): the real driver emitting the record on a
fresh and on a resumed launch, and the fingerprint guard refusing a poisoned
restore before any report is written. What this tier cannot show is stated
plainly below where it matters.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

from test_qk_report import _reject_constant
from test_qk_report_abort import _ABORT_HISTORY, _ABORT_IDENTITY, _abort_exc

from quantibbler import serialization
from quantibbler.qk_report import (
    COLLAPSE_LINE,
    OK,
    OK_LINE,
    abort_report,
    dump_report,
    exit_code,
    fidelity_verdict,
    probe_aborts,
    resume_record,
    stage_failure_report,
    verdict,
)

#: The admission members the restored checkpoint's own fingerprint record
#: carries. The builder threads this dict VERBATIM, so the lean fixture is free
#: to name any members; the structural claim (the members are the fingerprint
#: record's own provenance tuple plus ``solver_device``) is a property of the
#: driver's read and is asserted at the wired tier, where the real tuple is
#: importable.
_ADMISSION = {
    "fingerprint": "9f32ae6400000000000000000000000000000000000000000000000000000000",
    "model_identity": {
        "digest": "a1b2c3d400000000000000000000000000000000000000000000000000000000",
        "modules": {"layers.0.mlp.down_proj": [128, 64, "bfloat16"]},
    },
    "solver_backend": "torch",
    "compute_dtype": "float64",
    "solver": "admm",
    "rank": 512,
    "factor_mode": "binary",
    "admm_reg": 0.01,
    "rho_start": 1.0,
    "rho_end": 12.0,
    "solver_device": "cuda",
}

_RECORD_KEYS = (
    "restored_blocks",
    "checkpoint_dir",
    "image",
    "image_digest",
    "admitted_by_fingerprint",
    "resumed",
)


def _resumed() -> dict:
    return resume_record(
        resumed=True,
        restored_blocks=[1, 2, 3, 4, 5],
        checkpoint_dir="/out/checkpoints",
        image="quantibbler-ptq:rocm-official-v6b",
        image_digest="sha256:dcd9f7200000000000000000000000000000000000000000000000000000000",
        admitted_by_fingerprint=_ADMISSION,
    )


def _fresh() -> dict:
    return resume_record(
        resumed=False,
        restored_blocks=[],
        checkpoint_dir=None,
        image="unknown",
        image_digest="unknown",
        admitted_by_fingerprint=None,
    )


def test_scn_resume_record_on_resumed_run():
    """``resume_record_on_resumed_run``: a resumed launch declares its restart.

    The five fields the record exists for: the flag, the blocks a prior launch
    left on disk, the directory they came from, the launch environment's image
    identity, and the admission the restore was checked against.
    """
    record = _resumed()
    assert record["resumed"] is True
    assert record["restored_blocks"] == [1, 2, 3, 4, 5]
    assert record["checkpoint_dir"] == "/out/checkpoints"
    assert record["image"] == "quantibbler-ptq:rocm-official-v6b"
    assert record["image_digest"].startswith("sha256:")
    assert record["admitted_by_fingerprint"] == _ADMISSION
    # The record is report material: strict JSON, no non-standard tokens.
    text = json.dumps(record, allow_nan=False)
    assert json.loads(text, parse_constant=_reject_constant) == record


def test_scn_resume_record_fresh_run_consistent_shape():
    """``resume_record_fresh_run_consistent_shape``: same keys, empty values.

    A fresh launch carries the record too — with nothing restored — so a strict
    consumer never KeyErrors on either shape. ``checkpoint_dir`` is null rather
    than the directory this launch will write to: the field names where blocks
    were RESTORED FROM, and a fresh run restores from nowhere.
    """
    record = _fresh()
    assert record == {
        "resumed": False,
        "restored_blocks": [],
        "checkpoint_dir": None,
        "image": "unknown",
        "image_digest": "unknown",
        "admitted_by_fingerprint": None,
    }
    # The two shapes are single-shaped: identical key sets, so a reader that
    # indexes the record unconditionally works on both.
    assert set(record) == set(_resumed()) == set(_RECORD_KEYS)


def test_scn_resume_identity_is_evidence_not_an_assertion():
    """``resume_identity_is_evidence_not_an_assertion``: copied, never claimed.

    The admission members travel through the builder verbatim — a reader
    compares them against the launch trail himself — and no computed
    ``identical_bytes`` boolean is emitted: cross-reboot source-byte identity is
    a host-side diff across two tips, which the in-container driver (no ``git``)
    cannot measure, so a ``True`` there would be forged provenance.
    """
    record = _resumed()
    admission = record["admitted_by_fingerprint"]
    # Verbatim: not a re-derived subset, not a normalised copy.
    assert admission is _ADMISSION
    assert admission["model_identity"] == _ADMISSION["model_identity"]
    assert admission["fingerprint"] == _ADMISSION["fingerprint"]
    assert admission["solver_device"] == "cuda"
    blob = json.dumps(record)
    assert "identical_bytes" not in blob
    assert "identical_bytes" not in json.dumps(_fresh())


def test_scn_resume_record_is_a_record_not_a_gate():
    """``resume_record_is_a_record_not_a_gate``: garbage in it decides nothing.

    Adversarial garbage at every field — wrong types, a string where a list
    belongs — must leave the verdict, the exit code, the band edges and the
    report write byte-identical. The comparison is DISCRIMINATING: both records
    are written by the same ``dump_report`` and read by the same ``verdict``, so
    a gate that consulted the record would separate them.
    """
    well_formed = {"rank": 1, "ppl_ratio": 2.5, "status": OK, "resume": _resumed()}
    garbage = {
        "rank": 1,
        "ppl_ratio": 2.5,
        "status": OK,
        "resume": {
            "resumed": "banana",
            "restored_blocks": ["x"],
            "checkpoint_dir": 42,
            "image": None,
            "image_digest": 7,
            "admitted_by_fingerprint": "junk",
        },
    }
    decisions = []
    for report in (well_formed, garbage):
        payload, faithful = dump_report(report)
        assert faithful is True
        parsed = json.loads(payload, parse_constant=_reject_constant)
        status = verdict(parsed["ppl_ratio"])
        decisions.append((status, exit_code(status), parsed["status"], parsed["resume"]))
    assert decisions[0][:3] == decisions[1][:3] == (OK, 0, OK)
    # The record itself is written through unchanged — the sanitizer serializes
    # it, it does not repair or read it.
    assert decisions[0][3] == well_formed["resume"]
    assert decisions[1][3] == garbage["resume"]
    # The band edges are the pinned paper-derived constants, not anything the
    # record can move: OK_LINE closes `ok`, COLLAPSE_LINE closes `degraded`, and
    # one step past it is the only `failed`.
    assert serialization.VERDICT_BAND == (OK, "degraded", "failed")
    assert verdict(OK_LINE) == OK
    assert verdict(COLLAPSE_LINE) == "degraded"
    assert verdict(COLLAPSE_LINE + 1.0) == "failed"
    # ... and no gate accepts a record or a report at all: there is no parameter
    # for the record to arrive through, so "never read by a gate" is structural.
    for gate in (verdict, exit_code, probe_aborts, fidelity_verdict):
        params = set(inspect.signature(gate).parameters)
        assert not (params & {"resume", "record", "report"}), gate


def test_scn_resume_record_on_failure_paths():
    """``resume_record_on_failure_paths``: the declaration survives an early death.

    The record is built from LAUNCH state and threaded into the identity fields,
    so a run that dies before the end-of-run gate still declares its restart —
    which is the whole point: the restart is most interesting on a run that did
    not finish.
    """
    # The identity a real abort carries is the driver's shared `_run_identity`
    # payload, which holds the record; the fixture stands in for it here, and the
    # wired tier asserts the real payload carries it.
    identity = dict(_ABORT_IDENTITY, resume=_resumed())
    aborted = abort_report(_abort_exc(), identity, _ABORT_HISTORY)
    failed = stage_failure_report(
        SimpleNamespace(
            stage="tune_fp",
            block="layers.3",
            reason="zero_first_step_gradient",
            gradient_norm=0.0,
            initial_residual_norm=0.25,
        ),
        dict(identity, probe_abort=None),
        [],
    )
    for shape in (aborted, failed):
        assert set(shape["resume"]) == set(_RECORD_KEYS)
        assert shape["resume"]["resumed"] is True
        assert shape["resume"]["restored_blocks"]
    assert aborted["status"] == "failed" and failed["status"] == "failed"


def test_a_report_written_before_the_record_existed_still_parses():
    """Back-compat: the key is an extra, and the schema version did not move.

    The v6b artifact's own report carries no such key, so a reader built now
    must still accept it — and a reader built BEFORE this record must still
    accept a report carrying it. Both directions follow from the report model
    allowing extras and from the version staying put, which is what this pins.
    """
    legacy = {
        "status": "ok",
        "model": "Qwen/Qwen3-0.6B",
        "rank": 1,
        "solver": "admm",
        "ppl_ratio": 2.5,
        "wall_seconds": 7338.5,
    }
    assert "resume" not in legacy
    parsed = serialization.QuantizeReport.model_validate(legacy)
    assert parsed.ppl_ratio == 2.5
    payload, faithful = dump_report(legacy)
    assert faithful is True
    # No key is invented on the way out either: absence stays absence.
    assert "resume" not in json.loads(payload, parse_constant=_reject_constant)
    # The record rides as an EXTRA beside the contract, so the version that a
    # reader keys its shape decisions on does not move.
    assert serialization.SCHEMA_VERSION == 1
    assert "resume" not in serialization.QuantizeReport.model_fields
