"""The stage journal through the REAL driver (qk-fix-tunefp-teacher post-fire row).

Scenario: ``required_stage_execution_reads_the_same_record``. The unit tier
(``tests/ptq/test_stage_journal.py``) proves the resume mechanism against the
pipeline; this file proves the SHIPPED WIRING and the shipped payloads:

* the driver writes the journal beside the checkpoints, one line per event;
* every line it writes passes the reader's own shape gate (the production
  payloads, sanitized by ``_reportable_stage_payload``, are what the reader will
  be handed on a real resume);
* a resumed driver run's condition record is IDENTICAL to the uninterrupted
  run's — the statement that makes a tail-resumed artifact's label mean the same
  thing as a clean run's;
* with the journal removed, the five required-stage conditions of the stages a
  resume does not re-run fall, which is the behavior before the journal existed
  and the reason the fix is owed. Nothing is synthesized.

The label itself is the conjunction of the conditions (a fact of the landed
verdict, not a property of the journal), so the condition-record equality above
is asserted rather than a label equality: this drive uses the gate-shaped
budgets, whose measured polish is a deviation the label reads regardless of the
resume.

Tier: three real driver runs on the tiny fixture (the resume runs restore every
block from disk, so they re-run no solve).
"""

from __future__ import annotations

import json

import pytest
from test_driver_tune_seq_len import (  # noqa: F401  (pytest fixtures)
    _report,
    _run_driver,
    offline_corpus,
    tiny_model,
)

from quantibbler.ptq.checkpoints import stage_log_path
from quantibbler.qk_report import stage_journal_record

#: The required stages that only the interrupted launch executes — the five the
#: journal exists to recover so a resumed run is not falsely denied.
_SOLVE_CONDITIONS = (
    "required_capture_executed",
    "required_tune_fp_executed",
    "required_lb_admm_executed",
    "required_ste_executed",
    "required_sign_extract_pack_executed",
)

_REQUIRED_CONDITIONS = (
    "required_phase1_statistics_executed",
    *_SOLVE_CONDITIONS,
    "required_tune_scales_kd_executed",
)


def _journal_lines(ckpt_dir):
    path = stage_log_path(ckpt_dir)
    assert path.is_file(), f"the driver wrote no stage journal at {path}"
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


@pytest.mark.integration
def test_a_resumed_driver_run_records_every_stage_that_fired(
    offline_corpus,  # noqa: ANN001, F811
    tiny_model,  # noqa: ANN001, F811
    tmp_path,  # noqa: ANN001
):
    out_dir = tmp_path / "journal-artifact"
    ckpt_dir = out_dir / "checkpoints"

    proc = _run_driver(offline_corpus, tiny_model, out_dir)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    fresh = _report(out_dir)
    assert all(fresh["fidelity_conditions"][key] for key in _REQUIRED_CONDITIONS), fresh[
        "fidelity_conditions"
    ]

    lines = _journal_lines(ckpt_dir)
    for line in lines:
        # The production payloads — tensors already collapsed by the recorder —
        # are exactly what a resume hands the reader's shape gate.
        stage_journal_record(line)
    names = [line["event"] for line in lines]
    for stage in ("capture", "tune_fp", "lb_admm", "ste", "sign_extract_pack", "tune_scales_kd"):
        assert stage in names, f"{stage} is not journaled: {sorted(set(names))}"

    # --- the resume, journal intact: the record must not degrade -----------
    proc = _run_driver(offline_corpus, tiny_model, out_dir)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    resumed = _report(out_dir)
    assert resumed["fidelity_conditions"] == fresh["fidelity_conditions"], (
        "a resumed run's condition record differs from the uninterrupted run's: the journal did "
        "not restore the evidence of the stages the resume skipped"
    )

    # --- the journal removed: the pre-journal behavior, exactly ------------
    stage_log_path(ckpt_dir).unlink()
    proc = _run_driver(offline_corpus, tiny_model, out_dir)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    without = _report(out_dir)
    fell = [
        key
        for key in _REQUIRED_CONDITIONS
        if without["fidelity_conditions"][key] != fresh["fidelity_conditions"][key]
    ]
    assert set(fell) == set(_SOLVE_CONDITIONS), (
        f"with no journal the conditions that fell are {sorted(fell)}; expected exactly the five "
        "stages a resume does not re-run (Phase-1 and the scale stage fire on every launch)"
    )
