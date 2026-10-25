"""The Algorithm 1 stage map, per-stage row time, and the ``paper_faithful`` verdict.

Split out of ``tests/test_qk_report.py`` (surface 5 of 6); every test body is
verbatim, and the docstrings keep their scenario IDs: the stage inventory
(qk-fix-tunefp-teacher), per-stage row time (qk-ptq-speedups, scenario
``ste_tunefp_gpu_zero_algorithm_change``), the computed label
(`label_flips_on_runtime_deviations` / `label_flips_on_mechanism_deviations`),
the required-stage evidence (`required_stage_execution_reads_the_same_record`),
and the calibration geometry deviations (qk-geometry-conformance).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_qk_report_abort import (
    _ABORT_HISTORY,
    _ABORT_IDENTITY,
    _abort_exc,
    _completion_report_keys,
)

from quantibbler import qk_report
from quantibbler.qk_report import (
    FAILED,
    abort_report,
    dump_report,
    fidelity_conditions,
    fidelity_verdict,
    geometry_deviations,
    stage_failure_report,
    stage_map,
    stage_row_times,
)

# --------------------------------------------------------------------------
# qk-fix-tunefp-teacher: the Algorithm 1 stage map and the fail-closed report
# --------------------------------------------------------------------------


def _stage_fixture() -> list:
    return [
        ("phase1_statistics", {"linears": ["attn.q"], "num_linears": 7, "chunk_rows": None}),
        ("capture", {"teacher_source": "frozen_reference_copy", "y_b_stream": "fp"}),
        ("tune_fp", {"gradient_norm": 1.5, "initial_residual_norm": 0.4, "steps": 8}),
        ("lb_admm", {"polish_rounds": 4}),
        (
            "ste",
            {
                "per_linear": {
                    "attn.q": {"declared": "paper_ste", "zero_crossings": False},
                    "attn.o": {"declared": "discrete_flip_pass", "flip_pass": {"accepted_u": 1}},
                }
            },
        ),
        ("sign_extract_pack", {"extracted": 7}),
        ("tune_scales_kd", {"artifact": True}),
    ]


def test_stage_map_resolves_every_algorithm1_stage_and_names_deviations():
    """The faithful-run question, answered from the record alone."""
    mapped = stage_map(_stage_fixture())
    for key in (
        "phase1_statistics",
        "capture",
        "tune_fp",
        "lb_admm",
        "ste",
        "sign_extract_pack",
        "tune_scales_kd",
    ):
        assert key in mapped, key
        assert mapped[key]["algorithm1_lines"], key
        assert mapped[key]["description"], key
    # Phase 1 is its own pre-loop pass now, not a by-product of the
    # per-block capture.
    assert mapped["phase1_statistics"]["invocations"] == 1
    assert mapped["tune_fp"]["invocations"] == 1
    assert mapped["tune_fp"]["algorithm1_lines"].startswith("12")
    assert mapped["ste"]["algorithm1_lines"] == "19"
    assert mapped["tune_scales_kd"]["algorithm1_lines"] == "26-27"
    assert mapped["ste"]["discrete_flip_pass"] is True
    assert mapped["tune_fp"]["vacuous_events"] == 0
    # Deviations are NAMED, not silent: the extra polish stage and the
    # informed-assumption objective both carry text. The statistics stage
    # carries NONE — the hoist removed the compressed-prefix source deviation.
    assert "polish" in mapped["lb_admm"]["deviation"]
    assert "informed assumption" in mapped["tune_fp"]["deviation"]
    assert mapped["phase1_statistics"]["deviation"] is None
    assert "compressed-prefix" not in json.dumps(mapped["phase1_statistics"])


def test_stage_map_names_missing_stages_as_not_executed_never_omits_them():
    """A stage that did not run is recorded as such, so a gap cannot hide."""
    empty = stage_map([])
    assert set(empty) == set(stage_map(_stage_fixture()))
    assert empty["capture"]["executed"] is False
    assert empty["tune_fp"]["executed"] is False
    assert empty["tune_fp"]["invocations"] == 0
    assert empty["ste"]["discrete_flip_pass"] is False
    assert empty["phase1_statistics"]["executed"] is False


# --------------------------------------------------------------------------
# qk-ptq-speedups: per-stage row-time, carried by the stage inventory
# --------------------------------------------------------------------------


def test_stage_map_reports_row_time_per_stage_row():
    """Scenario ``ste_tunefp_gpu_zero_algorithm_change``: per-stage row-time is
    reported — one TuneFP row per fire, one STE row per refined linear."""
    stages = [
        ("tune_fp", {"elapsed_s": 12.0, "steps": 8}),
        ("tune_fp", {"elapsed_s": 6.0, "steps": 8}),
        ("lb_admm", {"polish_rounds": 0}),
        ("ste", {"per_linear": {"attn.q": {"elapsed_s": 3.0}, "attn.o": {"elapsed_s": 9.0}}}),
    ]
    mapped = stage_map(stages)
    assert mapped["tune_fp"]["rows"] == 2
    assert mapped["tune_fp"]["elapsed_s"] == 18.0
    assert mapped["tune_fp"]["row_time_s"] == 9.0
    # The STE rows are the refined LINEARS, which is the unit the stage's cost is
    # quoted in — not the block's invocations.
    assert mapped["ste"]["rows"] == 2
    assert mapped["ste"]["elapsed_s"] == 12.0
    assert mapped["ste"]["row_time_s"] == 6.0
    # A stage that ran without recording a time reports null, never 0 s: a zero
    # reads as "this stage was free" where the truth is "nobody measured it".
    assert mapped["lb_admm"]["rows"] == 1
    assert mapped["lb_admm"]["elapsed_s"] is None
    assert mapped["lb_admm"]["row_time_s"] is None
    # A stage that did not run reports no rows at all.
    assert mapped["tune_scales_kd"]["rows"] == 0
    assert mapped["tune_scales_kd"]["row_time_s"] is None


def test_stage_row_times_do_not_count_a_skipped_fire_or_a_null_reading():
    """A skipped fire is not a row, and an unreadable time is not a free stage."""
    stages = [
        ("tune_fp", {"skipped": "block_carries_no_second_target", "elapsed_s": 0.5}),
        ("tune_fp", {"elapsed_s": 4.0}),
        ("capture", {"elapsed_s": float("inf")}),
        ("ste", {"per_linear": {"a": {"elapsed_s": 2.0}, "b": {}}}),
    ]
    times = stage_row_times(stages)
    assert times["tune_fp"] == {"rows": 1, "elapsed_s": 4.0, "row_time_s": 4.0}
    # The STE record's own time is not substituted for its rows' missing readings.
    assert times["ste"] == {"rows": 2, "elapsed_s": 2.0, "row_time_s": 2.0}
    assert times["capture"] == {"rows": 1, "elapsed_s": None, "row_time_s": None}


def test_stage_row_times_read_both_record_shapes():
    """The driver carries ``(event, info)`` pairs and the payload spells them as
    dicts; one reader, both shapes (the ``_stage_pair`` contract)."""
    pairs = [("tune_fp", {"elapsed_s": 2.0})]
    dicts = [{"event": "tune_fp", "elapsed_s": 2.0}]
    assert stage_row_times(pairs) == stage_row_times(dicts)
    assert stage_row_times([]) == {}


# --------------------------------------------------------------------------
# qk-fix-tunefp-teacher: the paper_faithful label is a COMPUTED verdict
# --------------------------------------------------------------------------


def _clean_stages() -> list:
    """The mechanism-clean record of a faithful run (no deviations at all)."""
    return [
        ("phase1_statistics", {"num_linears": 7}),
        ("capture", {"teacher_source": "frozen_reference_copy", "y_b_stream": "fp"}),
        (
            "tune_fp",
            {
                "gradient_norm": 1.5,
                "initial_residual_norm": 0.4,
                "steps": 8,
                "marker": "tune_fp_loss: informed_assumption",
            },
        ),
        ("lb_admm", {"polish_rounds": 0}),
        (
            "ste",
            {
                "per_linear": {
                    "attn.q": {"declared": "paper_ste", "zero_crossings": True},
                    "mlp.down_proj": {"declared": "paper_ste", "zero_crossings": False},
                }
            },
        ),
        ("sign_extract_pack", {"extracted": 7}),
        ("tune_scales_kd", {"artifact": True}),
    ]


#: The stages the paper's method cannot skip, spelled out here so the test
#: states the set independently of the module it is auditing.
_REQUIRED_STAGES = (
    "phase1_statistics",
    "capture",
    "tune_fp",
    "lb_admm",
    "ste",
    "sign_extract_pack",
    "tune_scales_kd",
)


def test_fidelity_label_is_true_only_when_every_condition_holds():
    """The rehearsal shape: flag + clean geometry + clean mechanism -> true."""
    label, conditions = fidelity_verdict("admm", [], _clean_stages(), requested=True)
    assert label is True
    assert set(conditions) == {
        # Lane 2's stream-provenance condition: the verdict now denies a faithful
        # label when a capture lacks fp-stream provenance, so it is part of the
        # exact condition set a clean run carries.
        "capture_is_fp_stream",
        "paper_faithful_flag",
        "no_runtime_deviations",
        "polish_off",
        "no_discrete_flip_pass",
        "no_scale_only_rescope",
        "no_vacuous_teacher",
        "no_stage_failure",
        "solver_is_admm",
        *(f"required_{key}_executed" for key in _REQUIRED_STAGES),
    }
    assert all(conditions.values()), conditions


def test_fidelity_label_flips_on_runtime_deviations():
    """`label_flips_on_runtime_deviations`: any deviation denies the claim."""
    deviations = geometry_deviations(2048, 512, 16)
    label, conditions = fidelity_verdict("admm", deviations, _clean_stages(), requested=True)
    assert label is False
    assert conditions["no_runtime_deviations"] is False
    # ... and the deviation list is what denied it, not some other condition.
    assert all(value for key, value in conditions.items() if key != "no_runtime_deviations")


@pytest.mark.parametrize(
    ("name", "mutate", "denied"),
    [
        (
            "polish-ran",
            lambda stages: [
                ("lb_admm", {"polish_rounds": 4}) if e == "lb_admm" else (e, i) for e, i in stages
            ],
            "polish_off",
        ),
        (
            "discrete-flip-pass-fired",
            lambda stages: [
                (
                    e,
                    {
                        "per_linear": {
                            "attn.o": {
                                "declared": "discrete_flip_pass",
                                "flip_pass": {"accepted_u": 1},
                            }
                        }
                    },
                )
                if e == "ste"
                else (e, i)
                for e, i in stages
            ],
            "no_discrete_flip_pass",
        ),
        (
            "scale-only-rescope",
            lambda stages: [
                (
                    e,
                    {
                        "per_linear": {
                            "attn.o": {
                                "declared": "scale_only",
                                "rescope_marker": "ste_scale_only_declared",
                            }
                        }
                    },
                )
                if e == "ste"
                else (e, i)
                for e, i in stages
            ],
            "no_scale_only_rescope",
        ),
        (
            "vacuous-teacher",
            lambda stages: [*stages, ("tune_fp_vacuous_teacher", {"gradient_norm": 0.0})],
            "no_vacuous_teacher",
        ),
    ],
)
def test_fidelity_label_flips_on_mechanism_deviations(name, mutate, denied):  # noqa: ANN001
    """`label_flips_on_mechanism_deviations`: each condition is recorded."""
    label, conditions = fidelity_verdict("admm", [], mutate(_clean_stages()), requested=True)
    assert label is False, name
    assert conditions[denied] is False, name
    assert all(value for key, value in conditions.items() if key != denied), (name, conditions)


def test_fidelity_label_flips_on_a_non_admm_solver_and_on_a_stage_failure():
    """The production SVD ablation cannot carry the label; nor can a failure."""
    label, conditions = fidelity_verdict("svd", [], _clean_stages(), requested=True)
    assert label is False and conditions["solver_is_admm"] is False
    # An unrequested claim is false no matter how clean everything else is.
    label, conditions = fidelity_verdict("admm", [], _clean_stages(), requested=False)
    assert label is False and conditions["paper_faithful_flag"] is False
    # A tripped stage gate denies the claim even on a clean-looking record.
    label, conditions = fidelity_verdict(
        "admm", [], _clean_stages(), requested=True, stage_failure=True
    )
    assert label is False and conditions["no_stage_failure"] is False


def test_informed_assumptions_do_not_flip_the_label():
    """Informed-assumption markers are documentation, not deviations."""
    stages = _clean_stages()
    marker_stages = [
        (
            e,
            {**i, "marker": "tune_fp_loss: informed_assumption"} if e == "tune_fp" else i,
        )
        for e, i in stages
    ]
    label, _ = fidelity_verdict("admm", [], marker_stages, requested=True)
    assert label is True
    # The marker is nowhere in the condition record's inputs.
    assert "informed" not in json.dumps(
        fidelity_conditions("admm", [], marker_stages, requested=True)
    )


def test_fidelity_conditions_read_the_report_shaped_stage_records():
    """The report spells stages `{"event": ...}`; the verdict must read both."""
    report_shaped = [{"event": event, **info} for event, info in _clean_stages()]
    label, conditions = fidelity_verdict("admm", [], report_shaped, requested=True)
    assert label is True and all(conditions.values())


# --------------------------------------------------------------------------
# qk-fix-tunefp-teacher (planner ruling 2026-10-21): the claim also owes
# EVIDENCE that every required stage fired. An empty record used to certify
# nothing while the label read green — a label with no content behind it is
# the failure class this diff cycle exists to kill.
# Scenario `required_stage_execution_reads_the_same_record`.
# --------------------------------------------------------------------------


def _required_conditions(conditions: dict) -> dict:
    """The required-stage slice of a condition record."""
    return {key: value for key, value in conditions.items() if key.startswith("required_")}


def test_the_required_set_is_the_stage_inventorys_own_column():
    """The required set is DATA in `_STAGE_LINES`, never a second list.

    The set is pinned twice, independently: the inventory's `required` column
    against this file's literal, and the condition keys the verdict reports
    against that same column — so the stage map and the verdict cannot drift.
    """
    marked = {key for key, _lines, _desc, _dev, required in qk_report._STAGE_LINES if required}
    assert marked == set(_REQUIRED_STAGES)
    label, conditions = fidelity_verdict("admm", [], _clean_stages(), requested=True)
    assert set(_required_conditions(conditions)) == {f"required_{key}_executed" for key in marked}
    assert label is True


def test_the_verdict_reads_the_required_column_as_data(monkeypatch):  # noqa: ANN001
    """Mutating the inventory moves the conditions with it.

    Proof that the checked set is not a hardcoded copy beside the check: relax
    one stage's `required` marker and its condition disappears from the record.
    """
    relaxed = tuple(
        (key, lines, description, deviation, required and key != "lb_admm")
        for key, lines, description, deviation, required in qk_report._STAGE_LINES
    )
    monkeypatch.setattr(qk_report, "_STAGE_LINES", relaxed)
    stages = [entry for entry in _clean_stages() if entry[0] != "lb_admm"]
    label, conditions = fidelity_verdict("admm", [], stages, requested=True)
    assert "required_lb_admm_executed" not in conditions
    assert all(_required_conditions(conditions).values())
    assert label is True


def test_an_empty_stage_record_certifies_nothing():
    """The reproduced hole: `("admm", [], [])` returned EVERY condition green."""
    label, conditions = fidelity_verdict("admm", [], [], requested=True)
    assert label is False
    required = _required_conditions(conditions)
    assert required == {f"required_{key}_executed": False for key in _REQUIRED_STAGES}
    # ... and it was the missing evidence that denied the claim, not something
    # else the record happened not to carry.
    assert all(value for key, value in conditions.items() if key not in required)


def test_a_dropped_required_stage_is_named_by_its_condition():
    """The poison direction: a required stage that never fired names itself."""
    for dropped in _REQUIRED_STAGES:
        stages = [entry for entry in _clean_stages() if entry[0] != dropped]
        label, conditions = fidelity_verdict("admm", [], stages, requested=True)
        required = _required_conditions(conditions)
        assert label is False, dropped
        assert required[f"required_{dropped}_executed"] is False, dropped
        # Only that stage's condition fell: a dropped stage names ITSELF.
        assert [key for key, value in required.items() if not value] == [
            f"required_{dropped}_executed"
        ], dropped
        assert all(value for key, value in conditions.items() if key not in required), dropped


def test_a_skip_marked_event_is_recorded_not_executed():
    """`tune_fp` carrying `skipped` fired nothing (the degenerate-block path)."""
    stages = [
        ("tune_fp", {"skipped": "no_still_fp_remainder", "steps": 0, "vacuous": False})
        if event == "tune_fp"
        else (event, info)
        for event, info in _clean_stages()
    ]
    label, conditions = fidelity_verdict("admm", [], stages, requested=True)
    assert label is False
    assert conditions["required_tune_fp_executed"] is False
    assert all(
        value
        for key, value in _required_conditions(conditions).items()
        if key != "required_tune_fp_executed"
    )


def test_an_fp_measurement_event_is_recorded_not_executed():
    """The driver appends `tune_scales_kd` even in `factor_mode='fp'`.

    Its payload's `artifact` field is the honesty: the stage is skipped
    entirely for a measurement, so the event is recorded but did not execute.
    """
    stamp = {"factor_mode": "fp", "artifact": False}
    stages = [
        ("tune_scales_kd", stamp) if event == "tune_scales_kd" else (event, info)
        for event, info in _clean_stages()
    ]
    label, conditions = fidelity_verdict("admm", [], stages, requested=True)
    assert label is False
    assert conditions["required_tune_scales_kd_executed"] is False
    assert all(
        value
        for key, value in _required_conditions(conditions).items()
        if key != "required_tune_scales_kd_executed"
    )


def test_a_per_block_skip_does_not_false_deny_a_run_level_execution():
    """The count is run-level: one genuine fire survives a later block's skip."""
    stages = [*_clean_stages(), ("tune_fp", {"skipped": "no_still_fp_remainder", "steps": 0})]
    label, conditions = fidelity_verdict("admm", [], stages, requested=True)
    assert label is True and all(conditions.values()), conditions


def test_no_diagnostic_run_is_denied_by_the_required_stage_conditions():
    """These conditions never become the denier of a diagnostic run."""
    label, conditions = fidelity_verdict("admm", [], _clean_stages(), requested=False)
    assert label is False
    assert conditions["paper_faithful_flag"] is False
    assert all(_required_conditions(conditions).values())
    # An fp-mode measurement legitimately skipped the scale stage; it is denied
    # by its own (absent) claim, not newly by a required-stage condition.
    measurement = [
        ("tune_scales_kd", {"factor_mode": "fp", "artifact": False})
        if event == "tune_scales_kd"
        else (event, info)
        for event, info in _clean_stages()
    ]
    label, conditions = fidelity_verdict("admm", [], measurement, requested=False)
    assert label is False
    assert conditions["paper_faithful_flag"] is False


def test_abort_and_stage_failure_reports_carry_a_false_verdict_with_reasons():
    """A run that did not finish cannot claim fidelity, and says which why."""
    identity = dict(_ABORT_IDENTITY, paper_faithful=True)
    aborted = abort_report(_abort_exc(), identity, _ABORT_HISTORY)
    assert aborted["paper_faithful"] is False
    assert aborted["fidelity_conditions"]["paper_faithful_flag"] is True
    assert aborted["fidelity_conditions"]["no_stage_failure"] is False
    # The identity fields are not mutated: what travels there is the REQUESTED
    # flag, and the report's label is the verdict.
    assert identity["paper_faithful"] is True

    failure = stage_failure_report(
        SimpleNamespace(
            stage="tune_fp",
            block="layers.3",
            reason="zero_first_step_gradient",
            gradient_norm=0.0,
            initial_residual_norm=0.25,
        ),
        identity,
        [{"event": "capture", "y_b_stream": "fp"}],
    )
    assert failure["paper_faithful"] is False
    assert failure["fidelity_conditions"]["no_stage_failure"] is False
    assert failure["fidelity_conditions"]["paper_faithful_flag"] is True
    # The required-stage conditions ride EVERY report path, so the verdict is
    # auditable from whichever shape the run left behind. This failure record
    # holds only its one fired stage: that one is green and the missing
    # evidence names itself.
    for shape in (aborted, failure):
        assert set(shape["fidelity_conditions"]) >= {
            f"required_{key}_executed" for key in _REQUIRED_STAGES
        }
    assert failure["fidelity_conditions"]["required_capture_executed"] is True
    assert failure["fidelity_conditions"]["required_sign_extract_pack_executed"] is False


def test_stage_failure_report_is_failed_and_carries_both_guard_norms():
    """A fidelity-claiming run that trips the gate leaves forensics, not a crash."""
    exc = SimpleNamespace(
        stage="tune_fp",
        block="layers.3",
        reason="zero_first_step_gradient",
        gradient_norm=0.0,
        initial_residual_norm=0.25,
    )
    report = stage_failure_report(
        exc,
        {"model": "fixture", "paper_faithful": True, "stages": []},
        [{"event": "capture", "y_b_stream": "fp"}],
    )
    assert report["status"] == FAILED
    assert report["artifact_exported"] is False
    # The fidelity claim is recorded FALSE, never left standing.
    assert report["paper_faithful"] is False
    assert report["stage_failure"] == {
        "stage": "tune_fp",
        "block": "layers.3",
        "reason": "zero_first_step_gradient",
        "gradient_norm": 0.0,
        "initial_residual_norm": 0.25,
    }
    # output == input, both sides carrying the same payload: this asserts the
    # report echoes its stages verbatim, which is invariant under the capture
    # fixture gaining its stream-provenance field.
    assert report["stages"] == [{"event": "capture", "y_b_stream": "fp"}]
    for key in ("ppl_quant_native", "ppl_ratio", "effective_bpw", "qpk1_layer_bytes"):
        assert report[key] is None, key
    payload, faithful = dump_report(report)
    assert faithful is True
    assert json.loads(payload)["paper_faithful"] is False


def test_stage_failure_report_carries_the_completion_key_set_plus_its_own():
    """Strict consumers see every completion key on the stage-failure shape too."""
    exc = SimpleNamespace(
        stage="tune_fp",
        block="layers.0",
        reason="no_in_block_layer_factorized",
        gradient_norm=0.0,
        initial_residual_norm=0.0,
    )
    report = stage_failure_report(exc, dict(_ABORT_IDENTITY, probe_abort=None), [])
    assert set(_completion_report_keys()) <= set(report)
    assert set(report) - set(_completion_report_keys()) == {"stage_failure"}


def test_geometry_deviations_name_a_split_tune_geometry_and_row_count():
    """The stage map covers stages; the geometry they ran at is named too.

    A complete stage map over a SLICED tune window would make
    "paper-faithful" a green label over an artifact that is not the
    protocol's — the same defect class the whole week has been about.
    """
    assert geometry_deviations(2048, 2048, 128) == []
    # Unset tune geometry resolves to the eval geometry at the call site, so a
    # None here is also a deviation from the protocol.
    split = geometry_deviations(2048, 512, 128)
    assert [entry["kind"] for entry in split] == ["tune_geometry_split"]
    assert "2048" in split[0]["detail"] and "512" in split[0]["detail"]

    rows = geometry_deviations(2048, 2048, 16)
    assert [entry["kind"] for entry in rows] == ["calibration_rows"]

    both = geometry_deviations(2048, 512, 16)
    assert [entry["kind"] for entry in both] == ["tune_geometry_split", "calibration_rows"]
    # The protocol's own row count is 128 (16 batches x 8), so the default is
    # a clean geometry.
    assert geometry_deviations(2048, 2048, 128) == []

    # A non-finite geometry is a named deviation, never a silent pass.
    bad = geometry_deviations(float("nan"), 2048, 128)
    assert [entry["kind"] for entry in bad] == ["unmeasured_eval_geometry"]
