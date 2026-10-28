"""The Algorithm 1 stage map, per-stage row time, and the ``paper_faithful`` verdict.

Split out of ``tests/test_qk_report.py`` (surface 5 of 6); every test body is
verbatim, and the docstrings keep their scenario IDs: the stage inventory,
per-stage row time, the computed label
(`label_flips_on_runtime_deviations` / `label_flips_on_mechanism_deviations`),
the required-stage evidence (`required_stage_execution_reads_the_same_record`),
and the calibration geometry deviations.
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

from quantibbler import qk_report, serialization
from quantibbler.qk_report import (
    FAILED,
    abort_report,
    config_matches_paper_regime,
    dump_report,
    fidelity_conditions,
    fidelity_verdict,
    geometry_deviations,
    paper_regime_conditions,
    stage_failure_report,
    stage_map,
    stage_row_times,
)

# --------------------------------------------------------------------------
# The Algorithm 1 stage map and the fail-closed report
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
                    "attn.o": {
                        "declared": "scale_only",
                        "rescope_marker": "ste_scale_only_declared",
                        "zero_crossings": True,
                    },
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
    # The key stays in the schema and is never true: the mechanism that emitted
    # a `flip_pass` row is deleted, so a scale-only row is the honest shape.
    assert mapped["ste"]["discrete_flip_pass"] is False
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
# Per-stage row-time, carried by the stage inventory
# --------------------------------------------------------------------------


def test_stage_map_reports_row_time_per_stage_row():
    """Scenario ``ste_tunefp_gpu_zero_algorithm_change``: per-stage row-time is
    reported — one row per fire, and an STE row is one TARGET's fire (its
    per-linear map carries the one linear it refined)."""
    stages = [
        ("tune_fp", {"elapsed_s": 12.0, "steps": 8}),
        ("tune_fp", {"elapsed_s": 6.0, "steps": 8}),
        ("lb_admm", {"polish_rounds": 0}),
        ("ste", {"elapsed_s": 3.0, "per_linear": {"attn.q": {}}}),
        ("ste", {"elapsed_s": 9.0, "per_linear": {"mlp.up_proj": {}}}),
    ]
    mapped = stage_map(stages)
    assert mapped["tune_fp"]["rows"] == 2
    assert mapped["tune_fp"]["elapsed_s"] == 18.0
    assert mapped["tune_fp"]["row_time_s"] == 9.0
    # The STE rows are the FIRES, not a sum over the refined linears: since the
    # per-target cycle each fire refines exactly one linear, so its wall time is
    # that fire's own clock and the row time is the mean over fires.
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
    # The STE unit is the fire, and this record carries no time for it: a
    # per-linear reading is not its row, and an unreadable record stays null
    # rather than borrowing one.
    assert times["ste"] == {"rows": 1, "elapsed_s": None, "row_time_s": None}
    assert times["capture"] == {"rows": 1, "elapsed_s": None, "row_time_s": None}


def test_stage_row_times_read_both_record_shapes():
    """The driver carries ``(event, info)`` pairs and the payload spells them as
    dicts; one reader, both shapes (the ``_stage_pair`` contract)."""
    pairs = [("tune_fp", {"elapsed_s": 2.0})]
    dicts = [{"event": "tune_fp", "elapsed_s": 2.0}]
    assert stage_row_times(pairs) == stage_row_times(dicts)
    assert stage_row_times([]) == {}


# --------------------------------------------------------------------------
# The paper_faithful label is a COMPUTED verdict about the DECLARED config
# --------------------------------------------------------------------------


def _clean_stages() -> list:
    """The mechanism-clean record of a regime run (no deviations at all)."""
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
                "scope": "per_target",
                "per_linear": {
                    "attn.q": {"declared": "paper_ste", "zero_crossings": True},
                    "mlp.down_proj": {"declared": "paper_ste", "zero_crossings": False},
                },
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


#: A declared config that IS the paper regime — the values [NQ] App. C
#: prescribes, with the device on an accelerator. Spelled out here so this file
#: states the regime independently of the module it audits.
_PAPER_REGIME_CONFIG: dict = {
    "solver": "admm",
    "compute_dtype": "bf16",
    "factor_mode": "binary",
    "ablation": "none",
    "layer_scope": "all",
    "admm_iter": 400,
    "rho_end": 1.0,
    "device": "cuda",
}

#: The eight config-match conditions, spelled out here for the same reason.
_PAPER_REGIME_CONDITIONS = (
    "config_solver_is_admm",
    "config_compute_dtype_is_bf16",
    "config_factor_mode_is_binary",
    "config_ablation_is_none",
    "config_layer_scope_is_all",
    "config_admm_iter_matches_regime",
    "config_admm_rho_end_matches_regime",
    "config_device_is_accelerator",
)

#: One declared field off the regime per axis, paired with the condition that
#: must deny it. Every axis is covered, so a field that silently stops being
#: checked has to break a case here.
_REGIME_DEVIATIONS = (
    ("config_solver_is_admm", {"solver": "svd"}),
    ("config_compute_dtype_is_bf16", {"compute_dtype": "float32"}),
    ("config_factor_mode_is_binary", {"factor_mode": "fp"}),
    ("config_ablation_is_none", {"ablation": "untuned-signs"}),
    ("config_layer_scope_is_all", {"layer_scope": "mlp"}),
    ("config_admm_iter_matches_regime", {"admm_iter": 200}),
    ("config_admm_rho_end_matches_regime", {"rho_end": 4.0}),
    ("config_device_is_accelerator", {"device": "cpu"}),
)

#: A declared config that deviates (the SVD warm start, the measured
#: production ablation): the shape a DIAGNOSTIC run carries.
_DIAGNOSTIC_CONFIG = {**_PAPER_REGIME_CONFIG, "solver": "svd"}


def _regime(**overrides) -> dict:
    """The paper-regime config with the named fields declared otherwise."""
    return {**_PAPER_REGIME_CONFIG, **overrides}


def test_scn_all_conditions_satisfied_yields_true():
    """`all_conditions_satisfied_yields_true`: the regime config carries it."""
    label, conditions = fidelity_verdict("admm", [], _clean_stages(), config=_PAPER_REGIME_CONFIG)
    assert label is True
    assert all(conditions.values()), conditions


def test_scn_condition_set_names_every_config_field():
    """`condition_set_names_every_config_field`: the exact set a clean run carries.

    One condition per paper-regime config field, plus every stage-level
    condition the verdict already reported — and no `paper_faithful_flag`: the
    request flag is gone from the record, not merely always-false in it.
    """
    label, conditions = fidelity_verdict("admm", [], _clean_stages(), config=_PAPER_REGIME_CONFIG)
    assert label is True
    assert set(conditions) == {
        "capture_is_fp_stream",
        "no_runtime_deviations",
        "polish_off",
        "no_discrete_flip_pass",
        "no_scale_only_rescope",
        "no_vacuous_teacher",
        "no_stage_failure",
        "solver_is_admm",
        "ste_per_target",
        *_PAPER_REGIME_CONDITIONS,
        *(f"required_{key}_executed" for key in _REQUIRED_STAGES),
    }
    assert all(conditions.values()), conditions
    # The module's own field tuple is the one this file just pinned: a ninth
    # regime field cannot join the conjunction without a condition here.
    assert set(qk_report.PAPER_REGIME_CONFIG_FIELDS) == set(_PAPER_REGIME_CONFIG)
    assert set(paper_regime_conditions(_PAPER_REGIME_CONFIG)) == set(_PAPER_REGIME_CONDITIONS)


@pytest.mark.parametrize(("denied", "overrides"), _REGIME_DEVIATIONS)
def test_scn_a_deviating_config_field_names_the_deviation(denied, overrides):  # noqa: ANN001
    """Every axis: the condition for the declared field is the one that falls.

    The user never asserts fidelity — the machine compares the DECLARED config
    against the regime — so a run that deviates must say WHICH field it
    deviates on, from the condition record alone.
    """
    label, conditions = fidelity_verdict(
        overrides.get("solver", "admm"), [], _clean_stages(), config=_regime(**overrides)
    )
    assert label is False, denied
    # The solver axis denies in two places by construction: the config condition
    # and the stage-level `solver_is_admm` read the same declaration. Every
    # other axis denies exactly once, and nothing else falls.
    extra = {"solver_is_admm"} if "solver" in overrides else set()
    assert {key for key, value in conditions.items() if value is False} == {denied} | extra


def test_scn_solver_mismatch_names_the_deviation():
    """`solver_mismatch_names_the_deviation`: `--solver svd` denies the label."""
    label, conditions = fidelity_verdict("svd", [], _clean_stages(), config=_regime(solver="svd"))
    assert label is False
    assert conditions["config_solver_is_admm"] is False
    assert conditions["solver_is_admm"] is False


def test_scn_compute_dtype_mismatch_names_the_deviation():
    """`compute_dtype_mismatch_names_the_deviation`: float32 is not the regime."""
    label, conditions = fidelity_verdict(
        "admm", [], _clean_stages(), config=_regime(compute_dtype="float32")
    )
    assert label is False
    assert conditions["config_compute_dtype_is_bf16"] is False


def test_scn_admm_iter_mismatch_names_the_deviation():
    """`admm_iter_mismatch_names_the_deviation`: anything but App. C's 400."""
    label, conditions = fidelity_verdict("admm", [], _clean_stages(), config=_regime(admm_iter=200))
    assert label is False
    assert conditions["config_admm_iter_matches_regime"] is False


def test_scn_device_mismatch_names_the_deviation():
    """`device_mismatch_names_the_deviation`: an explicit CPU run is a deviation."""
    label, conditions = fidelity_verdict("admm", [], _clean_stages(), config=_regime(device="cpu"))
    assert label is False
    assert conditions["config_device_is_accelerator"] is False


def test_scn_no_request_flag_anywhere():
    """`no_request_flag_anywhere`: no fidelity-request value exists to carry.

    The verdict takes a CONFIG, never a request: the condition record cannot
    name a `paper_faithful_flag` key, and the label is computed for a config
    that never asked for anything.
    """
    _, conditions = fidelity_verdict("admm", [], _clean_stages(), config=_PAPER_REGIME_CONFIG)
    assert "paper_faithful_flag" not in conditions
    # An unlabelled config is not a regime match either: absence of a
    # declaration denies, it never silently claims.
    assert config_matches_paper_regime(None) is False
    assert config_matches_paper_regime({}) is False
    label, conditions = fidelity_verdict("admm", [], _clean_stages(), config=None)
    assert label is False
    assert all(conditions[key] is False for key in _PAPER_REGIME_CONDITIONS)


def test_scn_label_denied_by_stage_condition_still_reports_config_match():
    """`label_denied_by_stage_condition_still_reports_config_match`.

    A regime-matching config with a vacuous-teacher event: every config
    condition stays TRUE, the stage condition falls, and both facts are visible
    in the record — which is what makes the denial diagnosable.
    """
    stages = [*_clean_stages(), ("tune_fp_vacuous_teacher", {"gradient_norm": 0.0})]
    label, conditions = fidelity_verdict("admm", [], stages, config=_PAPER_REGIME_CONFIG)
    assert label is False
    assert conditions["no_vacuous_teacher"] is False
    assert all(conditions[key] is True for key in _PAPER_REGIME_CONDITIONS)


def test_fidelity_label_flips_on_runtime_deviations():
    """`label_flips_on_runtime_deviations`: any deviation denies the claim."""
    deviations = geometry_deviations(2048, 512, 16)
    label, conditions = fidelity_verdict(
        "admm", deviations, _clean_stages(), config=_PAPER_REGIME_CONFIG
    )
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
    label, conditions = fidelity_verdict(
        "admm", [], mutate(_clean_stages()), config=_PAPER_REGIME_CONFIG
    )
    assert label is False, name
    assert conditions[denied] is False, name
    assert all(value for key, value in conditions.items() if key != denied), (name, conditions)


def test_fidelity_label_flips_on_a_retired_block_joint_ste_scope():
    """An STE row declaring the RETIRED `block_joint` scope denies the label.

    Spec delta r4 (S-2): the fidelity conditions record the STE scope. The
    per-target cycle is the AMEND path (`ste_per_target: true`); a `block_joint`
    marker — the joint-pass scope this AMEND replaces — on a paper-regime run is
    a mechanism-class deviation, named by the `ste_per_target` condition falling
    rather than by a second key that could read green while the label did not.
    """
    stages = [
        (event, {**info, "scope": "block_joint"} if event == "ste" else info)
        for event, info in _clean_stages()
    ]
    label, conditions = fidelity_verdict("admm", [], stages, config=_PAPER_REGIME_CONFIG)
    assert label is False
    assert conditions["ste_per_target"] is False
    assert all(value for key, value in conditions.items() if key != "ste_per_target")


def test_fidelity_label_flips_on_a_non_admm_solver_and_on_a_stage_failure():
    """A non-regime config cannot carry the label; nor can a tripped stage gate."""
    label, conditions = fidelity_verdict("svd", [], _clean_stages(), config=_DIAGNOSTIC_CONFIG)
    assert label is False and conditions["config_solver_is_admm"] is False
    # A declared-config deviation denies too, however clean the mechanism is.
    label, conditions = fidelity_verdict(
        "admm", [], _clean_stages(), config=_regime(compute_dtype="float64")
    )
    assert label is False and conditions["config_compute_dtype_is_bf16"] is False
    # A tripped stage gate denies the claim even on a clean-looking record.
    label, conditions = fidelity_verdict(
        "admm", [], _clean_stages(), config=_PAPER_REGIME_CONFIG, stage_failure=True
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
    label, _ = fidelity_verdict("admm", [], marker_stages, config=_PAPER_REGIME_CONFIG)
    assert label is True
    # The marker is nowhere in the condition record's inputs.
    assert "informed" not in json.dumps(
        fidelity_conditions("admm", [], marker_stages, config=_PAPER_REGIME_CONFIG)
    )


def test_fidelity_conditions_read_the_report_shaped_stage_records():
    """The report spells stages `{"event": ...}`; the verdict must read both."""
    report_shaped = [{"event": event, **info} for event, info in _clean_stages()]
    label, conditions = fidelity_verdict("admm", [], report_shaped, config=_PAPER_REGIME_CONFIG)
    assert label is True and all(conditions.values())


# --------------------------------------------------------------------------
# The claim also owes
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
    label, conditions = fidelity_verdict("admm", [], _clean_stages(), config=_PAPER_REGIME_CONFIG)
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
    # The reader resolves the inventory in its own module, so the patch must
    # land there — patching the re-export in the package namespace is inert.
    monkeypatch.setattr(qk_report.stages, "_STAGE_LINES", relaxed)
    stages = [entry for entry in _clean_stages() if entry[0] != "lb_admm"]
    label, conditions = fidelity_verdict("admm", [], stages, config=_PAPER_REGIME_CONFIG)
    assert "required_lb_admm_executed" not in conditions
    assert all(_required_conditions(conditions).values())
    assert label is True


def test_an_empty_stage_record_certifies_nothing():
    """The reproduced hole: `("admm", [], [])` returned EVERY condition green."""
    label, conditions = fidelity_verdict("admm", [], [], config=_PAPER_REGIME_CONFIG)
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
        label, conditions = fidelity_verdict("admm", [], stages, config=_PAPER_REGIME_CONFIG)
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
    label, conditions = fidelity_verdict("admm", [], stages, config=_PAPER_REGIME_CONFIG)
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
    label, conditions = fidelity_verdict("admm", [], stages, config=_PAPER_REGIME_CONFIG)
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
    label, conditions = fidelity_verdict("admm", [], stages, config=_PAPER_REGIME_CONFIG)
    assert label is True and all(conditions.values()), conditions


def test_no_diagnostic_run_is_denied_by_the_required_stage_conditions():
    """These conditions never become the denier of a diagnostic run."""
    label, conditions = fidelity_verdict("admm", [], _clean_stages(), config=_DIAGNOSTIC_CONFIG)
    assert label is False
    assert conditions["config_solver_is_admm"] is False
    assert all(_required_conditions(conditions).values())
    # An fp-mode measurement legitimately skipped the scale stage; it is denied
    # by its own declared config, not newly by a required-stage condition.
    measurement = [
        ("tune_scales_kd", {"factor_mode": "fp", "artifact": False})
        if event == "tune_scales_kd"
        else (event, info)
        for event, info in _clean_stages()
    ]
    label, conditions = fidelity_verdict("admm", [], measurement, config=_DIAGNOSTIC_CONFIG)
    assert label is False
    assert conditions["config_solver_is_admm"] is False


def test_abort_and_stage_failure_reports_carry_a_false_verdict_with_reasons():
    """A run that did not finish cannot claim fidelity, and says which why."""
    identity = dict(_ABORT_IDENTITY)
    aborted = abort_report(_abort_exc(), identity, _ABORT_HISTORY)
    assert aborted["paper_faithful"] is False
    # Both halves of the denial are visible: the stage-failure flag, and the
    # declared config the recorded abort run carried (svd, cpu).
    assert aborted["fidelity_conditions"]["no_stage_failure"] is False
    assert aborted["fidelity_conditions"]["config_solver_is_admm"] is False
    assert aborted["fidelity_conditions"]["config_device_is_accelerator"] is False
    # The identity fields are not mutated: the report's label is computed from
    # the declaration the identity carries, never written back over it.
    assert identity["fidelity_config"] == _ABORT_IDENTITY["fidelity_config"]

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
    assert failure["fidelity_conditions"]["config_solver_is_admm"] is False
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
        {"model": "fixture", "fidelity_config": _PAPER_REGIME_CONFIG, "stages": []},
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
    parsed = json.loads(payload)
    assert parsed["paper_faithful"] is False
    # Read back through the contract model: the fail-closed shape is the same
    # report every other path writes, so it clears the same contract.
    assert parsed["status"] == FAILED
    assert serialization.QuantizeReport.model_validate(parsed).status == FAILED


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

    A complete stage map over a SLICED tune window would make the fidelity
    label a green label over an artifact that is not the protocol's — the same
    defect class the whole week has been about.
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
