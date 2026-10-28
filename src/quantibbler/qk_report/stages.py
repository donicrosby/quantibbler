"""The stage inventory and its readers — kept in one module so they cannot drift."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

#: The paper-regime config fields, in report order: `("solver",
#: "compute_dtype", "factor_mode", "ablation", "layer_scope", "admm_iter",
#: "rho_end", "device")`.
#:
#: ONE list. The verdict's `config_*` conditions, the CI referee's re-derivation
#: of the conjunction, and the runtime gates' regime test all read this tuple and
#: :func:`paper_regime_conditions` rather than a second hand-kept name list — a
#: copy of the field set is how a ninth regime field silently loses its
#: condition. The values [NQ] App. C prescribes are the predicates below.
PAPER_REGIME_CONFIG_FIELDS: tuple[str, ...] = (
    "solver",
    "compute_dtype",
    "factor_mode",
    "ablation",
    "layer_scope",
    "admm_iter",
    "rho_end",
    "device",
)

#: The STE row-scope marker the per-target cycle RETIRED, kept here so an
#: old report stays READABLE rather than being denied anonymously: an STE row
#: that still declares this joint-pass scope (Algorithm 1 line 19's block-scoped
#: call, the reading the 2026-10-28 AMEND ruling overrides) is a mechanism-class
#: deviation, and :func:`fidelity_conditions`'s ``ste_per_target`` condition is
#: the field that names it. The per-target cycle's own rows declare
#: ``per_target`` (``ptq._tuning.STE_PER_TARGET_SCOPE``); the value is spelled
#: here, not imported, because ``qk_report`` does not depend on ``ptq``.
STE_BLOCK_JOINT_SCOPE = "block_joint"


def paper_regime_conditions(config: Any) -> dict[str, bool]:
    """The eight ``config_*`` conditions for one declared-config mapping.

    The label is computed against the config the run DECLARED — the machine
    checks the config, the user never asserts it — so this is where the
    declared-vs-paper comparison lives, as data rather than as a boolean baked
    into the caller.

    ``config`` is a mapping of the fields named by
    :data:`PAPER_REGIME_CONFIG_FIELDS`; anything else (``None``, a partial
    record, a field of the wrong type) is a DEVIATION, never a pass. An
    unlabelled config must not happen to satisfy the regime: absence of a
    declaration is not a declaration of the regime's value.

    ``device`` is the device the run actually resolved to, so ``"cpu"`` is the
    one value that denies it — an unset/unknown device denies too (nothing was
    labelled), while ``"cuda"``/``"cuda:0"`` satisfy it. ``rho_end`` is the
    run's effective endpoint: the interval is the live spec's, and a value that
    is not a number is a deviation rather than a crash.
    """
    fields = config if isinstance(config, Mapping) else {}
    rho_end = fields.get("rho_end")
    try:
        rho_in_interval = 0.0 <= float(rho_end) <= 1.0
    except (TypeError, ValueError):
        rho_in_interval = False
    device = fields.get("device")
    return {
        "config_solver_is_admm": fields.get("solver") == "admm",
        "config_compute_dtype_is_bf16": fields.get("compute_dtype") == "bf16",
        "config_factor_mode_is_binary": fields.get("factor_mode") == "binary",
        "config_ablation_is_none": fields.get("ablation") == "none",
        "config_layer_scope_is_all": fields.get("layer_scope") == "all",
        "config_admm_iter_matches_regime": fields.get("admm_iter") == 400,
        "config_admm_rho_end_matches_regime": rho_in_interval,
        "config_device_is_accelerator": (
            isinstance(device, str) and bool(device) and device.lower() != "cpu"
        ),
    }


def config_matches_paper_regime(config: Any) -> bool:
    """The config-match conjunction, over a declared-config mapping.

    The runtime gates and the CI referee both ask this one question, so the
    answer cannot differ between the stage that runs and the script that audits:
    a run that matches the regime on every declared field IS a paper-regime run.
    """
    return all(paper_regime_conditions(config).values())


def _stage_pair(entry: Any) -> tuple[Any, Any]:
    """Normalize one stage record to ``(event, info)``.

    The driver carries stages as ``(event, info)`` tuples while the report
    payload spells them ``{"event": ..., **info}``; the fidelity conditions
    have to read both shapes so the label is computed off the same record on
    every path.
    """
    if isinstance(entry, dict):
        return entry.get("event"), entry
    if isinstance(entry, (tuple, list)) and len(entry) == 2:
        return entry[0], entry[1]
    return None, None


def fidelity_conditions(
    solver: Any,
    runtime_deviations: Any,
    stages: Any,
    *,
    config: Any,
    stage_failure: bool = False,
) -> dict[str, bool]:
    """The named conditions behind the ``paper_faithful`` verdict.

    The reported label is a COMPUTED verdict, never a forwarded flag
    (scenarios ``label_flips_on_runtime_deviations`` /
    ``label_flips_on_mechanism_deviations``): it is the declared config's match
    against the paper regime AND an empty deviation list AND no
    mechanism-class deviation AND evidence that every required stage actually
    executed, and every condition is recorded so a reader can audit the label
    from the artifact alone. ``config`` is the declared-config mapping
    :func:`paper_regime_conditions` reads — one condition per regime field, so a
    deviating config names the field it deviates on instead of denying the label
    anonymously. The mechanism list mirrors the spec exactly — greedy-polish
    rounds, a STE scale-only re-scope marker, a vacuous-teacher event, a stage
    failure, a solver other than LB-ADMM, and an STE row still declaring the
    retired ``block_joint`` scope (``ste_per_target`` denied).

    Required-stage execution (scenario
    ``required_stage_execution_reads_the_same_record``) is read off this same
    normalized record: the set is the ``required`` column of ``_STAGE_LINES``,
    never a second list, and one condition per required stage is reported so a
    missing stage names itself. Nothing here is vacuous — an empty or
    incomplete record denies every required stage, which is the point: a
    consistent-but-empty record must not certify anything.

    An EXECUTION is an event that fired, not merely one that was recorded. So
    a skip-marked event does not count (``tune_fp`` carrying ``skipped`` — the
    per-target cycle's ``block_carries_no_second_target`` path on a block with
    fewer than two quantizable linears — is a recorded skip), and
    neither does ``tune_scales_kd`` when its payload says the run is the
    ``factor_mode='fp'`` measurement, where the driver appends the event
    unconditionally even though the stage is skipped entirely.

    The count is run-level (a stage that fired anywhere in the run executed),
    so a per-block legitimate skip cannot false-deny a faithful run.

    Informed-assumption markers (the TuneFP loss formula) are documentation,
    not deviations, and no condition here reads them.
    """
    polish_rounds = 0
    # The discrete sign-flip mechanism is deleted, so no STE row can carry the
    # key this reads -- the condition below is satisfied by construction. The
    # read stays as the tripwire for an emitter that re-enters the vocabulary:
    # a key nobody can write today is exactly what a canary must still see.
    flip_pass = False
    rescope = False
    vacuous = False
    ste_scopes: set[str] = set()
    captures: list[dict] = []
    executions: dict[str, int] = {}
    for entry in stages or []:
        event, info = _stage_pair(entry)
        if event is None:
            continue
        fields = info if isinstance(info, dict) else {}
        skipped = bool(fields.get("skipped"))
        if not skipped and (event != "tune_scales_kd" or fields.get("artifact")):
            executions[event] = executions.get(event, 0) + 1
        if event == "lb_admm" and isinstance(info, dict):
            try:
                polish_rounds = max(polish_rounds, int(info.get("polish_rounds") or 0))
            except (TypeError, ValueError):
                # An unreadable round count is a deviation-shaped fact: treat
                # it as polish having happened rather than as zero.
                polish_rounds = max(polish_rounds, 1)
        elif event == "ste" and isinstance(info, dict):
            # The STE row carries its SCOPE (the refinement unit). An absent
            # scope is not collected -- a pre-scope writer is read by the
            # required-stage condition, not denied here for a field it never had.
            scope = info.get("scope")
            if scope is not None:
                ste_scopes.add(str(scope))
            for rec in (info.get("per_linear") or {}).values():
                if not isinstance(rec, dict):
                    continue
                if "flip_pass" in rec:
                    flip_pass = True
                if rec.get("declared") == "scale_only" or rec.get("rescope_marker"):
                    rescope = True
        elif event == "capture":
            captures.append(fields)
        elif event == "tune_fp_vacuous_teacher":
            vacuous = True
    conditions = {
        "no_runtime_deviations": not bool(runtime_deviations),
        "polish_off": polish_rounds == 0,
        "no_discrete_flip_pass": not flip_pass,
        "no_scale_only_rescope": not rescope,
        "no_vacuous_teacher": not vacuous,
        # A paper_faithful claim owes evidence that the TEACHER TARGET came
        # from the fp-stream capture. Before this condition every capture event was
        # accepted as a teacher source, so a run whose teacher read the compressed
        # stream was labelled faithful -- the fidelity hole the lane closes.
        #
        # Scoped to captures that EXIST: a record with no capture event is already
        # denied by required_capture_executed, and making this condition deny it too
        # would flip the label on records whose provenance was never at issue.
        # A capture present with no y_b_stream field is a pre-Lane-2 writer or a
        # mislabelled feed, and that denies: absence of provenance on a capture that
        # exists is not evidence of an fp-stream teacher.
        "capture_is_fp_stream": not captures or all(c.get("y_b_stream") == "fp" for c in captures),
        "no_stage_failure": not bool(stage_failure),
        "solver_is_admm": solver == "admm",
        # The STE SCOPE condition (spec delta r4, S-2): the refinement unit is
        # the per-target linear, and the record SHALL carry it. `ste_per_target`
        # is the positive condition -- it holds unless an STE row declares the
        # RETIRED `block_joint` scope (the one `block_joint` marker the spec names;
        # written `ste_block_joint` in the delta's condition vocabulary), which on
        # a paper-regime run is a mechanism-class deviation, not a measurement.
        # Scoped to rows that carry a scope, exactly like `capture_is_fp_stream`:
        # a record with no STE row is already denied by `required_ste_executed`,
        # so demanding the marker here too would flip the label on records whose
        # scope was never at issue. A row that DOES carry the marker denies.
        "ste_per_target": STE_BLOCK_JOINT_SCOPE not in ste_scopes,
    }
    # The declared-config half of the verdict, one condition per regime field.
    conditions.update(paper_regime_conditions(config))
    for key, _lines, _description, _deviation, required in _STAGE_LINES:
        if required:
            conditions[f"required_{key}_executed"] = executions.get(key, 0) > 0
    return conditions


def fidelity_verdict(
    solver: Any,
    runtime_deviations: Any,
    stages: Any,
    *,
    config: Any,
    stage_failure: bool = False,
) -> tuple[bool, dict[str, bool]]:
    """``(label, conditions)`` for a run — the label is the conjunction.

    Returns the label alongside the condition record so every report path can
    carry both: a reader sees WHICH condition denied the claim, not merely
    that it was denied.
    """
    conditions = fidelity_conditions(
        solver,
        runtime_deviations,
        stages,
        config=config,
        stage_failure=stage_failure,
    )
    return all(conditions.values()), conditions


#: Every stage the driver can execute, mapped to its [NQ] Algorithm 1 lines.
#:
#: The last column is the ``required`` marker: the stages the paper's method
#: cannot skip, so a ``paper_faithful`` claim owes evidence that each one
#: actually FIRED (scenario ``required_stage_execution_reads_the_same_record``).
#: It lives here, as a column of the single stage inventory, and never as a
#: second list beside the verdict, so the stage map and the verdict cannot
#: drift apart.
_STAGE_LINES: tuple[tuple[str, str, str, str | None, bool], ...] = (
    (
        "phase1_statistics",
        "2-5",
        "global calibration statistics (RobustDiag preconditioners)",
        None,
        True,
    ),
    ("capture", "9-10", "compressed-prefix inputs X_b and teacher outputs Y_b", None, True),
    (
        "tune_fp",
        "12 (Figure 3 per-target cycle)",
        "TuneFP: error-propagation mitigation",
        "placement read from Figure 3 as a PER-TARGET-LINEAR cycle: tune the "
        "still-FP remainder, factorize the target, refine the target, with "
        "target 1 factorized first so the pass fires N-1 times per block (the "
        "earlier once-per-block reading was falsified by the 64,151x fire); "
        "the internal loss formula is an informed assumption (the corpus "
        "names the call but not its objective)",
        True,
    ),
    (
        "lb_admm",
        "14-17",
        "LB-ADMM low-rank binary initialization",
        "the greedy bit-flip polish (qk-root-cause line of work) is an EXTRA "
        "discrete stage the paper does not describe; its round count travels "
        "in the lb_admm event and is 0 for the untuned-signs ablation",
        True,
    ),
    (
        "ste",
        "19",
        "TuneLatentSTE: per-target factorized-component refinement",
        "one pass per TARGET linear, inside the factorization loop (one "
        "optimizer, one block-output loss at each fire), the target then "
        "hardened -- the reading Figure 3 and [NQ] Sec 3.2 Step 3's singular "
        '"the current target linear layer" prescribe (the per-target reading '
        "governs); Algorithm 1 line 19's block-scoped pseudocode (:780) is the "
        "DECLARED tension it overrides. The [NQ] App. C factorized-stage budget "
        "is consumed afresh for each of the block's linears, so a block of N "
        "targets spends the regime N times",
        True,
    ),
    ("sign_extract_pack", "20-23", "sign extraction and packing", None, True),
    (
        "tune_scales_kd",
        "26-27",
        "TuneScalesKD: global scale-only KD against the FP teacher",
        "packed binaries are frozen during scale tuning (verified against "
        "TuneScalesKD semantics); skipped entirely when the run is a "
        "measurement (factor_mode='fp' exports no artifact)",
        True,
    ),
)


#: The payload keys each stage event ALWAYS carries, keyed by event name.
#:
#: The verdict is payload-BLIND by design: a required stage counts as executed
#: on its event NAME alone, because the in-process recorder cannot emit a name
#: without the emitter having run. That is sound only while the record is
#: in-process. Once a record can be RE-READ from disk — the stage journal beside
#: the solve checkpoints, which seeds a resumed run's sink — a name-only line is
#: forgeable by corruption, and a name-only line satisfies every required-stage
#: condition while attesting nothing.
#:
#: This table is the shape such a line must carry: one entry per event the
#: recorder can emit, listing the payload keys that event's emitter always
#: sends. It is keyed by the SAME vocabulary ``_STAGE_LINES`` holds (a test pins
#: the containment in both directions), so a stage cannot join the inventory
#: without a shape, and a reader cannot be handed a stage record it has no shape
#: for.
#:
#: It is a SHAPE check, not a signature: a hand-written line that copies the keys
#: still passes it. Binding a journal to the run whose checkpoints it sits beside
#: is the fingerprint's job (``checkpoints.check_fingerprint``), not this
#: table's.
_STAGE_PAYLOAD_KEYS: dict[str, tuple[str, ...]] = {
    "phase1_statistics": ("linears", "num_linears"),
    # The stream provenance labels are the pair the verdict's
    # `capture_is_fp_stream` condition reads, so a capture with no payload at
    # all must not be readable as one.
    "capture": ("capture_index", "teacher_source", "x_b_stream", "y_b_stream"),
    # The marker is the TuneFP loss formula's informed-assumption note; a
    # skip-marked record carries it too, so it is required either way.
    "tune_fp": ("capture_index", "marker", "steps", "tunable"),
    "lb_admm": ("polish_rounds",),
    "ste": ("capture_index", "per_linear", "total_steps"),
    "sign_extract_pack": ("extracted", "factor_mode", "layers"),
    "tune_scales_kd": ("artifact", "factor_mode"),
    # Off-inventory events. Neither is a row of the stage inventory (nothing
    # counts their execution), but the recorder emits both, so a journal that
    # carries one must still be readable. The two norms ARE the vacuous-teacher
    # event's evidence (A3), which is why both are required here.
    "tune_loss_uniform_fallback": ("block", "reason"),
    "tune_fp_vacuous_teacher": ("block", "gradient_norm", "initial_residual_norm", "reason"),
}


def _non_finite_path(value: Any, path: str = "info") -> str | None:
    """The path of the first non-finite number inside a payload, or ``None``.

    The MONOTONE-LEVERAGE corruption class (see the repo skill's corruption
    self-test rule): a measurement that saturates to ``inf``/``NaN`` — or a
    literal large enough to saturate on parse, e.g. ``1e400`` — dominates any
    artifact state for a nonzero entry, so it is the class a corruption
    self-test must use rather than a sign flip or a small-entry tweak. The
    required-stage conditions are payload-blind, so the journal's gate is the
    one place such a value could reach a verdict; refusing it here keeps a
    saturated line from being read as an execution. ``bool`` is not a number
    here (``isinstance(True, float)`` is False), and JSON has already turned
    every number into ``int``/``float``, so no numpy scalar can appear.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            found = _non_finite_path(item, f"{path}.{key}")
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _non_finite_path(item, f"{path}[{index}]")
            if found is not None:
                return found
    elif isinstance(value, float) and not math.isfinite(value):
        return path
    return None


def stage_journal_record(record: Any) -> tuple[str, dict]:
    """``(event, info)`` for one stage-journal line, or ``ValueError``.

    The journal beside the solve checkpoints is the ONLY path by which a stage
    record can be re-read from disk and fed back into ``fidelity_conditions``,
    so it is the only place where the payload-blindness described on
    ``_STAGE_PAYLOAD_KEYS`` could be reached by a record whose shape nobody
    vouched for. Every line therefore has to name an event the recorder can
    emit AND carry the payload that event always sends; a name-only record —
    the shape a corrupted or hand-written line most easily takes — is refused
    rather than accepted as an execution. A line whose payload carries a
    non-finite number is refused for the same reason the shape gate exists: a
    saturated measurement is the MONOTONE-LEVERAGE corruption class, and it
    must deny or raise, never bless a verdict on its name alone.

    This is a validator, not a reader: the file/line context and the
    fingerprint guard belong to the caller (``checkpoints.read_stage_log``).
    """
    if not isinstance(record, dict):
        raise ValueError("a stage journal line must be a JSON object")
    event = record.get("event")
    if not isinstance(event, str):
        raise ValueError("a stage journal line must name its event as a string")
    shape = _STAGE_PAYLOAD_KEYS.get(event)
    if shape is None:
        raise ValueError(
            f"event {event!r} is not one the stage recorder can emit; refusing to read a "
            "stage record from an unregistered name"
        )
    info = record.get("info")
    if not isinstance(info, dict):
        raise ValueError(f"event {event!r} carries no payload object")
    missing = [key for key in shape if key not in info]
    if missing:
        raise ValueError(
            f"event {event!r} is missing the payload keys {missing}; a name-only record "
            "would satisfy every required-stage condition while attesting nothing"
        )
    saturated = _non_finite_path(info)
    if saturated is not None:
        raise ValueError(
            f"event {event!r} carries a non-finite number at {saturated}; a saturated "
            "measurement is the monotone-leverage corruption class and must not be read "
            "as an execution"
        )
    return event, info


def _row_seconds(fields: Any) -> float | None:
    """A stage row's own wall time, or ``None`` when the row did not record one.

    Never 0 for a missing reading: a zero row time is a positive claim ("this
    stage cost nothing"), which is exactly the shape a silently un-instrumented
    stage would produce (same rule as the null-not-zero unmeasured numerics).
    """
    value = fields.get("elapsed_s") if isinstance(fields, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def stage_row_times(stages: list) -> dict[str, dict]:
    """Per-stage wall time and its cost per ROW, read off the stage records.

    ONE row per unit the stage's cost is quoted in (qk-ptq-speedups, scenario
    ``ste_tunefp_gpu_zero_algorithm_change``): a ``tune_fp`` row is one fire and
    a ``ste`` row is one TARGET's fire -- since the per-target cycle (AMEND
    2026-10-28) TuneLatentSTE runs once per linear, so its wall time belongs to
    that one linear and the row count is the block's fire count. (Pre-AMEND
    reports carry one block-joint ``ste`` row per block instead; their totals
    are still the totals the pricing quotes, their per-row means are not
    comparable.) Every stage's row is therefore one invocation, with the time
    the record itself carries.

    A recorded ``tune_fp`` skip is NOT a row: it did no work and spent no time,
    and counting it would deflate the stage's per-row cost. Rows without a
    reading are counted and their time is ``None`` — see :func:`_row_seconds`.
    ``row_time_s`` is the mean over the rows that DID record a time, so an
    unmeasured row dilutes nothing.
    """
    rows: dict[str, list[float | None]] = {}
    for entry in stages or []:
        event, info = _stage_pair(entry)
        if event is None:
            continue
        fields = info if isinstance(info, dict) else {}
        if event == "tune_fp" and fields.get("skipped"):
            rows.setdefault("tune_fp", [])
            continue
        rows.setdefault(event, []).append(_row_seconds(fields))
    out: dict[str, dict] = {}
    for event, spent in rows.items():
        known = [value for value in spent if value is not None]
        out[event] = {
            "rows": len(spent),
            "elapsed_s": sum(known) if known else None,
            "row_time_s": (sum(known) / len(known)) if known else None,
        }
    return out


def stage_map(stages: list) -> dict:
    """Map every executable stage to its Algorithm 1 lines, naming deviations.

    ``stages`` is the recorder's event list (``(event, info)`` pairs). The
    result answers the faithful-run question directly: which stages of
    Algorithm 1 lines 0-28 actually executed, and which parts of the letter
    the pipeline deviates from.

    Each entry also carries its measured row time (``rows``, ``elapsed_s``,
    ``row_time_s``) beside its Algorithm 1 lines — see
    :func:`stage_row_times`. The timing lives in THIS record rather than a
    parallel one, so a stage's declared lines and its measured cost cannot
    drift apart.
    """
    executed = [event for event, _ in stages]
    counts: dict[str, int] = {}
    for event in executed:
        counts[event] = counts.get(event, 0) + 1
    times = stage_row_times(stages)
    out: dict[str, dict] = {}
    for key, lines, description, deviation, _required in _STAGE_LINES:
        fired = counts.get(key, 0)
        out[key] = {
            "algorithm1_lines": lines,
            "description": description,
            "invocations": fired,
            "executed": bool(fired),
            "deviation": deviation,
            "rows": times.get(key, {}).get("rows", 0),
            "elapsed_s": times.get(key, {}).get("elapsed_s"),
            "row_time_s": times.get(key, {}).get("row_time_s"),
        }
    out["tune_fp"]["vacuous_events"] = counts.get("tune_fp_vacuous_teacher", 0)
    # Schema compatibility, never a live reading: the mechanism that emitted
    # `flip_pass` is deleted, so this key stays in the schema for readers that
    # know it and can only ever be False.
    out["ste"]["discrete_flip_pass"] = any(
        isinstance(info, dict)
        and any(
            isinstance(rec, dict) and "flip_pass" in rec
            for rec in (info.get("per_linear") or {}).values()
        )
        for event, info in stages
        if event == "ste"
    )
    return out
