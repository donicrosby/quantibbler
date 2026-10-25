"""The stage inventory and its readers — kept in one module so they cannot drift."""

from __future__ import annotations

import math
from typing import Any


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
    requested: Any,
    stage_failure: bool = False,
) -> dict[str, bool]:
    """The named conditions behind the ``paper_faithful`` verdict.

    The reported label is a COMPUTED verdict, never a forwarded flag
    (scenarios ``label_flips_on_runtime_deviations`` /
    ``label_flips_on_mechanism_deviations``): it is the flag AND an empty
    deviation list AND no mechanism-class deviation AND evidence that every
    required stage actually executed, and every condition is recorded so a
    reader can audit the label from the artifact alone. The mechanism list
    mirrors the spec exactly — greedy-polish rounds, a fired STE discrete flip
    pass, a STE scale-only re-scope marker, a vacuous-teacher event, a stage
    failure, and a solver other than LB-ADMM.

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
    flip_pass = False
    rescope = False
    vacuous = False
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
        "paper_faithful_flag": bool(requested),
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
    }
    for key, _lines, _description, _deviation, required in _STAGE_LINES:
        if required:
            conditions[f"required_{key}_executed"] = executions.get(key, 0) > 0
    return conditions


def fidelity_verdict(
    solver: Any,
    runtime_deviations: Any,
    stages: Any,
    *,
    requested: Any,
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
        requested=requested,
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
        "TuneLatentSTE: joint block-level factorized-component refinement",
        "one pass per block, jointly over every factorized linear (one "
        "optimizer, one block-output loss) -- a DECLARED deviation from "
        'Algorithm 1 line 19\'s "the current target linear layer" prose; the '
        "[NQ] App. C factorized-stage budget is consumed once for the whole "
        "block; the bounded discrete flip pass is a named deviation when its "
        "budget is nonzero",
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
    ``ste_tunefp_gpu_zero_algorithm_change``): a ``tune_fp`` row is one fire,
    a ``ste`` row is one BLOCK fire -- the stage runs one joint pass over every
    factorized linear, so its wall time belongs to the block and dividing it by
    linear would report a share no clock observed. Every stage's row is
    therefore one invocation, with the time the record itself carries.

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
