"""End-to-end quality verdict for real-model production runs.

Change: ``qk-e2e-quality-gate``. This module is deliberately TORCH-FREE and
lives in the compressor package (not in ``tools/quantize_hf_model.py``) so
the verdict and sanitizer get real suite coverage in a lean environment —
the driver imports torch at module top, so helpers kept there would silently
skip. The sanitizer previously lived only in a REPL transcript; it must not
again.

The band
--------

Both edges are DERIVED from the pinned papers' own measurements, not chosen.
Anchors are [LB2] Sec. 5.2 Table 1 (Llama-3-8B, WikiText-2): STBLLM's
documented method collapse at 0.55 bpp (PPL 241.96) against the FP16
reference (PPL 6.10), and the functional envelope at the extreme end of the
budget range, LittleBit-2 at 0.1 bpp (PPL 23.74) against the same 6.10.

    COLLAPSE_LINE = ceil(241.96 / 6.10) = 40
    OK_LINE       = ceil(23.74 / 6.10)  = 4

Reaching the papers' own documented collapse for a method in this regime is
a failed run. The historical best ratio ever measured in this repo is 943.49
(grep ``artifacts/*/quantize_report.json``) — ~24x past the collapse line.
Under this gate every historical run classifies ``failed``; that is the
point.

Calibration-rule debt (declared in the change's design.md): the repo's
gate-calibration rule wants the band set from production healthy extremes,
and none exist — every production extreme is a failure. The band is
paper-anchored instead, and the owning change makes recalibration from the
first ``ok`` artifact a mandatory spec update.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

__all__ = [
    "COLLAPSE_LINE",
    "STEPS_EFFECT_EPS",
    "OK_LINE",
    "MIN_PROBE_DEPTH",
    "FAILED",
    "DEGRADED",
    "OK",
    "verdict",
    "exit_code",
    "json_safe",
    "dump_report",
    "probe_ratio",
    "probe_aborts",
    "abort_report",
    "stage_failure_report",
    "stage_map",
    "geometry_deviations",
    "fidelity_conditions",
    "fidelity_verdict",
    "provenance",
]

# [LB2] Sec. 5.2 Table 1 anchors (Llama-3-8B, WikiText-2).
_STBLLM_COLLAPSE_PPL = 241.96  # 0.55 bpp — documented method collapse
_LITTLEBIT2_FUNCTIONAL_PPL = 23.74  # 0.1 bpp — functional envelope
_FP16_REFERENCE_PPL = 6.10  # the FP16 baseline both are measured against

# Derived, never hand-tuned: ceil(241.96 / 6.10) and ceil(23.74 / 6.10).
# tests/test_qk_report.py asserts the arithmetic of both edges.
COLLAPSE_LINE: float = float(math.ceil(_STBLLM_COLLAPSE_PPL / _FP16_REFERENCE_PPL))
OK_LINE: float = float(math.ceil(_LITTLEBIT2_FUNCTIONAL_PPL / _FP16_REFERENCE_PPL))

# Minimum relative reduction of a stage's own objective for the stage to count
# as having done work: a record whose ``residual_after`` is not at least this
# much below its ``residual_before`` is an inert stage, however many steps it
# booked. One constant, shared by the report gate and the stage tests — a
# second threshold anywhere would be a hand-tuned re-pin (the same rule
# ``COLLAPSE_LINE`` already carries).
#
# The measured pair this exists to reject: the rehearsal recorded
# ``1.9e-04 -> 1.9e-04`` on a full TuneFP budget — display-identical, i.e. no
# measurable work — and every gate in the pipeline passed it. Real runs clear
# it by two orders of magnitude (the fatal fire's TuneFP moved
# ``0.3184 -> 0.1973``, 38%).
STEPS_EFFECT_EPS: float = 1e-3

FAILED = "failed"
DEGRADED = "degraded"
OK = "ok"


def verdict(ratio: Any) -> str:
    """Classify an end-to-end perplexity ratio into the three-value band.

    ``ok`` for ``ratio <= OK_LINE``, ``degraded`` for
    ``OK_LINE < ratio <= COLLAPSE_LINE``, ``failed`` above ``COLLAPSE_LINE``.

    The missing / non-finite check runs FIRST and is the only thing that can
    catch a NaN ratio: every comparison against NaN is False, so a bare
    ``ratio > COLLAPSE_LINE`` gate silently passes the exact shape that
    r137-v3/v4 wrote into their reports. A missing (``None``) or non-finite
    ratio is a measurement or numerics defect in the artifact path — there is
    no legitimate successful run that produces one.
    """
    try:
        finite = ratio is not None and math.isfinite(ratio)
    except TypeError:
        # Non-numeric junk is a missing measurement, not a good one.
        finite = False
    if not finite:
        return FAILED
    value = float(ratio)
    if value > COLLAPSE_LINE:
        return FAILED
    if value > OK_LINE:
        return DEGRADED
    return OK


def exit_code(status: str) -> int:
    """Process exit code for a verdict: 0 for ``ok``/``degraded``, else 1.

    The artifact and report are written regardless of verdict — the gate
    fails the *process*, never the forensic evidence.
    """
    return 0 if status in (OK, DEGRADED) else 1


def json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats with ``None``.

    Numpy scalars are converted first, through ``np.generic.item()``: an
    ``np.float32(np.inf)`` is not a Python ``float``, so it survived an
    ``isinstance(obj, float)`` check, reached ``json.dumps(...,
    allow_nan=False)`` and raised ``TypeError`` — no report at all on exactly
    the weirdest runs, which are the ones forensics exists for. ``np.int64``
    and friends are covered by the same conversion, since they are not
    JSON-serializable either.

    A bare ``NaN`` in the report is (a) not valid JSON — strict parsers
    reject it, and Python's own ``allow_nan=False`` refuses to emit it — and
    (b) silently defeats any downstream ``value > MAX`` check. Non-finite
    measurements are recorded as ``null`` instead, paired with a
    ``status: failed`` verdict as the durable trace.
    """
    if isinstance(obj, np.generic):  # numpy scalar -> Python scalar
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def dump_report(report: dict) -> tuple[str, bool]:
    """Serialize a report to strict JSON. Always returns a payload.

    Returns ``(payload, faithful)``. ``faithful`` is ``False`` only when the
    report could not be serialized as measured — in which case ``payload`` is
    the minimal failed report instead of nothing at all. A value the
    sanitizer did not recognize must not cost the verdict: the caller is
    required to fail the process whenever ``faithful`` is ``False``, so the
    durable trace and the exit code can never disagree.
    """
    try:
        return json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n", True
    except (TypeError, ValueError) as exc:
        fallback = {
            "status": FAILED,
            "report_error": f"{type(exc).__name__}: {exc}",
        }
        return json.dumps(fallback, indent=2, allow_nan=False) + "\n", False


def _as_finite_float(value: Any) -> float | None:
    """Coerce to a finite Python float, or ``None`` if that is not possible."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


#: Prefix probes begin no earlier than this depth. Block-0-alone prefixes are
#: the noisiest, and block-0 damage surfaces amplified at depth >= 2 anyway
#: (the forensics' compounding pattern).
MIN_PROBE_DEPTH = 2


def probe_ratio(prefix_ppl: Any, fp_prefix_ppl: Any) -> float | None:
    """Quantized-prefix / fp-prefix ratio at the SAME depth.

    Dividing by the fp prefix at the same depth is the whole trick: it divides
    out the depth-dependent PPL floor, so the ratio isolates the damage
    quantization inflicted rather than the fact that a 2-block prefix predicts
    worse than a 28-block one.

    ``None`` when either side is missing, non-finite, or the fp baseline is
    zero. A perplexity of zero is impossible and a non-finite one is a
    numerics defect, so neither is a ratio — it is a defective measurement,
    and the caller fails closed on it.
    """
    prefix = _as_finite_float(prefix_ppl)
    baseline = _as_finite_float(fp_prefix_ppl)
    if prefix is None or baseline is None or baseline == 0.0:
        return None
    return prefix / baseline


def probe_aborts(depth: int, prefix_ppl: Any, fp_prefix_ppl: Any) -> bool:
    """Whether the progress probe must halt the run at this block boundary.

    Runs the SAME comparator as the end-of-run gate — ``verdict()`` over the
    shared ``COLLAPSE_LINE`` — with the missing/non-finite check first, so
    "collapsed" means the same thing at depth 3 as at depth 28 and there is
    no second copy of the line to drift.

    Kill-only by construction: it returns ``True`` only for the gate's
    ``failed`` verdict. A probe can never report success. A 2-batch prefix
    supports *death* (every dead run measured 20x+ past the line, far outside
    slice noise) but it can never support *success* — a lucky prefix ratio
    below ``OK_LINE`` on a model that is degraded deeper is exactly the case
    this asymmetry forbids acting on. ``ok`` and ``degraded`` come only from
    the full held-out measurement.

    A defective measurement (missing, non-finite, or a zero baseline) fails
    closed: a false positive costs a re-run, a false negative costs the
    hour-plus run this probe exists to kill.
    """
    if depth is None or depth < MIN_PROBE_DEPTH:
        return False
    ratio = probe_ratio(prefix_ppl, fp_prefix_ppl)
    if ratio is None:
        return True
    return verdict(ratio) == FAILED


#: End-of-run numerics that do not exist when the probe aborted the run before
#: the end-of-run gate measured anything. On an abort report they are written as
#: JSON ``null`` — never omitted, never invented — so both report paths carry an
#: identical key set and a strict consumer cannot KeyError on either shape.
_ABORT_UNMEASURED = ("ppl_quant_native", "ppl_ratio", "effective_bpw", "qpk1_layer_bytes")


def abort_report(exc: Any, identity_fields: dict, history: list) -> dict:
    """The report for a probe-aborted run — built without torch, weights, or IO.

    This is the catch→report→status construction, kept here (not inline in the
    torch-importing driver) so it has repeatable suite coverage in a lean
    environment. The caller writes the result through ``dump_report`` on the
    same path as every other report, so the abort report is sanitized and
    strict-parse-clean exactly like a completion report.

    ``exc`` is duck-typed on purpose: the abort exception lives in
    ``quantibbler.ptq.config``, which imports this module for ``COLLAPSE_LINE``,
    so importing it back would be a cycle. Its attributes are read directly,
    with no ``getattr`` defaults, so a wrong object fails loudly rather than
    producing a report full of nulls.

    ``identity_fields`` is everything the driver knows at the moment of the
    abort — model identity, regime, the ``probe`` block without its history,
    wall time. ``history`` is the probe trajectory so far: an aborted run keeps
    it, because the trajectory that ended in a collapse is the autopsy nobody
    otherwise has. The unmeasured end-of-run numerics are forced to ``None``
    and the status to ``FAILED`` here rather than at the call site, so "an
    aborted run is never reported healthy" is structural instead of copied by
    hand — a caller that smuggles ``status`` or ``artifact_exported`` into the
    identity fields cannot overwrite either one.
    """
    report = dict(identity_fields)
    probe = dict(report.get("probe") or {})
    probe["history"] = [dict(entry) for entry in history]
    report["probe"] = probe
    report["probe_abort"] = {
        "block": exc.block,
        "depth": exc.depth,
        "prefix_ppl": exc.prefix_ppl,
        "fp_prefix_ppl": exc.fp_prefix_ppl,
        "ratio": exc.ratio,
        "line": exc.line,
    }
    report["artifact_exported"] = False
    report["status"] = FAILED
    for key in _ABORT_UNMEASURED:
        report[key] = None
    # The fidelity claim is a computed verdict on this path too, and a run
    # that did not reach the end of the protocol cannot carry it: the
    # stage-failure flag is set, so the label falls however clean the rest of
    # the record looks. `paper_faithful` in the identity fields is the
    # REQUESTED flag; it is replaced here by the verdict.
    label, conditions = fidelity_verdict(
        report.get("solver"),
        report.get("runtime_deviations"),
        report.get("stages"),
        requested=report.get("paper_faithful"),
        stage_failure=True,
    )
    report["paper_faithful"] = label
    report["fidelity_conditions"] = conditions
    return report


def stage_failure_report(exc: Any, identity_fields: dict, stages: list) -> dict:
    """The report for a run that failed a fail-closed stage gate (torch-free).

    Used by the TuneFP vacuity guard (qk-fix-tunefp-teacher A3): a run that
    CLAIMS paper fidelity must fail rather than continue on a stage that did
    not execute, and it must still leave a forensically complete report.
    Same key set as the completion and abort reports (unmeasured end-of-run
    numerics as ``null``, ``artifact_exported`` false, status ``failed``), so
    a strict consumer cannot KeyError on this shape either.

    ``exc`` is duck-typed on purpose, exactly like :func:`abort_report`: its
    attributes are read directly with no defaults, so a wrong object fails
    loudly instead of producing a report full of nulls.
    """
    report = dict(identity_fields)
    report["stage_failure"] = {
        "stage": exc.stage,
        "block": exc.block,
        "reason": exc.reason,
        "gradient_norm": exc.gradient_norm,
        "initial_residual_norm": exc.initial_residual_norm,
    }
    # The fidelity claim is recorded FALSE, never left standing: the run
    # reached a stage whose execution the paper's method requires and it did
    # not happen. `paper_faithful` in the identity fields is the REQUESTED
    # flag; the verdict below replaces it, with the condition record beside it.
    label, conditions = fidelity_verdict(
        report.get("solver"),
        report.get("runtime_deviations"),
        stages,
        requested=report.get("paper_faithful"),
        stage_failure=True,
    )
    report["paper_faithful"] = label
    report["fidelity_conditions"] = conditions
    report["stages"] = list(stages)
    report["artifact_exported"] = False
    report["status"] = FAILED
    for key in _ABORT_UNMEASURED:
        report[key] = None
    return report


def geometry_deviations(seq_len: Any, tune_seq_len: Any, calibration_rows: Any = None) -> list:
    """Named runtime deviations from [NQ]'s calibration geometry.

    The stage map covers STAGES; this covers the geometry those stages ran at,
    because a "paper-faithful" claim over a sliced tune window would otherwise
    be the same class of defect as every other green label over a deviant
    artifact: the paper's protocol is a sequence length of 2048
    (``docs/papers/nanoquant-2602.06694v3.md``, "Calibration used 128 samples
    from the WikiText-2 dataset ... with a sequence length of 2048"), so a run
    that tunes on shorter windows is NOT running the protocol even when every
    stage fires with App. C budgets.

    Returns a list of ``{kind, detail}`` entries; empty means the geometry is
    the protocol's.
    """
    out: list[dict] = []
    eval_len = _as_finite_float(seq_len)
    tune_len = _as_finite_float(tune_seq_len)
    if eval_len is None:
        out.append(
            {
                "kind": "unmeasured_eval_geometry",
                "detail": "the evaluation sequence length is not a finite number",
            }
        )
        return out
    if tune_len is None or tune_len != eval_len:
        out.append(
            {
                "kind": "tune_geometry_split",
                "detail": (
                    f"evaluation runs at seq {eval_len:g} while block reconstruction "
                    f"tunes on {tune_len:g}-token windows sliced from the same rows — the "
                    "token count is preserved but the per-step attention context is not "
                    "the protocol's; [NQ] states one sequence length (2048) for both"
                ),
            }
        )
    # Sequence-counted by contract (qk-geometry-conformance): the driver's
    # ``--calibration-seqs`` knob counts sequences ([NQ] Sec 3.4/App. C) and its
    # counter reports the same unit, so this comparison is sequences against the
    # protocol's 128 — no batch factor belongs here.
    rows = _as_finite_float(calibration_rows)
    if rows is not None and rows != 128:
        out.append(
            {
                "kind": "calibration_rows",
                "detail": (
                    f"{rows:g} calibration rows against the protocol's 128 ([NQ] Sec 3.4/App. C)"
                ),
            }
        )
    return out


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
        elif event == "tune_fp_vacuous_teacher":
            vacuous = True
    conditions = {
        "paper_faithful_flag": bool(requested),
        "no_runtime_deviations": not bool(runtime_deviations),
        "polish_off": polish_rounds == 0,
        "no_discrete_flip_pass": not flip_pass,
        "no_scale_only_rescope": not rescope,
        "no_vacuous_teacher": not vacuous,
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
        "TuneLatentSTE: factorized-component refinement",
        "per-target scheduling follows Figure 3's per-target cycle; each "
        "scoped linear consumes the full [NQ] App. C budget (epochs x "
        "(n // batch)) on the just-factorized linear; the bounded discrete "
        "flip pass is a named deviation when its budget is nonzero",
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


def stage_map(stages: list) -> dict:
    """Map every executable stage to its Algorithm 1 lines, naming deviations.

    ``stages`` is the recorder's event list (``(event, info)`` pairs). The
    result answers the faithful-run question directly: which stages of
    Algorithm 1 lines 0-28 actually executed, and which parts of the letter
    the pipeline deviates from.
    """
    executed = [event for event, _ in stages]
    counts: dict[str, int] = {}
    for event in executed:
        counts[event] = counts.get(event, 0) + 1
    out: dict[str, dict] = {}
    for key, lines, description, deviation, _required in _STAGE_LINES:
        fired = counts.get(key, 0)
        out[key] = {
            "algorithm1_lines": lines,
            "description": description,
            "invocations": fired,
            "executed": bool(fired),
            "deviation": deviation,
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


def provenance(
    solver: str,
    scale_dtype: str,
    *,
    rho_end: float | None = None,
    rho_end_source: str | None = None,
) -> dict:
    """Report provenance fields derived from the ACTUAL run arguments.

    The driver used to hardcode ``regime.artifact`` as
    ``"qpk1: +-1 bitplanes + float16 scales"`` while ``--scale-dtype`` had
    already defaulted to float32, so every forensic report lied about its own
    artifact, and ``solver`` (the only field distinguishing the r512-svd line
    from the r137 line) was never recorded at all. These strings are emitted
    from the arguments rather than from a constant, so a report cannot
    disagree with the run it describes.
    """
    return {
        "solver": solver,
        # The penalty endpoint is run-defining and has no CLI flag, so it
        # arrives by environment. Recording the value alone would be half the
        # story: two attribution runs differ in exactly this number, and a row
        # that says 1e-1 without saying whether config or environment authored
        # it cannot be audited. Value AND source, or the field lies by omission.
        "rho_end": rho_end,
        "rho_end_source": rho_end_source,
        "scale_dtype": scale_dtype,
        "regime_artifact": f"qpk1: +-1 bitplanes + {scale_dtype} scales",
        "dtype_note": (f"solver {solver} (float64 compute) / scales {scale_dtype} / artifact qpk1"),
    }


#: The evaluator behind ``ppl_quant_native``. Named in the report so a reader
#: never has to infer which harness produced the number.
EVAL_EVALUATOR = "wikitext2_test_blockwise_ppl"


def evaluation_record(
    *,
    evaluator: str,
    source_tokens: Any,
    seq_len: Any,
    selected_blocks: Any,
    batch: Any,
    selection_seed: Any,
    selection_fingerprint: str,
) -> dict:
    """The EVALUATION verdict, recorded beside the method-fidelity verdict.

    [NQ] names WikiText-2 PPL and ``lm_eval=0.4.9`` but the pinned corpus does
    not define coverage or windowing, so "paper-faithful" cannot mean "the
    paper's table" — that is a claim about the harness, not about the method.
    Two consequences, both structural here:

    * The record states exactly how much of the split was scored: source
      tokens, complete blocks, the dropped remainder, selected blocks/tokens,
      scored next-token predictions, and both coverage fractions. ``eval_rows``
      alone records the numerator, which is why an artifact could not prove
      what it covered.
    * ``paper_table_comparable`` is ``False`` unconditionally: this evaluator
      scores ONE batch-aligned window of a shuffled block sample, so its PPL is
      an internal baseline/quantized ratio, not a table row.

    Comparability deliberately does NOT enter ``runtime_deviations`` or
    ``fidelity_verdict``: those are scoped to the quantization method and the
    calibration protocol. The two verdicts sit side by side in the report
    instead of being folded into one boolean, so a faithful-labelled run whose
    pedestal is a 27%-coverage sample says so in its own field.
    """
    tokens = int(source_tokens)
    seq = int(seq_len)
    picked = int(selected_blocks)
    per_batch = int(batch)
    full_blocks = tokens // seq if seq > 0 else 0
    remainder = tokens - full_blocks * seq if seq > 0 else tokens
    selected_tokens = picked * seq if seq > 0 else 0
    return {
        "evaluator": evaluator,
        "windowing": "one batch-aligned window of a seeded, no-replacement block sample",
        "source_tokens": tokens,
        "seq_len": seq,
        "full_blocks": full_blocks,
        "remainder_tokens": remainder,
        "selected_blocks": picked,
        "selected_tokens": selected_tokens,
        # Next-token scoring: row ``t`` predicts ``t+1``, so the last position
        # of each row carries no target.
        "scored_predictions": picked * (seq - 1) if seq > 0 else 0,
        "coverage_of_full_blocks": (picked / full_blocks) if full_blocks else None,
        "coverage_of_source_tokens": (selected_tokens / tokens) if tokens else None,
        "selection_seed": selection_seed,
        # Ordered-index hash: which blocks, in which order, without shipping a
        # 144-element list into the report.
        "selection_fingerprint": selection_fingerprint,
        "full_local_window_batches": (full_blocks // per_batch) if per_batch > 0 else 0,
        "paper_table_comparable": False,
        "caveat": (
            f"{max(full_blocks - picked, 0)} complete block(s) and {remainder} trailing "
            "token(s) are not scored; the PPL here is an internal "
            "baseline/quantized ratio, not a row of a published table"
        ),
    }
