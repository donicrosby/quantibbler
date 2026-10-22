#!/usr/bin/env python3
"""The referee's report assertions, as pure functions — no torch, no weights, no IO.

Two CI jobs assert on a real ``quantize_report.json`` produced by a real driver
process: the tiny offline smoke (tiny random-init fixture, plumbing-only) and the
weights-cached abort leg (real Qwen3-0.6B checkpoint, detection proof). Both
kinds of assertion live here rather than inline in the workflow shell or inside a
torch-importing runner, so they are unit-testable in the lean tier: the referee's
own logic gets refereed instead of only running when CI happens to reach it.

The strict parse is the point of ``load_report``: a report carrying ``NaN`` or
``Infinity`` is not a report this repo accepts (``json.dumps`` would have written
it only via a bug), so the parser rejects those constants instead of letting
Python's permissive default read them as floats.
"""

from __future__ import annotations

import json
import math
from typing import Any

from quantibbler.qk_report import (
    COLLAPSE_LINE,
    DEGRADED,
    FAILED,
    OK,
    STEPS_EFFECT_EPS,
    exit_code,
)

ABORT_FIELDS = ("block", "depth", "prefix_ppl", "fp_prefix_ppl", "ratio", "line")
#: The evidence a long-running stage must leave behind per fire.
STAGE_EVIDENCE_FIELDS = ("residual_before", "residual_after", "gradient_norm")
_VALID_STATUSES = (OK, DEGRADED, FAILED)


class ReportError(ValueError):
    """The payload is not a strict-parseable JSON object."""


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite constant {name!r} reached the report")


def load_report(payload_text: str) -> dict:
    """Strict-parse a driver report. Raises ``ReportError`` on NaN/Infinity/garbage."""
    try:
        report = json.loads(payload_text, parse_constant=_reject_constant)
    except ValueError as exc:
        raise ReportError(f"strict parse rejected the payload: {exc}") from exc
    if not isinstance(report, dict):
        raise ReportError(f"the payload is a {type(report).__name__}, not a report object")
    return report


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def smoke_findings(payload_text: str, rc: int, expect: dict) -> list[str]:
    """Everything wrong with the tiny-fixture smoke run. Empty list == contract held.

    Plumbing only: this fixture is random-init, so its next-token distribution is
    already uniform (measured rank-4 ratio ~1.0) and no detection claim rides on
    it. What it must prove is that the gate's code path executed end-to-end at
    runtime and that the report and the exit code agree.
    """
    try:
        report = load_report(payload_text)
    except ReportError as exc:
        return [f"the smoke report is not acceptable: {exc}"]

    problems: list[str] = []
    status = report.get("status")
    if status not in _VALID_STATUSES:
        problems.append(f"report carries no usable verdict status: {status!r}")
    elif rc != exit_code(status):
        problems.append(f"process exit {rc} != verdict_exit_code({status!r}) = {exit_code(status)}")
    if report.get("probe_abort") is not None:
        problems.append("the healthy tiny fixture reported a probe abort")
    if report.get("artifact_exported") is not True:
        problems.append("a completed run must report artifact_exported: true")
    if report.get("ppl_quant_native") is None:
        problems.append("a completed run must carry the end-of-run quantized perplexity")
    for key, want in expect.items():
        got = report.get(key)
        if got != want:
            problems.append(f"provenance {key}: report says {got!r}, the run asked for {want!r}")
    # NOT wired to ``stage_evidence_findings`` yet, deliberately: the gate is
    # unit-tested against real-shaped records, but the smoke leg's own artifact
    # has not been measured against ``STEPS_EFFECT_EPS``, and the tiny fixture
    # is a near-converged regime (measured on the sibling 3-block fixture: a
    # full 256-step TuneFP budget moved the capture residual 3.96e-06 ->
    # 4.38e-06, i.e. UP). Wiring a gate that fires on the plumbing-only fixture
    # would be a false alarm, and the fixture-honesty rule is that the fixture
    # gets a live objective rather than the threshold moving. Measure the smoke
    # artifact first, then wire.
    return problems


def stage_evidence_findings(payload_text: str) -> list[str]:
    """Every stage record that cannot show its own work. Empty == evidence held.

    TuneFP and TuneLatentSTE now carry ``residual_before``/``residual_after``
    (full-capture objective) and ``gradient_norm`` (first step's backward) per
    fire, so "did this stage do anything?" is answered from the artifact
    instead of inferred. The reduction test uses the shared
    ``STEPS_EFFECT_EPS``: a record whose objective did not measurably move is an
    inert stage however many steps it booked. The shape this rejects is real —
    the rehearsal recorded a full TuneFP budget as ``1.9e-04 -> 1.9e-04`` and
    every gate passed it, because the flip counters alone cannot distinguish an
    effective stage from an inert one.
    """
    try:
        report = load_report(payload_text)
    except ReportError as exc:
        return [f"the report is not acceptable: {exc}"]

    stages = report.get("stages")
    if not isinstance(stages, list):
        return [f"the report carries no stage list: {stages!r}"]

    problems: list[str] = []

    def _check(kind: str, label: Any, rec: Any) -> None:
        if not isinstance(rec, dict):
            problems.append(f"{kind} record {label!r} is not an object")
            return
        missing = [f for f in STAGE_EVIDENCE_FIELDS if not _finite(rec.get(f))]
        if missing:
            problems.append(
                f"{kind} record {label!r} is missing a finite {missing}: "
                "the stage cannot show its own work"
            )
            return
        before, after = rec["residual_before"], rec["residual_after"]
        if after > before * (1.0 - STEPS_EFFECT_EPS):
            problems.append(
                f"{kind} record {label!r} moved its objective {before!r} -> "
                f"{after!r}, short of the shared STEPS_EFFECT_EPS "
                f"{STEPS_EFFECT_EPS!r}: an inert stage"
            )

    for entry in stages:
        if isinstance(entry, dict):
            event, info = entry.get("event"), entry
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            event, info = entry
        else:
            continue
        if not isinstance(info, dict):
            continue
        if event == "ste":
            per = info.get("per_linear")
            if not isinstance(per, dict) or not per:
                problems.append("ste stage carries no per_linear rows to audit")
                continue
            for name, rec in per.items():
                _check("ste", name, rec)
        elif event == "tune_fp":
            if info.get("skipped"):
                # A recorded skip is not an executed stage: the per-target
                # cycle legitimately has no pass on a block with one linear.
                continue
            _check("tune_fp", info.get("label") or "tune_fp", info)
    return problems


def abort_findings(
    payload_text: str,
    rc: int,
    elapsed_s: float,
    *,
    max_wall_s: float = 600.0,
    max_depth: int = 3,
) -> list[str]:
    """Everything wrong with the weights-cached abort leg. Empty list == proof held.

    This is the leg's whole reason to exist: the catch -> report -> exit wiring
    executed on a real checkpoint, under CI, with the numbers asserted from the
    real process. ``max_depth`` encodes "aborts before block 3" — the recorded
    collapse dies at depth 2 with 26 of 28 blocks never solved, so a run that
    reaches depth 3 is not the fail-fast path being proved.
    """
    try:
        report = load_report(payload_text)
    except ReportError as exc:
        return [f"the abort report is not acceptable: {exc}"]

    problems: list[str] = []
    status = report.get("status")
    if status != FAILED:
        problems.append(f"an aborted run must report status {FAILED!r}, got {status!r}")
    if rc == 0:
        problems.append("the aborted run exited 0 — a failed run must exit nonzero")
    elif status == FAILED and rc != exit_code(FAILED):
        problems.append(f"process exit {rc} != verdict_exit_code({FAILED!r}) = {exit_code(FAILED)}")
    if report.get("artifact_exported") is not False:
        problems.append("an aborted run exports nothing: artifact_exported must be false")
    for key in ("ppl_quant_native", "ppl_ratio", "effective_bpw", "qpk1_layer_bytes"):
        if report.get(key) is not None:
            problems.append(f"{key} is {report[key]!r}: an aborted run has no end-of-run numerics")

    abort = report.get("probe_abort")
    if not isinstance(abort, dict):
        problems.append(f"probe_abort is missing or not an object: {abort!r}")
        return problems
    missing = sorted(set(ABORT_FIELDS) - set(abort))
    if missing:
        problems.append(f"probe_abort is missing fields {missing}")

    depth = abort.get("depth")
    if not isinstance(depth, int) or isinstance(depth, bool) or depth >= max_depth:
        problems.append(
            f"the run aborted at depth {depth!r}: this leg must abort before block {max_depth}"
        )
    if abort.get("line") != COLLAPSE_LINE:
        problems.append(
            f"probe_abort.line is {abort.get('line')!r} but the shared COLLAPSE_LINE is "
            f"{COLLAPSE_LINE!r} — a second threshold would be a hand-tuned re-pin"
        )
    ratio = abort.get("ratio")
    if not _finite(ratio) or ratio <= COLLAPSE_LINE:
        problems.append(f"probe_abort.ratio {ratio!r} is not a finite crossing of {COLLAPSE_LINE}")
    for key in ("prefix_ppl", "fp_prefix_ppl"):
        value = abort.get(key)
        if not _finite(value) or value <= 0:
            problems.append(f"probe_abort.{key} {value!r} is not a positive finite perplexity")

    history = (report.get("probe") or {}).get("history")
    if not isinstance(history, list) or not history:
        problems.append(
            "the abort report carries no probe history: the trajectory that collapsed "
            "is the autopsy"
        )
    elif history[-1].get("depth") != depth or history[-1].get("block") != abort.get("block"):
        problems.append(
            f"history does not end at the aborting block: {history[-1]!r} vs depth "
            f"{depth!r}/{abort.get('block')!r}"
        )

    if elapsed_s >= max_wall_s:
        problems.append(
            f"the run took {elapsed_s:.0f}s — this leg must fail in minutes, not tens of "
            f"minutes (< {max_wall_s:.0f}s)"
        )
    reported_wall = report.get("wall_seconds")
    if not _finite(reported_wall) or reported_wall >= max_wall_s:
        problems.append(f"report wall_seconds {reported_wall!r} is not below {max_wall_s:.0f}s")
    return problems


def report_findings(problems: list[str], label: str) -> int:
    """Emit findings as workflow errors and return the process exit code."""
    for problem in problems:
        print(f"::error::{label}: {problem}")
    if problems:
        print(f"{label}: FAILED ({len(problems)} finding(s))")
        return 1
    print(f"{label}: the asserted contract held")
    return 0
