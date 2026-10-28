"""The report builders: abort, stage-failure, geometry, provenance, and evaluation."""

from __future__ import annotations

from typing import Any

from .band import FAILED, _as_finite_float
from .stages import fidelity_verdict

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
    it, because the trajectory that ended in a collapse is the forensic record nobody
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
    # the record looks. The verdict is derived from the declared config the
    # caller carries (`fidelity_config`), never from a self-asserted flag.
    label, conditions = fidelity_verdict(
        report.get("solver"),
        report.get("runtime_deviations"),
        report.get("stages"),
        config=report.get("fidelity_config"),
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
    # The fidelity claim is computed from the declared config and recorded,
    # never left standing: the run reached a stage whose execution the paper's
    # method requires and it did not happen. The condition record sits beside
    # the label so this run's denial is auditable field by field.
    label, conditions = fidelity_verdict(
        report.get("solver"),
        report.get("runtime_deviations"),
        stages,
        config=report.get("fidelity_config"),
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


def resume_record(
    *,
    resumed: bool,
    restored_blocks: Any,
    checkpoint_dir: str | None,
    image: str,
    image_digest: str,
    admitted_by_fingerprint: dict | None,
) -> dict:
    """The top-level ``resume`` record: a reader-facing restart declaration.

    A resumed launch's ``wall_seconds`` covers the post-resume segment only, so
    the number alone cannot tell a reader that the full solve was not measured
    end to end. This record names the discontinuity from launch state the driver
    already holds: whether a prior launch's completed blocks were restored,
    which blocks, from which directory, under which container image, and the
    fingerprint admission the restore was checked against.

    It is a RECORD, never a gate — no verdict, band edge, exit code, stage
    condition or fail-fast reads it — and it is emitted on every report shape
    (completion, probe-abort, stage-failure) with the same key set, so a strict
    consumer never KeyErrors on one shape. It rides beside the identity fields
    rather than inside the serialized report model, so an older build still
    parses a newer report and a report written before this record existed still
    parses here.

    ``admitted_by_fingerprint`` is the restored checkpoint's own admission
    record, threaded verbatim from what ``check_fingerprint`` read — the caller
    copies the real record rather than re-deriving a subset. No
    ``identical_bytes`` claim is computed: whether two launches ran identical
    source bytes is a host-side measurement (a diff across two tips) that the
    container, which has no ``git``, cannot make. Absence of the claim is the
    honest shape, never a ``True`` the driver cannot support.
    """
    return {
        # Derived at the call site from the completed-block scan, so the flag
        # and the list it describes cannot disagree.
        "resumed": bool(resumed),
        "restored_blocks": list(restored_blocks),
        "checkpoint_dir": checkpoint_dir,
        "image": image,
        "image_digest": image_digest,
        "admitted_by_fingerprint": admitted_by_fingerprint,
    }


def geometry_deviations(seq_len: Any, tune_seq_len: Any, calibration_rows: Any = None) -> list:
    """Named runtime deviations from [NQ]'s calibration geometry.

    The stage map covers STAGES; this covers the geometry those stages ran at,
    because a label over a sliced tune window would otherwise be the same class
    of defect as every other green label over a deviant
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
    # Sequence-counted by contract: the driver's
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


def provenance(
    solver: str,
    scale_dtype: str,
    *,
    remainder_dtype: str = "fp32",
    rho_end: float | None = None,
    rho_end_source: str | None = None,
    rho_start: float | None = None,
    rho_start_source: str | None = None,
    admm_reg: float | None = None,
    admm_reg_source: str | None = None,
    solver_backend: str | None = None,
    solver_backend_source: str | None = None,
    solver_device: str | None = None,
) -> dict:
    """Report provenance fields derived from the ACTUAL run arguments.

    These strings are emitted from the run arguments rather than from a
    constant, so a report cannot disagree with the run it describes (a past
    hardcode shipped reports mislabeling their own storage dtype, and
    ``solver`` went unrecorded entirely).
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
        # Regime identity: the schedule start and the stabilizer floor
        # are both run-defining, and the floor is env-overridable. Same rule as
        # rho_end -- value AND source, or the row cannot be audited.
        "rho_start": rho_start,
        "rho_start_source": rho_start_source,
        "admm_reg": admm_reg,
        "admm_reg_source": admm_reg_source,
        # The solver arm and the device it ran on. "torch" alone under-identifies
        # a report: cpu and cuda are different numerics, and a report that names
        # only the library is the same mislabelling class as an unlabelled rho.
        "solver_backend": solver_backend,
        "solver_backend_source": solver_backend_source,
        "solver_device": solver_device,
        "scale_dtype": scale_dtype,
        # The container declares TWO storage dtypes: the per-layer scales and
        # the FP remainder. Naming only the scales under-identifies the bytes —
        # the same run can ship an fp32 or a bf16 remainder — so both are
        # recorded from the run's arguments, never from a literal.
        "remainder_dtype": remainder_dtype,
        "regime_artifact": (
            f"qpk1: +-1 bitplanes + {scale_dtype} scales + {remainder_dtype} remainder"
        ),
        "dtype_note": (
            f"solver {solver} (float64 compute) / scales {scale_dtype} / "
            f"remainder {remainder_dtype} / artifact qpk1"
        ),
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
    not define coverage or windowing, so the fidelity label cannot mean "the
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
