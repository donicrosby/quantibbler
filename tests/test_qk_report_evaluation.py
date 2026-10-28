"""The EVALUATION verdict, recorded beside the method-fidelity verdict.

Split out of ``tests/test_qk_report.py`` (surface 6 of 6); every test body is
verbatim, and the docstrings keep their scenario IDs:
`evaluation_coverage_is_artifact_provable` and
`evaluation_limit_does_not_masquerade_as_method_deviation`, plus
`label_requires_fp_stream_teacher_provenance`. The pedestal's coverage is
recorded beside the method verdict, never inside it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_qk_report_abort import _ABORT_HISTORY, _ABORT_IDENTITY, _abort_exc
from test_qk_report_stages import _clean_stages

from quantibbler.qk_report import (
    EVAL_EVALUATOR,
    abort_report,
    evaluation_record,
    fidelity_verdict,
    geometry_deviations,
    stage_failure_report,
)

# ---------------------------------------------------------------------------
# Evaluation verdict: the
# pedestal's COVERAGE is recorded beside the method verdict, never inside it.
# ---------------------------------------------------------------------------


def test_evaluation_record_states_the_pinned_wikitext_coverage():
    """`evaluation_coverage_is_artifact_provable`: the pinned WT2 numbers.

    ``EVAL_SEQS=18`` is the maximal complete-batch window at seq 2048 on
    298,938 tokens: 145 complete blocks, one 1,978-token remainder, 144 picked.
    The record must say all of that — ``eval_rows`` alone (144) is a numerator
    with no denominator, which is exactly how a sample passes as a split.
    """
    record = evaluation_record(
        evaluator=EVAL_EVALUATOR,
        source_tokens=298_938,
        seq_len=2048,
        selected_blocks=144,
        batch=8,
        selection_seed=1,
        selection_fingerprint="0123456789abcdef",
    )
    assert record["full_blocks"] == 145
    assert record["remainder_tokens"] == 1_978
    assert record["selected_blocks"] == 144
    assert record["selected_tokens"] == 294_912
    # Row t predicts t+1: the last position of each row carries no target.
    assert record["scored_predictions"] == 294_768
    assert record["coverage_of_full_blocks"] == pytest.approx(144 / 145)
    assert record["coverage_of_source_tokens"] == pytest.approx(294_912 / 298_938)
    assert record["full_local_window_batches"] == 18
    assert record["selection_seed"] == 1
    assert record["selection_fingerprint"] == "0123456789abcdef"
    assert record["paper_table_comparable"] is False
    assert "not scored" in record["caveat"]


def test_full_block_coverage_still_is_not_a_paper_table_row():
    """Even at 100% block coverage the harness is not the paper's table.

    A batch-aligned window over a seeded block sample is not ``lm_eval`` on the
    whole split, so comparability is ``False`` unconditionally — the flag must
    not become True just because coverage reached 1.0 (that would make it a
    function of one lucky corpus size).
    """
    record = evaluation_record(
        evaluator=EVAL_EVALUATOR,
        source_tokens=144 * 2048,
        seq_len=2048,
        selected_blocks=144,
        batch=8,
        selection_seed=7,
        selection_fingerprint="ffffffffffffffff",
    )
    assert record["coverage_of_full_blocks"] == 1.0
    assert record["coverage_of_source_tokens"] == 1.0
    assert record["paper_table_comparable"] is False


#: A declared config that IS the paper regime: the verdict is computed from the
#: declaration, so a clean-label test has to declare one.
_PAPER_REGIME_CONFIG = {
    "solver": "admm",
    "compute_dtype": "bf16",
    "factor_mode": "binary",
    "ablation": "none",
    "layer_scope": "all",
    "admm_iter": 400,
    "rho_end": 1.0,
    "device": "cuda",
}


def test_an_undersampled_eval_window_does_not_enter_the_method_verdict():
    """`evaluation_limit_does_not_masquerade_as_method_deviation`.

    A 40-row window on WT2 covers 27.6% of the test split. That is a real
    honesty problem — but it is an EVALUATION fact, so it is named in the
    evaluation record and stays out of ``runtime_deviations`` and out of the
    method-fidelity conjunction, which the ruling scopes to the quantization
    method and the calibration protocol. If a later hand re-folds eval
    coverage into either, this test fails.
    """
    record = evaluation_record(
        evaluator=EVAL_EVALUATOR,
        source_tokens=298_938,
        seq_len=2048,
        selected_blocks=40,
        batch=8,
        selection_seed=1,
        selection_fingerprint="deadbeefdeadbeef",
    )
    assert record["coverage_of_source_tokens"] < 0.3
    # The geometry check has no eval-coverage condition, by design.
    assert geometry_deviations(2048, 2048, 128) == []
    label, conditions = fidelity_verdict("admm", [], _clean_stages(), config=_PAPER_REGIME_CONFIG)
    assert label is True
    assert "evaluation" not in conditions
    assert "eval_coverage" not in conditions
    assert all(conditions.values())


def test_non_completion_report_shapes_carry_the_evaluation_record():
    """`evaluation_coverage_is_artifact_provable` on the non-completion paths."""
    record = evaluation_record(
        evaluator=EVAL_EVALUATOR,
        source_tokens=298_938,
        seq_len=2048,
        selected_blocks=144,
        batch=8,
        selection_seed=1,
        selection_fingerprint="0123456789abcdef",
    )
    identity = dict(_ABORT_IDENTITY, fidelity_config=_PAPER_REGIME_CONFIG, evaluation=record)
    aborted = abort_report(_abort_exc(), dict(identity, probe={}), _ABORT_HISTORY)
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
    for report in (aborted, failure):
        assert report["evaluation"] == record
        # ... and the record is not a fidelity condition on those paths either.
        assert "evaluation" not in report["fidelity_conditions"]
        assert report["paper_faithful"] is False


def test_scn_label_requires_fp_stream_teacher_provenance():
    """Scenario ``label_requires_fp_stream_teacher_provenance`` (Phase-2 gated).

    A computed fidelity label owes evidence that the teacher target came from the
    FP-stream capture. Today no condition in the verdict can see a capture's stream
    provenance, so a run whose teacher read the COMPRESSED stream is labelled
    faithful -- the fidelity hole this closes.

    Isolated by positive control, not by hand: the baseline record satisfies all
    fifteen conditions, so each sub-case below varies ONLY the capture provenance
    field and any negative label is load-bearing by construction. Without the
    baseline, a negative assert can pass because some unrelated condition is false,
    which is exactly how the first draft of this test went soft.
    """
    import quantibbler.qk_report as R

    keys = [k for k, *_ in R._STAGE_LINES]
    # tune_scales_kd counts as executed only when its payload carries an artifact
    # (the ``artifact`` clause in ``fidelity_conditions``), so a uniform payload
    # cannot satisfy all fifteen and the baseline is per-stage rather than generic.
    # ``capture`` is excluded here and supplied by _with_capture below: a base
    # entry for it would be an unprovenanced second capture, and the provenance
    # condition correctly denies the label when ANY capture lacks provenance.
    # Carrying a generic base entry alongside the real one made the positive
    # control fail for a reason that had nothing to do with the sub-cases.
    base = [
        (k, {"artifact": "scales.json"} if k == "tune_scales_kd" else {"executed": True})
        for k in keys
        if k != "capture"
    ]

    def _with_capture(**prov):
        return [*base, ("capture", {"capture_index": 1, **prov})]

    # --- positive control: every condition satisfied, fp-stream provenance -----
    baseline = _with_capture(teacher_source="fp", x_b_stream="fp", y_b_stream="fp")
    label0, conds0 = R.fidelity_verdict("admm", [], baseline, config=_PAPER_REGIME_CONFIG)
    assert label0 is True, (
        "the positive control does not satisfy the verdict: "
        f"{sorted(k for k, v in conds0.items() if not v)}. Every sub-case below would "
        "be riding on this condition, so stop until the baseline is clean."
    )

    # --- (a) capture carries no stream provenance at all ----------------------
    label_a, conds_a = R.fidelity_verdict("admm", [], _with_capture(), config=_PAPER_REGIME_CONFIG)
    assert label_a is False, (
        "a capture with no stream provenance was labelled faithful; the label "
        "must require evidence that the teacher came from the fp stream"
    )
    assert any("stream" in n for n in conds_a), (
        f"no stream-aware condition exists in the verdict: {sorted(conds_a)}"
    )

    # --- (b) the capture explicitly came from the compressed stream -----------
    comp = _with_capture(teacher_source="compressed", x_b_stream="compressed")
    label_b, conds_b = R.fidelity_verdict("admm", [], comp, config=_PAPER_REGIME_CONFIG)
    assert label_b is False, "a compressed-stream teacher capture was labelled faithful"
    assert any("stream" in n for n in conds_b), (
        f"no stream-aware condition exists in the verdict: {sorted(conds_b)}"
    )

    # --- (c) fp-stream provenance is the baseline: covered by the positive control,
    # so no second assertion is needed -- label0 is True above with only these
    # fields differing from (a) and (b).

    # --- (d) the documented dual-stream deviation must not flip the label -----
    label_d, conds_d = R.fidelity_verdict(
        "admm",
        [],
        _with_capture(
            teacher_source="fp",
            x_b_stream="fp",
            y_b_stream="fp",
            deviation="dual_stream_teacher_target",
        ),
        config=_PAPER_REGIME_CONFIG,
    )
    assert label_d is True, (
        "the documented dual-stream deviation flipped the label; a declared deviation "
        f"is not a fidelity failure: {sorted(k for k, v in conds_d.items() if not v)}"
    )
