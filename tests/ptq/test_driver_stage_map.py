"""qk-fix-tunefp-teacher: the paper-faithful run record, end to end.

The claimable artifact is produced by a REAL driver run on a local tiny
Qwen3, and the report must show, from its own bytes, that the stages the
paper's method requires actually executed — with every deviation named.

What these tests CANNOT show, said plainly: that the resulting model is
good. A random-init tiny model has near-uniform logits, so the PPL ratio is
~1 whether the pipeline is faithful or not. The assertion here is about the
STAGE RECORD (which stages ran, with what numbers, and which deviations are
declared), not about artifact quality — that is the faithful run's job, at
the proof scale, on the real model.

Fixtures are imported from the sibling driver test so the offline corpus and
the tiny Qwen3 are built exactly once per session.
"""

from __future__ import annotations

import json

from test_driver_tune_seq_len import (  # noqa: F401  (pytest fixtures)
    _report,
    _run_driver,
    offline_corpus,
    tiny_model,
)


def test_paper_faithful_run_records_the_algorithm1_stage_map(
    offline_corpus,  # noqa: ANN001, F811
    tiny_model,  # noqa: ANN001, F811
    tmp_path,  # noqa: ANN001
):
    out_dir = tmp_path / "paper-faithful"
    proc = _run_driver(
        offline_corpus,
        tiny_model,
        out_dir,
        extra=["--paper-faithful", "--ste-flip-pass-budget", "2"],
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    report = _report(out_dir)

    assert report["paper_faithful"] is True
    assert report["ste_flip_pass_budget"] == 2

    mapped = report["stage_map"]
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
    # Every Algorithm 1 stage the driver owns actually executed.
    for key in ("capture", "tune_fp", "lb_admm", "ste", "sign_extract_pack", "tune_scales_kd"):
        assert mapped[key]["executed"], (key, mapped[key])
    # One TuneFP pass per block, never a vacuous one.
    assert mapped["tune_fp"]["invocations"] == mapped["capture"]["invocations"]
    assert mapped["tune_fp"]["vacuous_events"] == 0
    # Deviations are NAMED in the record, not silent.
    assert "polish" in mapped["lb_admm"]["deviation"]
    assert "informed assumption" in mapped["tune_fp"]["deviation"]
    assert "compressed-prefix" in mapped["phase1_statistics"]["deviation"]

    tune = [stage for stage in report["stages"] if stage["event"] == "tune_fp"]
    assert tune
    for stage in tune:
        assert stage["steps"] > 0, "TuneFP did not execute a single step"
        assert stage["gradient_norm"] > 0.0, "the objective gradient was zero"
        assert stage["vacuous"] is False
        assert stage["teacher_source"] == "frozen_reference_copy"
        assert isinstance(stage["teacher_fingerprint"], str) and stage["teacher_fingerprint"]
        assert stage["tunable"], "TuneFP had no still-FP remainder to tune"
        assert stage["weight_delta_rel"] > 0.0, "the fp weights never moved (Step 1 inert)"

    # The teacher capture names its provenance, and the teacher tensors were
    # never persisted (memory-only cache).
    captures = [stage for stage in report["stages"] if stage["event"] == "capture"]
    assert captures
    for stage in captures:
        assert stage["teacher_source"] == "frozen_reference_copy"
        assert stage["teacher_fingerprint"]
        assert stage["capture_index"] is not None
        assert isinstance(stage["x_b"], str), "a tensor reached the report payload"

    # TuneLatentSTE recorded per-linear, per-factor counts and scale movement
    # for every scoped linear, and every zero-flip linear carries a
    # declaration (never a silent no-op).
    ste = [stage for stage in report["stages"] if stage["event"] == "ste"]
    assert ste
    for stage in ste:
        assert stage["per_linear"], "no per-linear STE record"
        for name, rec in stage["per_linear"].items():
            assert {"flips_u", "flips_v", "s1_rel_move", "s2_rel_move"} <= set(rec), name
            if rec["allocated_steps"] > 0:
                assert rec["declared"] in {"paper_ste", "discrete_flip_pass", "scale_only"}, name
                if rec["declared"] == "scale_only":
                    assert rec["rescope_marker"] == "ste_scale_only_declared", name
            else:
                assert rec["declared"] == "not_scheduled_zero_budget", name


def test_diagnostic_run_is_not_labeled_paper_faithful(
    offline_corpus,  # noqa: ANN001, F811
    tiny_model,  # noqa: ANN001, F811
    tmp_path,  # noqa: ANN001
):
    """Without the claim the report says so, and the stage map still exists."""
    out_dir = tmp_path / "diagnostic"
    proc = _run_driver(offline_corpus, tiny_model, out_dir)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    report = _report(out_dir)
    assert report["paper_faithful"] is False
    assert report["ste_flip_pass_budget"] == 0
    assert "stage_map" in report and "stages" in report
    assert "stage_failure" not in report
    # Strict JSON: no bare NaN/Infinity reached the payload.
    assert json.dumps(report, allow_nan=False)
