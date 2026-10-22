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
    BATCH,
    SEQ_LEN,
    _report,
    _run_driver,
    offline_corpus,
    tiny_model,
)

from quantibbler.qk_report import EVAL_EVALUATOR


def _max_eval_batches(
    offline_corpus,  # noqa: ANN001, F811
    tiny_model,  # noqa: ANN001, F811
    seq_len: int = SEQ_LEN,
) -> int:
    """The maximal complete-batch eval window for the FIXTURE corpus.

    Mirrors the driver's own sizer exactly (same parquet read, same ``"\\n"``
    join, same tokenizer call, same ``// seq_len // BATCH`` arithmetic) so the
    faithful preflight — which compares ``--eval-seqs`` against the number the
    driver computes at run time — is satisfied by construction rather than by a
    hardcoded guess that would rot if the fixture corpus changed.
    """
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    shard = sorted(offline_corpus.glob("**/wikitext-2-raw-v1/test-*.parquet"))[0]
    lines = pq.read_table(shard, columns=["text"]).column("text").to_pylist()
    text = "\n".join(line for line in lines if line.strip())
    tokenizer = AutoTokenizer.from_pretrained(str(tiny_model))
    ids = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
    return len(ids) // seq_len // BATCH


def test_paper_faithful_run_records_the_algorithm1_stage_map(
    offline_corpus,  # noqa: ANN001, F811
    tiny_model,  # noqa: ANN001, F811
    tmp_path,  # noqa: ANN001
):
    """The whole faithful surface, rehearsed on the tiny Qwen3.

    Every faithful-path pre-fire ruling is exercised here at once: the App. C
    budgets, the paper's LB-ADMM solver at the App. C 400-iteration schedule,
    the greedy polish disabled, the Phase-1 statistics hoisted, the full
    per-linear STE budget with `declared: paper_ste` on every scoped linear,
    and the protocol's 128 calibration rows so the computed fidelity verdict
    has no runtime deviation to deny it.

    The `--calibration-seqs 16` is load-bearing, not incidental: [NQ]'s
    protocol is 128 rows (16 batches x 8), and the verdict flips false on any
    other count — which is why the assertion below is `== []`, not "empty or
    named".
    """
    out_dir = tmp_path / "paper-faithful"
    max_eval = _max_eval_batches(offline_corpus, tiny_model)
    proc = _run_driver(
        offline_corpus,
        tiny_model,
        out_dir,
        extra=[
            "--paper-faithful",
            "--solver",
            "admm",
            "--max-admm-iter",
            "400",
            "--calibration-seqs",
            "16",
            # A faithful claim must use the maximal local eval window; the
            # driver refuses anything smaller at preflight.
            "--eval-seqs",
            str(max_eval),
        ],
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    report = _report(out_dir)

    # --- the label is a COMPUTED verdict, and here every condition holds ---
    assert report["paper_faithful"] is True
    assert report["fidelity_conditions"] == {
        "paper_faithful_flag": True,
        "no_runtime_deviations": True,
        "polish_off": True,
        "no_discrete_flip_pass": True,
        "no_scale_only_rescope": True,
        "no_vacuous_teacher": True,
        "no_stage_failure": True,
        "solver_is_admm": True,
        # The evidence half of the claim: every `required` stage of the
        # inventory FIRED in this run (a skip-marked event would not count).
        "required_phase1_statistics_executed": True,
        "required_capture_executed": True,
        "required_tune_fp_executed": True,
        "required_lb_admm_executed": True,
        "required_ste_executed": True,
        "required_sign_extract_pack_executed": True,
        "required_tune_scales_kd_executed": True,
    }
    assert report["ste_flip_pass_budget"] == 0
    assert report["calibration_rows"] == 128
    assert report["tune_seq_len"] == report["seq_len"]
    assert report["runtime_deviations"] == [], report["runtime_deviations"]
    assert report["solver"] == "admm"

    # --- the EVALUATION verdict, beside the method verdict -----------------
    # The pedestal is the one geometry the label cannot see: on a faithful run
    # the eval window MUST be the maximal local one, and the record states what
    # that covered instead of leaving `eval_rows` as a numerator with no
    # denominator.
    record = report["evaluation"]
    assert record["evaluator"] == EVAL_EVALUATOR
    assert record["selected_blocks"] == max_eval * BATCH
    assert record["source_tokens"] >= record["selected_tokens"]
    assert record["full_blocks"] * SEQ_LEN <= record["source_tokens"]
    assert record["coverage_of_full_blocks"] == (record["selected_blocks"] / record["full_blocks"])
    assert record["coverage_of_source_tokens"] == (
        record["selected_tokens"] / record["source_tokens"]
    )
    assert record["scored_predictions"] == record["selected_blocks"] * (SEQ_LEN - 1)
    assert record["selection_seed"] == report["seed"] + 1
    assert len(record["selection_fingerprint"]) == 16
    # SEPARATION, asserted where both verdicts are visible at once: the harness
    # is never paper-table-comparable, while the method verdict is true and
    # carries no eval-derived deviation. If evaluation were folded into the
    # conjunction these two could not disagree.
    assert record["paper_table_comparable"] is False
    assert report["paper_faithful"] is True
    assert not any("eval" in str(d.get("kind", "")) for d in report["runtime_deviations"])
    assert "evaluation" not in report["fidelity_conditions"]

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
    # Phase 1 is its own pre-loop pass now, and it carries NO deviation: the
    # compressed-prefix statistics source is gone.
    assert mapped["phase1_statistics"]["executed"] is True
    assert mapped["phase1_statistics"]["invocations"] == 1
    assert mapped["phase1_statistics"]["deviation"] is None
    assert "compressed-prefix" not in json.dumps(mapped["phase1_statistics"])
    # Per-target cycle (Figure 3, re-ruled): TuneFP fires once per target
    # AFTER the first — target 1 is factorized first, because a pass before it
    # has nothing factorized and is therefore bitwise-vacuous. So the count is
    # (factorized linears) - (blocks), and it is DERIVED from the packed layer
    # lists rather than pinned to a literal. The once-per-block reading this
    # replaced asserted `== capture.invocations`.
    packed = [stage for stage in report["stages"] if stage["event"] == "sign_extract_pack"]
    factorized = sum(len(stage["layers"]) for stage in packed)
    assert factorized, "no packed layer lists to derive the cycle count from"
    assert mapped["tune_fp"]["invocations"] == factorized - mapped["capture"]["invocations"]
    assert mapped["tune_fp"]["vacuous_events"] == 0
    # ... and the extra-paper discrete stages never fired on this path.
    assert mapped["ste"]["discrete_flip_pass"] is False
    # Deviations that remain are NAMED in the record, not silent.
    assert "polish" in mapped["lb_admm"]["deviation"]
    assert "informed assumption" in mapped["tune_fp"]["deviation"]
    # The stage map's own reading was falsified by the 64,151x fire: both cells
    # must describe the per-target cycle now, and the overruled reading must not
    # survive anywhere in them.
    assert "PER-TARGET" in mapped["tune_fp"]["deviation"], (
        "the tune_fp stage-map cell still carries the overruled placement reading"
    )
    assert "once per block" not in mapped["tune_fp"]["deviation"]
    assert "per-target" in mapped["ste"]["deviation"], (
        "the ste stage-map cell still carries the overruled scheduling reading"
    )

    tune = [stage for stage in report["stages"] if stage["event"] == "tune_fp"]
    assert tune
    for stage in tune:
        assert stage["steps"] > 0, "TuneFP did not execute a single step"
        assert stage["gradient_norm"] > 0.0, "the objective gradient was zero"
        assert stage["vacuous"] is False
        assert stage["teacher_source"] == "frozen_reference_copy"
        assert isinstance(stage["teacher_fingerprint"], str) and stage["teacher_fingerprint"]
        assert stage["tunable"], "TuneFP had no still-FP remainder to tune"
    # The per-target cycle fires N-1 times, so the fires are SMALLER than the
    # once-per-block reading's single pass, and the late ones carry a
    # single-target remainder on a fixture whose block output is already fit to
    # ~1e-4. There Adam's update underflows to exactly zero (the step falls
    # below the float32 ulp of the parameter), so "every fire moved the weights"
    # is not the invariant. The invariant is TuneFP's own guard distinction: a
    # fire with a LIVE gradient must move the weights, and a fire that moved
    # nothing must have had a dead one — gradient-zero with a residual on the
    # board would be a detached student, not an underflow.
    assert any(stage["weight_delta_rel"] > 0.0 for stage in tune), (
        "not one TuneFP fire moved the fp weights: Step 1 is inert run-wide"
    )
    for stage in tune:
        if stage["weight_delta_rel"] == 0.0:
            assert stage["gradient_norm"] < 1e-9, (
                f"a fire with a live gradient ({stage['gradient_norm']!r}) moved "
                "nothing: an inert stage, not an update underflow"
            )
        else:
            assert stage["gradient_norm"] > 0.0, (
                "the weights moved with an identically-zero gradient"
            )

    # The teacher capture names its provenance, and the teacher tensors were
    # never persisted (memory-only cache).
    captures = [stage for stage in report["stages"] if stage["event"] == "capture"]
    assert captures
    for stage in captures:
        assert stage["teacher_source"] == "frozen_reference_copy"
        assert stage["teacher_fingerprint"]
        assert stage["capture_index"] is not None
        assert isinstance(stage["x_b"], str), "a tensor reached the report payload"

    # Greedy polish is OFF on the faithful path and the artifact proves it.
    admm = [stage for stage in report["stages"] if stage["event"] == "lb_admm"]
    assert admm and all(stage["polish_rounds"] == 0 for stage in admm)

    # TuneLatentSTE: FULL per-linear budget (no pooled split), `paper_ste` on
    # every scoped linear, and the measured crossings still on record.
    ste = [stage for stage in report["stages"] if stage["event"] == "ste"]
    assert ste
    for stage in ste:
        assert stage["per_linear"], "no per-linear STE record"
        assert "allocation" not in stage, "a block-pooled allocation survived"
        for name, rec in stage["per_linear"].items():
            assert {"flips_u", "flips_v", "s1_rel_move", "s2_rel_move"} <= set(rec), name
            assert rec["scheduled_steps"] == stage["total_steps"], name
            assert rec["steps"] == rec["scheduled_steps"], name
            # The paper path declares its measurement; it never re-scopes.
            assert rec["declared"] == "paper_ste", name
            assert "flip_pass" not in rec and "rescope_marker" not in rec, name
            assert isinstance(rec["zero_crossings"], bool), name


def test_a_faithful_claim_on_the_svd_ablation_is_denied_by_the_computed_verdict(
    offline_corpus,  # noqa: ANN001, F811
    tiny_model,  # noqa: ANN001, F811
    tmp_path,  # noqa: ANN001
):
    """`label_flips_on_mechanism_deviations`, on a real driver run.

    The production SVD warm start is the measured ablation and cannot carry
    the paper's label, whatever the flag says.
    """
    out_dir = tmp_path / "faithful-claim-on-svd"
    proc = _run_driver(
        offline_corpus,
        tiny_model,
        out_dir,
        extra=[
            "--paper-faithful",
            "--calibration-seqs",
            "16",
            "--eval-seqs",
            str(_max_eval_batches(offline_corpus, tiny_model)),
        ],
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    report = _report(out_dir)
    assert report["paper_faithful"] is False
    assert report["fidelity_conditions"]["paper_faithful_flag"] is True
    assert report["fidelity_conditions"]["solver_is_admm"] is False
    # It is the solver, not the geometry: the row count IS the protocol's.
    assert report["runtime_deviations"] == []


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
    # A diagnostic may use a smaller eval window — but the record says so, in
    # its own field, while the method verdict stays out of it entirely.
    record = report["evaluation"]
    assert record["paper_table_comparable"] is False
    assert record["selected_blocks"] == report["eval_rows"]
    assert record["coverage_of_source_tokens"] < 1.0
    assert record["full_local_window_batches"] > record["selected_blocks"] // BATCH
    assert "evaluation" not in report["fidelity_conditions"]
    # Strict JSON: no bare NaN/Infinity reached the payload.
    assert json.dumps(report, allow_nan=False)


def test_a_split_tune_geometry_is_named_as_a_runtime_deviation(
    offline_corpus,  # noqa: ANN001, F811
    tiny_model,  # noqa: ANN001, F811
    tmp_path,  # noqa: ANN001
):
    """`--tune-seq-len` shorter than the eval geometry is a NAMED deviation.

    The proof-scale run needs a bounded tune window (seq 2048 whole OOMs at the
    container ceiling), which slices the calibration rows into shorter windows.
    Every stage then fires with the paper's budgets while the protocol's own
    single sequence length is gone — so the report must name it, not let a
    complete stage map imply a faithful protocol.
    """
    out_dir = tmp_path / "split-geometry"
    proc = _run_driver(
        offline_corpus,
        tiny_model,
        out_dir,
        seq_len=32,
        extra=[
            "--paper-faithful",
            "--solver",
            "admm",
            "--max-admm-iter",
            "400",
            "--tune-seq-len",
            "16",
            "--eval-seqs",
            str(_max_eval_batches(offline_corpus, tiny_model)),
        ],
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    report = _report(out_dir)
    kinds = [entry["kind"] for entry in report["runtime_deviations"]]
    assert "tune_geometry_split" in kinds, report["runtime_deviations"]
    detail = next(
        entry["detail"]
        for entry in report["runtime_deviations"]
        if entry["kind"] == "tune_geometry_split"
    )
    assert "32" in detail and "16" in detail
    assert report["tune_seq_len"] == 16 and report["seq_len"] == 32
    # ... and the computed verdict DENIES the claim because of it. The label
    # was requested and the mechanism is clean (admm, no polish, no extra
    # discrete stage): the geometry alone is what falls.
    assert report["paper_faithful"] is False
    conditions = report["fidelity_conditions"]
    assert conditions["paper_faithful_flag"] is True
    assert conditions["no_runtime_deviations"] is False
    assert all(value for key, value in conditions.items() if key != "no_runtime_deviations"), (
        conditions
    )


def test_a_faithful_claim_is_refused_at_preflight_on_an_undersampled_eval_window(
    offline_corpus,  # noqa: ANN001, F811
    tiny_model,  # noqa: ANN001, F811
    tmp_path,  # noqa: ANN001
):
    """`faithful_fire_uses_maximal_local_evaluation_window`, at preflight.

    Bug Hunter's should-fix: the method label could not see the eval geometry,
    so a faithful-flagged run on a 27%-coverage sample published a green claim.
    The refusal is a usage error (nonzero exit, named reason, the number to
    use), and it fires at preflight — nothing was quantized.
    """
    max_eval = _max_eval_batches(offline_corpus, tiny_model)
    assert max_eval > 1, "fixture corpus too small to exercise an undersampled window"
    out_dir = tmp_path / "undersampled-eval"
    proc = _run_driver(
        offline_corpus,
        tiny_model,
        out_dir,
        extra=[
            "--paper-faithful",
            "--calibration-seqs",
            "16",
            "--eval-seqs",
            str(max_eval - 1),
        ],
    )
    assert proc.returncode != 0
    assert "--paper-faithful requires the maximal local evaluation window" in proc.stderr
    assert f"--eval-seqs {max_eval}" in proc.stderr
    assert f"got {max_eval - 1}" in proc.stderr
    # Preflight means preflight: no report, no artifact, no partial claim.
    assert not (out_dir / "quantize_report.json").exists()
    assert not list(out_dir.glob("layers/*.qpk1"))


def test_an_eval_window_past_the_corpus_is_still_refused_by_the_sizer(
    offline_corpus,  # noqa: ANN001, F811
    tiny_model,  # noqa: ANN001, F811
    tmp_path,  # noqa: ANN001
):
    """One past the maximal window is the sizer's overshoot refusal, unchanged.

    The preflight compares against `full_blocks // BATCH`; asking for more than
    the corpus holds must still fail in the sizer (which raises on overshoot
    rather than silently clamping), so the two refusals do not shadow each
    other and the maximal number is a hard edge, not a floor with slack.
    """
    max_eval = _max_eval_batches(offline_corpus, tiny_model)
    out_dir = tmp_path / "oversampled-eval"
    proc = _run_driver(
        offline_corpus,
        tiny_model,
        out_dir,
        extra=["--paper-faithful", "--eval-seqs", str(max_eval + 1)],
    )
    assert proc.returncode != 0
    assert "wikitext yields" in proc.stderr
    assert f"need {(max_eval + 1) * BATCH}" in proc.stderr
    assert not (out_dir / "quantize_report.json").exists()
