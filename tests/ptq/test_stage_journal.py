"""The stage journal beside the solve checkpoints (qk-fix-tunefp-teacher
post-fire row).

Scenario: ``required_stage_execution_reads_the_same_record``, with the delta's
``teacher_target_is_persisted_and_recovered_on_resume`` for the persistence
discipline the journal rides (fingerprint-bound, refused on mismatch). The
adoption of the journal's own model (`qk-serialization`) adds
``producers_emit_through_their_model`` and ``journal_reader_refuses_an_unread_version``.

The gap, stated as the record it produces: a run that solved a block, died in
the pipeline tail, and resumes has a stage record holding only the stages THE
RESUME emitted — Phase-1 statistics, and (in the driver) the scale stage. Every
required stage the restored blocks executed reads false, so a legitimate run
loses its fidelity label. The journal is the fix; these tests are its evidence
at the layer where resume lives, so the equivalence asserted below is between
two records of the SAME pipeline, not between a record and a hand-built one.

Tier note: the run here is the tiny api fixture, on the numpy oracle, with
checkpointing on. At THIS fixture every block carries a single quantizable
linear, so TuneFP's per-target cycle records a `skipped` event (`tune_fp` with
``skipped``) and cannot count as executed — a property of the fixture, not of
the journal. The seven-of-seven record is asserted at the real driver
(``tests/ptq/test_driver_stage_journal.py``).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from test_api import _calib, _TinyApiModel
from test_checkpoints import _cfg

import quantibbler.ptq.block_recon as block_recon
import quantibbler.ptq.scale_calib as scale_calib
from quantibbler import serialization as ser
from quantibbler.ptq.api import quantize_model
from quantibbler.ptq.checkpoints import (
    append_stage_event,
    read_stage_log,
    stage_log_path,
)
from quantibbler.qk_report import fidelity_conditions, stage_journal_record
from quantibbler.serialization import ArtifactSchemaError

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The required stages a crash in the pipeline TAIL leaves to the journal: the
#: ones the interrupted launch executed inside the block loop. Phase-1
#: statistics re-runs on the resumed launch (it is not checkpointed), and the
#: scale stage is emitted after the block loop by the ``calibrate_scales`` call
#: site in ``api.py`` — so the resume re-runs (and re-emits) both, and neither
#: needs the journal to attest it.
_SOLVE_STAGES = ("capture", "tune_fp", "lb_admm", "ste", "sign_extract_pack")


@pytest.fixture(scope="module")
def driver_module():
    """The driver module, for its REAL stage recorder + payload sanitizer.

    Loaded from source the way ``test_driver_probe.py`` loads it, so the journal
    these tests write is written by the production recorder (tensors collapsed
    to type markers, one append per event) rather than by a test double.
    """
    path = REPO_ROOT / "tools" / "quantize_hf_model.py"
    spec = importlib.util.spec_from_file_location("qk_stage_journal_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity_model() -> _TinyApiModel:
    """The model whose identity the fingerprint binds (qk-fp-model-identity).

    ``read_stage_log`` / ``resume_stage_events`` are fingerprint-guarded LIVE and
    the fingerprint now binds the model as well as the config, so a reader has to
    name the model it is reading for. The journals here are written by ``_launch``
    on this exact fixture, so its identity is the one to pass.
    """
    return _TinyApiModel(seed=3)


def _tail_crash(monkeypatch) -> None:
    """Crash AFTER the last block was solved and checkpointed.

    The card's gap is a run that dies in the pipeline tail — every block on
    disk, the run not completed — so the interruption is placed at Phase 3
    (scale calibration), the first stage after the block loop.
    """

    def exploding(*_a, **_kw):
        raise RuntimeError("simulated tail crash: scale calibration")

    monkeypatch.setattr(scale_calib, "calibrate_scales", exploding)


def _launch(cfg, batches, ckpt_dir, driver_module, *, seed: int = 3):
    """One launch of the pipeline with checkpointing and the journal enabled."""
    sink: list = []
    recorder = driver_module._stage_recorder(sink, ckpt_dir)
    quantize_model(
        _TinyApiModel(seed=seed),
        batches,
        cfg,
        checkpoint_dir=ckpt_dir,
        recorder=recorder,
        stage_sink=sink,
    )
    return sink


def _names(sink) -> list[str]:
    return [event for event, _info in sink]


def test_a_tail_resumed_record_matches_the_uninterrupted_record(
    tmp_path, monkeypatch, driver_module
):
    """The fix, as an EQUIVALENCE between two records of the same pipeline.

    Interrupt a faithful tail: solve both blocks, die at scale calibration.
    Every block is then on disk, so the resume re-runs NO solve — the solver is
    rigged to explode if it is called — and the only events the resumed launch
    can emit itself are Phase-1 statistics and the scale stage (both re-run on
    every launch; neither is checkpointed). Everything else in its record comes
    from the journal, and the record has to read exactly like the uninterrupted
    run's, which is what makes a tail-resumed artifact's label mean the same
    thing as a clean run's.
    """
    torch.manual_seed(0)
    cfg = _cfg()
    batches = _calib(seed=7)

    uninterrupted = _launch(cfg, batches, tmp_path / "clean", driver_module)

    ckpt_dir = tmp_path / "ckpts"
    _tail_crash(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated tail crash"):
        _launch(cfg, batches, ckpt_dir, driver_module)
    monkeypatch.undo()

    journal = read_stage_log(ckpt_dir, cfg, _identity_model())
    journal_names = _names(journal)
    assert set(_SOLVE_STAGES) <= set(journal_names), (
        f"the interrupted launch journaled {journal_names}: the block loop's stages are what a "
        "tail resume cannot re-emit"
    )

    def exploding(*_a, **_kw):
        raise RuntimeError("solver must not be called on a fully-checkpointed resume")

    monkeypatch.setattr(block_recon, "reconstruct_block", exploding)
    resumed = _launch(cfg, batches, ckpt_dir, driver_module)
    monkeypatch.undo()

    assert _names(resumed) == [*journal_names, "phase1_statistics", "tune_scales_kd"], (
        "the resumed record is not exactly the seeded journal plus the stages this launch re-emits"
    )
    assert resumed[: len(journal)] == journal, (
        "the journal was not seeded verbatim at the head of the resumed record"
    )
    assert _names(resumed).count("phase1_statistics") == 1, (
        "Phase-1 statistics appears more than once: the seed read the resumed launch's own prefix"
    )

    clean = fidelity_conditions("admm", [], uninterrupted, requested=True)
    after_resume = fidelity_conditions("admm", [], resumed, requested=True)
    assert after_resume == clean, (
        "a tail-resumed run's condition record differs from the uninterrupted run's: the journal "
        "did not restore the evidence of the stages it skipped"
    )
    for key in ("capture", "lb_admm", "ste", "sign_extract_pack"):
        assert after_resume[f"required_{key}_executed"] is True, key


def test_a_missing_journal_denies_exactly_as_before_the_journal_existed(
    tmp_path, monkeypatch, driver_module
):
    """Never synthesize: no journal means the record is only what this launch did.

    This is the pre-journal behavior, pinned so the fix cannot quietly grow into
    "assume the stages ran": the required-stage conditions of the stages the
    restored blocks executed all fall, which is the honest false-deny the
    journal exists to replace.
    """
    torch.manual_seed(0)
    cfg = _cfg()
    batches = _calib(seed=7)
    ckpt_dir = tmp_path / "ckpts"

    _tail_crash(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated tail crash"):
        _launch(cfg, batches, ckpt_dir, driver_module)
    monkeypatch.undo()

    assert read_stage_log(ckpt_dir, cfg, _identity_model()), (
        "the interrupted launch left no journal to remove"
    )
    stage_log_path(ckpt_dir).unlink()

    def exploding(*_a, **_kw):
        raise RuntimeError("solver must not be called on a fully-checkpointed resume")

    monkeypatch.setattr(block_recon, "reconstruct_block", exploding)
    resumed = _launch(cfg, batches, ckpt_dir, driver_module)
    monkeypatch.undo()

    assert _names(resumed) == ["phase1_statistics", "tune_scales_kd"]
    conditions = fidelity_conditions("admm", [], resumed, requested=True)
    for key in _SOLVE_STAGES:
        assert conditions[f"required_{key}_executed"] is False, (
            f"required_{key}_executed is True with no journal: nothing was synthesized, so a "
            "stage the resume did not re-run cannot be attested"
        )
    assert conditions["required_phase1_statistics_executed"] is True, (
        "Phase-1 re-runs on every launch and must still count"
    )


def test_the_journal_is_read_only_behind_the_fingerprint(tmp_path, driver_module):
    """The fingerprint guard, live, and the per-line shape check behind it."""
    torch.manual_seed(0)
    cfg = _cfg()
    batches = _calib(seed=7)
    ckpt_dir = tmp_path / "ckpts"
    _launch(cfg, batches, ckpt_dir, driver_module)
    good_line = stage_log_path(ckpt_dir).read_text(encoding="utf-8").splitlines()[0]

    stale = dataclasses.replace(cfg, rank=cfg.rank + 1)
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        read_stage_log(ckpt_dir, stale, _identity_model())

    # A name-only line: the shape a corrupted or hand-written record takes, and
    # the one that would satisfy every required-stage condition on its name.
    for poison, expected in (
        (json.dumps({"event": "lb_admm"}), "carries no payload object"),
        (json.dumps({"event": "lb_admm", "info": {}}), "missing the payload keys"),
        (json.dumps({"event": "capture", "info": {}}), "missing the payload keys"),
        (json.dumps({"event": "lb_adm", "info": {"polish_rounds": 0}}), "unregistered name"),
        # MONOTONE-LEVERAGE (the repo skill's corruption class: the entry is
        # multiplied by 2^16 or saturates to inf/NaN, so the value dominates any
        # artifact state). The key set is intact — only the measurement is
        # poisoned — which is exactly the line a shape-only gate would bless.
        (
            json.dumps({"event": "lb_admm", "info": {"polish_rounds": float("inf")}}),
            "non-finite number",
        ),
        (
            json.dumps({"event": "lb_admm", "info": {"polish_rounds": float("nan")}}),
            "non-finite number",
        ),
        # A literal that saturates on PARSE: json.loads("1e400") is inf, so the
        # line is syntactically valid JSON and still a poison. Built as a raw
        # string because json.dumps cannot emit the literal without producing
        # the non-standard Infinity token instead.
        ('{"event": "lb_admm", "info": {"polish_rounds": 1e400}}', "non-finite number"),
        (good_line[:-12], "not JSON"),
    ):
        stage_log_path(ckpt_dir).write_text(poison + "\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match=expected):
            read_stage_log(ckpt_dir, cfg, _identity_model())

    # The control, on the same bytes: the untouched journal still reads.
    stage_log_path(ckpt_dir).write_text("\n".join([good_line, ""]), encoding="utf-8")
    assert read_stage_log(ckpt_dir, cfg, _identity_model())  # a trailing blank line is not a record


def _reject_constant(token):
    raise AssertionError(f"non-standard JSON token {token!r} written to the journal")


def test_the_writer_keeps_the_journal_strict_json(tmp_path):
    """The WRITE half of the monotone-leverage guard.

    ``json.dumps``' default ``allow_nan=True`` emits the non-standard tokens
    ``NaN``/``Infinity`` for non-finite floats — strict JSON parsers reject
    them, and a resume that must read its own journal would crash on its own
    legitimate degenerate metric. ``append_stage_event`` sanitizes first
    (non-finite -> ``null``) and refuses to emit a non-standard token, so the
    line stays strict JSON and the reader sees an honest unmeasured ``null``
    instead of a saturated value that would dominate every comparison.
    """
    ckpt_dir = tmp_path / "ckpts"
    append_stage_event(
        ckpt_dir,
        "tune_fp_vacuous_teacher",
        {
            "block": "layers.0",
            "gradient_norm": float("nan"),
            "initial_residual_norm": float("inf"),
            "reason": "zero_first_step_gradient",
        },
    )
    raw = stage_log_path(ckpt_dir).read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw, (
        "the writer emitted a non-standard JSON token"
    )
    # Strict parse: the stdlib's strict default rejects the tokens, so a parse
    # that succeeds here proves the line is strict JSON.
    line = json.loads(raw.strip(), parse_constant=_reject_constant)
    assert line["info"]["gradient_norm"] is None
    assert line["info"]["initial_residual_norm"] is None
    # The attestation survives the sanitization: the event and its keys remain.
    event, info = stage_journal_record(line)
    assert event == "tune_fp_vacuous_teacher"
    assert info["reason"] == "zero_first_step_gradient"


def test_a_fresh_start_drops_the_journal_of_the_fingerprint_it_replaces(tmp_path, driver_module):
    """The other half of the binding: a dir that starts over cannot inherit.

    A fresh start rewrites ``fingerprint.json``, so the journal on disk at that
    moment belongs to a config nobody can resume as; it is dropped rather than
    left beside the new fingerprint for a later resume to read.
    """
    torch.manual_seed(0)
    cfg = _cfg()
    batches = _calib(seed=7)
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    append_stage_event(ckpt_dir, "lb_admm", {"polish_rounds": 3, "launch": "dead"})

    sink = _launch(cfg, batches, ckpt_dir, driver_module)

    journal = read_stage_log(ckpt_dir, cfg, _identity_model())
    assert [info for _e, info in journal if info.get("launch") == "dead"] == [], (
        "a fresh start kept the journal of the fingerprint it replaced"
    )
    assert _names(journal), "the fresh launch's own events are journaled"
    assert "lb_admm" in _names(sink)


def test_scn_producers_emit_through_their_model_journal_line(tmp_path):
    """The writer emits the entry MODEL, so every line is self-describing.

    A journal that cannot say which shape it was written in is one a later
    reader has to guess at; the model supplies the version, and the line it
    writes is one the model accepts on the way back in.
    """
    ckpt_dir = tmp_path / "ckpts"
    append_stage_event(ckpt_dir, "lb_admm", {"polish_rounds": 3})

    line = json.loads(stage_log_path(ckpt_dir).read_text(encoding="utf-8").strip())
    assert line["schema_version"] == ser.SCHEMA_VERSION
    entry = ser.StageJournalEntry.model_validate(line)
    assert (entry.event, entry.info) == ("lb_admm", {"polish_rounds": 3})
    # One vocabulary: the report module's shape gate reads the same line.
    assert stage_journal_record(line) == ("lb_admm", {"polish_rounds": 3})


def test_the_writer_refuses_a_line_the_reader_would_refuse(tmp_path):
    """The gate runs at the APPEND, not only at the read.

    A refused line is a run poisoning its own resume: the next launch inherits a
    journal that raises on the way in. The model refuses it where it is written
    — a payload-blind event name and an unregistered one alike.
    """
    ckpt_dir = tmp_path / "ckpts"
    with pytest.raises(ValidationError):
        append_stage_event(ckpt_dir, "lb_admm", {})
    with pytest.raises(ValidationError):
        append_stage_event(ckpt_dir, "lb_adm", {"polish_rounds": 0})
    assert not stage_log_path(ckpt_dir).exists(), (
        "a refused event wrote a journal line the reader would then refuse"
    )


@pytest.mark.parametrize("version", [0, 2, "1"])
def test_scn_journal_reader_refuses_an_unread_version(tmp_path, driver_module, version):
    """An unread journal version refuses by NAME, behind the fingerprint guard.

    These lines seed a resumed run's stage record, so a shape this build cannot
    vouch for must refuse with ``ArtifactSchemaError`` rather than being parsed
    as the current one — including a version that is not an exact integer, which
    must not coerce into a match.
    """
    torch.manual_seed(0)
    cfg = _cfg()
    batches = _calib(seed=7)
    ckpt_dir = tmp_path / "ckpts"
    _launch(cfg, batches, ckpt_dir, driver_module)

    record = json.loads(stage_log_path(ckpt_dir).read_text(encoding="utf-8").splitlines()[0])
    record["schema_version"] = version
    stage_log_path(ckpt_dir).write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactSchemaError, match="schema_version"):
        read_stage_log(ckpt_dir, cfg, _identity_model())
