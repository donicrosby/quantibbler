"""The stage journal's shape contract (qk-fix-tunefp-teacher post-fire row).

Scenario: ``required_stage_execution_reads_the_same_record``. That scenario's
record is read from ONE normalized source, and every required-stage condition
counts an event's NAME — deliberately payload-blind, because an in-process
recorder cannot emit a name without the emitter having run. The property is
sound exactly as long as the record is in-process, which is why the required
conditions may not be handed a record re-read from disk unvalidated: the stage
journal beside the solve checkpoints (``checkpoints.read_stage_log``) is that
path now, and ``stage_journal_record`` is the gate on it.

This file pins the gate's two halves, lean (no torch): the payload shape table
covers the whole stage inventory and nothing else, and a line that names a stage
without carrying its payload is refused rather than counted as an execution.

What these tests do NOT show: that a journal on disk belongs to the run reading
it. That binding is the checkpoint fingerprint's job and is tested with the
reader (``tests/ptq/test_stage_journal.py``); the shape check here is a SHAPE
check, not a signature — the last test pins that limit rather than leaving it to
a reader's imagination.
"""

from __future__ import annotations

import pytest

from quantibbler import qk_report as R

#: One real payload per REQUIRED stage, trimmed from a real driver run's
#: ``quantize_report.json`` ``stages`` (tiny Qwen3 fixture, ``--paper-faithful``
#: with ``--checkpoint-dir`` set). Trimmed, not invented: the keys are what the
#: emitters always send, which is the contract the reader enforces, and the
#: values are the shape a faithful run really records (``polish_rounds`` 0 is
#: what the paper-faithful path records — the polish is off).
_REAL_PAYLOADS: dict[str, dict] = {
    "phase1_statistics": {
        "linears": ["layers.0.inner.self_attn.q_proj"],
        "num_linears": 22,
        "chunk_rows": 1,
    },
    "capture": {
        "capture_index": 1,
        "teacher_fingerprint": "b40cefc23c8eeba005449169a1e1a11f6e8587b7352819db1e6b116a57f5edc4",
        "teacher_source": "frozen_reference_copy",
        "x_b_stream": "compressed",
        "y_b_stream": "fp",
    },
    "tune_fp": {
        "capture_index": 1,
        "marker": "tune_fp_loss: weighted_mse_nq_3012 (weight source: informed_assumption)",
        "steps": 16,
        "tunable": ["inner.mlp.down_proj", "inner.mlp.gate_proj"],
    },
    "lb_admm": {"polish_rounds": 0},
    "ste": {
        "capture_index": 1,
        "per_linear": {"inner.self_attn.q_proj": {"declared": "paper_ste"}},
        "total_steps": 96,
    },
    "sign_extract_pack": {
        "extracted": 7,
        "factor_mode": "binary",
        "layers": ["inner.self_attn.q_proj"],
    },
    "tune_scales_kd": {"artifact": True, "factor_mode": "binary"},
}

#: The two events the recorder emits that are NOT rows of the stage inventory:
#: a reader must still be able to read a journal carrying one, and neither is
#: counted by any condition. Payloads as the emitters send them.
_DEVIATION_PAYLOADS: dict[str, dict] = {
    "tune_loss_uniform_fallback": {
        "block": "layers.0",
        "missing_canonical_layer": "inner.mlp.down_proj",
        "requested_mode": "terminal_grad",
        "reason": "absent_canonical_layer",
    },
    "tune_fp_vacuous_teacher": {
        "block": "layers.0",
        "gradient_norm": 0.0,
        "initial_residual_norm": 0.0,
        "reason": "zero_first_step_gradient",
        "capture_index": 1,
        "teacher_source": "frozen_reference_copy",
        "teacher_fingerprint": "b40cefc23c8eeba005449169a1e1a11f6e8587b7352819db1e6b116a57f5edc4",
    },
}


def _required_names() -> list[str]:
    return [key for key, _l, _d, _dev, required in R._STAGE_LINES if required]


def _name_only_records() -> list[dict]:
    """The seven required events as a NAME-ONLY record, and nothing else.

    The two fields the verdict itself reads are the only payload present — a
    capture's stream provenance and the scale stage's artifact flag — so what is
    left is a record that names every required stage and attests no work: no
    steps, no per-linear counts, no linears, no extracted layer list.
    """
    records = []
    for key in _required_names():
        record: dict = {"event": key}
        if key == "capture":
            record["y_b_stream"] = "fp"
        if key == "tune_scales_kd":
            record["artifact"] = True
        records.append(record)
    return records


# --- the table and the inventory are the same vocabulary ----------------


def test_every_inventory_stage_has_a_journal_payload_shape():
    """The journal's shape table covers the verdict's inventory exactly.

    Both directions, because either half of a one-way containment is a hole:
    a stage the verdict counts but the reader has no shape for is a stage whose
    re-read record cannot be validated, and a shape for a name the recorder
    cannot emit is a name the reader would accept on faith.
    """
    inventory = [key for key, *_rest in R._STAGE_LINES]
    assert set(inventory) <= set(R._STAGE_PAYLOAD_KEYS), "an inventory stage has no shape"
    assert set(_required_names()) <= set(R._STAGE_PAYLOAD_KEYS), "a REQUIRED stage has no shape"
    off_inventory = set(R._STAGE_PAYLOAD_KEYS) - set(inventory)
    assert off_inventory == set(_DEVIATION_PAYLOADS), (
        f"off-inventory journal names are {sorted(off_inventory)}, expected exactly the two "
        "deviation events the recorder emits"
    )
    # Every required stage's shape is a non-empty tuple of key names: an empty
    # shape would validate a name-only line, which is the whole hazard.
    for key in _required_names():
        assert R._STAGE_PAYLOAD_KEYS[key], f"required stage {key!r} has an empty payload shape"


def test_the_verdict_alone_blesses_a_name_only_record():
    """The MOTIVATION for the gate, pinned where it is visible.

    A record that names all seven required stages and attests no work reaches a
    green label — the payload-blind property sec-eng named on the landed
    verdict. It is unreachable in production only because no record was ever
    re-read from disk; it becomes reachable the moment a journal is read without
    a shape check, which is why ``stage_journal_record`` exists. If this test
    ever goes green in the other direction (the label falls here), the reader's
    gate has been replaced by a stronger property — re-derive this test, do not
    delete it.
    """
    names_only = _name_only_records()
    label, conditions = R.fidelity_verdict("admm", [], names_only, requested=True)
    assert label is True
    assert all(conditions.values()), conditions
    # The JOURNAL form of those same names is `(event, {})` — nothing to attest
    # with — and the gate is what keeps it from ever reaching the verdict that
    # just blessed the flat form.
    for record in names_only:
        with pytest.raises(ValueError, match="missing the payload keys"):
            R.stage_journal_record({"event": record["event"], "info": {}})


# --- the gate: real payloads pass, corrupted ones raise -----------------


@pytest.mark.parametrize("event", sorted(_REAL_PAYLOADS))
def test_a_real_payload_validates_and_round_trips(event):
    line = {"event": event, "info": dict(_REAL_PAYLOADS[event])}
    assert R.stage_journal_record(line) == (event, _REAL_PAYLOADS[event])


@pytest.mark.parametrize("event", sorted(_DEVIATION_PAYLOADS))
def test_the_off_inventory_deviation_events_are_readable(event):
    """They are not counted, but a journal carrying one must still be readable."""
    line = {"event": event, "info": dict(_DEVIATION_PAYLOADS[event])}
    assert R.stage_journal_record(line) == (event, _DEVIATION_PAYLOADS[event])
    assert event not in [key for key, *_rest in R._STAGE_LINES]


def test_a_real_record_from_every_required_stage_validates_as_one_run():
    """The whole inventory at once: what a resumed run's journal must contain."""
    records = [{"event": key, "info": dict(_REAL_PAYLOADS[key])} for key in _REAL_PAYLOADS]
    parsed = [R.stage_journal_record(record) for record in records]
    label, conditions = R.fidelity_verdict("admm", [], parsed, requested=True)
    assert label is True
    assert all(conditions.values()), conditions


def _corrupted_lines() -> list[tuple[str, object, str]]:
    """``(reason, line, the error text the validator must raise with)``.

    The corruption class is DECISIVE, not a subtle tweak: each entry removes the
    payload that carries the evidence, or the event name itself, or saturates a
    measurement to ``inf``/``NaN`` — the MONOTONE-LEVERAGE class the repo skill
    names (``src/quantibbler/ptq/e2e.py``: "An exponent flip is monotone
    leverage: the entry is multiplied by 2^16 or saturates to inf/NaN ... sign
    flips and small-entry flips are sub-noise"). A name-only line removes the
    evidence entirely; a saturated value dominates every comparison and would
    otherwise reach the verdict as an execution. (Cf.
    ``test_scn_e2e_gate_detects_corruption``: a sign-bit flip only tripped the
    artifact gate by luck because the fixture is noise-dominated, so that test
    uses the high-exponent class instead.)
    """
    return [
        ("name-only line", {"event": "lb_admm", "info": {}}, "missing the payload keys"),
        ("payload missing entirely", {"event": "ste"}, "carries no payload object"),
        (
            "payload gutted to one key",
            {"event": "capture", "info": {"y_b_stream": "fp"}},
            "missing the payload keys",
        ),
        (
            "payload is not an object",
            {"event": "lb_admm", "info": [0]},
            "carries no payload object",
        ),
        ("event is not a string", {"event": 12, "info": {}}, "must name its event as a string"),
        (
            "unknown event name",
            {"event": "lb_adm", "info": {"polish_rounds": 0}},
            "not one the stage recorder can emit",
        ),
        ("line is not an object", ["lb_admm", {}], "must be a JSON object"),
        # MONOTONE-LEVERAGE: the measurement saturates rather than loses its key.
        (
            "payload saturated to +inf",
            {"event": "lb_admm", "info": {"polish_rounds": float("inf")}},
            "non-finite number at info.polish_rounds",
        ),
        (
            "payload saturated to -inf",
            {"event": "lb_admm", "info": {"polish_rounds": float("-inf")}},
            "non-finite number at info.polish_rounds",
        ),
        (
            "payload saturated to NaN",
            {"event": "lb_admm", "info": {"polish_rounds": float("nan")}},
            "non-finite number at info.polish_rounds",
        ),
        (
            "nested saturated measurement",
            {
                "event": "tune_fp_vacuous_teacher",
                "info": {
                    "block": "layers.0",
                    "gradient_norm": float("inf"),
                    "initial_residual_norm": 0.0,
                    "reason": "zero_first_step_gradient",
                },
            },
            "non-finite number at info.gradient_norm",
        ),
    ]


@pytest.mark.parametrize("reason,line,message", _corrupted_lines(), ids=lambda v: str(v)[:38])
def test_a_corrupted_line_is_refused(reason, line, message):
    with pytest.raises(ValueError, match=message):
        R.stage_journal_record(line)


def test_the_shape_check_is_not_a_signature():
    """The HONEST limit, pinned so nobody reads more into the gate.

    A line that copies the keys of a real payload validates whatever its values
    are. What keeps a journal honest is the fingerprint of the directory it sits
    in and the operator's control of that directory — not this table. Stated as
    a test so the boundary is a fact rather than an impression.
    """
    forged = {"event": "lb_admm", "info": {"polish_rounds": 0, "forged": "yes"}}
    assert R.stage_journal_record(forged) == ("lb_admm", forged["info"])
