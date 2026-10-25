"""qk-serialization: the host-side artifact models roundtrip and refuse loudly.

Spec: ``openspec/changes/qk-serialization/specs/qk-serialization/spec.md`` (the
requirement of record for the models' behaviour) and, per model, the owning
contract cited in ``quantibbler/serialization.py``'s docstrings.

Scenarios covered (IDs verbatim from the delta):

- ``models_roundtrip_dict_json_model_identity``
- ``models_preserve_unmodelled_keys``
- ``unknown_schema_version_refuses_loudly``
- ``absent_schema_version_reads_as_current``
- ``wrong_field_type_is_rejected``
- ``ledger_row_labels_track_the_bar``

Torch-free by construction: the models live outside the torch-importing driver,
so these run in the lean tier (same reason ``tests/test_qk_report.py`` and
``tests/test_qk_bar.py`` do).
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from quantibbler import qk_bar, qk_report
from quantibbler import serialization as ser

# --------------------------------------------------------------------------
# Samples — one well-formed payload per surface, taken from the artifact each
# model describes rather than invented (the ledger sample is the
# ``ledger-format.md`` illustration; the probe samples are the two real writer
# shapes; the launch sample is the crumb line's first line).
# --------------------------------------------------------------------------

REPORT = {
    "model": "Qwen/Qwen3-0.6B",
    "rank": 8,
    "solver": "admm",
    "scale_dtype": "float64",
    "seq_len": 2048,
    "tune_seq_len": 512,
    "factor_mode": "binary",
    "ablation": "none",
    "layer_scope": "all",
    "ppl_baseline_bf16": 12.5,
    "ppl_quant_native": 30.1,
    "ppl_ratio": 2.408,
    "effective_bpw": 0.79,
    "qpk1_layer_bytes": 123456,
    "artifact_exported": True,
    "paper_faithful": True,
    "wall_seconds": 12.3,
    "status": "ok",
    "regime": {"artifact": "qpk1: +-1 bitplanes + float64 scales", "solver": "float64"},
}

LEDGER = {
    "entry_id": "qwen3-0p6b-a1b2c3d-nanoquant_0.80_bits-20261102",
    "model_id": "Qwen/Qwen3-0.6B",
    "commit": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
    "targeted_row": "nanoquant_0.80_bits",
    "eval_protocol": {
        "corpus": "wikitext-2-raw-v1",
        "partition": "test",
        "seq_len": 2048,
        "tune_seq_len": 512,
        "calibration_samples": 128,
    },
    "measured": {
        "ppl": 34.12,
        "ratio": 1.0098,
        "ratio_vs_baseline": 2.6954,
        "effective_bpw": 0.79,
        "row_ppl": 33.79,
        "row_bits": 0.80,
        "bar_ceiling": 38.8585,
    },
    "tolerance_provenance": "provisional 1.15 (qk-nq-recreation-bar spec pin)",
    "bar_met": True,
    "operational_gate_verdict": {
        "band": "ok",
        "report_path": "artifacts/qwen3-0p6b-r08/quantize_report.json",
    },
    "recorded_at": "2026-11-02T18:40:11Z",
    "run_command": "uv run python tools/quantize_hf_model.py --model Qwen/Qwen3-0.6B",
    "report_path": "artifacts/qwen3-0p6b-r08/quantize_report.json",
    "artifact_path": "artifacts/qwen3-0p6b-r08/qpk1",
    "acceptance": {"decision": "met", "decided_by": "@user", "decided_at": "2026-11-02T19:02:00Z"},
    "notes": "format illustration only; not a measurement",
}

# quantibbler.ptq.block_recon._record_solver_probe's row.
PROBE_ROW = {
    "name": "model.layers.0.self_attn.q_proj",
    "solver": "admm",
    "compute_dtype": "float64",
    "image_digest": "sha256:deadbeef",
    "device": "cpu",
    "rho_end": 0.1,
    "rho_end_source": "env",
    "rho_start": 0.0,
    "rho_start_source": "config",
    "admm_reg": 0.03,
    "admm_reg_source": "config",
    "solver_backend": "torch",
    "solver_backend_source": "config",
    "solver_device": "cpu",
    "n_iter": 40,
    "converged": True,
    "w_fro": 11.2,
    "wq_raw_fro": 10.9,
    "wq_pack_fro": 11.0,
    "cos_raw": 0.84,
    "cos_pack": 0.45,
    "rel_raw": 0.54,
    "rel_pack": 0.89,
    "rank": 8,
    "ts": 1761412345.5,
}

# tools/lane3_reg_sweep.py's row: the divergent shape the model must also accept
# (design.md OQ-2 — resolving the divergence belongs to the adoption card).
SWEEP_ROW = {
    "name": "layers.0.mlp.down_proj",
    "reg": 0.03,
    "reg_source": "env",
    "rho_start": 0.0,
    "rho_end": 0.1,
    "normalization": "column",
    "n_iter": 40,
    "converged": True,
    "cos_raw": 0.7,
    "rel_raw": 0.71,
    "min_eigs": 0.001,
    "residual_final": 0.02,
    "residual_stall": 1.05,
    "ridge_scale_mean": 0.5,
    "w_fro": 9.9,
    "provisional": True,
    "ts": 1761412345.5,
}

JOURNAL = {"event": "lb_admm", "info": {"polish_rounds": 0}}

LAUNCH = {
    "at": 1761412345.0,
    "pid": 4242,
    "argv": "--device cpu --model Qwen/Qwen3-0.6B --out /out/run-a",
    "rev": "cfa3311de7066cd88a7cf9457f7d9ece66fb9e3d",
    "out": "/out/run-a",
    "owner": "t_0a8f5936",
    "card": "t_0a8f5936",
    "launch_id": "20261025T181500Z-4242",
    "commit": "cfa3311",
}

#: (model, well-formed payload, a payload whose declared type is wrong).
SURFACES = (
    (ser.QuantizeReport, REPORT, {**REPORT, "rank": "eight"}),
    (ser.AcceptanceLedgerEntry, LEDGER, {**LEDGER, "commit": "a1b2c3d"}),
    (ser.SolverProbeRow, PROBE_ROW, {**PROBE_ROW, "ts": "later"}),
    (ser.SolverProbeRow, SWEEP_ROW, {**SWEEP_ROW, "n_iter": "many"}),
    (ser.StageJournalEntry, JOURNAL, {"event": "not_a_stage", "info": {}}),
    (ser.LaunchAttributionRecord, LAUNCH, {**LAUNCH, "pid": "forty-two"}),
)


def _roundtrip(model_cls, payload):
    """dict -> model -> json -> model, and assert the two models are equal."""
    first = model_cls.model_validate(payload)
    blob = json.loads(json.dumps(first.model_dump(mode="json")))
    second = model_cls.model_validate(blob)
    assert second == first
    return first


# --------------------------------------------------------------------------
# Scenario: models_roundtrip_dict_json_model_identity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model_cls,payload,_bad", SURFACES)
def test_scn_models_roundtrip_dict_json_model_identity(model_cls, payload, _bad):
    first = _roundtrip(model_cls, payload)
    assert first.schema_version == ser.SCHEMA_VERSION


def test_scn_models_roundtrip_dict_json_model_identity_preserves_every_key():
    """The modelled keys all survive, not only the ones the test asserts on."""
    for model_cls, payload, _bad in SURFACES:
        dumped = model_cls.model_validate(payload).model_dump(mode="json")
        for key in payload:
            assert dumped[key] == payload[key], (model_cls.__name__, key)


# --------------------------------------------------------------------------
# Scenario: models_preserve_unmodelled_keys
# --------------------------------------------------------------------------


def test_scn_models_preserve_unmodelled_keys():
    report = _roundtrip(ser.QuantizeReport, {**REPORT, "probe": {"history": [1, 2, 3]}})
    assert report.model_dump(mode="json")["probe"] == {"history": [1, 2, 3]}

    row = _roundtrip(ser.SolverProbeRow, {**PROBE_ROW, "future_field": "kept"})
    assert row.model_dump(mode="json")["future_field"] == "kept"

    launch = _roundtrip(ser.LaunchAttributionRecord, {**LAUNCH, "image_digest": "sha256:x"})
    assert launch.model_dump(mode="json")["image_digest"] == "sha256:x"


# --------------------------------------------------------------------------
# Scenario: unknown_schema_version_refuses_loudly
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model_cls,payload,_bad", SURFACES)
def test_scn_unknown_schema_version_refuses_loudly(model_cls, payload, _bad):
    with pytest.raises(ser.ArtifactSchemaError) as excinfo:
        model_cls.model_validate({**payload, "schema_version": ser.SCHEMA_VERSION + 1})
    message = str(excinfo.value)
    assert model_cls.__name__ in message
    assert str(ser.SCHEMA_VERSION + 1) in message


@pytest.mark.parametrize("junk", ["1", 1.0, True, None])
def test_scn_unknown_schema_version_refuses_loudly_on_a_non_integer_version(junk):
    """A version that is not an exact integer must not coerce into a match."""
    with pytest.raises(ser.ArtifactSchemaError):
        ser.QuantizeReport.model_validate({**REPORT, "schema_version": junk})


# --------------------------------------------------------------------------
# Scenario: absent_schema_version_reads_as_current
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model_cls,payload,_bad", SURFACES)
def test_scn_absent_schema_version_reads_as_current(model_cls, payload, _bad):
    assert "schema_version" not in payload
    assert model_cls.model_validate(payload).schema_version == ser.SCHEMA_VERSION


# --------------------------------------------------------------------------
# Scenario: wrong_field_type_is_rejected
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model_cls,_payload,bad", SURFACES)
def test_scn_wrong_field_type_is_rejected(model_cls, _payload, bad):
    with pytest.raises(ValidationError):
        model_cls.model_validate(bad)


def test_scn_wrong_field_type_is_rejected_on_an_unknown_key():
    """The closed surfaces refuse an undeclared key rather than dropping it."""
    with pytest.raises(ValidationError):
        ser.AcceptanceLedgerEntry.model_validate({**LEDGER, "not_a_ledger_field": 1})
    with pytest.raises(ValidationError):
        ser.StageJournalEntry.model_validate({**JOURNAL, "extra": 1})


def test_scn_wrong_field_type_is_rejected_on_a_name_only_journal_line():
    """A journal line that satisfies no event's payload shape is refused."""
    with pytest.raises(ValidationError):
        ser.StageJournalEntry.model_validate({"event": "lb_admm", "info": {}})


# --------------------------------------------------------------------------
# Scenario: ledger_row_labels_track_the_bar
# --------------------------------------------------------------------------


def test_scn_ledger_row_labels_track_the_bar():
    assert tuple(row[0] for row in qk_bar.BAR_ROWS) == ser.LEDGER_ROW_LABELS
    assert ser.AcceptanceLedgerEntry.model_validate(LEDGER).targeted_row in ser.LEDGER_ROW_LABELS
    with pytest.raises(ValidationError):
        ser.AcceptanceLedgerEntry.model_validate({**LEDGER, "targeted_row": "fp16_baseline"})


def test_scn_ledger_row_labels_track_the_bar_on_the_cross_field_rules():
    """The format's own rules: met requires bar_met; equivalence requires the argument."""
    with pytest.raises(ValidationError):
        ser.AcceptanceLedgerEntry.model_validate({**LEDGER, "bar_met": False})
    equivalence = {
        **LEDGER,
        "bar_met": False,
        "acceptance": {"decision": "equivalence", "decided_by": "@user", "decided_at": "T"},
        "protocol_deviation": {"deviation": "seq_len 512", "equivalence_argument": "surrogate"},
    }
    assert ser.AcceptanceLedgerEntry.model_validate(equivalence).bar_met is False
    with pytest.raises(ValidationError):
        ser.AcceptanceLedgerEntry.model_validate(
            {
                **equivalence,
                "protocol_deviation": {"deviation": "seq_len 512", "equivalence_argument": ""},
            }
        )


# --------------------------------------------------------------------------
# Pins: one vocabulary, not two.
# --------------------------------------------------------------------------


def test_verdict_band_is_the_report_modules_band():
    assert ser.VERDICT_BAND == (qk_report.OK, qk_report.DEGRADED, qk_report.FAILED)


def test_schema_version_is_positive_integer():
    assert isinstance(ser.SCHEMA_VERSION, int) and ser.SCHEMA_VERSION >= 1
