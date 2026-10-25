"""Versioned pydantic models for the host-side artifact surfaces.

Change: ``qk-serialization`` (foundation card ``t_0a8f5936``, @user request
2026-10-25). These models are the EXECUTABLE FORM of contracts that already
exist in the spec store and in this package — they are not a new format, and
this module writes nothing: the producers keep emitting the bytes they emit
today until the adoption cards land.

One model per persisted surface, with its owning contract:

* :class:`QuantizeReport` — ``openspec/specs/qk-ptq/spec.md``, requirement
  "quantize_report.json SHALL be provenance-complete and strict-JSON-parseable";
  writer ``quantibbler.qk_report.dump_report``.
* :class:`AcceptanceLedgerEntry` — ``qk-nq-recreation-bar``'s
  ``ledger-format.md`` (archived), scenario ``bar_recorded_not_gated``
  (``openspec/specs/qk-ptq/spec.md``); file ``docs/acceptance-ledger.jsonl``.
* :class:`SolverProbeRow` — ``openspec/specs/qk-ptq/spec.md`` ("every probe row
  SHALL carry row-time ...", "every solver-probe row record backend and
  dtype"), plus the ``qk-nanoquant-alignment-lane3`` regime-field delta; writer
  ``quantibbler.ptq.block_recon._record_solver_probe`` (and the sibling sweep
  writer ``tools/lane3_reg_sweep.py``).
* :class:`StageJournalEntry` — ``qk-fix-tunefp-teacher``'s stage-journal
  requirement; writer ``quantibbler.ptq.checkpoints.append_stage_event``, shape
  table ``qk_report._STAGE_PAYLOAD_KEYS``.
* :class:`LaunchAttributionRecord` — ``qk-launch-attribution``'s spec deltas
  (the launch crumb line, and the owner-or-refuse requirement); writer
  ``tools/quantize_hf_model.py`` ``main``'s first ``_bread`` call.

Torch-free on purpose: like ``qk_report`` and ``qk_bar``, this module must be
importable and testable in a lean environment — the driver imports torch at
module top, so anything kept there silently skips instead of running.

Version policy
--------------

Every model carries ``schema_version`` (:data:`SCHEMA_VERSION`). An artifact
naming a version this build does not read REFUSES loudly with
:class:`ArtifactSchemaError`, naming the model and the version; it is never
parsed as if it were the current shape. An artifact carrying NO
``schema_version`` reads as the current one, so the corpora already on disk
keep loading — "old artifacts either load or refuse loudly", never silently
misparse.

``extra`` is chosen per surface by whether the writer emits a closed set:
``forbid`` where the owning contract enumerates the fields (the ledger row, the
stage-journal line), ``allow`` where a writer emits keys beyond the modelled
contract today (the report, the probe row) — dropping those would be exactly
the silent misparse this module exists to prevent.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .qk_bar import BAR_ROWS
from .qk_report import DEGRADED, FAILED, OK, stage_journal_record

__all__ = [
    "SCHEMA_VERSION",
    "VERDICT_BAND",
    "LEDGER_ROW_LABELS",
    "ArtifactSchemaError",
    "ArtifactModel",
    "RegimeRecord",
    "QuantizeReport",
    "EvalProtocol",
    "Measured",
    "OperationalGateVerdict",
    "Acceptance",
    "ProtocolDeviation",
    "AcceptanceLedgerEntry",
    "SolverProbeRow",
    "StageJournalEntry",
    "LaunchAttributionRecord",
]

#: The single schema version every persisted model declares. Bump it when a
#: model's shape changes in a way an older reader could misparse.
SCHEMA_VERSION: int = 1

#: The three-value band, copied from ``qk_report`` so there is one vocabulary:
#: ``tests/test_serialization.py`` pins this tuple equal to the module's
#: ``OK``/``DEGRADED``/``FAILED`` constants.
VERDICT_BAND: tuple[str, str, str] = (OK, DEGRADED, FAILED)

#: The report's band, and the COPY the ledger records. Written as a literal so
#: it is the model's field type; the equality to ``qk_report``'s constants is
#: pinned by a test rather than asserted here.
_VerdictBand = Literal["ok", "degraded", "failed"]

#: The acceptance-ledger row vocabulary, DERIVED from the bar's single source of
#: truth — never a second hand-kept list (``qk_bar``'s docstring: the format
#: "names the labels but never re-lists the values").
LEDGER_ROW_LABELS: tuple[str, ...] = tuple(row[0] for row in BAR_ROWS)


class ArtifactSchemaError(RuntimeError):
    """An artifact names a ``schema_version`` this build does not read.

    Deliberately NOT a ``ValueError``: pydantic converts those from a validator
    into a generic ``ValidationError``, and the point of this refusal is that a
    reader can catch the shape-version failure by its own name.
    """


class ArtifactModel(BaseModel):
    """Base for every persisted host-side artifact model.

    Carries ``schema_version`` and the loud refusal for an unknown one. The
    validator runs ``mode="before"`` so a version that is not an exact integer
    (``"1"``, ``1.0``, ``True``) is refused rather than coerced into a match.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION

    @field_validator("schema_version", mode="before")
    @classmethod
    def _refuse_unknown_version(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value != SCHEMA_VERSION:
            raise ArtifactSchemaError(
                f"{cls.__name__}: schema_version {value!r} is not the version this build "
                f"reads ({SCHEMA_VERSION}); refusing to parse the artifact as the current shape"
            )
        return value


class RegimeRecord(BaseModel):
    """The report's ``regime`` block.

    Spec: ``openspec/specs/qk-ptq/spec.md``, requirement "quantize_report.json
    SHALL be provenance-complete and strict-JSON-parseable" — ``regime.artifact``
    is emitted from the run's actual scale dtype, never a hardcoded literal.

    Only ``artifact`` is declared: the block's other members (``baseline_eval``,
    ``solver``, ``kd_and_ste``) are prose the requirement does not name, so they
    ride as extras and the block roundtrips byte-for-byte instead of growing
    ``null`` members the writer never emitted.
    """

    model_config = ConfigDict(extra="allow")

    artifact: str


class QuantizeReport(ArtifactModel):
    """``quantize_report.json`` — the per-run measurement of record.

    Spec: ``openspec/specs/qk-ptq/spec.md``, requirement "quantize_report.json
    SHALL be provenance-complete and strict-JSON-parseable" (``solver`` and
    ``scale_dtype`` from the run's arguments, ``status`` from the verdict);
    writer ``quantibbler.qk_report.dump_report``.

    ``extra="allow"``: the writer emits keys beyond this contract — including
    the documented crash report, which carries only ``status`` and
    ``report_error`` — so unmodelled keys ride through instead of being lost.
    """

    model_config = ConfigDict(extra="allow")

    status: _VerdictBand
    model: str | None = None
    rank: int | None = None
    solver: str | None = None
    scale_dtype: str | None = None
    seq_len: int | None = None
    tune_seq_len: int | None = None
    factor_mode: str | None = None
    ablation: str | None = None
    layer_scope: str | None = None
    ppl_baseline_bf16: float | None = None
    ppl_quant_native: float | None = None
    ppl_ratio: float | None = None
    effective_bpw: float | None = None
    qpk1_layer_bytes: int | None = None
    artifact_exported: bool | None = None
    paper_faithful: bool | None = None
    wall_seconds: float | None = None
    report_error: str | None = None
    regime: RegimeRecord | None = None


class EvalProtocol(BaseModel):
    """The ledger row's ``eval_protocol`` object.

    Spec: ``qk-nq-recreation-bar`` ``ledger-format.md`` (archived),
    ``docs/acceptance-ledger.jsonl``. ``seq_len == 2048`` is the paper protocol
    ([NQ] Sec. 4.2); ``calibration_samples`` is 128 in [NQ].
    """

    model_config = ConfigDict(extra="forbid")

    corpus: str
    partition: str
    seq_len: int
    tune_seq_len: int
    calibration_samples: int


class Measured(BaseModel):
    """The ledger row's ``measured`` object — the numbers judged, plus copies.

    Spec: ``qk-nq-recreation-bar`` ``ledger-format.md``, ``measured``. The
    copies (``row_ppl``, ``row_bits``, ``bar_ceiling``) are what the row was
    judged against, recorded so the judgment travels with the number.
    """

    model_config = ConfigDict(extra="forbid")

    ppl: float
    ratio: float
    ratio_vs_baseline: float
    effective_bpw: float
    row_ppl: float
    row_bits: float
    bar_ceiling: float


class OperationalGateVerdict(BaseModel):
    """The ledger row's COPY of the run verdict, never a recomputation.

    Spec: ``qk-nq-recreation-bar`` ``ledger-format.md``,
    ``operational_gate_verdict``; scenario ``bar_distinct_from_landing_gate``
    (``openspec/specs/qk-ptq/spec.md``) — the band stays the ONLY per-run
    verdict, and the ledger only records it.
    """

    model_config = ConfigDict(extra="forbid")

    band: _VerdictBand
    report_path: str


class Acceptance(BaseModel):
    """The ledger row's human gate: the acceptance decision.

    Spec: ``qk-nq-recreation-bar`` ``ledger-format.md``, ``acceptance`` and the
    "rules the format enforces" section.
    """

    model_config = ConfigDict(extra="forbid")

    decision: Literal["met", "not_met", "equivalence"]
    decided_by: str
    decided_at: str


class ProtocolDeviation(BaseModel):
    """The ledger row's optional ``protocol_deviation``.

    Spec: ``qk-nq-recreation-bar`` ``ledger-format.md`` — present iff the eval
    geometry deviates from [NQ]'s, and then the equivalence argument is
    mandatory, not assumed.
    """

    model_config = ConfigDict(extra="forbid")

    deviation: str
    equivalence_argument: str


class AcceptanceLedgerEntry(ArtifactModel):
    """One row of ``docs/acceptance-ledger.jsonl``.

    Spec: ``qk-nq-recreation-bar``'s ``ledger-format.md`` (archived), scenario
    ``bar_recorded_not_gated`` (``openspec/specs/qk-ptq/spec.md``): when an
    acceptance run completes, a ledger entry records model id, commit hash,
    targeted row, eval protocol, measured PPL, ratio, effective_bpw, and the
    operational-gate verdict — the bar being a documented human-gated acceptance
    criterion, not a per-run gate verdict.

    The cross-field rules the format states are enforced here: ``met`` requires
    ``bar_met``, and ``equivalence`` requires NOT ``bar_met`` plus a non-empty
    equivalence argument. ``targeted_row`` is validated against
    :data:`LEDGER_ROW_LABELS`, derived from ``qk_bar.BAR_ROWS``.
    """

    model_config = ConfigDict(extra="forbid")

    entry_id: str
    model_id: str
    commit: str
    targeted_row: str
    eval_protocol: EvalProtocol
    measured: Measured
    tolerance_provenance: str
    bar_met: bool
    operational_gate_verdict: OperationalGateVerdict
    recorded_at: str
    run_command: str
    report_path: str
    artifact_path: str
    acceptance: Acceptance
    protocol_deviation: ProtocolDeviation | None = None
    notes: str | None = None
    supersedes: str | None = None

    @field_validator("commit")
    @classmethod
    def _full_git_revision(cls, value: str) -> str:
        if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(
                "commit must be the full 40-hex `git rev-parse HEAD` of the tree that "
                "produced the artifact; a short or non-hex revision cannot be resolved"
            )
        return value

    @field_validator("targeted_row")
    @classmethod
    def _row_is_a_bar_row(cls, value: str) -> str:
        if value not in LEDGER_ROW_LABELS:
            raise ValueError(
                f"targeted_row {value!r} is not a qk_bar.BAR_ROWS label "
                f"({list(LEDGER_ROW_LABELS)}); the FP16 baseline is not a target and there "
                "is no row below the lowest budget"
            )
        return value

    @model_validator(mode="after")
    def _decision_agrees_with_bar_met(self) -> AcceptanceLedgerEntry:
        decision = self.acceptance.decision
        if decision == "met" and not self.bar_met:
            raise ValueError("decision 'met' requires bar_met true")
        if decision == "equivalence":
            if self.bar_met:
                raise ValueError(
                    "decision 'equivalence' requires bar_met false — a recorded protocol "
                    "deviation is never a flip of bar_met"
                )
            deviation = self.protocol_deviation
            argument = deviation.equivalence_argument if deviation else ""
            if not argument:
                raise ValueError(
                    "decision 'equivalence' requires a non-empty equivalence_argument in "
                    "protocol_deviation"
                )
        return self


class SolverProbeRow(ArtifactModel):
    """One line of a ``solver_probe.jsonl`` / ``lane3_reg_sweep.jsonl`` corpus.

    Spec: ``openspec/specs/qk-ptq/spec.md`` — "launch.env, the checkpoint
    fingerprint, and every solver-probe row record backend and dtype", and
    "every probe row SHALL carry row-time so the measured win replaces the
    estimate" — plus the ``qk-nanoquant-alignment-lane3`` regime-field delta
    (``rho_start``/``admm_reg`` value AND source). Writer:
    ``quantibbler.ptq.block_recon._record_solver_probe``.

    The required fields are the ones BOTH writers always emit; the rest are
    optional because the two writers disagree today (the ``t_2522c684`` finding:
    ``block_recon`` does not carry ``min_eigs`` while ``tools/lane3_reg_sweep.py``
    does). Resolving that divergence belongs to the adoption card, so this model
    refuses neither row — see ``design.md`` OQ-2. ``extra="allow"`` preserves any
    key a writer adds before the model catches up.
    """

    model_config = ConfigDict(extra="allow")

    # Both writers.
    name: str
    ts: float
    rho_end: float | None = None
    n_iter: int | None = None
    converged: bool | None = None
    cos_raw: float | None = None
    rel_raw: float | None = None
    w_fro: float | None = None
    # block_recon._record_solver_probe.
    solver: str | None = None
    compute_dtype: str | None = None
    image_digest: str | None = None
    device: str | None = None
    rho_end_source: str | None = None
    rho_start: float | None = None
    rho_start_source: str | None = None
    admm_reg: float | None = None
    admm_reg_source: str | None = None
    solver_backend: str | None = None
    solver_backend_source: str | None = None
    solver_device: str | None = None
    wq_raw_fro: float | None = None
    wq_pack_fro: float | None = None
    cos_pack: float | None = None
    rel_pack: float | None = None
    rank: int | None = None
    # tools/lane3_reg_sweep.py (the divergent writer — design.md OQ-2).
    reg: float | None = None
    reg_source: str | None = None
    normalization: str | None = None
    min_eigs: float | None = None
    residual_final: float | None = None
    residual_stall: float | None = None
    ridge_scale_mean: float | None = None
    provisional: bool | None = None


class StageJournalEntry(ArtifactModel):
    """One line of the stage journal (``stage-events.jsonl``).

    Spec: ``qk-fix-tunefp-teacher``'s stage-journal requirement — the journal is
    the only path by which a stage record can be re-read from disk and fed back
    into ``fidelity_conditions``, so every line must name an event the recorder
    can emit AND carry the payload that event always sends. The shape check
    itself is NOT re-implemented here: this model delegates to
    ``qk_report.stage_journal_record`` so there is one vocabulary
    (``_STAGE_PAYLOAD_KEYS``), not a second table to drift. Writer:
    ``quantibbler.ptq.checkpoints.append_stage_event``; reader:
    ``checkpoints.read_stage_log`` (whose fingerprint guard stays authoritative).
    """

    model_config = ConfigDict(extra="forbid")

    event: str
    info: dict[str, Any]

    @model_validator(mode="after")
    def _must_be_a_recordable_stage(self) -> StageJournalEntry:
        stage_journal_record({"event": self.event, "info": self.info})
        return self


class LaunchAttributionRecord(ArtifactModel):
    """The launch crumb's first line — the run's self-attribution.

    Spec: ``qk-launch-attribution``'s deltas — "The launch crumb line SHALL
    attribute the run to code, process, and output target" (pid, argv, git rev,
    resolved out dir) and "A driver run SHALL name its owner or refuse to start"
    (``owner=``, ``card=<card|unset>``). The epoch-first shape (:attr:`at` comes
    before the payload) is ``qk-progress-instrumentation``'s requirement
    ``progress_sink_routing``: every crumb line is ``{epoch} …`` so the
    watchdog's stall detector can read it. Writer: ``tools/quantize_hf_model.py``
    ``main``'s first ``_bread`` call.

    ``extra="allow"``: the launch.env identity written beside the crumb
    (``launch_id``, ``commit``, ``image_digest``, ...) rides through as extras.
    """

    model_config = ConfigDict(extra="allow")

    at: float
    pid: int
    argv: str
    rev: str
    out: str
    owner: str
    card: str
