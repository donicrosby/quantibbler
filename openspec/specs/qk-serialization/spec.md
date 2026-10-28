# qk-serialization Specification

## Purpose

Define the versioned pydantic host-side artifact models: the quantize report, acceptance-ledger rows, solver-probe JSONL rows, stage-journal entries, and launch-attribution records — each model's shape owned by the spec that introduced it, this capability only declares the executable form.

## Requirements

### Requirement: Host-side artifact models SHALL be the executable form of the already-specified surfaces

Quantibbler SHALL express the schemas of its persisted host-side artifacts as
versioned pydantic models in `quantibbler.serialization`, one model per surface:
the quantize report, acceptance-ledger rows, solver-probe JSONL rows,
stage-journal entries, and launch-attribution records. Each model docstring
SHALL cite the spec section or change doc that already owns its shape. The
models ARE that contract: this capability introduces no new format.

#### Scenario: models_roundtrip_dict_json_model_identity

- **WHEN** a well-formed dict for a surface is validated into its model, dumped
  to JSON, and parsed back into the model,
- **THEN** the parsed model equals the first, for every model in the module

#### Scenario: models_preserve_unmodelled_keys

- **WHEN** a report or a solver-probe row carries a key the model does not
  declare,
- **THEN** that key survives the dict → model → JSON → model roundtrip rather
  than being dropped

### Requirement: An artifact naming an unread schema_version SHALL refuse with a named error

Every persisted model SHALL carry a `schema_version` field. An artifact whose
`schema_version` is not the version this build reads SHALL raise the named
error `ArtifactSchemaError`, naming the model and the version, and SHALL NOT be
parsed as the current shape. An artifact carrying no `schema_version` SHALL
read as the current version, so corpora written before versioning keep loading.

#### Scenario: unknown_schema_version_refuses_loudly

- **WHEN** any model is validated from a payload naming a `schema_version` other
  than this build's, including a non-integer one,
- **THEN** validation raises `ArtifactSchemaError` naming the model and the
  offending version, and no object is constructed

#### Scenario: absent_schema_version_reads_as_current

- **WHEN** any model is validated from a payload that omits `schema_version`,
- **THEN** the constructed model reports this build's `SCHEMA_VERSION`

### Requirement: A mis-shaped artifact SHALL be rejected rather than coerced

A value of the wrong type in a declared field SHALL fail validation instead of
being coerced into a plausible one, so a corrupt or drifted artifact cannot
pass as a good one.

#### Scenario: wrong_field_type_is_rejected

- **WHEN** a payload gives a declared field a value of the wrong type,
- **THEN** validation raises `pydantic.ValidationError` for every model

### Requirement: Producers SHALL emit every surface through its model

The solver-probe writer and the stage-journal writer SHALL serialize through the
model that owns their surface; the driver's launch line SHALL be rendered FROM its
`LaunchAttributionRecord` — every identity field is the record's own value — with
its text and leading epoch byte-identical. The stage-journal reader SHALL validate
every line through `StageJournalEntry`: a line naming an unread `schema_version`
SHALL raise `ArtifactSchemaError` before any event is seeded into a resume.

#### Scenario: producers_emit_through_their_model

- **WHEN** a producer writes its surface
- **THEN** the bytes written are the model's own, and re-reading them through
  that model yields the values the producer was given, for the probe row, the
  journal line, and the launch record alike

#### Scenario: launch_line_is_rendered_from_its_record

- **WHEN** the driver starts and writes the trail's first line
- **THEN** every identity field on that line is the `LaunchAttributionRecord`'s
  own value, and the line's leading epoch is the record's `at`

#### Scenario: journal_reader_refuses_an_unread_version

- **WHEN** the journal is re-read for a resume and a line names a
  `schema_version` this build does not read
- **THEN** the read raises `ArtifactSchemaError` naming the model and the
  version, and no event is seeded into the resume

### Requirement: Every probe row SHALL carry the min_eigs floor

Both solver-probe writers — `quantibbler.ptq._probe._record_solver_probe` and
`tools/lane3_reg_sweep.py` — SHALL emit the `min_eigs` floor, and
`SolverProbeRow` SHALL require it, so no row can silently omit the quantity
`specs/qk-ptq` names for every probed row. An unmeasured floor SHALL read as
`nan`, never `0.0`, which would claim a singular system.

#### Scenario: every_probe_row_carries_the_min_eigs_floor

- **WHEN** either probe writer appends a row whose solve left no measured floor
- **THEN** the row carries `min_eigs` (`nan` rather than `0.0`), and a row that
  omits the field fails `SolverProbeRow` validation

### Requirement: The ledger row model SHALL take its row vocabulary from qk_bar

The acceptance-ledger row model SHALL derive its accepted `targeted_row`
labels from `qk_bar.BAR_ROWS` at validation time. It SHALL NOT hold a second,
hand-maintained list of labels, and a label outside that vocabulary SHALL be
rejected.

#### Scenario: ledger_row_labels_track_the_bar

- **WHEN** a ledger row names a `targeted_row` that is one of `qk_bar.BAR_ROWS`
  labels,
- **THEN** it validates, and a label outside that set is rejected

### Requirement: The report writer SHALL validate every emitted report against the report model

`quantibbler.qk_report.dump_report` SHALL validate the report against
`QuantizeReport` before serializing it, SHALL emit the writer's own keys, values
and order plus `schema_version`, and SHALL write the minimal failed report with
`faithful` false when validation or serialization fails — never raising, never
dropping a key the report contract requires as an explicit `null`.

#### Scenario: report_payload_validates_against_the_model

- **WHEN** a report dict is serialized through `dump_report`,
- **THEN** the emitted JSON parses, validates against `QuantizeReport`, and
  carries this build's `schema_version`

#### Scenario: report_payload_keeps_the_writers_keys_and_order

- **WHEN** a report carrying the writer's key set is serialized,
- **THEN** every existing key keeps its value, shape and relative order, the
  unmeasured end-of-run numerics stay explicit `null`s, and `schema_version` is
  the only added key

#### Scenario: report_payload_model_rejection_falls_back_to_the_failed_stub

- **WHEN** validation or serialization of a report fails,
- **THEN** `dump_report` returns the minimal failed report with `faithful` false
  and never raises
