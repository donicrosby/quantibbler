# qk-serialization — Spec Delta

## ADDED Requirements

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

### Requirement: The ledger row model SHALL take its row vocabulary from qk_bar

The acceptance-ledger row model SHALL derive its accepted `targeted_row`
labels from `qk_bar.BAR_ROWS` at validation time. It SHALL NOT hold a second,
hand-maintained list of labels, and a label outside that vocabulary SHALL be
rejected.

#### Scenario: ledger_row_labels_track_the_bar

- **WHEN** a ledger row names a `targeted_row` that is one of `qk_bar.BAR_ROWS`
  labels,
- **THEN** it validates, and a label outside that set is rejected
