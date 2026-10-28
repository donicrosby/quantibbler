# qk-ptq delta: qk-artifact-blob

## ADDED Requirements

### Requirement: The exporter SHALL write a single-file blob

The real-model driver SHALL expose `--container-format {blob,dir}`
and SHALL emit a `.qpk` blob when `blob` is named (default) and a
directory when `dir` is named. The blob path SHALL render the header
JSON (B2), section table (B3), and aligned payloads (B4) per the blob
specification, with the remainder at the declared `--remainder-dtype`.
The report SHALL record the emitted blob's sha256 (B6 external
fingerprint). The directory path SHALL preserve the pre-change export
behaviour.

#### Scenario: BW-1.1 — A blob export loads in the Rust loader

- **WHEN** the driver runs with the default `--container-format blob` on
  a tiny model
- **THEN** the output `.qpk` file passes every loader validation gate
  (qk-server BL-1.1 and BL-2.1).

#### Scenario: BW-1.2 — Directory export is still accepted

- **WHEN** the driver runs with `--container-format dir`
- **THEN** the output directory loads in the legacy loader path
  (qk-server BL-1.2) and its content is byte-identical to a pre-change
  directory export.

#### Scenario: BW-1.3 — The report records the emitted blob's sha256

- **WHEN** the driver exports a blob
- **THEN** `quantize_report.json` carries the blob's sha256 — the hash
  is recorded outside the container, never as a header field (B6).

### Requirement: The same model exported twice SHALL produce identical blobs

The same model packed twice to blob SHALL produce byte-identical files.
The writer SHALL use the fixed canonical section order, SHALL zero all
padding bytes, and SHALL emit no wall-clock timestamp in the header.
This SHALL be pinned by packing a model twice and asserting
`sha256(blob1) == sha256(blob2)`.

#### Scenario: BW-2.1 — Pack-twice sha256 is equal

- **WHEN** a tiny model is exported to blob, then exported again from
  the same state
- **THEN** `sha256(file1) == sha256(file2)`.

### Requirement: The repack utility SHALL convert a directory to a blob

`tools/repack_remainder.py` SHALL gain a dir→blob leg: given a
directory artifact, it SHALL produce a single `.qpk` file carrying every
layer stream verbatim (SF-2.1 byte-identical round-trip bytes), the
remainder at the declared or converted dtype, the tokenizer files as
raw sections, and a complete header + section table. Non-remainder
content SHALL be byte-identical to the source directory.

#### Scenario: BW-3.1 — A dir→blob repack loads

- **WHEN** the committed server-model directory fixture is repacked to
  blob with `--container-format blob`
- **THEN** the output `.qpk` loads in the Rust loader and produces a
  `ModelDirectory` identical field-for-field to the one loaded from the
  source directory (qk-server BL-1.1 + BL-2.1).

#### Scenario: BW-3.2 — Layer streams are byte-identical

- **WHEN** a directory is repacked to blob
- **THEN** the bytes of every layer stream section in the blob equal the
  bytes of the corresponding `.qpk1` file in the source directory —
  verified by extracting each section and comparing sha256.

### Requirement: The repack utility SHALL convert a blob's remainder dtype

The repack SHALL accept a `.qpk` blob as input and produce a new `.qpk`
blob with the remainder converted to a different declared dtype (fp32
→ bf16, through the shared RNE cast helper from the parent change). The
header's `remainder_dtype` and `weight_tying` suffix SHALL be rewritten
to agree. Section offsets and lengths SHALL be recomputed. Layer
streams and tokenizer sections SHALL be carried over byte-identically.

#### Scenario: BW-4.1 — A blob→blob fp32→bf16 conversion loads

- **WHEN** the committed blob fixture (fp32 remainder) is repacked with
  `--remainder-dtype bf16`
- **THEN** the output blob loads (qk-server BL-1.1), its
  `remainder_dtype` reads `"bf16"`, and the loaded remainder values
  satisfy the parent change's RDP-2.1 half-ulp bound.

#### Scenario: BW-4.2 — Layer streams survive blob→blob unchanged

- **WHEN** a blob is repacked to a new blob with a different remainder
  dtype
- **THEN** the byte ranges of every layer stream section in the output
  blob are identical to the input blob's corresponding ranges (except
  at different offsets), and the tokenizer raw sections are also
  byte-identical.
