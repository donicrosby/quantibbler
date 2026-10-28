# qk-representation delta: qk-artifact-blob

## MODIFIED Requirements

### Requirement: Non-goals of the v0.1 representation

A qpk1 stream SHALL be exactly one layer: the format SHALL NOT define a
model-level checkpoint, manifest, layer registry, or multi-layer
container. The multi-layer container is the separate QKPB blob defined
in this change; it wraps qpk1 streams. The format SHALL NOT define GPU
tensor or inference layouts in v0.1 (config Architecture: GPU compute
stays in PyTorch; Rust kernels are CPU-only). It SHALL NOT compress or
quantize the scale vectors beyond the tagged float16/float32 storage,
and SHALL NOT store continuous pre-sign latent factors, optimizer
state, or calibration statistics.

#### Scenario: SF-6.1 — Stream extent is strictly one layer

- **WHEN** a stream contains serialized data for more than one layer, or
  any bytes beyond the single layer's declared extent
- **THEN** deserialization rejects the stream (trailing-garbage rule),
  and the library provides no API that concatenates or registers multiple
  layers into one qpk1 artifact.

#### Scenario: SF-6.2 — No GPU layout is defined

- **WHEN** a consumer needs a GPU- or kernel-specific working layout
- **THEN** that layout is defined outside this capability; qpk1 defines
  only the serialized byte stream, and no qpk1 field encodes a GPU format
  or device placement.

## ADDED Requirements

### Requirement: The blob SHALL carry a distinct magic and format version

A qpk1 model blob (distinct from the per-layer qpk1 stream) SHALL begin
with the four-byte magic `QKPB` followed by a little-endian u32 format
version whose value is 1. A reader SHALL refuse any file whose first
four bytes differ from `QKPB` or whose declared format version is
outside the supported set.

#### Scenario: BF-1.1 — Recognized magic and version are accepted

- **WHEN** a blob begins with `QKPB` followed by little-endian u32 `1`
- **THEN** the parser proceeds to the header.

#### Scenario: BF-1.2 — Bad magic is refused

- **WHEN** a file's first four bytes are anything other than `QKPB`
  (e.g. `BLOB`, `QPK1`, all-zero, or truncated to fewer than four bytes)
- **THEN** the parser refuses with a diagnostic naming the observed bytes.

#### Scenario: BF-1.3 — Unsupported version is refused

- **WHEN** a file's magic matches but the u32 version is 0, 2, or any
  value outside {1}
- **THEN** the parser refuses with a diagnostic naming the version.

### Requirement: The blob header SHALL be length-prefixed JSON

After the magic and version, a u32 LE byte length SHALL prefix a
UTF-8 JSON header. Required fields SHALL be exactly: model_id,
vocab_size, d_model, n_layers, n_heads, n_kv_heads, head_dim,
eos_token_id, bos_token_id, rms_norm_eps, rope_theta, norm_slots,
layers, weight_tying, remainder_dtype. The header SHALL carry no
wall-clock timestamp. A reader SHALL refuse a truncated or non-JSON
header and a missing or wrong-typed required field, and SHALL ignore
unknown fields.

#### Scenario: BF-2.1 — A full-model header round-trips every field

- **WHEN** a blob is written with declared fields for model_id,
  vocab_size, d_model, n_layers, n_heads, n_kv_heads, head_dim,
  eos_token_id, bos_token_id, rms_norm_eps, rope_theta, norm_slots,
  layers, weight_tying, and remainder_dtype
- **THEN** a parser recovers every field at its written value.

#### Scenario: BF-2.2 — No wall-clock timestamp is present

- **WHEN** a blob header is serialized
- **THEN** it SHALL contain no timestamp field (wall-clock or Unix epoch);
  the header JSON carries only declared metadata and an optional
  `created_by` provenance string, which SHALL name the producing tool,
  never a timestamp.

#### Scenario: BF-2.3 — A truncated or non-JSON header refuses

- **WHEN** the declared header byte length extends past EOF, the bytes
  are not valid UTF-8, or the bytes do not parse as JSON
- **THEN** the parser refuses with a named diagnostic.

#### Scenario: BF-2.4 — A missing required field or wrong-typed field refuses

- **WHEN** the header JSON parses but any field of the closed required
  set is absent, or a required field's JSON type disagrees with the
  declared schema (e.g. `d_model` is a string)
- **THEN** the parser refuses with a diagnostic naming the field and
  the expected type.

### Requirement: The section table SHALL list every payload and SHALL be validated fail-closed

A u32 LE section count SHALL precede self-delimiting entries:
per entry, a u8 name length + UTF-8 name + u8 dtype tag + u64 LE
offset + u64 LE length. Names SHALL be declared layers, the remainder
names, or tokenizer.*; tags SHALL be {qpk1_v1=0, qpk1_v2=1,
fp32_le=2, bf16_le=3, raw=4}. Fail-closed: unknown name or tag,
duplicate name, checked offset+length past EOF, overlap,
non-4096-aligned offset, or table truncated by EOF each yields a
named refusal.

#### Scenario: BF-3.1 — Unknown dtype tag refuses

- **WHEN** any section entry's dtype tag is outside 0–4
- **THEN** the parser refuses with a diagnostic naming the tag value.

#### Scenario: BF-3.2 — Offset-plus-length past EOF or u64 overflow refuses

- **WHEN** any section's offset + length, computed in checked
  arithmetic, exceeds the file size, or the addition itself overflows
  u64 (e.g. offset 2^64−100, length 1000)
- **THEN** the parser refuses with a diagnostic naming the section.

#### Scenario: BF-3.3 — Overlapping sections refuse

- **WHEN** two sections' byte ranges [offset, offset+length) intersect
- **THEN** the parser refuses with a diagnostic naming both sections.

#### Scenario: BF-3.4 — Duplicate section names refuse

- **WHEN** two section entries share the same name
- **THEN** the parser refuses with a diagnostic naming the duplicate
  name.

#### Scenario: BF-3.5 — Non-4096-aligned offsets refuse

- **WHEN** any section's offset is not a multiple of 4096
- **THEN** the parser refuses with a diagnostic naming the section and
  its offset.

#### Scenario: BF-3.6 — A section table truncated by EOF refuses

- **WHEN** the declared section count's entries run past EOF (the file
  ends mid-entry)
- **THEN** the parser refuses with a diagnostic naming the section table
  and the truncated entry's byte position.

#### Scenario: BF-3.7 — An unknown section name refuses

- **WHEN** a section name is outside the closed vocabulary (not a
  declared layer, remainder name, or `tokenizer.*`)
- **THEN** the parser refuses with a diagnostic naming the name.

### Requirement: Section byte ranges SHALL occupy the payload region exactly

Every section's byte range [offset, offset+length) SHALL lie
entirely after the section table's final byte, and the greatest
section end SHALL equal the file size: the magic/version/header bytes
and the table's own bytes are not valid section targets, and no byte
may follow the last section. Either violation SHALL yield a named
refusal.

#### Scenario: BF-7.1 — A section aliasing the metadata region refuses

- **WHEN** a section entry declares a range intersecting the bytes from
  file start through the section table's end (e.g. offset 0, or an
  offset inside the table)
- **THEN** the parser refuses with a diagnostic naming the section and
  its offset.

#### Scenario: BF-7.2 — Trailing bytes after the last section refuse

- **WHEN** the file contains any byte beyond the greatest section's
  offset+length (trailing garbage)
- **THEN** the parser refuses with a diagnostic naming the byte offset
  where the extraneous bytes begin.

### Requirement: Payload sections SHALL be page-aligned

Every section payload SHALL begin at a byte offset that is a multiple
of 4096, the host's page size, so an mmap consumer can access any
section at native page granularity without copying. Padding bytes
between regions SHALL each be written as zero.

#### Scenario: BF-4.1 — All section payload offsets are 4096-aligned

- **WHEN** a valid blob is written
- **THEN** every `offset` field in the section table is divisible by 4096.

#### Scenario: BF-4.2 — Padding bytes are zero

- **WHEN** bytes between the header's last byte and the section table's
  start, between the section table's end and the first section offset,
  and between successive section payload extents, are examined
- **THEN** every such byte is `0x00`.

### Requirement: Section entries SHALL appear in a fixed canonical order

Section entries SHALL appear in order: every layer stream in
block-then-projection order (W_q, W_k, W_v, W_o, W_gate, W_up, W_down
per block, matching the header's `layers` array), then the remainder
sections (embedding, lm_head when untied, norms), then tokenizer raw
sections.

#### Scenario: BF-5.1 — The section table follows the canonical order

- **WHEN** a blob is written for a model with declared layers
  `["blocks.0.W_q", "blocks.0.W_k", ..., "blocks.0.W_down"]`, remainder
  items `["embedding", "norms"]`, and a tokenizer
- **THEN** the section table lists entries in exactly that order.

### Requirement: The same model packed twice SHALL produce identical bytes

The same model packed twice with the same inputs SHALL produce
byte-identical blob files. No field SHALL depend on wall-clock time
(BF-2.2), filesystem walk order, or host identity. This SHALL be
pinned by packing a model twice and asserting the two blobs' sha256
digests are equal.

#### Scenario: BF-6.1 — Pack-twice hash equality

- **WHEN** a model is packed to a blob, then packed again from the same
  inputs
- **THEN** `sha256sum` of the two blobs produces identical digests.
