# qk-representation Specification

## Purpose

Define the qpk1 binary stream and model-directory contracts that safely carry factorized linear layers from the Python compressor to pure-Rust consumers.

## Requirements

### Requirement: The qpk1 container serializes one factorized linear layer

A qpk1 stream SHALL represent exactly one factorized linear layer `W ~= diag(s1) @ (Ub @ Vb^T) @ diag(s2)` together with its per-linear bias, with binary `Ub`, `Vb` and full-precision scales `s1`, `s2` ([NQ] Sec 3.1 Eq. 1). All fields SHALL be little-endian. Version 2 carries the bias field and is the only version a serializer emits. The canonical layout is specified in the scenarios below.

#### Scenario: SF-1.1 — Serialized size follows the documented formula

- **WHEN** any layer is serialized with scale itemsize `w` bytes
- **THEN** the stream length is exactly
  `24 + (2*dout + din) * w + ceil(dout*r/8) + ceil(din*r/8)` bytes, and a
  version-1 stream — the same layer without the bias vector — is exactly
  `dout * w` bytes shorter.

#### Scenario: SF-1.2 — Dtype tag selects scale precision

- **WHEN** a stream carrying dtype tag 1 or 2 is deserialized
- **THEN** the scale vectors and the bias vector are read as float16 (tag 1)
  or float32 (tag 2), and any other tag value is rejected.

#### Scenario: SF-1.3 — Known layer serializes to the exact expected bytes

- **WHEN** a layer with `dout = 3`, `din = 5`, `r = 7`, float16 scales, known
  factor and scale values, and the known bias `[1.0, -0.5, 2.0]` is serialized
- **THEN** the stream is `24 + (2*3 + 5)*2 + ceil(3*7/8) + ceil(5*7/8) = 54`
  bytes long, begins with the bytes `QPK1` followed by the u32 little-endian
  values `2`, `1`, `3`, `5`, `7`, and matches the documented byte layout
  field-for-field at the documented offsets.

#### Scenario: SF-1.4 — Malformed headers are rejected

- **WHEN** a byte stream with wrong magic, a version other than 1 or 2, an
  unknown dtype tag, `r = 0`, `r > min(dout, din)`, a body short by the bias
  vector's bytes, truncated length, or trailing bytes is deserialized
- **THEN** deserialization fails with the documented error type and no
  partial object is returned.

### Requirement: Lossless round-trip

Serialization and deserialization SHALL be mutually inverse at the byte
level: `serialize(deserialize(b)) == b` for every valid qpk1 stream `b`, and
`deserialize(serialize(Ub, Vb, s1, s2, bias))` SHALL recover the packed
bitplanes bit-exactly, the scale vectors exactly as stored under the stream's
dtype, and the bias vector exactly as stored.

#### Scenario: SF-2.1 — Byte-identical round-trip

- **WHEN** any valid qpk1 stream is deserialized and re-serialized
- **THEN** the resulting byte string equals the original byte string
  exactly, for both float16 and float32 scale dtypes, bias included.

#### Scenario: SF-2.2 — Factors and scales are recovered exactly

- **WHEN** binary factors `Ub`, `Vb`, scale vectors `s1`, `s2` and a bias
  vector are serialized and then deserialized
- **THEN** the recovered factors compare bit-exactly to `Ub`, `Vb`, and the
  recovered scales and bias equal the stored values exactly under the
  stream's dtype (no float32 stream loses precision; a float16 stream returns
  exactly the float16-representable values that were stored).

### Requirement: Canonical bit packing of binary factors

Binary factors SHALL be packed row-major at one bit per entry: element
`(i, j)` occupies stream bit `i*r + j` of its matrix's bitplane; bit `k`
SHALL be stored in byte `k // 8` at bit position `k % 8` counted from the
least-significant bit (little-endian bit order within bytes). The mapping
SHALL be `+1 -> bit 1` and `-1 -> bit 0`, per [NQ] Sec 3.1 packing stage
(c) ("-1 -> 0, +1 -> 1"). Unused high bits of the final byte of each
bitplane SHALL be written as zero by serializers.

#### Scenario: SF-3.1 — Row-major little-endian layout is deterministic

- **WHEN** a factor matrix with known signs at a non-byte-aligned shape
  (e.g. `dout = 3`, `r = 5`) is packed
- **THEN** each element's bit lands at byte `(i*r + j) // 8`, bit
  `(i*r + j) % 8` from the LSB, and the bitplane matches a hand-computed
  reference byte string.

#### Scenario: SF-3.2 — Sign mapping follows +1 to bit 1

- **WHEN** a factor row of all `+1` and a factor row of all `-1` are
  packed at `r` a multiple of 8
- **THEN** the all-`+1` row packs to bytes `0xFF` and the all-`-1` row
  packs to bytes `0x00`.

### Requirement: BPW accounting from serialized bytes

Effective bits per weight SHALL be computed only from the serialized byte
length of the qpk1 stream: `BPW = 8 * len(bytes) / (dout * din)`, counting
the header, scale vectors, bias vector, and padding — never from parameter
counts or analytical estimates (config Method invariants; [NQ] App. F
counting discipline; [LB] App. D).

#### Scenario: SF-4.1 — Measured BPW uses the full stream length

- **WHEN** a layer is serialized and its BPW is computed
- **THEN** the BPW equals `8 * len(serialized bytes) / (dout * din)`,
  including header, bias and padding bytes, and this value is the one
  reported by the library's BPW function.

#### Scenario: SF-4.2 — Measured BPW matches the analytical byte-size formula

- **WHEN** the analytical size
  `24 + (2*dout + din) * itemsize + ceil(dout*r/8) + ceil(din*r/8)` bytes is
  evaluated for the same `(dout, din, r, dtype)`
- **THEN** the measured BPW from the serialized bytes and the BPW recomputed
  from the analytical size agree exactly (same integer byte count).

#### Scenario: SF-4.3 — Sub-1-bit at target ranks

- **WHEN** a standard LLM linear-layer shape (e.g. `dout = 4096`,
  `din = 11008`) is serialized at a rank `r` chosen by the rank planning
  formula for a target `b < 1.0` with float16 scales
- **THEN** the measured BPW from serialized bytes is strictly less than 1.0
  (storage dominated by packed binary bits; [NQ] Sec 4.3, App. F.5).

### Requirement: Rank planning formula

The library SHALL provide a rank planning formula inverting the qpk1
byte-size model: given `dout`, `din`, scale itemsize `w` bytes, and target
BPW `b`, the real-valued rank is
`r(b) = (b * dout * din - 8 * (24 + (2*dout + din) * w)) / (dout + din)`,
rounded to an integer and re-checked against the measured BPW. It is a
planning aid only; the authoritative BPW remains the byte-derived value.

#### Scenario: SF-5.1 — Rank selection hits a target BPW

- **WHEN** `rank_for_target_bpw(dout, din, b, dtype)` is called with a
  feasible target `b` (positive numerator)
- **THEN** the returned integer rank, when serialized, measures a BPW within
  one rank-step's rounding of `b`, and the re-checked derivation from the
  measured bytes is consistent.

#### Scenario: SF-5.2 — Formula reduces to the paper storage model

- **WHEN** header and rounding terms are set aside
- **THEN** the byte model reduces at `w = 2` to
  `r*(dout+din) + 16*(dout+din)` bits for the weight storage, matching
  [NQ] App. F.5 Eq. 58–59 and [LB] App. D Eq. 16 with the residual-path
  factor of 2 and the latent-scale row removed, plus exactly `16*dout` bits
  for the bias vector.

### Requirement: Non-goals of the v0.1 representation

A qpk1 stream SHALL be exactly one layer: the format SHALL NOT define a model-level checkpoint, manifest, layer registry, or multi-layer container (that is the separate QKPB blob). It SHALL NOT define GPU tensor or inference layouts in v0.1 (GPU compute stays in PyTorch; Rust kernels are CPU-only). It SHALL NOT compress or quantize the scale vectors beyond tagged float16/float32 storage, and SHALL NOT store continuous pre-sign latent factors, optimizer state, or calibration statistics.

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

### Requirement: The version-2 layout SHALL carry the per-linear bias vector

A version-2 qpk1 stream SHALL carry the layer's `(dout,)` bias vector in the
stream's tagged dtype, written between the scale vectors and the `Ub`
bitplane. Serializers SHALL emit version 2 and SHALL always write the field:
an absent argument writes the zero vector, never an omitted field. A bias
whose width is not `dout` SHALL be rejected.

#### Scenario: SF-1.5 — The bias vector is a fixed part of the layout

- **WHEN** a layer is serialized at version 2 with and without an explicit
  bias argument, at any `(dout, din, r)` and either scale dtype
- **THEN** both streams have the same length and the same bytes outside the
  bias field, the omitted form carries the zero vector, and a wrong-width
  bias raises the documented error.

#### Scenario: SF-2.3 — The bias vector round-trips byte-exactly

- **WHEN** a version-2 stream is deserialized and its bias, factors and
  scales are re-serialized
- **THEN** the bytes equal the original stream, and changing only the bias
  changes only the bias field's bytes.

### Requirement: A version-1 stream SHALL read as a pre-bias artifact

A version-1 stream SHALL carry no bias vector and SHALL remain readable: the
reader SHALL return it with no bias field, which is the zero vector a
biasless linear applies. The version word alone selects the layout, and any
value other than 1 or 2 SHALL be rejected rather than defaulted.

#### Scenario: SF-1.6 — Legacy reading, and no silent default

- **WHEN** a pinned pre-bias stream is deserialized
- **THEN** it parses with no bias field and its factors and scales are
  recovered exactly, while the same bytes under a version-3 word and a
  version-2 body short by the bias vector's bytes are both rejected.
