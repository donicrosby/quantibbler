# qk-representation delta: qk-bias-carry

## MODIFIED Requirements

### Requirement: The qpk1 container serializes one factorized linear layer

A qpk1 stream SHALL represent exactly one factorized linear layer
`W ~= diag(s1) @ (Ub @ Vb^T) @ diag(s2)` together with its per-linear bias,
with binary `Ub`, `Vb` and full-precision scales `s1`, `s2` ([NQ] Sec 3.1
Eq. 1). The canonical layout SHALL be a 24-byte u32 header (magic `'QPK1'`,
version, dtype tag, dims `dout`, `din`, `r`), then the scale vectors in the
tagged dtype, then the `(dout,)` bias vector in that dtype, then the packed
uint8 `Ub`/`Vb` bitplanes; all elements little-endian. Version 2 carries the
bias field and is the only version a serializer emits.

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

## ADDED Requirements

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
