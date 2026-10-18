# qk-representation Specification

## ADDED Requirements

### Requirement: The qpk1 container serializes one factorized linear layer

A qpk1 stream SHALL represent exactly one factorized linear layer
`W ~= diag(s1) @ (Ub @ Vb^T) @ diag(s2)` with binary `Ub`, `Vb`
and full-precision scales `s1`, `s2` ([NQ] Sec 3.1 Eq. 1; [LB]
Sec 3.1 Eq. 4; latent scale folded in).

The stream SHALL use the canonical layout: a 24-byte u32 header (magic
`'QPK1'`, version `1`, dtype tag, dims `dout`, `din`, `r`), then the
scale vectors in the tagged dtype, then the packed uint8 `Ub` and `Vb`
bitplanes; all elements little-endian.

#### Scenario: SF-1.1 — Serialized size follows the documented formula

- **WHEN** any layer is serialized with scale itemsize `w` bytes
- **THEN** the stream length is exactly
  `24 + (dout + din) * w + ceil(dout*r/8) + ceil(din*r/8)` bytes.

#### Scenario: SF-1.2 — Dtype tag selects scale precision

- **WHEN** a stream carrying dtype tag 1 or 2 is deserialized
- **THEN** the scale vectors are read as float16 (tag 1) or float32
  (tag 2), and any other tag value is rejected.

#### Scenario: SF-1.3 — Known layer serializes to the exact expected bytes

- **WHEN** a layer with `dout = 3`, `din = 5`, `r = 7`, float16 scales,
  and known factor and scale values is serialized
- **THEN** the stream is `24 + 2*(3+5) + ceil(3*7/8) + ceil(5*7/8) = 24 + 16 + 3 + 5 = 48` bytes long, begins with the bytes `QPK1` followed by
  the u32 little-endian values `1`, `1`, `3`, `5`, `7`, and matches the
  documented byte layout field-for-field at the documented offsets.

#### Scenario: SF-1.4 — Malformed headers are rejected

- **WHEN** a byte stream with wrong magic, version other than 1, unknown
  dtype tag, `r = 0`, `r > min(dout, din)`, truncated length, or trailing
  bytes is deserialized
- **THEN** deserialization fails with the documented error type and no
  partial object is returned.

### Requirement: Lossless round-trip

Serialization and deserialization SHALL be mutually inverse at the byte
level: `serialize(deserialize(b)) == b` for every valid qpk1 stream `b`,
and `deserialize(serialize(Ub, Vb, s1, s2))` SHALL recover the packed
bitplanes bit-exactly and the scale vectors exactly as stored under the
stream's dtype.

#### Scenario: SF-2.1 — Byte-identical round-trip

- **WHEN** any valid qpk1 stream is deserialized and re-serialized
- **THEN** the resulting byte string equals the original byte string
  exactly, for both float16 and float32 scale dtypes.

#### Scenario: SF-2.2 — Factors and scales are recovered exactly

- **WHEN** binary factors `Ub`, `Vb` and scale vectors `s1`, `s2` are
  serialized and then deserialized
- **THEN** the recovered factors compare bit-exactly to `Ub`, `Vb` (equal
  packed bits, sign-agreement holds), and the recovered scales equal the
  stored values exactly under the stream's dtype (no float32 stream loses
  precision; a float16 stream returns exactly the float16-representable
  values that were stored).

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
length of the qpk1 stream: `BPW = 8 * len(bytes) / (dout * din)`,
counting the header, scale vectors, and padding — never from parameter
counts or analytical estimates (config Method invariants; [NQ] App. F
counting discipline; [LB] App. D).

#### Scenario: SF-4.1 — Measured BPW uses the full stream length

- **WHEN** a layer is serialized and its BPW is computed
- **THEN** the BPW equals `8 * len(serialized bytes) / (dout * din)`,
  including header and padding bytes, and this value is the one reported
  by the library's BPW function.

#### Scenario: SF-4.2 — Measured BPW matches the analytical byte-size formula

- **WHEN** the analytical size
  `24 + (dout + din) * itemsize + ceil(dout*r/8) + ceil(din*r/8)` bytes is
  evaluated for the same `(dout, din, r, dtype)`
- **THEN** the measured BPW from the serialized bytes and the BPW
  recomputed from the analytical size agree exactly (same integer byte
  count).

#### Scenario: SF-4.3 — Sub-1-bit at target ranks

- **WHEN** a standard LLM linear-layer shape (e.g. `dout = 4096`,
  `din = 11008`) is serialized at a rank `r` chosen by the rank planning
  formula for a target `b < 1.0` with float16 scales
- **THEN** the measured BPW from serialized bytes is strictly less than
  1.0 (storage dominated by packed binary bits; [NQ] Sec 4.3, App. F.5).

### Requirement: Rank planning formula

The library SHALL provide a rank planning formula inverting the qpk1
byte-size model: given `dout`, `din`, scale itemsize `w` bytes, and
target BPW `b`, the real-valued rank is
`r(b) = (b * dout * din - 8 * (24 + (dout + din) * w)) / (dout + din)`,
rounded to an integer and re-checked against the measured BPW. It is a
planning aid only; the authoritative BPW remains the byte-derived value.

#### Scenario: SF-5.1 — Rank selection hits a target BPW

- **WHEN** `rank_for_target_bpw(dout, din, b, dtype)` is called with a
  feasible target `b` (positive numerator)
- **THEN** the returned integer rank, when serialized, measures a BPW
  within one rank-step's rounding of `b`, and the re-checked
  `r(b)` derivation from the measured bytes is consistent.

#### Scenario: SF-5.2 — Formula reduces to the paper storage model

- **WHEN** header and rounding terms are set aside
- **THEN** the byte model reduces at `w = 2` to
  `r*(dout+din) + 16*(dout+din)` bits over `dout*din`, matching
  [NQ] App. F.5 Eq. 58–59 and [LB] App. D Eq. 16 with the residual-path
  factor of 2 and the latent-scale row removed.

### Requirement: Non-goals of the v0.1 representation

A qpk1 stream SHALL be exactly one layer: the format SHALL NOT define a
model-level checkpoint, manifest, layer registry, or multi-layer
container. The format SHALL NOT define GPU tensor or inference layouts in
v0.1 (config Architecture: GPU compute stays in PyTorch; Rust kernels are
CPU-only). It SHALL NOT compress or quantize the scale vectors beyond the
tagged float16/float32 storage, and SHALL NOT store continuous pre-sign
latent factors, optimizer state, or calibration statistics.

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
