# Change: qk-representation — qpk1 serialized container for sub-1-bit factorized weights

## Why

quantibbler's method invariant is that storage must be dominated by packed
binary bits and effective BPW is counted from serialized bytes, never from
parameter counts (openspec/config.yaml, Method invariants). That claim is
unverifiable without a canonical byte-level container: [NQ] Sec 4.3 and
Appendix F show that binary PTQ baselines lose the sub-1-bit regime to
metadata overhead (BiLLM 2.88 BPW, STBLLM 4.13 BPW), while NanoQuant achieves
genuine sub-1-bit storage precisely by minimizing metadata
([NQ] App. F.5, Eq. 57–59). [LB] App. D performs the same counting discipline
for the QAT path, deriving BPW from stored parameter sizes (Eq. 16–17).

The shared weight representation
`W ~= diag(s1) @ (Ub @ Vb^T) @ diag(s2)` with binary factors
`Ub in {-1,+1}^(dout x r)`, `Vb in {-1,+1}^(din x r)` and full-precision
scale vectors `s1 in R^dout`, `s2 in R^din` is fixed by
[NQ] Sec 3.1 (Eq. 1) and [LB] Sec 3.1 (Eq. 4, with the latent scale `l`
folded into the factors per config invariant). This change pins the single
serialized form of that representation — format name `qpk1` — plus the
lossless round-trip property and the byte-derived BPW accounting that the
config verification gates ("BPW measured from bytes") consume. Sibling lanes
qk-kernels (packing kernels) and qk-ptq (PTQ pipeline) produce and consume
qpk1; they require this contract to exist first.

## What Changes

- ADD capability `qk-representation`:
  - the **qpk1 container**: a 24-byte u32 metadata header (magic `'QPK1'`,
    `REPRESENTATION_VERSION = 1`, dtype tag, dims `dout`/`din`/`r`), per-factor
    scale vectors `s1`/`s2` (float16 default, float32 opt-in via dtype tag),
    and packed U/V bitplanes as uint8 arrays;
  - **canonical bit packing**: row-major, little-endian within bytes,
    `+1 -> bit 1`, `-1 -> bit 0`, zero padding to byte boundaries;
  - the **lossless round-trip property**: serialize -> deserialize is the
    identity on bytes, and factors/scales are recovered exactly;
  - **BPW accounting from serialized bytes**:
    `BPW = 8 * len(bytes) / (dout * din)`, header and padding included;
  - the **rank-vs-BPW formula** given `dout`/`din`/`r`, consistent with
    [NQ] App. F.5 Eq. 59 modulo the exact header term;
  - **strict single-layer extent**: a qpk1 stream is exactly one factorized
    linear layer — nothing more, nothing less.

- Shared constants fixed repo-wide (sibling lanes must use verbatim):
  format name `qpk1`; `REPRESENTATION_VERSION = 1`; bit order row-major
  little-endian within bytes; `+1` maps to bit 1 and `-1` to bit 0; packed
  factor arrays are uint8 with a u32 metadata header (magic `'QPK1'`,
  version u32, dtype tag u32, dims). BPW is always computed from serialized
  byte length, never parameter counts.

- **Invariant-touching**: this change defines the storage format named in
  config Method invariants and constrains the kernels contract. See Impact
  for the compatibility statement.

## Impact

- **Affected specs**: `qk-representation` (new capability spec; no existing
  specs modified — `openspec/specs/` is empty).
- **Affected code (future implementation landing against this spec)**:
  a new Python module in the `quantibbler` package implementing
  serialize/deserialize/BPW/rank-selection on the NumPy reference path; the
  **qk-kernels** lane implements matching CPU pack/unpack kernels (referenced
  by name only; semantics fixed by this spec and the sign-agreement
  invariant: packed bits identical across NumPy and Rust paths). The
  **qk-ptq** lane consumes qpk1 at "binary parameters are frozen and packed"
  time ([NQ] Sec 3.3).
- **Compatibility impact**: qpk1 is the first storage format in v0.1 — there
  is no prior on-disk format to break. Forward compatibility is gated by
  `REPRESENTATION_VERSION`: readers must reject headers whose version is not
  1, and any future layout change bumps the version rather than mutating
  meaning in place. Because the format is invariant-touching, any change
  here invalidates stored fixtures and packed-output comparisons in the
  qk-kernels and qk-ptq lanes simultaneously; both lanes must bump together.

## Non-goals

- **No model-level checkpoint format.** qpk1 serializes exactly one
  factorized linear layer. No multi-layer container, no top-level model
  manifest, no config/tokenizer serialization. (The strict-extent requirement
  enforces the single-layer scope at the format level.)
- **No GPU formats in v0.1.** No GPU tensor layouts, no inference-kernel
  data formats. GPU/array compute stays in PyTorch and Rust kernels are
  CPU-only in v0.1 (config Architecture).
- **No compression beyond fixed-rate bit packing.** No entropy coding, no
  sparsity bitmaps, no shared-codebook tricks — these would complicate the
  exact byte-level BPW accounting that is the point of this capability.
- **No scale quantization.** `s1`/`s2` are stored full precision (float16
  default, float32 opt-in), matching the papers' FP16 scale accounting
  ([LB] App. D; [NQ] App. F.5). Quantizing the scales is a future format
  version, not a v0.1 concern.
- **Not a training format.** Continuous latent factors (pre-sign `U`, `V`
  in [LB] Sec 3.1), optimizer state, and calibration statistics are out of
  scope; qpk1 stores the frozen, packed result.
