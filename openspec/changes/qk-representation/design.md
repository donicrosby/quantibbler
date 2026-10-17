# Design: qk-representation (qpk1)

Context: config.yaml fixes the shared representation
`W ~= diag(s1) @ (Ub @ Vb^T) @ diag(s2)` ([NQ] Sec 3.1; [LB] Sec 3.1 Eq. 4
with the latent scale `l` folded into the factors) and requires storage to be
dominated by packed binary bits with BPW counted from serialized bytes. This
document records the byte-level decisions, the alternatives discarded, open
questions, and informed assumptions.

## Decisions

### D1. Single-layer container, fixed 24-byte header

A qpk1 stream is exactly one factorized linear layer. Layout:

| offset | size | field |
|---|---|---|
| 0 | 4 | magic `b'QPK1'` |
| 4 | 4 | representation version, u32 LE, `= 1` |
| 8 | 4 | dtype tag, u32 LE (1 = float16 scales, 2 = float32 scales) |
| 12 | 4 | `dout`, u32 LE |
| 16 | 4 | `din`, u32 LE |
| 20 | 4 | `r`, u32 LE |
| 24 | 2*dout*itemsize | `s1`, float16/float32 (per dtype tag) |
| — | 2*din*itemsize | `s2`, float16/float32 (per dtype tag) |
| — | ceil(dout*r/8) | packed `Ub` bitplane, uint8 |
| — | ceil(din*r/8) | packed `Vb` bitplane, uint8 |

Total serialized size:
`24 + (dout + din) * itemsize + ceil(dout*r/8) + ceil(din*r/8)` bytes.

All multi-byte scalars and array elements are little-endian; the header is
exactly the u32 metadata block shared with the sibling lanes (magic
`'QPK1'`, version u32, dtype tag u32, dims).

Rationale: a fixed-size header makes byte-offset math trivial and lets the
round-trip and BPW requirements be checked without a parser. The paper
storage model counts the two scale vectors plus the packed bits
([NQ] App. F.5 Eq. 58: `r(n+m) + 16(n+m)` bits; [LB] App. D: sign matrices
1 bit/element, scale vectors 16 bits/element); the only additions over the
papers are the per-layer header and byte-rounding padding, both counted in
BPW by construction. Per-layer granularity matches the papers' unit of
analysis (BPW is derived per linear layer, [LB] App. D Eq. 16) and defers
any model-level aggregation — explicitly a non-goal.

### D2. Canonical bit packing: row-major, little-endian within bytes, +1 = bit 1

Each factor matrix is packed row-major: element `(i, j)` of `Ub` is bit
`i*r + j` of the bit stream; bit `k` lives in byte `k // 8` at bit position
`k % 8` counting from the least-significant bit (little-endian bit order
within bytes). `+1` maps to bit `1`, `-1` to bit `0` ([NQ] Sec 3.1 stage
(c): "-1 -> 0, +1 -> 1"). The final byte of each bitplane is zero-padded.

Rationale: this is the config-mandated canonical order (shared constant).
Row-major matches NumPy C-order semantics on the reference path, so the
executable specification is a one-liner with `numpy.unpackbits`; the sign
mapping matches [NQ] Fig. 2 stage (c). Zero padding keeps unassigned bits
deterministic so byte equality (the round-trip and sign-agreement checks)
is well-defined.

### D3. Scale storage: float16 default, float32 opt-in, per-layer dtype tag

`s1` and `s2` are stored as float16 by default; a writer may select float32
by setting dtype tag = 2. Both vectors in one stream share one dtype — no
mixed-scale streams.

Rationale: the papers' BPW accounting assumes FP16 scales ([LB] App. D:
"Three FP16 scaling vectors ... require 16 bits per element"; [NQ] App. F.5
Eq. 58: `16(n+m)` bits). Storing float16 by default keeps the realized BPW
comparable to the papers' figures. Float32 exists because config requires
solvers to compute in float64 by default and float32 to be a documented
opt-in accelerator; a float32 escape hatch lets round-trip tests distinguish
"exact under the stored dtype" from "rounding at serialize time" without a
format change. The config numerics rule governs computation; serialization
precision is a separate, explicitly-tagged choice.

### D4. Round-trip is exact by construction and checked at byte level

Serialize(deserialize(b)) == b for every valid byte stream `b`, and
deserialize(serialize(factors, scales)) recovers `Ub`, `Vb` bit-exactly and
`s1`, `s2` exactly under the stream's dtype (float32 streams: exact; float16
streams: exactly the float16 values stored). The property is defined on
bytes — not on float comparisons — because packed bits and stored scale
values are the actual payload.

Rationale: config lists the round-trip property as the correctness anchor
for this capability, and the kernels gate compares packed bits
("sign-agreement invariant: packed bits identical"). A byte-level identity
is strictly stronger and implementation-agnostic.

### D5. BPW from serialized bytes only

`BPW = 8 * len(serialized_bytes) / (dout * din)`. Header, scales, padding
— everything in the stream — is inside `len(serialized_bytes)`. There is no
"analytical" BPW and no parameter-count formula in the API.

Rationale: config invariant ("effective BPW is counted from serialized
bytes, never from parameter counts"), adopted from the papers' counting
discipline: [NQ] App. F counts "all stored bits including binarized weights,
reconstruction coefficients, and any auxiliary flags" and shows baselines
failing sub-1-bit precisely through uncounted metadata ([NQ] Sec 4.3:
BiLLM 2.88, STBLLM 4.13 BPW vs GPTQ W2g64 2.28); [NQ] App. F.1 defines
`BPW = B_WQ / B_WFP * 16 = B_WQ / (n*m)`, i.e. stored bits over the fp16
baseline element count. The end-to-end smoke gate ("BPW measured from
bytes") consumes exactly this definition.

### D6. Rank-vs-BPW formula (analytical planning formula, BPW still measured)

For planning rank choice given `dout`, `din`, target `b`:

```
r(b)  = (b * dout * din - 8 * (24 + (dout + din) * itemsize))
        / (dout + din)                     ... real-valued, round and re-check
BPW(r) = 8 * (24 + (dout + din) * itemsize
              + ceil(dout*r/8) + ceil(din*r/8)) / (dout * din)
```

With the header and rounding terms dropped this converges to
`r(n+m) + 16(n+m)` over `n*m` — [NQ] App. F.5 Eq. 59; with the `[l]` latent
scale row (`+16r`) and the residual-path factor of 2 added it is [LB]
App. D Eq. 16–17 (LittleBit counts a residual compensation path with
identical structure; quantibbler's single-path container matches the
"factor of 2 removed" note in [LB] App. D).

Rationale: the papers derive exactly this inversion to pick ranks per layer
([LB] App. D Example 1: 4096x4096, b=0.55 -> r=546). The formula is a
*planning* aid; the authoritative BPW for any artifact remains
`8 * len(bytes) / (dout * din)` per D5, and tasks require the two to agree
within rounding.

### D7. Sub-1-bit feasibility at target ranks

At float16 scales the fixed per-layer overhead is
`24 + 2*(dout + din)` bytes. The gap to 1.0 BPW is:
`BPW(r) < 1.0` iff `8*(ceil(dout*r/8) + ceil(din*r/8)) + overhead_bits <
dout*din`, where `overhead_bits = 8*(24 + 2*(dout+din))`. For every LLM
linear layer shape this holds with large margin at moderate r: e.g.
4096x11008 at r = 100 -> 24 + 30,208 + 188,800 = 219,032 serialized
bytes -> BPW ~ 0.0389 (the [NQ] App. F.5 paper model gives the same
value at this shape; the delta is exactly the 24-byte header, byte
rounding being zero here, and it is counted). [LB] App. D Example 2
lands at b ~ 0.1 with r = 133 on this shape under its two-path
counting. The spec pins: for target ranks r satisfying the D6 formula
at b < 1.0 with the standard LLM shapes, the measured BPW must be
< 1.0 — verified by property test, not by example.

### D8. Strict validation, fail-closed reader

`deserialize` rejects: bad magic, version != 1, unknown dtype tag, `r == 0`,
any dim exceeding u32, truncation (stream shorter than the header-declared
size), and trailing garbage (stream longer). `r` must satisfy
`r <= min(dout, din)`. No silent best-effort loads.

Rationale: the round-trip property is only meaningful over exactly the valid
set; fail-closed parsing keeps the byte-identity definition total and makes
the version gate from the proposal enforceable.

### D9. Version gate and extension policy

`REPRESENTATION_VERSION = 1` forever labels this exact layout. Additions
(new dtype tags) are made by allocating new tag values, which cannot break
byte offsets; structural changes bump the version and readers reject
foreign versions with a clear error. No reserved-but-undefined fields: every
header byte is specified (a v0.1 format with undefined padding bits invites
divergent fixtures across lanes).

## Discarded Alternatives

- **Single model-level container (zip/TarDB-style manifest of many layers).**
  Discarded: config forbids a model-level checkpoint format for this change
  (proposal Non-goals); per-layer streams compose into any future model
  format without constraining this one, and per-layer granularity matches
  the papers' per-layer BPW analysis ([LB] App. D; [NQ] App. F).
- **Reserving a latent/rank scale vector `s_mid in R^r` in the format**
  (the DBF layout, [NQ] App. F.4 Eq. 54–56). Discarded: [NQ] App. F.5 shows
  the two-scale system removes `16r` bits per layer and is the paper's
  stated BPW advantage; config fixes the two-scale representation. The
  latent scale of [LB] Eq. 4 is folded into the factors by the config
  invariant, so there is nothing to store.
- **Explicit valid-bit count or mask byte for the padding tail.** Discarded:
  the valid-bit count is derivable from `dout*r` and `din*r` in the header;
  storing it duplicates state and adds a divergence surface between lanes.
  Zero-padding (D2) already makes the tail deterministic.
- **Bit-packing columns instead of rows (column-major planes) for
  SIMD-friendlier kernels.** Discarded: config fixes row-major as the
  shared constant and the NumPy executable specification is C-ordered; the
  Rust kernels must match the NumPy path bit-exactly, so any transposition
  would live *inside* a kernel as an access pattern, not in the format.
- **Big-endian element order for network-byte-order portability.**
  Discarded: every producer/consumer is little-endian (x86-64/aarch64) and
  NumPy/Rust reference paths would each carry a swap; the papers specify no
  wire order, so there is nothing to stay compatible with.
- **Storing scales in bfloat16.** Discarded: float16 is what both papers'
  BPW derivations count ([LB] App. D; [NQ] App. F.5), and float16 round-trip
  is exact at the numpy level while bfloat16 would add a non-standard dtype
  dependency to the container for no storage win (same 16 bits).
- **Variable-length header (tag-length-value) for future extensibility.**
  Discarded: TLR headers make `len(bytes)` BPW accounting and byte-offset
  reasoning harder for zero v0.1 benefit; D9's dtype-tag allocation covers
  the realistic extension (new scale dtypes) without layout change.
- **Headerless raw arrays with sidecar metadata (npz/JSON).** Discarded:
  BPW "from serialized bytes" must have a single unambiguous byte artifact
  per layer; sidecars let metadata escape the count, recreating exactly the
  accounting failure mode the papers criticize in baselines ([NQ] Sec 4.3).

## Open Questions

- **OQ1 — Should float16 scale storage be replaced by a groupwise scheme
  (e.g. one scale per 64 input channels) in a future version?**
  Status: open. Owner: qk-representation lane (this change's owner) to
  decide at first format-version bump; input from qk-ptq on quality
  trade-offs ([NQ] App. F discusses block-scale overhead for baselines).
  v0.1 keeps per-layer full vectors.
- **OQ2 — Does the Rust kernels lane want an in-memory packed type distinct
  from the qpk1 byte stream (e.g. word-aligned u64 planes for CPU GEMM)?**
  Status: open. Owner: qk-kernels lane. This spec only fixes the serialized
  form; if kernels want a different working layout the conversion belongs
  behind their API, and the sign-agreement invariant applies to the packed
  bits as serialized.
- **OQ3 — Maximum supported `dout`/`din`/`r` (practical limits below the
  u32 ceiling, e.g. a 2^31-bit sanity bound).** Status: open. Owner:
  qk-representation lane, resolved when the first end-to-end smoke fixture
  sizes are known (config gate: tiny random transformer block). v0.1
  validates against the u32 ceiling only.

## Informed Assumptions

- **IA1 — The 24-byte fixed header's overhead is negligible against the
  sub-1-bit target for real LLM shapes.** Rationale: header plus float16
  scales cost `8*(24 + 2*(dout+din))` bits against a `dout*din`-bit
  budget; at 4096x4096 that is ~0.008 bits/weight against a 1.0 budget,
  and it *shrinks* relative to the payload as layers grow. If wrong (some
  consumer serializes many tiny layers), it surfaces as measured BPW
  stubbornly above the D6 prediction on small shapes — the property test
  comparing measured BPW to the formula flags it immediately, and
  small-layer support can be revisited via OQ3.
- **IA2 — Per-layer files/streams are an acceptable operational unit
  (users tar/key them by layer name themselves).** Rationale: the papers'
  storage analysis is per-layer ([LB] App. D; [NQ] App. F), and a model
  manifest is an explicit non-goal; no paper evidence bears on multi-layer
  packaging, so we assume deferring it costs nothing in v0.1. If wrong, it
  surfaces as user friction in the qk-ptq end-to-end flow (quantize ->
  serialize -> reload of a whole block), which would show up as a task
  failure or review feedback in that lane, not as a correctness bug here.
- **IA3 — Zero-padded tail bytes are required to be zero on write but
  readers do not enforce them.** Rationale: making readers reject nonzero
  padding would break byte-level round-trip tolerance to third-party
  writers for no correctness gain (padding bits are never read). If wrong
  (a lane produces garbage in padding that then shows up in byte-diff
  fixtures), it surfaces as flaky sign-agreement/round-trip diffs in the
  kernels lane, and we tighten the reader then.
- **IA4 — float16 (not bfloat16/float32-by-default) scale storage preserves
  end-to-end quality comparably to the papers' FP16 accounting.** Rationale:
  both papers count FP16 scales in their BPW models ([LB] App. D;
  [NQ] App. F.5), implying FP16 scale precision is adequate in their
  results; quantibbler's solvers still compute in float64 and round once at
  serialize time. If wrong, it surfaces as degraded smoke-gate logit deltas
  or perplexity regressions in qk-ptq evaluation, not as a format failure;
  the float32 tag (D3) is the standing escape hatch.

## Implementation Crystallization

### 2026-10-17 — Scope of the `r <= min(dout, din)` bound (D8)

During implementation (qk-representation tasks 1.2/2.1) the following
conflict surfaced: D8 and SF-1.4 require `deserialize` to reject
`r > min(dout, din)`, while SF-1.3 mandates the known-layer fixture
`dout = 3, din = 5, r = 7` and tasks 1.2/3.1 require its exact 48-byte
serialization (SF-1.3's own size arithmetic), and SF-2.1 requires
byte-identical round-trips. No monotone validity predicate on
`(dout, din, r)` can accept `r = 7` at `(3, 5)` while rejecting `r = 6`,
so the fixture cannot be reader-valid and round-tripping it would violate
SF-1.4's mandated rejection.

Resolution: the rank bound is enforced by the **reader only**, exactly as
D8's text scopes it ("`deserialize` rejects ..."). `serialize` and
`expected_size` validate structure (positive, u32-bounded dims; matching
ranks; +/-1 entries) but not rank feasibility; `rank_for_target_bpw` keeps
the bound as an infeasibility error since it is a planning aid for real
layers. The D4 round-trip property holds over exactly the reader-accepted
set (SF-2.1's own domain: "any valid qpk1 stream"), so byte-identity is
unaffected. The SF-1.3 fixture remains writer-only; the byte-level
round-trip anchor in the tests uses its rank-feasible twin `(3, 5, r = 3)`.
Concretely: `r > min(dout, din)` raises `Qpk1Error` from `deserialize`
(SF-1.4), never from `serialize`/`expected_size`. No spec.md change is
required: each scenario reads as before under this scoping.

### 2026-10-17 — Naming decisions the spec left silent

- **Error type:** the spec's "the documented error type" is implemented as
a single `Qpk1Error(ValueError)` raised by every writer/reader rejection
(bad magic, version != 1, dtype tag not in {1, 2}, non-positive or
u32-overflowing dims, rank bound, truncation, trailing bytes, non-±1
factor entries, non-storable scale dtype). One type keeps the reject
corpus (SF-1.4) assertable without leaking internal exception classes;
subclassing per failure mode is deferred until a consumer needs it.
- **Success type:** `deserialize` returns a `Qpk1Layer` dataclass with
attributes `Ub` (int8, ±1), `Vb` (int8, ±1), `s1`, `s2` (float per dtype
tag), `dtype`. The spec requires "no partial object" but never names the
success shape; the dataclass is the pinned public shape going forward.
- **Public surface:** `qpk1.__all__` pins constants, `Qpk1Error`,
`Qpk1Layer`, and the five functions, making SF-6.1's "no multi-layer
API" directly testable at the module boundary.
- **Test byte literals:** every exact-byte assertion in the tests was
derived from an independent bit-loop oracle written from D1/D2 prose
before the pack implementation existed (not from the code under test),
so the known-layer vectors are an independent check of the packing,
not a tautology.

### 2026-10-17 — Packing-domain note for consumers (continuous vs per-row)

The qpk1 container's `_pack_bits` is CONTINUOUS over the flattened matrix
(bit `i*r + j` for element (i, j) — `np.packbits(axis=None)`), which is
what SF-3.1 mandates. The qk-kernels `pack_bits` is PER-ROW (each row
zero-padded to a byte boundary), per [NQ] Fig. 2 and its own spec. The two
layouts coincide only when `r % 8 == 0`. Consumers bridging the two
domains (verified in the qk-server lane: container-to-row repack plus
`Vb^T` materialization at load) must convert explicitly; byte-level
fixture exchange is only valid within one domain.
