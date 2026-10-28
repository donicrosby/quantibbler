# qk-ptq delta: qk-artifact-dtype

## ADDED Requirements

### Requirement: The exporter SHALL write the remainder at the declared dtype

The real-model driver SHALL expose `--remainder-dtype {fp32, bf16}`
(default fp32) and SHALL write the embedding, lm_head, and norm sidecar
at the declared dtype. fp32 output SHALL be byte-identical to the
pre-change writer; bf16 SHALL use ONE shared round-to-nearest-even
conversion helper with a post-cast finiteness gate, and SHALL declare
the dtype in the manifest and the `weight_tying` suffix so the
declarations agree (qk-server RD-1).

#### Scenario: RDW-1.1 — The fp32 default is byte-identical
- **WHEN** the driver exports with the default `--remainder-dtype fp32`
- **THEN** the remainder files' bytes equal the pre-change writer's
  pinned bytes, and the manifest carries `dtype = "fp32"`

#### Scenario: RDW-1.2 — A bf16 export is a loadable declared artifact
- **WHEN** the driver exports with `--remainder-dtype bf16`
- **THEN** the output directory carries `[sidecar] dtype = "bf16"`, a
  matching `@bf16` weight-tying suffix, and bf16 remainder files, and
  the Rust loader accepts it (qk-server RD-2.1)

#### Scenario: RDW-1.3 — Non-finite values refuse on the bf16 path
- **WHEN** a bf16 export encounters a non-finite remainder value
- **THEN** the cast stage refuses with a named error; the fp32 export
  path is unchanged

#### Scenario: RDW-1.4 — The export self-check uses the declared itemsize
- **WHEN** the exporter self-checks the sidecar byte length
- **THEN** the check validates against the float count times the
  declared dtype's itemsize, so a miscount fails at export, not at
  server load

### Requirement: The report SHALL name the remainder dtype

The report's fidelity fields SHALL name the remainder storage dtype
from the run's arguments: a top-level `remainder_dtype` field, and
`regime.artifact` and the dtype note strings extended to name it —
emitted from the argument values, never from hardcoded literals.

#### Scenario: RDW-2.1 — A bf16 run records bf16
- **WHEN** a run executes with `--remainder-dtype bf16`
- **THEN** the report records `remainder_dtype: bf16` and the artifact
  and dtype-note strings name bf16, recorded from the argument

#### Scenario: RDW-2.2 — The default run records fp32 from the argument
- **WHEN** a run executes with the default remainder dtype
- **THEN** the report records `remainder_dtype: fp32` from the argument
  value — the bf16 literal never appears for an fp32 run

### Requirement: The repack utility SHALL convert the remainder without re-training

`tools/repack_remainder.py` SHALL convert an artifact's remainder
from fp32 to bf16 into a NEW directory, never mutating the source: layer
streams and tokenizer files SHALL be copied byte-identically, the
remainder re-encoded through the shared conversion helper, and the
manifests rewritten with the declared dtype. The output manifest SHALL
record the conversion provenance and SHALL carry no wall-clock
timestamps, so identical sources produce identical output bytes.

#### Scenario: RDP-1.1 — The repacked fixture loads
- **WHEN** the committed fp32 fixture is repacked to bf16
- **THEN** the output directory loads in the Rust loader with the
  pinned bf16 values, and its manifest carries `dtype = "bf16"` plus a
  `[repack]` record naming the source model id, the conversion
  (`remainder fp32 -> bf16`), and the sha256 of the source remainder
  files

#### Scenario: RDP-1.2 — An already-bf16 source refuses
- **WHEN** the source artifact's remainder is already declared bf16
- **THEN** the utility refuses with a named error naming the artifact

#### Scenario: RDP-1.3 — A non-finite source refuses
- **WHEN** any source remainder value is non-finite
- **THEN** the utility refuses with a named error; no output directory
  is written

#### Scenario: RDP-1.4 — Non-remainder files are byte-identical
- **WHEN** the repack copies the layer streams and tokenizer files
- **THEN** every non-remainder file in the output is byte-identical to
  its source counterpart

#### Scenario: RDP-1.5 — Repack is deterministic
- **WHEN** the same source is repacked twice
- **THEN** the two output directories are byte-identical

### Requirement: bf16 remainder conversion SHALL bound elementwise error at half an ulp

The fp32-to-bf16 conversion SHALL round to nearest even, so every
converted finite element satisfies |bf16(x) - x| <= 2^-8 |x| + 2^-134 —
and this bound, not the float64 1e-12 parity tolerance, SHALL govern
bf16 round-trip comparisons.

#### Scenario: RDP-2.1 — Elementwise round-trip within the half-ulp bound
- **WHEN** a deterministic fp32 matrix is converted to bf16 and
  compared elementwise
- **THEN** every element satisfies the 2^-8|x| + 2^-134 bound exactly
