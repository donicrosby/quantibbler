# qk-server delta: qk-artifact-dtype

## MODIFIED Requirements

### Requirement: The model directory SHALL declare its norm inventory

The exported model directory SHALL declare its per-block norm inventory
in `config.toml` as an ordered `norm_slots` array. The sidecar SHALL
hold, per block, each declared slot's vector in inventory order, then
the model-final norm, in the directory's declared remainder dtype —
fp32 little-endian when undeclared, bf16 little-endian when the
manifest `[sidecar]` block declares `dtype = "bf16"` (RD-1). The total
float count SHALL be derived from the declared inventory, never from any
model family's formula.

#### Scenario: N-1.0 — Slot vocabulary, widths, and layout
- **WHEN** an inventory is declared
- **THEN** slot names come from the closed set `input`,
  `post_attention`, `q_norm`, `k_norm`, `post_feedforward`, with no
  repeats and at least one slot; `input`, `post_attention`, and
  `post_feedforward` are `d_model` floats each; `q_norm` and `k_norm`
  are `head_dim` floats each (one shared vector per projection,
  broadcast across heads — Qwen3's post-projection Q/K RMSNorm
  convention); per-block vectors follow inventory order, then the final
  norm's `d_model` floats

### Requirement: The loader SHALL validate the norm sidecar fail-closed

The loader SHALL validate the sidecar byte length against the
inventory-derived count times the declared dtype's itemsize and reject
the directory fail-closed (per `qk-server` SR-1.2 semantics) on any
inventory violation (absent, unknown-named, repeated) or sidecar
violation (missing, truncated, miscounted at the declared itemsize).

#### Scenario: N-1.1 — Load succeeds with a complete sidecar
- **WHEN** the server loads a model directory whose sidecar matches
  `n_layers` blocks at `d_model` each, at its declared dtype's itemsize
- **THEN** loading completes and the engine reports ready

#### Scenario: N-1.2 — Missing or malformed sidecar fails closed
- **WHEN** the sidecar is absent, truncated, or wrong-count
- **THEN** the server never reports ready and exits non-zero naming the
  failing artifact

#### Scenario: N-1.3 — A second declared inventory of the same geometry loads
- **WHEN** two directories declare identical geometry but different
  inventories (e.g. `["input", "post_attention"]` vs
  `["input", "post_attention", "q_norm", "k_norm"]` vs a Gemma-style
  `["input", "post_attention", "post_feedforward"]`)
- **THEN** each loads iff its sidecar byte length equals its own
  inventory-derived count times its declared itemsize, and each is
  rejected against the other's count

## ADDED Requirements

### Requirement: The model directory SHALL declare its remainder dtype

The manifest `[sidecar]` block SHALL declare the remainder tensors'
dtype through an optional `dtype` key whose values are `fp32` or
`bf16`; an absent key SHALL read as `fp32`. The `weight_tying` suffix
SHALL accept `@fp32` or `@bf16` and SHALL equal the declared
`[sidecar]` dtype. Any value outside the declared set SHALL refuse the
load with a diagnostic.

#### Scenario: RD-1.1 — A declared bf16 directory is accepted
- **WHEN** a manifest declares `[sidecar] dtype = "bf16"` and its
  `weight_tying` line carries a matching `@bf16` suffix
- **THEN** the loader resolves bf16 for the embedding, lm_head, and norm
  sidecar and proceeds with loading

#### Scenario: RD-1.2 — An absent declaration reads fp32
- **WHEN** a pre-change-shaped manifest carries no `dtype` key
- **THEN** the loader reads the remainder as fp32 exactly as before the
  change — the directory loads with values identical to the pre-change
  loader

#### Scenario: RD-1.3 — Declaration disagreement refuses
- **WHEN** the `[sidecar]` dtype and the `weight_tying` suffix disagree
  (either direction: `@fp32` with bf16 declared, or `@bf16` with fp32
  declared)
- **THEN** the load refuses with a diagnostic naming both declarations

#### Scenario: RD-1.4 — Unknown tokens refuse
- **WHEN** the `dtype` key holds any value outside `fp32`/`bf16` (e.g.
  `f8`, `fp16`, `b16`, the empty string), or the `weight_tying` suffix
  holds any token outside `@fp32`/`@bf16` — including a value carrying
  no `@`-suffix at all
- **THEN** the load refuses with a diagnostic naming the offending token

#### Scenario: RD-1.5 — An unset weight_tying with a bf16 sidecar is accepted
- **WHEN** the manifest declares `[sidecar] dtype = "bf16"` and
  `config.toml` carries no `weight_tying` line (an untied head)
- **THEN** the loader reads the head file at bf16 like the other two
  remainder files, and loading proceeds

### Requirement: The loader SHALL read remainder tensors at the declared dtype

The loader SHALL resolve each remainder file's dtype from the
declaration only — never from file-size heuristics — and SHALL read the
embedding, lm_head, and norm sidecar at the declared itemsize (4 bytes
for fp32, 2 for bf16), validating the stat length against the
inventory-derived float count times that itemsize before any read.
Loaded values SHALL widen to f64 exactly (bf16 to f64 is exact).

#### Scenario: RD-2.1 — A bf16 fixture loads with pinned values
- **WHEN** the loader loads the committed bf16 twin fixture
- **THEN** every loaded remainder value equals its pinned f64 literal —
  the bf16-to-f64 widening is exact, and no fp32-sized read occurs

#### Scenario: RD-2.2 — A dtype/byte-length mismatch refuses both ways
- **WHEN** a bf16-declared file is actually fp32-sized, or an
  fp32-declared file is actually bf16-sized
- **THEN** the load refuses with a diagnostic naming the file, the
  declared dtype, and the expected byte count

#### Scenario: RD-2.3 — The fp32 fixture is unchanged by the change
- **WHEN** the committed fp32 fixture loads before and after the change
- **THEN** the loaded values are identical — the fp32 path's byte math
  is untouched

### Requirement: A bf16-remainder model SHALL decode within the declared bf16 tolerance

The decode logits of a bf16-remainder artifact SHALL sit within a
tolerance derived from bf16's 8-bit mantissa: per-element half-ulp 2^-8,
accumulated over the artifact's own tensor magnitudes per the design.md
formula — instantiated on the committed fixture as 2^-4 times
max(1, max fp32 logit). The float64 1e-12 parity tolerance SHALL NOT
govern bf16 comparisons.

#### Scenario: RD-3.1 — Fixture decode within the bf16 bound
- **WHEN** the bf16 twin fixture and its fp32 original decode the same
  token ids through the engine
- **THEN** the max absolute logit delta is within 2^-4 times
  max(1, max fp32 logit) — the bound computed from the fixture's own
  loaded tensors — and the max delta is nonzero, proving the bf16
  remainder was actually exercised
