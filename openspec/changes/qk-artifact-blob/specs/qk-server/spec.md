# qk-server delta: qk-artifact-blob

## ADDED Requirements

### Requirement: The server SHALL accept a single-file blob as model source

The server SHALL accept a `.qpk` single-file blob as its model
source. When the model path names a regular file, the loader SHALL
detect the blob format by its `QKPB` magic and load via the blob
path; a regular file without the magic SHALL refuse; when the path
names a directory, the loader SHALL fall back to the legacy directory
format (dual-read). The server SHALL report readiness only after the
blob passes every validation gate.

#### Scenario: BL-1.1 — A blob file is detected and loaded

- **WHEN** the server is started with a `ModelDirectory::load()` call
  targeting a `.qpk` file whose first four bytes are `QKPB`
- **THEN** the loader parses the blob header, validates the section table,
  constructs the `ModelDirectory`, and reports ready with the model's
  declared identity.

#### Scenario: BL-1.2 — A directory path still loads

- **WHEN** the server is started with a path that is a directory
  containing `manifest.toml` and a valid model
- **THEN** the legacy loader path executes and the server reports ready —
  the blob change does not break existing directories.

#### Scenario: BL-1.3 — A regular file without the magic refuses

- **WHEN** the model path names a regular file whose first four bytes
  are not `QKPB`
- **THEN** the loader refuses with a diagnostic naming the path — it
  never attempts a directory load or a partial parse.

### Requirement: The loader SHALL validate every blob section against the header metadata

The loader SHALL validate every blob section against the header
metadata before constructing any tensor. Layer-section byte lengths
SHALL match the stream's own declared size; remainder-section byte
lengths SHALL equal the inventory-derived float count (declared
norm_slots inventory, N-1.0 — never a family formula) times the
declared itemsize; every required section SHALL be present. Any
violation SHALL yield a named diagnostic naming the section and the
expected value.

#### Scenario: BL-2.1 — All required sections present

- **WHEN** a blob carries section entries for every declared layer, the
  embedding, norms, tokenizer.json, and (when present) an untied
  lm_head, each at the correct byte length for its dtype and geometry
- **THEN** the loader constructs a complete `ModelDirectory` with no
  errors.

#### Scenario: BL-2.2 — A missing required section refuses

- **WHEN** a blob's section table omits any declared layer name, the
  `"embedding"` entry, the `"norms"` entry, or the `"tokenizer.json"`
  entry
- **THEN** the loader refuses with a diagnostic naming the missing section.

#### Scenario: BL-2.3 — A wrong-length remainder section refuses

- **WHEN** a blob's `"embedding"` section has a byte length that is not
  `vocab_size * d_model * itemsize(remainder_dtype)`, or the `"norms"`
  section's length does not match
  `(per-block inventory count * n_layers + d_model) * itemsize(remainder_dtype)`,
  where the per-block count sums the declared `norm_slots` widths
  (`input`/`post_attention`/`post_feedforward` → `d_model`,
  `q_norm`/`k_norm` → `head_dim`) per N-1.0
- **THEN** the loader refuses with a diagnostic naming the section, the
  expected byte count, and the actual byte count.

### Requirement: The remainder dtype SHALL be read from the blob header only

The remainder dtype SHALL resolve from the header's `remainder_dtype`
field only. Remainder section tags SHALL equal the declared dtype
(fp32_le↔fp32, bf16_le↔bf16); layer section tags SHALL agree with the
stream's own version; either disagreement is a named refusal.
`weight_tying` SHALL be required: a `lm_head=<path>@{fp32,bf16}`
string, the sentinel, or null (untied head, no suffix); a carried
suffix SHALL agree with `remainder_dtype`, and null SHALL read the
head at the declared dtype.

#### Scenario: BL-3.1 — The declared dtype and weight_tying suffix agree

- **WHEN** a blob header declares `"remainder_dtype": "bf16"` and its
  `weight_tying` value carries a `@bf16` suffix
- **THEN** the loader resolves bf16 for all remainder sections and loads
  the model.

#### Scenario: BL-3.2 — A suffix disagreement refuses

- **WHEN** the header's `remainder_dtype` and `weight_tying` suffix do
  not match (either direction)
- **THEN** the loader refuses with a diagnostic naming both declarations.

#### Scenario: BL-3.3 — A remainder tag disagreeing with the header refuses

- **WHEN** a remainder section's dtype tag does not match the header's
  `remainder_dtype` (e.g. `remainder_dtype: fp32` with an `embedding`
  tagged `bf16_le`)
- **THEN** the loader refuses with a diagnostic naming the section, its
  tag, and the declared dtype.

#### Scenario: BL-3.4 — A layer tag disagreeing with the stream version refuses

- **WHEN** a layer section's tag names a qpk1 stream version that
  differs from the version its own stream header declares (e.g.
  `qpk1_v2` over a stream whose header declares version 1)
- **THEN** the loader refuses with a diagnostic naming the section, its
  tag, and the stream's declared version.

#### Scenario: BL-3.5 — A null weight_tying reads the head at the declared dtype

- **WHEN** the header's `weight_tying` is null and `remainder_dtype` is
  `"bf16"`
- **THEN** the loader reads the `"lm_head"` section at bf16 like the
  other remainder sections, and loading proceeds.

### Requirement: The loader SHALL mmap the blob with alignment validation

The loader SHALL map the blob file with mmap (or, on platforms without
mmap, a page-aligned read) rather than slurping the whole file into a
buffer. The 4096-byte alignment of every section offset SHALL be
asserted at load before any mmap call. Section payloads SHALL be
accessed through mapped views or page-aligned copies; the section table
itself SHALL be read into memory.

#### Scenario: BL-4.1 — A well-formed blob maps and loads

- **WHEN** a valid blob is presented to the loader
- **THEN** the file is opened, every section offset is verified
  4096-aligned, mmap is requested, and at least the layer-stream
  sections are accessed through the resulting mapped views.

#### Scenario: BL-4.2 — A misaligned blob refuses before mmap

- **WHEN** any section offset in a blob's table is not a multiple of
  4096
- **THEN** the loader refuses with a diagnostic naming the section and
  its offset — it never attempts to mmap the file.
