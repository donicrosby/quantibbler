# Design: qk-artifact-blob

Decisions B1–B12 plus the dual-read ruling are the design-silent choices this
change resolves. Grounded facts: the current loader reads a directory
(`loader.rs` `load(model_dir: &Path)`, 148 lines of directory traversal), the
per-layer streams carry their own 24-byte header with scale dtype, and the
parent change qk-artifact-dtype declares the remainder dtype in the TOML
manifests (design D2). The blob header subsumes both.

## B1 — Blob identity: magic + version (distinct from the stream)

The per-layer stream carries magic `QPK1` (lib.rs:86). The blob SHALL carry
its OWN magic `QKPB` (qpk1-blob) so a misdirected stream parser cannot
silently interpret blob bytes as a layer. Blob format version SHALL be u32
LE 1. A reader SHALL refuse any file whose magic differs or whose version
is outside {1}. This is the same fail-closed pattern as the stream parser
(SF-1.4, SR-1.2).

## B2 — Header: length-prefixed JSON (no TOML)

The header SHALL be a u32 LE byte length followed by that many UTF-8 JSON
bytes. JSON over TOML: the server already depends on `serde_json` (the API
surface is JSON-native); JSON has no multi-line string or table-syntax
footguns; and JSON deserialization errors are precise (byte offset). The
header carries every field the loader needs without consulting a separate
config:

- `model_id`: string. Identity for `/v1/models`.
- `vocab_size`, `d_model`, `n_layers`, `n_heads`, `n_kv_heads`, `head_dim`:
  usizes. Attention geometry (qk-server N-2 validation rules).
- `eos_token_id`, `bos_token_id`: u32.
- `rms_norm_eps`, `rope_theta`: f64, finite-positive (same validation as
  config.toml).
- `norm_slots`: [string]. Declared inventory (qk-server N-1 vocabulary).
- `layers`: [string]. Ordered layer names; the section table SHALL carry
  one entry per name.
- `weight_tying`: string|null, REQUIRED. The string grammar matches the
  parent: `"lm_head=<path>@{fp32,bf16}"` (untied head) or the tied
  sentinel `"lm_head=e1@{fp32,bf16}"`. In a blob the `<path>` component is
  directory-grammar provenance only — the blob loader never resolves it and
  keys the head on the `"lm_head"` section's presence. `null` SHALL mean an
  untied head with no suffix, read at `remainder_dtype` (the blob twin of
  the directory's absent `weight_tying` line, parent RD-1.5). Absence SHALL
  refuse (required field).
- `remainder_dtype`: "fp32"|"bf16". REQUIRED — every blob is an explicitly-
  declared new artifact; there is no legacy-blob absence rule.

The header SHALL contain NO wall-clock timestamp (B5 determinism). A reader
SHALL refuse missing required fields and wrong-typed required fields; a
reader SHALL ignore unknown fields (forward-compatible). The `created_by`
field (optional string) SHALL name the producing tool, never a timestamp —
provenance, not a behaviour-driver.

## B3 — Binary section table (self-delimiting entries)

After the header region's trailing zero-padding to the next 4K boundary, a
binary section table lists every payload section:

- 4 bytes: u32 LE section count.
- Per section: 1-byte u8 name_len (max 255) + that many UTF-8 name bytes +
  1-byte u8 dtype tag + 8-byte u64 LE offset + 8-byte u64 LE length (total
  1 + name_len + 1 + 8 + 8 bytes, variable).

**dtype tag vocabulary** (closed; any other value → named refusal at load):

| tag | label      | payload                                          |
|-----|------------|--------------------------------------------------|
| 0   | qpk1_v1    | qpk1 stream, version 1 (no bias)                 |
| 1   | qpk1_v2    | qpk1 stream, version 2 (bias-carry; current)      |
| 2   | fp32_le    | IEEE binary32 little-endian                      |
| 3   | bf16_le    | IEEE bfloat16 little-endian (parent D4)           |
| 4   | raw        | opaque bytes (tokenizer JSON, etc.)               |

Section-name vocabulary is closed: declared layer names, the fixed
remainder names (`"embedding"`, `"lm_head"`, `"norms"`), and
`tokenizer.*`. Any other name SHALL be refused at load (BF-3.7). The
exporter copies only `tokenizer.json` (the pinned `tokenizers` crate
is JSON-BPE); `vocab.json`/`merges.txt`-style files are never emitted.

Cross-checks pinned in qk-server BL-3: every remainder section's tag
SHALL equal the header's `remainder_dtype`, and every layer section's
tag SHALL agree with the stream's own declared version — disagreement
is a named refusal in both directions.

The brief's section-table grammar listed a per-section `shape` field.
Deliberately dropped: shapes derive from the header geometry (remainder
tensors) or are self-describing (layer streams carry their own header);
a stored shape would duplicate declarations and invite disagreement.

BF-7 hosts the payload-region rules (metadata disjointness from the
header/table region, no trailing bytes — BLK-1/2c of the review) as
its own requirement: BF-3's body sits at the strict 500-char cap with
the entry-validation list.

## B4 — Alignment: 4096 bytes (page size, not SIMD)

Payload SHALL start at a byte offset aligned to 4096 (the host's memory
page size on every platform the server targets — x86-64, ARM64, RISC-V).
Justification: `mmap` works from offset 0 at ANY internal alignment —
the OS maps whole pages regardless of the file offset's value, so a
64-byte or 16-byte alignment does not break mapping. What 4K-aligned
section offsets buy is page-isolated sections: each section begins on a
page boundary, so a page-granular mapping of any one section contains
only that section's bytes and can be handed to the engine as a
zero-copy view. A smaller alignment would let sections share pages,
forcing either a copy or exposure of a neighbour's bytes inside the
mapped page. 4K is the minimum alignment that guarantees per-section
zero-copy views on every target.

Cost: small models (committed fixture ~3 KB) pay up to ~50 KB with padding.
Acceptable — the fixtures are test artifacts, not distribution artifacts;
production models are tens of GB where 4K padding is noise.

## B5 — Canonical section ordering → deterministic bytes

Section entries SHALL appear in a fixed order, enforced by the writer:

1. Every layer, in the `layers` array order: per block, the seven
   projections `W_q`, `W_k`, `W_v`, `W_o`, `W_gate`, `W_up`, `W_down`.
2. Remainder: `"embedding"`, then `"lm_head"` (only when weight_tying is
   `"lm_head=<path>@..."` — an untied head), then `"norms"`.
3. Tokenizer: `"tokenizer.json"` and any auxiliary `"tokenizer.*"` files
   in name-lexicographic order.

Plus B2 (no timestamps) and B4 (zero padding) → every byte in the blob is
either content-deterministic or pad-zero. The whole-file sha256 is therefore
deterministic: pack twice, hashes equal.

## B6 — Fingerprint: external sha256 (no self-referential hash in the container)

The blob header DOES NOT carry a self-referential hash of the blob. The
section table provides structural integrity (truncation and overlap
detection). A consumer that needs a content hash computes `sha256(blob)`
externally; the writer records that hash in the quantize report (pinned
in qk-ptq BW-1.3: the report carries the emitted blob's sha256). A
self-referential hash inside the container would create a write-then-seek
cycle that complicates the writer and would need an exclusion zone — no
gain over external hashing. The deterministic-bytes contract (B5) guarantees
the same inputs produce the same hash every time.

## B7 — Dual-read vs migrate: dual-read, directory deprecated

**Decision: dual-read** — the loader accepts both blob (`.qpk` file) and
directory, auto-detected at load time. Directory format is documented as
legacy/deprecated; the blob is the canonical format going forward.

**Fixture-cost analysis (the decision criterion):** Dropping directory
support would force regeneration of every committed fixture (server-model,
bf16 twin) into blob form AND deletion/rewrite of every directory-loading
test — mere churn with no correctness benefit. Keeping dual-read lets
fixtures migrate incrementally and preserves the ability to load old
artifacts during the transition. The directory parser is isolated
(`loader.rs` `load()` — one entry point) and the blob parser adds a new
`load_blob()` path sharing the same `ModelDirectory` struct; the cost of
dual-read is one dispatch branch at the top of `ModelDirectory::load()`.

**Deprecation:** The directory format is NOT deleted in this change; the
spec delta ADDED requirements note the blob as canonical and the directory
as legacy. A future change may remove directory support after blob
stabilizes — that future change is out of scope here.

## B8 — Remainder dtype declaration in the blob header

The parent change's D2 places the remainder dtype in the directory's
`manifest.toml [sidecar] dtype` key and `config.toml weight_tying` suffix.
In the blob, both declarations live in the JSON header as `remainder_dtype`
and `weight_tying`. The agreement rule (parent D2: suffix SHALL equal
declared dtype) is reproduced in the blob loader, extended with two tag
cross-checks (BL-3): remainder tags SHALL equal `remainder_dtype`, layer
tags SHALL agree with the stream's own version. An untied head with a bf16
remainder uses `"lm_head"` as a bf16 section entry; the `weight_tying` path
component is provenance only (never resolved in the blob), and the
`remainder_dtype` field declares the itemsize. The parent's "every new
artifact carries the explicit dtype" rule holds: the blob's
`remainder_dtype` is REQUIRED, never absent.

## B9 — Tokenizer: raw sections

Tokenizer files (`tokenizer.json` — a JSON BPE vocabulary, consumed by the
`tokenizers` crate at server load) SHALL be stored as a `"tokenizer.json"`
raw section. The `tokenizer.json` section SHALL be required: the server
tokenizes immediately after load, and a blob without it would die at first
tokenize, so BL-2.2 refuses it. Any auxiliary tokenizer config files
(`tokenizer_config.json`, `special_tokens_map.json`, etc.) SHALL be stored
as additional `"tokenizer.*"` raw sections. The loader SHALL extract the
JSON bytes and hand them to the tokenizer crate; no parsing or validation
of tokenizer content is the loader's job beyond the structural blob checks.
The tokenizer sections SHALL follow the remainder in canonical order (B5).

## B10 — PyO3 face: blob reading for Python parity

The `quantibbler-kernels` crate's PyO3 face does NOT own blob reading
(the qpk1 crate does). The `quantibbler-qpk1` crate SHALL expose a
`BlobReader` / `parse_blob` function through its OWN PyO3 bindings so
Python tests can validate blob round-trips and cross-face byte identity.
The Python updater's `qpk1.py` stays stream-only (it reads individual
layer streams); blob operations go through the PyO3 bindings of the qpk1
crate, keeping the single-implementation invariant (one reader per
language, and the Rust reader is the canonical one for the blob format).

## B11 — Writer: blob export, default container format

The host exporter (`tools/quantize_hf_model.py`) SHALL gain a
`--container-format {blob,dir}` knob; the default SHALL be `blob` (the
canonical format). The directory path remains available for
testing/compat. The blob writer: (a) serializes header JSON with all
metadata fields; (b) writes per-layer qpk1 stream bytes verbatim (the
streams are already deterministic); (c) writes remainder tensors at the
declared dtype; (d) copies tokenizer files as raw sections; (e) computes
offsets per configuration; (f) writes section table; (g) pads to 4K;
(h) emits the whole file in one ordered pass.

## B12 — Repack legs

`tools/repack_remainder.py` gains two new legs (composing with the existing
dir→dir fp32→bf16 dtype conversion):

1. **dir→blob**: `--in <dir> --out <file.qpk> --container-format blob`.
   Reads the directory artifact, constructs the blob (header + section table
   + aligned payloads), emits. Non-remainder files are carried verbatim
   (stream bytes, tokenizer files). The remainder dtype stays as declared
   in the source directory.

2. **blob→blob (dtype conversion)**: `--in <file.qpk> --out <file.qpk>
   --remainder-dtype bf16`. Reads the blob, converts the remainder sections
   from fp32 to bf16 through the shared cast helper (parent D4), rewrites
   the header's `remainder_dtype`, recomputes section offsets/lengths,
   emits a new blob. Layer sections and tokenizer sections are
   byte-identical.

The existing dir→dir leg (parent D7) stays; the `--container-format`
knob selects the output format across all legs.

## Discarded alternatives

- **Embedded self-referential hash**: discarded — B6; structural
  integrity from section table is sufficient, and the writer becomes
  simpler (single-pass, no seek-back).
- **64-byte alignment**: discarded — B4; justified above: section
  isolation, not inability to mmap, is what 4K alignment buys.
- **16-byte alignment (SIMD)**: discarded — same reason; the serving
  path's memory layout is the engine's job (qk-serve-bf16), not the
  container's.
- **TOML header**: discarded — B2; JSON is already in the server's
  dependency tree (serde_json + axum), parsing is simpler, and the
  error format is more precise.
- **Fixed-width binary header instead of JSON**: discarded — a fixed
  struct cannot carry forward-compatible fields or optional entries
  without version bumps; JSON is self-describing.
- **Drop directory support**: discarded — B7; fixture-cost analysis
  shows pure churn.
- **Per-section alignment rather than page alignment**: discarded —
  4K is universal; per-platform tune (2M hugepages, 16K aarch64) is
  a premature optimisation with no measured benefit.
- **Section table in JSON rather than binary**: discarded — a JSON
  table with every section's name/dtype/offset/length would be a
  large array in a variable-length encoding; a binary table is fixed
  per entry and directly addressable.
- **Plain concatenation with an index at the end (tar-like)**: discarded
  — scanning to EOF for the index defeats mmap for random access.

## Open Questions

None open. Every design choice above is resolved with evidence (B1–B12).
One execution-time branch: the committed blob fixture is regenerated from
the current server-model fixture directory (word-aligned bitplanes
already stable under the bias-carry v2 wire). The fixture's blob form
replaces the directory form in the committed test data; provenance records
the conversion and the source-directory sha256.

## Implementation crystallization

### 2026-10-26 — the section table's start is DERIVED, never stored

B3 said the table follows "the header region's trailing zero-padding to the
next 4K boundary" but did not say whether the padding length is a field. It is
not: the reader computes `table_start = next_page(12 + header_len)`, so a
header length cannot describe two different tables and BLK-1's aliasing class
(BF-7.1) is a rule about offsets rather than a second declaration that could
disagree with the first. The parser checks the bytes in
`[header_end, table_start)` are zero (BF-4.2), then reads the table, then walks
the payload region's GAPS (table end → first section, between sections, last
section end → EOF) asserting each is zero and the last is empty (BF-7.2). The
walk is over ranges, never over a materialized buffer, so validating a tens-of-
GB blob costs no scratch memory.

### 2026-10-26 — alignment is 4096 (page), and the mmap is the load path

B4's choice stands as written: 4096 is the minimum alignment that gives each
section a page-isolated view, and the acceptance for it is executable — every
`offset` in the committed fixture's table is a multiple of 4096
(`offsets_are_4096_aligned_bf4_1`) and a blob with one misaligned offset is
refused by name before any map is requested — the parser reads the magic,
header, and section table from the file with pread, asserts every offset's
4096-alignment, and only then maps the file (`misaligned_blob_refused_bl4_2`;
the ordering is pinned by `misaligned_blob_refused_before_any_mmap`, which
injects an mmap probe and asserts it was never called).
The loader opens the file and maps it (`memmap2`, BL-4); section payloads are
sliced from the map and copied only where a value is widened to f64.

### 2026-10-26 — dual-read, and where the "tied head" is decided in a blob

B7's dual-read decision is implemented at one dispatch point:
`ModelDirectory::load(path)` stats the path, runs the directory loader for a
directory, and for a regular file reads four bytes and takes the blob path only
on `QKPB` — a regular file without the magic is refused with its path named and
never falls back to a directory attempt (BL-1.3). The legacy directory loader
keeps its name and its bytes; the blob path composes the SAME
`ModelDirectory`, so a directory and the blob packed from it load
field-for-field identically (pinned in `loads_blob_file_bl1_1_bl2_1`).

The head, though, is decided differently in a blob than in a directory: B2 says
the blob loader "keys the head on the `lm_head` section's presence", so a
`lm_head` SECTION means an untied head with its own bytes and its absence means
the head IS the embedding. This resolves the ambiguity B2 and BL-2.2 leave
between "section present" and "the header DECLARES an untied head" (a JSON null
or a non-sentinel path): the section decides, and a header that declares an
untied head while shipping no section is refused as a missing required section
(BL-2.2). The tied sentinel `lm_head=e1@…` names the embedding, so a blob whose
header carries it ships no `lm_head` section — exactly what the committed
fixture does.

### 2026-10-26 — the host writer is one module; Python reads through the Rust face

`src/quantibbler/blob.py` is the only Python code that knows the blob layout
(B11): the exporter, the fixture generator and the repack utility all compose
`pack`/`write_blob_from_directory`. Its dtype codec import is function-local,
so packing a directory whose remainder is already at its declared dtype costs
no numpy/torch import (the fixture generator packs verbatim bytes). READING is
the Rust reader through the qpk1 crate's own PyO3 face (B10), built by maturin
from `crates/quantibbler-qpk1/pyproject.toml` and consumed as a dev dependency
so `uv sync` yields `quantibbler_qpk1` (a missing one is a sync error, not a
skipped tier). There is no second blob parser.

### 2026-10-26 — the driver keeps its working directory and emits the blob beside it

B11 asks the exporter for a `--container-format {blob,dir}` knob with `blob`
default; it does not say whether the working directory the report lives in
survives. It does: the driver's export step writes the directory as before and
then, in blob mode, packs it to `<out>.qpk` (the distribution artifact) and
records `{"format", "path", "sha256"}` under the report's `container` key. The
directory is the build scratch (`quantize_report.json`, checkpoints,
breadcrumbs live there) and every existing driver assertion reads it, so
deleting it in blob mode would be churn with no correctness gain; the
single-file blob is what a consumer downloads. The hash is recorded OUTSIDE the
container (B6), never as a header field.

### 2026-10-26 — the repack knob: the input decides the container, the dtype defaults split

B12's `--container-format` selects the OUTPUT; absent, the container follows
the INPUT (a blob source is never written back as a directory — that would be a
silent downgrade). `--remainder-dtype` absent means the tool's historical
`fp32 -> bf16` for a DIRECTORY output (the legacy dir→dir leg, whose RDP
battery is untouched) and the source's DECLARED dtype for a BLOB output — a
dir→blob repack carries the remainder as declared, and blob→blob converts only
when asked (an ask that names the dtype already declared is refused, not
silently a no-op).

### 2026-10-26 — the fixture is committed in both shapes

`tools/gen_server_fixtures.py` now also emits `tests/fixtures/server-model.qpk`
from the committed directory, with `created_by = "quantibbler-gen-server-fixtures"`
(constant provenance, never a timestamp — BF-2.2). The directory stays
committed beside it so the dual-read path keeps a fixture of each shape and the
two loads are compared field-for-field in the Rust suite.

## Review revision (2026-10-26)

Spec review t_6bd9f411 (4 blockers, 7 should-fixes, 5 nits) resolved. Char-budget-forced
placements documented:

- **BF-7** hosts metadata-disjoint and trailing-bytes rules — BLK-1's
  region-aliasing and BLK-2c's trailing-garbage did not fit into BF-3's body
  under the 500-char ADDED-requirement cap; BF-7 is the payload-region
  integrity requirement.
- **BL-3** carries dtype-tag cross-checks (BLK-4) and weight_tying null
  pinning (BLK-3) in a rewritten body ≤ 493 chars; the body drops the
  non-normative trailing sentence about directory-manifest reproduction.

All resolved per the review's line-level findings. Original patch sha256
6f482e59f16ade95e3e18fa4b32f1fcd1a45eeb028795e918dab62cce65f57f8.

### 2026-10-26 — BL-4.2 ordering: the alignment assert runs on a pread prefix

The first implementation parsed from the mapped bytes, so `loader.rs` called
`Mmap::map` before `Blob::parse` and the design's "refused by name before any
map is requested" was false. The fix phase-splits `Blob::parse`: the prefix
phase reads the magic, header JSON, header padding, and section table through a
`Read` (the loader passes the open file, so the reads are pread), validates
every metadata rule — including BL-4.2's 4096-alignment — and returns the
`Blob`; the payload phase, `Blob::validate_payload`, checks section bounds,
inter-payload zero gaps, and trailing bytes on the map. The loader calls the
prefix phase first, refuses a misaligned blob before requesting the map, and
maps only after the prefix passes. The ordering is executable:
`misaligned_blob_refused_before_any_mmap` injects an mmap probe and asserts it
was never called; a regression that maps first makes the probe count 1 and
fails. The PyO3 face (`python.rs`) uses the same order. `Blob::parse_bytes`
preserves the whole-slice entry point for callers that already hold the file
(the tests).
