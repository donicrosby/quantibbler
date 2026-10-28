# Proposal: qk-artifact-blob

## Why

The qpk1 artifact is currently a model directory (`config.toml` + `manifest.toml` +
per-layer qpk1 streams + remainder sidecars + tokenizer files). Distribution means
shipping a directory tree — hard to version, hard to checksum, and impossible to
mmap. Per @user's 2026-10-25 directive, the qpk1 container becomes a single-file
blob: one model part to download, GGUF-style header + section table + aligned
payloads, so the server can mmap rather than slurp.

## What Changes

1. **Single-file container (qk-representation).** A blob file begins with the
   `QKPB` magic (4 bytes) + u32 LE format version 1 + a length-prefixed UTF-8
   JSON header carrying every declared metadata field (model identity, geometry,
   norm inventory, weight_tying, the qk-artifact-dtype remainder dtype, and the
   ordered layer-name list). A binary section table follows — self-delimiting
   entries of `(name, dtype tag, offset, length)` — listing every payload section.
   Payloads are 4096-byte-page-aligned for mmap. Layer streams are carried
   verbatim (their own `QPK1` binary wire); remainder tensors are raw IEEE
   little-endian at the declared dtype; tokenizer files are stored as raw
   sections. No change to the per-layer stream.
2. **Deterministic bytes.** Fixed section order, zero padding, no wall-clock
   timestamp in the header. Pack twice → identical blob bytes (whole-file sha256
   equality).
3. **Loader fail-closed (qk-server).** Bad magic, unknown version, truncated
   header, truncated section table, offset+length u64 overflow, overlapping
   sections, misaligned offsets, sections aliasing the header/table region,
   trailing bytes, unknown dtype tags or section names, duplicate section
   names, section-tag/declaration disagreements, missing required sections
   → all named refusals with the failing artifact named. Dual-read: the
   loader detects `.qpk` blob vs legacy directory; directory format stays
   readable (deprecated).
4. **Writer + repack legs (qk-ptq).** The exporter emits the blob by default
   (`--container-format blob`); `tools/repack_remainder.py` gains dir→blob
   and blob→blob (dtype-conversion) legs.
5. **Both faces.** Rust loader (`crates/quantibbler-qpk1` + server) and host
   writer (`tools/quantize_hf_model.py`); the parity fixture (currently a
   directory) is regenerated in blob form and committed with provenance;
   the Python face reads blob sections through the PyO3 bindings.

## Impact

- **Invariant-touching: yes — the container format.** The per-layer qpk1
  stream is untouched. The blob is a new wrapping container (its own magic
  `QKPB`, distinct from the stream magic `QPK1`); the directory format is
  not retired in place (v6 artifact untouched; repack produces new artifacts).
- **Affected capabilities**: `qk-representation` (1 MODIFIED + 7 ADDED: the
  blob format definition), `qk-server` (4 ADDED: blob load + dual-read +
  validation + mmap), `qk-ptq` (4 ADDED: blob export + repack legs +
  deterministic output).
- **Composes with qk-artifact-dtype**: the remainder dtype declaration
  (`remainder_dtype` + `weight_tying` suffix) rides in the blob JSON header
  rather than directory TOML manifests; the agreement rule, vocabulary, and
  loader semantics are identical.
- **Affected code (implementation)**: `crates/quantibbler-qpk1/src/` (blob
  parser), `crates/quantibbler-server/src/model/` (blob load path),
  `tools/quantize_hf_model.py` (blob export), `tools/repack_remainder.py`
  (new legs), `tools/gen_server_fixtures.py` (blob fixture), committed blob
  fixture, PyO3 bindings.

## Paper anchor

No method change: no solver, stage, budget, or geometry surface is touched.
The container format is a distribution decision, not a method element. The
mmap alignment dimension (4096) is justified from the host's page size, not
a paper figure.

## Non-goals

- No rewrite of the v6 directory artifact in place; repack produces new
  artifacts.
- No change to the per-layer qpk1 stream (qk-representation stream
  requirements untouched; layer streams are carried verbatim in blob
  sections).
- No layer-stream dtype change (qk-kernels-dtype-generic owns kernel
  genericity).
- No other remainder dtypes beyond fp32/bf16 (qk-artifact-dtype scope).
- No server-side serving semantics beyond the container load path
  (qk-serve-bf16 owns the serving dtype path).
- No re-fire of the real-model run; no in-place repack.
- No GPU/device info in the container (the blob is a portable serialized
  artifact).
