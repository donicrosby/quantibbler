# qk-serialization — host-side artifact schemas as versioned pydantic models

## Why

The repo persists five host-side artifacts as hand-rolled `dict` → `json`:
the quantize report, acceptance-ledger rows, solver-probe JSONL rows,
stage-journal entries, and launch-attribution records. Each shape lives in a
different place — a writer function, a module docstring, a format note, or a
shape table — so no reader can ask "is this artifact well formed?" without
re-deriving the answer from the producer. @user asked (2026-10-25) for the
migration to pydantic; this change is its foundation: the dependency, the
models, and their tests. Producer adoption follows in separate cards so the
two files reorganized in parallel (`test_qk_report.py`, `block_recon.py`)
are adopted in their post-split layout instead of being churned twice.

## What Changes

- Add `pydantic` (v2) as a core dependency of the Python compressor package.
- New module `src/quantibbler/serialization.py` holding one model per surface.
  Each docstring cites the spec section or change doc that already owns the
  shape, so the models are the executable form of an existing contract.
- `schema_version` on every persisted model, with a named-error refusal for an
  unknown version and a load-as-current reading for a pre-versioning artifact.
- Roundtrip + reject tests per model (torch-free, so they run in the lean
  environment).

## Non-goals

- **qpk1 container bytes** (packed bitplanes, scales, norm sidecars): a
  cross-implementation byte contract with the Rust loader, untouched.
- **`config.toml`**: the Rust fail-closed loader is its consumer.
- **Cross-face fixture JSONs** (byte-exact parity vectors) and anything the
  Rust side reads.
- **No producer wiring**: this change defines and tests the models; the report,
  ledger, probe, journal and launch writers keep emitting exactly the bytes
  they emit today until their adoption cards land.
- **No format change**: not touching the kernels contract or the storage
  format, so this change is not invariant-touching.

## Impact

- `pyproject.toml` + `uv.lock`: `+pydantic>=2.14,<3` (and its Rust-wheel backend
  `pydantic-core`).
- `src/quantibbler/serialization.py` (new), `tests/test_serialization.py` (new).
- No emitted artifact bytes change; no consumer changes.
