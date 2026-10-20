# quantibbler — contributor notes

Spec-first repo. `openspec/config.yaml` is the authoritative statement of
architecture, method invariants, and verification gates — read it before
proposing or implementing anything.

## Spec discipline (no drift)

- The spec is the oracle: `openspec/changes/<id>/specs/<capability>/spec.md`
  defines the testable contract for its capability. Before changing
  behavior anywhere, read the spec that owns it — requirements and
  scenarios there are contracts, not suggestions.
- Run `openspec list` before starting work. If an active change already
  covers the surface you're about to touch, work within that change;
  never start a parallel change over the same capability.
- Behavior changes land WITH their spec delta in the SAME commit — never
  ship code that contradicts or silently outpaces the spec. If
  implementation reveals an ambiguity or forces a design-silent choice,
  resolve it into the owning change's `design.md` (dated "Implementation
  crystallization" entry) alongside the code fix, and re-run
  `openspec validate <id> --strict`.
- Checkbox state is evidence: a task is checked only after the command
  that verifies it actually ran green. Never pre-check.
- Specs fold into `openspec/specs/` only via `openspec archive <id>` when
  the change genuinely completes — not by hand, not early.

## Layout

- `src/quantibbler/` — Python package (compressor/trainer)
- `crates/quantibbler-kernels/` — Rust kernels; rlib for the server, PyO3
  face for Python (uv workspace member)
- `crates/quantibbler-qpk1/` — qpk1 container crate: stream reader +
  hardened model-directory loader (path containment, stat-before-read,
  checked arithmetic, fail-closed config). Consumed by the server; the
  Python face is `src/quantibbler/qpk1.py` (same byte contract).
- `crates/quantibbler-server/` — pure-Rust OpenAI-compatible server
  (OpenAI-compatible endpoints over qpk1 model dirs; see
  openspec/changes/qk-server)
- `openspec/` — specs and active changes
- `docs/papers/` — pinned reference papers
- `tools/` — utilities

## Tooling: uv

All Python tooling goes through uv — never pip, never bare `python`:

- `uv sync --all-groups --extra kernels` — create/refresh the environment;
  also builds the Rust extension via maturin (workspace member).
- `uv run pytest` — test suite.
- `uv run maturin develop` — rebuild the extension after editing Rust code.

## Gates

Every implementation change runs, before landing:

1. `pre-commit run --all-files` (ruff check + format, whitespace hygiene,
   `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`)
2. `uv run pytest`
3. `uv run maturin build`
4. `cargo test` (in `crates/quantibbler-kernels/`)

Install the hook once per clone: `uv run pre-commit install`.

## Run launch protocol

Every container run launch (PTQ runs, GPU probes, benchmarks) is
announced in the team room BEFORE firing — the exact launch line or the
wrapper's `DRY_RUN` output — never after. Unannounced launches leave
unattributable corpses in the forensic trail: on 2026-10-27 a second
seq-2048-class tune attempt from an unknown session on this host died
at the 32g cap (kernel-provable: container `2d2d4b6df1e0`, 33,480,724 kB,
15:56:27Z) and its launcher is unestablishable post-mortem (`--rm`,
no journal, events not persisted). One launch, one announcement, one
name.

## Method sources

Methods are implemented from the pinned papers in `docs/papers/`; cite the
paper section, not a third-party repository.
