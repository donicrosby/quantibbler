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
- `crates/quantibbler-server/` — pure-Rust OpenAI-compatible server
  (planned: see openspec/changes/qk-server)
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

## Method sources

Methods are implemented from the pinned papers in `docs/papers/`; cite the
paper section, not a third-party repository.
