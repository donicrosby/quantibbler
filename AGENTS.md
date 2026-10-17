# quantibbler — contributor notes

Spec-first repo. `openspec/config.yaml` is the authoritative statement of
architecture, method invariants, and verification gates — read it before
proposing or implementing anything.

## Layout

- `src/quantibbler/` — Python package
- `crates/quantibbler-kernels/` — Rust/PyO3 extension (uv workspace member)
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
