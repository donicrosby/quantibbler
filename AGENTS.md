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
- The openspec CLI is invoked via `npx --yes @fission-ai/openspec` (or
  `openspec` if globally installed); verify with `npx --yes @fission-ai/openspec --version`.
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

## Comment law

A comment earns its line only if it explains WHY: a non-obvious constraint, a
gotcha, or a formula anchored to the pinned paper (paper-section citations
stay — see `openspec/config.yaml`). It dies if it narrates provenance or
process (change ids, ticket/card ids, "user stated/wants/requires", ruling
narration, incident post-mortems), restates the code beside it, or echoes a
spec file. Strip the citation, keep the substance: the durable rule and its
measured numbers stay, the run ledger does not. Enforced at review; the suite
gates behavior, not prose.

## Environment law

uv owns the environment. There is one `.venv`, created and refreshed by
`uv sync`; bespoke venvs are slop and are deleted on sight. Never `pip` into a
second environment, never run bare `python`. Disposition: `.venv-bh` (5.6 GB)
and `.venv-suite` (1.2 GB) were deleted 2026-10-25 — zero references in any
tracked file, zero live holders.

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

- `uv sync` — create/refresh the environment: the product's own dependencies
  (numpy plus the PTQ pipeline's torch/transformers/tokenizers/safetensors/
  pyarrow) and the default `dev` group (pytest, pytest-xdist, pre-commit,
  maturin, pyyaml, and the `quantibbler-kernels` extension — which is also the
  one product extra, since the NumPy references run where it is absent); also
  builds the Rust extension via maturin (workspace member). A bare `uv sync` +
  bare `uv run pytest` runs the FULL suite: there is no extra to remember and no
  tier that soft-skips when a needed dependency is missing — a missing one is a
  sync error that fails loudly at collection.
- `uv run pytest` — test suite.
- `uv run maturin develop --manifest-path crates/quantibbler-kernels/Cargo.toml` — rebuild the extension after editing Rust code.
- `UV_PROJECT_ENVIRONMENT=/opt/venv` is exported on this box, so `uv sync` /
  `uv run` target that SHARED venv from every worktree — a worktree does NOT get
  its own `.venv`. Prefix `UV_PROJECT_ENVIRONMENT=<worktree>/.venv` for an
  isolated one, or use `uv run --no-sync` with `PYTHONPATH=<worktree>/src` and
  assert `quantibbler.__file__`.

## Gates

Every implementation change runs, before landing:

1. `pre-commit run --all-files` (ruff check + format, whitespace hygiene,
   `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`)
2. `uv run pytest`
3. `uv run maturin build --manifest-path crates/quantibbler-kernels/Cargo.toml`
4. `cargo test` (in `crates/quantibbler-kernels/`)

Install the hook once per clone: `uv run pre-commit install`.

## Run launch protocol

Every container run launch (PTQ runs, GPU probes, benchmarks) is
announced in the team room BEFORE firing — the exact launch line or the
wrapper's `DRY_RUN` output — never after. Unannounced launches leave
unattributable corpses in the forensic trail: on 2026-10-20 a second
seq-2048-class tune attempt from an unknown session on this host died
at the 32g cap (kernel-provable: container `2d2d4b6df1e0`, 33,480,724 kB,
15:56:27Z) and its launcher is unestablishable post-mortem (`--rm`,
no journal, events not persisted). One launch, one announcement, one
name.

**Since 2026-10-25 that rule is enforced, not remembered.** Every fire must
carry `QKB_LAUNCH_OWNER` — plus `QKB_LAUNCH_CARD` (the owning kanban card) for a
board fire — or it does not start: `tools/run-ptq-container.sh` refuses an
ownerless launch on the host and stamps `qkb_launch_owner` / `qkb_launch_card`
as container labels beside `qkb_launch_id`, and `tools/quantize_hf_model.py`
refuses the same launch from inside the container, which is the only thing that
stops a direct `docker run`. The owner is written at fire time into the trail's
first line, the wrapper's `=== run …` header and `launch.env`, so an
unattributable corpse cannot exist by construction. A launcher holding only the
label vocabulary may pass it as env (`qkb_launch_owner` / `qkb_launch_card`);
docker never shows a container its own labels, so a direct `docker run` MUST
carry `-e QKB_LAUNCH_OWNER=…`. The incident this closes: at 18:10:32Z on
2026-10-25 container `quantibbler-ptq-run` ran a paper-faithful PTQ under launch
id `qk-pertarget-v4-20261025T1815Z` with no owner stamp, no announcement and no
GO (killed by the orchestrator at 18:31:24Z), and nothing in the launch path
could say whose run it was.

## Method sources

Methods are implemented from the pinned papers in `docs/papers/`; cite the
paper section, not a third-party repository.
