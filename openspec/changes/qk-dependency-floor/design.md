# Design: qk-dependency-floor

## Context

The ruling being implemented (@user via the orchestrator's dependency card,
2026-10-25): *"If we NEED torch et al to run unit tests, they shouldn't be an
extra (or any soft gating). An extra is strictly for extra parts of the program
YOU DON'T NEED."* The PTQ pipeline is the product; the suite includes its tier.

## Decision 1 — where the pipeline's dependencies live

**Decision: `[project].dependencies`** (the product's own declared set), not
`[dependency-groups].dev`.

The card's scope bullet said "dev dependency group". That placement is not
implementable without violating two contracts of record:

1. `openspec/config.yaml` (Component decomposition) defines the Python package
   as the COMPRESSOR/TRAINER: the PTQ pipeline is what it is *for*. A declared
   dependency set of `numpy` alone cannot run its own pipeline, so a dev-only
   home leaves the product's runtime deps undeclared — the same "needed part
   living somewhere optional" the ruling removes.
2. The run images install the PRODUCT without dev tooling and then assert it
   works: `uv sync --locked --no-dev --no-editable --all-extras` followed by
   `python -c "import torch, transformers; ..."` in `Dockerfile.ptq:53-66` and
   `Dockerfile.ptq-rocm:73-104`, and `Dockerfile.ptq-rocm-official:57-78`
   exports with `--no-dev --all-extras` and asserts
   `import torch, transformers` at build time. With the deps in a dev group those
   builds fail *by construction* (the assertion is fail-closed, so the images
   could not ship at all), and `pip install quantibbler` would yield a
   compressor that cannot quantize.

Because they are product deps, `uv sync` (product deps + the default `dev`
group) is exactly what the acceptance asks for: a bare `uv sync` + bare
`uv run pytest` runs the full tier, ptq included, with no extra flag and no
skip-when-absent path.

**Delta reported, not hidden:** the card's bullet named a different table. The
invariant wins, and the deviation is recorded here, in the completion metadata
and in a card comment so the orchestrator can overrule it in one line (moving the
five requirements back into `[dependency-groups].dev` is a pure TOML move — the
`uv.lock` churn, the conftest deletion and the CI legs are unchanged by it *except*
that the run images would then need their dependency source re-thought).

## Decision 2 — the removable extra

`kernels` stays the only extra, and is justified as genuinely-optional product
surface: `src/quantibbler/kernels/__init__.py` documents the NumPy references as
the executable specification, so the Rust extension is opt-in for a product user
(and `openspec/specs/qk-kernels/spec.md` requires the Rust pytest leg to skip
cleanly when it is absent). It is ALSO added to the `dev` group, because the
suite exercises the parity tier and the acceptance forbids skip-when-absent
paths: with the extension present, a bare `uv sync` + `uv run pytest` reports
zero skips. An extra and a dependency group are not exclusive, and neither home
contradicts the other.

## Decision 3 — what replaces the lean-python CI leg

Deleting `tests/ptq/conftest.py`'s `pytest_ignore_collect` hook makes a
torch-free *product* install impossible (torch is a product dep now), so the
`lean-python` leg's two claims die together: there is no longer an interpreter
that proves the PTQ deps absent while running the suite, and the lean tier's
documented 16-skip count (the kernels Rust leg) has no tier to belong to.

**Decision:** the leg becomes `product-floor`, which carries the ruling's own
assertion instead of its inverse — a fresh venv built from the project's declared
dependencies alone must import the PTQ lane (`numpy, torch, transformers,
tokenizers, safetensors, pyarrow`) AND must NOT import the harness
(`pytest, yaml`). This is the referee that keeps the classification honest: if
torch (or any pipeline dep) moves back into an extra or a dev group, that job
reds instead of the change silently re-opening the soft gate the ruling closed.

**Discarded alternatives:**

- *Delete the leg entirely.* Discarded: the deps' classification would then be
  unrefereed (the full job installs the dev group, so it passes either way), and
  the qk-ci capability would be strictly weaker than before.
- *Keep the leg and assert the floor REFUSES the suite (rc≠0 + ModuleNotFoundError).*
  Discarded as a new control for less signal: with the skip-when-absent hook
  gone, a missing dep already fails loudly at collection in every leg, and the
  full job's `--expect-skipped 0` parse covers the hole; the positive
  classification assertion serves the ruling better.
- *Keep torch/transformers/… as an extra and add them to `dev` as well.*
  Discarded: the ruling forbids the extra ("they shouldn't be an extra"), and
  the run images' `--all-extras` would then be the deps' only justification.

## Implementation crystallization (2026-10-25)

Measured while implementing, so none of the above is assumed:

- Live-registry floors (PyPI JSON, 2026-10-25): torch 2.14.1, transformers
  5.20.2, tokenizers 0.23.3, safetensors 0.8.0, pyarrow 26.0.0, numpy 2.5.4,
  pytest 9.1.1, pytest-xdist 3.8.0, pre-commit 4.7.0, maturin 1.15.0, pyyaml
  6.0.3. Under the freshness rule (one major, or one minor for 0.x) the stale
  floors were pyarrow `>=18` → `>=25` (its `<24` cap also excluded the current
  release outright), tokenizers `>=0.21` → `>=0.22`, safetensors `>=0.5` →
  `>=0.7`. torch and transformers were already within one major and were left
  alone. `uv lock` re-resolved exactly one version: pyarrow 23.0.1 → 26.0.0.
- The product-floor job's two assertions were rehearsed in the sandbox against
  real environments (a `uv pip install -e .` venv and the shared dev venv), not
  written from the docs — see `tasks.md` for the recorded runs.
- Delta mechanics, measured: a `## MODIFIED Requirements` block replaces the
  whole requirement, and `openspec validate qk-dependency-floor --strict` refuses
  one that omits scenarios the current spec still has (`MODIFIED "CI SHALL
  execute the full suite, never a skipped skeleton" omits scenario(s) ... archive
  refuses to drop them`). The scenario loss is deliberate here, so the requirement
  is REMOVED (full original block, kept for audit) and re-ADDED under the new
  name; the map between the two names is this paragraph.
- `validate --specs --strict` is 6 passed / 1 failed at this revision, and NOT
  because of this change: `spec/qk-ci`'s abort-leg requirement is 591 characters,
  over the 500-character limit, and was pushed there by `b3ed881` (the abort-leg
  600s→3600s backstop landing of 2026-10-25). The acceptance's "strict validate
  7/7" is therefore unattainable on this trunk without trimming a landed SHALL
  from another lane; it is reported rather than fixed here (Open Question OQ-2).
- The lock keeps torch on the `pytorch-cpu` index (`2.14.1+cpu`), now reached
  through an unscoped `[tool.uv.sources]` entry because torch is a product dep
  rather than an extra's member; the ROCm images keep replacing that wheel with
  an AMD-index install afterwards, which is unchanged.
- **Environment hazard, recorded:** `UV_PROJECT_ENVIRONMENT=/opt/venv` is
  exported in this box's sessions, so a bare `uv sync` targets the SHARED venv
  no matter which worktree it runs in — the worktree law's expectation ("uv sync
  gives the worktree its own .venv") does not hold as written. Worktree-local
  runs therefore assert `quantibbler.__file__` explicitly and use
  `uv run --no-sync`; the hazard belongs to the env-law card, not to this change.
- The ptq tier's determinism pins moved from the deleted hook to conftest
  import time (`_pin_determinism()` at module scope, beside the existing
  OPENBLAS assignment), so the thread count is still fixed before the first
  reduction the divergence gate reads.

## Open questions

- **OQ-1 (owner: planner).** `openspec/specs/qk-ptq/spec.md` phrases three
  scenarios around "the suite runs in a lean environment without torch"
  (`abort_report_construction_lean_tested`, `probe_math_lean_tested`,
  `verdict_helpers_torch_free_tested`) and cites "the repo's tier conftest
  pattern". Their substance survives — the covered helpers are proven torch-free
  by subprocess import (`tests/test_qk_report.py::test_verdict_helpers_torch_free_tested`
  asserts `'torch' not in sys.modules`) — but the *configuration* they name is
  gone with this change. Rewording a folded SHALL is a spec change, so it is
  reported rather than done from inside this change's delta. Status: OPEN.
- **OQ-2 (owner: orchestrator, routing to the abort-leg lane).** `validate
  --specs --strict` is 6 passed / 1 failed on this trunk (pre-existing, see the
  crystallization entry): `spec/qk-ci`'s abort-leg requirement is 591 characters
  after `b3ed881`. This change's acceptance names "strict validate 7/7", so the
  clause needs either that requirement trimmed/split by its own lane or the
  acceptance restated. Status: OPEN.

## Informed assumptions

- **The run images do not need the harness.** `--no-dev` in every Dockerfile
  plus the DEV=1 pytest knob in `Dockerfile.ptq-rocm`. If wrong, the images would
  need a group install rather than the product floor — the product-floor CI leg
  is the assertion that would surface it first.
- **ubuntu-latest carries a Rust toolchain.** Unchanged from before this change
  (the full-python leg already synced the `kernels` extra there), now also relied
  on by the smoke/abort legs through the dev group. A runner without cargo would
  red those legs at the build step, loudly.
