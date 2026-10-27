# Proposal: qk-dependency-floor

## Why

Dependency classification drifted against the ruling that an extra is strictly
for a part of the PROGRAM a product user may not need (@user, 2026-10-25):
torch, transformers, tokenizers, safetensors and pyarrow sat in the `torch`/`hf`
extras even though the PTQ pipeline IS the product, so the suite could only see
them when an extra was synced by hand and `tests/ptq/conftest.py` retired the
whole tier as "ignored" whenever they were absent — a green suite with a hole in
it, which is the exact failure the CI referee was written to refuse.

The deps are therefore the product's own declared dependencies: the run images
already install the product with `uv sync --no-dev --all-extras` and assert
`import torch, transformers` (Dockerfile.ptq, Dockerfile.ptq-rocm-official), and
`pip install quantibbler` must yield a working compressor. A dev-group home
would break both; the only remaining extra is `kernels`, which is genuinely
optional product surface because the NumPy references in
`src/quantibbler/kernels/` are the verification fixtures at the declared dtype and run where the
Rust extension is absent.

## What Changes

- `pyproject.toml`: torch/transformers/tokenizers/safetensors/pyarrow move into
  `[project].dependencies`; the `torch` and `hf` extras are deleted; `kernels`
  stays the one extra (justified) and also joins the `dev` group so a bare
  `uv sync` exercises the parity tier instead of skipping it; floors re-verified
  against the live registry (pyarrow 18→25, tokenizers 0.21→0.22,
  safetensors 0.5→0.7).
- `tests/ptq/conftest.py`: the `pytest_ignore_collect` skip-when-absent hook is
  deleted. A missing needed dependency is a sync error and fails loudly at
  collection; the determinism pins (OPENBLAS thread count, one torch thread)
  remain.
- `.github/workflows/ci.yml`: the `lean-python` leg (a torch-free suite run plus
  a skip-count assertion) becomes `product-floor` — it installs the project's
  declared dependencies alone and proves BOTH halves of the split: the PTQ lane
  imports from them, and the test harness does not. The three `uv sync` legs
  stop naming extras and sync the locked project instead.
- `openspec/config.yaml`: the dependency-classification rule is stated where the
  component decomposition lives, so "is this an extra or a product dep" is
  answerable from the config rather than from memory.
- `AGENTS.md` and the `quantibbler` skill: the env/gate command forms follow.

## Impact

- **Affected specs**: `qk-ci` — the `lean-python` requirement is REMOVED and
  re-ADDED under a new name (`CI SHALL run the full suite on the locked project
  environment and referee the product/dev dependency split`): the extras-sync
  clause becomes the locked-sync clause, and the two lean-python scenarios die
  with the requirement they belonged to, because the configuration they named — a
  torch-free product floor running the suite — no longer exists.
- **Affected code**: `pyproject.toml`, `uv.lock`, `tests/ptq/conftest.py`,
  `.github/workflows/ci.yml`, `tests/test_ci_referee.py`, plus live
  command-form docstrings in `tools/quantize_hf_model.py`,
  `tools/lane3_reg_sweep.py` and `tests/ptq/test_launch_attribution.py`.
- **Invariant-touching (dependency policy)**: the shipped dependency set
  changes and `uv.lock` churns. Compatible in the sense that matters — the run
  images, `pip install quantibbler` and the CI legs all keep resolving the same
  wheels, and no qpk1 byte contract is touched.
- **Not touched**: no kernel, server, qpk1 or report-schema behavior; no
  accelerator leg's `integration` marker (hardware gating is not a dependency
  question) and no test assertion is relaxed.

## Non-goals

- Not a dependency *upgrade* campaign: only the floors that violated the
  freshness rule moved, and the lock resolves the same torch/transformers it did
  before (pyarrow is the one resolved version that changed).
- Not a CI re-design: the product-floor leg replaces the capability the
  lean-python leg carried (the suite's dependencies cannot be optional) and no
  other job, pin or gate is re-scoped.
- Not a packaging experiment: no new extra is invented, and the `kernels` extra
  keeps its meaning and its install path.
