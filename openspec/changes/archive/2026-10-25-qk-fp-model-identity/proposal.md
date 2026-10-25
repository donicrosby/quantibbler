# Proposal: qk-fp-model-identity

## Why

The resume fingerprint binds `PTQConfig` + the quantibbler version + `git describe`
— and **nothing about the model**. Two models quantized with the same configuration
therefore hash into the same fingerprint, and a checkpoint directory written for one
of them is accepted as valid for the other.

Measured on the landed bytes (`/opt/venv`, torch 2.14.1+cpu, no accelerator; probes
archived beside the ticket):

1. `config_fingerprint` is a function of the config alone — the config does not
   describe the model at all — so it is **identical for two different models**
   (`config_fingerprint identical across models: True`).
2. `check_fingerprint(ckpt_dir, cfg)` therefore **PASSES** when the directory holds
   another model's block archives (`RESULT [check_fingerprint]: PASSED -- no model
   identity in the guard`).
3. **Silent chimera, end to end.** Model B (`embed+lin1+act+head`) resumed a directory
   seeded by model A (`embed+lin1+lin2+head`) with the solver rigged to raise if
   called: the run **completed**, `[checkpoint] block 1 (lin1): restored from
   checkpoint, skipping solve`, `reconstruct_block` calls `0`, and the resulting
   artifact differed from a fresh run of B. B's quantized block is A's solve and
   nothing anywhere says so. A fingerprint that cannot see the model is what makes
   this a success rather than a crash.
4. When the two models' shapes happen to disagree the failure is real but anonymous:
   `RuntimeError: The size of tensor a (6) must match the size of tensor b (8)` raised
   inside `apply_layer_states` — frames away from the cause, never a fingerprint
   verdict, so the operator learns nothing about *why* the directory is wrong.

**The amplifier is the driver's default.** `tools/quantize_hf_model.py` defaults
`--checkpoint-dir` to the shared, stable, mutable `artifacts/ptq-checkpoints`. Two
models launched by default collide in that one directory without anyone *choosing*
to share it, so the defect is reachable by doing nothing.

## What Changes

- **The resume fingerprint gains a model-identity component**, derived from the LIVE
  model at fingerprint-write time: the quantized-parameter scope's module inventory,
  each module's `(dout, din)`, and its weight dtype. `check_fingerprint` refuses a
  mismatch and **names the diverged component(s)**. Deliberately **not** the HF repo
  id: a moved or renamed local path must not invalidate a real checkpoint, and the
  shape/inventory digest is the identity that matters.
- **The driver's default becomes explicitly per-run**: without `--checkpoint-dir` the
  launch checkpoints into a fresh temporary directory, so two defaulted launches never
  share block archives. A resumable launch names its directory explicitly, which is
  the only way to opt into sharing.

## Impact

- **Affected specs**: `qk-ptq`, one ADDED requirement (no MODIFIED; sibling changes
  carry live deltas under this capability).
- **Affected code**: `src/quantibbler/ptq/checkpoints.py` (identity component +
  refusal), `src/quantibbler/ptq/block_recon.py` (hand the live model to the
  fingerprint write/check), `tools/quantize_hf_model.py` (per-run default), tests.
- **Not invariant-touching**: no qpk1 format, kernel, or server change. Block archives
  are untouched and still load. Resume compatibility: a directory written before this
  change carries no identity component and now refuses (fail closed) — it predates a
  binding that cannot be reconstructed after the fact, and the honest reading of
  "cannot verify" is to refuse; such a directory is deleted (or given to a fresh run)
  exactly as a stale-config directory is.

## Non-goals

- **Not closing the same-shape / different-weights case** (two checkpoints of the same
  architecture, e.g. a base model and a fine-tune of it). The shape/inventory digest
  cannot see it, and binding weights is a different mechanism. The per-run default
  removes the *accidental* path to it; sharing one directory across same-shape models
  stays an explicit operator choice. Recorded as residual risk in `design.md`.
- No content hash of the block archives, no hand-maintained hash/anchor file, no
  identity stamp nothing reads (the user's standing ruling: version control plus the CI
  gates already answer "did the code change").
- No migration or repair of existing checkpoint directories; no change to the
  empty/partial-archive guard of `qk-fix-fp-resume-guard` (archived — cited, not
  modified).
- No new config field and no new env knob: the identity is derived, never configured.
