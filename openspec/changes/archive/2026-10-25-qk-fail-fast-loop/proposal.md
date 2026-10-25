# Proposal: qk-fail-fast-loop

## Why

The user's current loop is ~90 minutes per explosion: a run quantizes all
28 blocks, polishes, calibrates, and only then measures end-to-end PPL —
so a directionally dead artifact is discovered at minute ~90, after the
full solver cost, and the new e2e gate then correctly refuses it. The
forensics showed *where* death begins (block 0 mostly, amplified
downstream); the loop is only slow because **nothing measures the
quantized prefix while it is being built**. Divergence still costs an
hour-plus, not minutes — that is the gap this change closes. Fast
failure is also the enabler for the paper-recreation goal ([NQ] parity):
cycle time, not gate strength, is what makes 90-minute ashes expensive.

The blocks are reconstructed sequentially, in order, writing back into the
host model as they land (so the driver's write-back host already IS the
quantized prefix at any depth). Each new block's reconstruction cost is
the bulk of the run; one cheap prefix perplexity at each boundary trades
seconds against tens of minutes of doomed work per failed block.

## What Changes

- **`PTQConfig.progress_probe`** (new callable field, default `None`):
  invoked after each block's reconstruction (post-STE, post-write-back)
  with the live host model and the prefix depth. May raise a dedicated
  `QuantizationAbort`-style exception to halt the pipeline. Excluded from
  checkpoint fingerprints (a callable is code-state, not solver-state; a
  resume with a probe re-runs probes for the remaining blocks).
- **Driver-side probe** in `tools/quantize_hf_model.py`: at startup it
  precomputes fp-prefix baselines (one chunked forward per depth, fp model
  frozen, so this is cheap and done once); per block it measures the
  quantized prefix's PPL on a small slice of the held-out eval batches and
  compares against the fp-prefix at the same depth through the **same
  comparator and collision line as the e2e gate** (`qk_report.verdict`,
  `COLLAPSE_LINE = 40`). Above the line → abort naming block and depth.
- **Knobs** on the driver: `--probe-interval` (default 1 = every block) and
  `--probe-batches` (default 2). The ratio line itself stays the derived
  constant — no per-run threshold escape hatch.
- **Suite tests**: probe decision math lives in `qk_report` (torch-free,
  lean-tested); torch-tier tests in `tests/ptq/` assert invocation order,
  that the abort fires on an exploded prefix and not on a healthy one, and
  that absence of the callable changes nothing (back-compat).

## Impact

- **Affected specs**: `qk-ptq` (ADDED-requirement delta under
  `changes/qk-fail-fast-loop/specs/qk-ptq/spec.md`; distinct surface from
  `qk-e2e-quality-gate` — end-of-run verdict vs in-run progress probe, and
  from `qk-divergence-guards` — solver-level bands vs eval-level probe).
- **Affected code**: `src/quantibbler/ptq/config.py` + `block_recon.py`
  (hook invocation + fingerprint exclusion), `tools/quantize_hf_model.py`
  (probe implementation, shared with the e2e gate's changes — see
  dependency note below), `src/quantibbler/qk_report.py` (probe decision),
  `tests/`.
- **Not invariant-touching**: no qpk1 grammar, packing, kernels, or server
  changes. No new dependency. The probe never mutates model state.
- **Runtime cost**: bounded at roughly half of one full-model PPL pass
  spread over the run (2 of ~40 eval batches per block, prefixes shorter
  than full depth); measured numbers land in the design crystallization.
- **False-positive exposure**: a shallow noisy prefix can trip the line —
  cost is a re-run, and the design records mitigations (min probe depth,
  fp-prefix baseline at the same depth so depth noise is divided out).

## Non-goals

- No change to the solver stages themselves (no early LB-ADMM exits, no
  polish-stopping heuristics) — the probe only *observes* the prefix.
- No change to the e2e gate's band, verdicts, or exit policy; the probe
  borrows the collapse line, it does not redefine it.
- No wall-clock budget aborts (`--max-minutes`-style) — clock time is a
  poor proxy for divergence and would couple the gate to machine load.
- No parallel-block execution or solver-speed work.

## Dependency note (tracks the repo stacking rule)

The affected driver (`tools/quantize_hf_model.py`) is currently modified
by the unlanded `qk-e2e-quality-gate` change. Implementation of this
change therefore starts AFTER that change lands (spec+code in the same
commit per change). This proposal and its spec delta are safe to review
now; the code tasks are sequenced in `tasks.md` accordingly.
