# Proposal: qk-serve-bf16

## Why

The server currently loads only directory-format artifacts and serves every
remainder at f32 — norms, embedding, lm-head, and the engine's accumulation
all hardcode single-precision. The parent changes make the kernels
dtype-generic (bf16 + f32 first-class) and the artifact a single-file blob
with a declared `remainder_dtype`. The server must consume both: load the
blob via mmap, read the dtype from the header, and serve bf16 remainders at
native precision through the dtype-generic kernels, with fp32 serving
regression-free.

## What Changes

1. **Blob load path.** The server accepts a `.qpk` blob file (auto-detected
   at load); mmap where 4K alignment permits, read fallback with a declared
   reason. The loader resolves every section's dtype from the header's
   section table; remainder sections load at their declared element width.
2. **Dtype-declared remainder.** `remainder_dtype` from the blob header
   drives the engine's working dtype: norm application, logits head, KV
   cache, and attention use the declared dtype for their operands, with f32
   accumulation where the kernels prescribe it (bf16 operand → f32
   accumulator). The server never guesses a dtype from byte size.
   `remainder_dtype` absent → loud refusal at startup.
3. **Kernel dispatch.** The engine routes packed-matvec calls to the
   dtype-aware kernel entry points (`packed_matvec_dtype` /
   `packed_matmul_dtype`), passing the resolved `Dtype` tag so the
   fp32/bf16 accumulation paths are selected per artifact.
4. **fp32 no-regression.** Existing serving tests pass identically on an
   fp32 blob artifact; the fp32 path through the dtype-aware kernels is
   the existing f64-accumulated path (unchanged byte contract).
5. **Serving parity.** A bf16 remainder artifact and the fp32 repack of
   the same model produce temperature-0 generations with token-stream
   equivalence on a fixed probe set; logit agreement within the kernel's
   declared bf16 tolerance (backward 1e-2 of row term scale, per
   qk-kernels-dtype-generic D6).
6. **Chat template.** `POST /v1/chat/completions` applies the model's
   Qwen3 chat template to the incoming `messages` array before
   tokenization (the comparison corpus for benchmarking is chat-format
   scenarios; template drift breaks prompt-token parity).
7. **Prefill timing.** Non-streaming responses report `timings.prompt_ms`
   (llama.cpp-equivalent: tokenization excluded; first prefill forward to
   last prefill forward — per the t_72781a70 addendum so PP t/s is
   computed identically across comparison legs).

## Impact

- **Invariant-touching: yes — server loader and engine surfaces.** The
  server gains a second model-input format (blob) and a second serving
  precision (bf16 operands). The fp32 path is byte-preserved.
- **Affected specs**: `qk-server` — 1 MODIFIED (startup load) + 9 ADDED
  requirements (blob load, mmap, dtype dispatch, chat template, prefill
  timing, parity).
- **Affected code**: `crates/quantibbler-server/src/model/mod.rs` (blob
  load), `crates/quantibbler-server/src/engine/mod.rs` (dtype dispatch,
  norm/logits/attention dtype), `crates/quantibbler-server/src/cli.rs`
  (`--artifact` knob over `--model-dir`), `crates/quantibbler-server/src/
  api/generate.rs` (chat template, timing), `crates/quantibbler-server/
  Cargo.toml` (one new dep: `minijinja` — pure Rust, zero mandatory
  transitive deps; C/C++ ban intact + `binary_purity` gate stays green).
  Fixture test data: committed bf16 blob twin of the server-model fixture.

## Paper anchor

No method change: no solver, stage, budget, or geometry surface is touched.
The dtype genericity is a deployment decision governed by the kernel
tolerance declaration (qk-kernels-dtype-generic D6). Clean-room law: no
implementation source consulted.

## Non-goals

- No performance measurement (qk-bench-server owns the speedup measurement).
- No fp16 remainder dtype (qk-artifact-dtype scope is fp32/bf16).
- No GPU-specific serving path; the CPU path is the only path.
- No hot-swap of artifacts at runtime; the artifact is frozen at startup.
- No change to the sampler (remains greedy argmax).
- No change to the directory load path beyond keeping it functional.
