# Tasks: qk-serve-bf16

## Task 1 — Blob load path (Rust)

- [ ] Implement `model::load_blob(path: &Path) -> Result<LoadedModel, String>`
  in `crates/quantibbler-server/src/model/mod.rs`: detect `.qpk` file,
  open it, attempt `memmap2::Mmap::map`, fall back to `std::fs::read` on
  failure with a WARN naming the reason, parse header + section table,
  validate every section per the blob spec (fail-closed), load
  remainder at the declared section dtype into f32 or `half::bf16` per
  the resolved `ServingDtype`.
- **Verification**: `cargo test -p quantibbler-server -- blob_load` —
  loads the committed blob fixture, asserts `LoadedModel` fields equal
  the directory-loaded same model.

## Task 2 — CLI: `--artifact` over `--model-dir`

- [ ] Add `--artifact <PATH>` to `Cli`; make it required; keep
  `--model-dir` as a deprecated alias with a deprecation notice to
  stderr. Supplying both `--artifact` and `--model-dir` SHALL be
  refused at parse with a usage error naming both flags (fail-closed,
  scenario SR-1.5). Route to `model::load` (auto-detect) when the path
  is a blob file, `model::load_dir` when legacy directory.
- **Verification**: `cargo run -p quantibbler-server -- --artifact
  tests/fixtures/server-model-blob.qpk --port 19999` starts and reports
  ready. `--model-dir <dir>` with the legacy directory still works.
  Supplying both flags exits non-zero with a usage error.

## Task 3 — ServingDtype resolution and kernel dispatch

- [ ] Define `ServingDtype` enum (`Fp32`, `Bf16`) in
  `crates/quantibbler-server/src/model/mod.rs`. Parse from blob header
  `remainder_dtype` (required; absent → refusal). Store on
  `ModelWeights`. In `PackedLayer.apply`, select the kernel entry point:
  `Fp32` → existing `cpu_matvec` (unchanged), `Bf16` →
  `core::packed_matvec_dtype` with `Dtype::Bf16`. The scales/bias stay
  f32.
- **Verification**: `cargo test -p quantibbler-server -- layer_apply` —
  the fixture layer's fp32 path is byte-identical to pre-change; bf16
  path produces results within the declared tolerance.

## Task 4 — Remainder dtype in engine (norms, logits, attention)

- [ ] Read `self.serving_dtype` at the norm/logits/attention call sites:
  when `Bf16`, load remainder as `bf16`, convert norm weights and
  embedding/lm_head to `bf16` at load. RMSNorm: operand bf16, computed
  in f32, result cast to bf16. Logits head: dot-product in f32
  accumulator.
- **Verification**: `cargo test -p quantibbler-server -- engine_bf16` —
  a tiny bf16 fixture model produces logits within the bf16 tolerance
  of the fp32 twin.

## Task 5 — Chat template

- [ ] Read the committed fixture tokenizer.json's `chat_template` field
  via serde_json (the `tokenizers` crate 0.23.2 does not expose
  arbitrary metadata — verified against vendored source). Render the
  Qwen3 template with `minijinja` 3.0.3 (latest stable per crates.io
  2026-10-26; default features, zero mandatory transitive deps, no
  C/C++ deps) with the pinned context `{messages,
  add_generation_prompt: true, enable_thinking: false}`. Tokenize the
  rendered string. On template failure return 400 (BS-5.2). On success,
  assert the rendered prompt string AND the tokenized prompt IDs equal
  the committed goldens (BS-5.1). Run `cargo tree -p
  quantibbler-server -e normal --locked` to confirm minijinja's runtime
  dep set is empty and the binary_purity test stays green.
- **Verification**: `cargo test -p quantibbler-server -- chat_template`
  — passes against the golden files (scenarios BS-5.1 / BS-5.2);
  `cargo test -p quantibbler-server -- binary_purity` stays green.

## Task 6 — Prefill timing in responses

- [ ] Add `timings: Option<Timings>` to `ChatCompletionResponse` and
  `CompletionResponse`. `Timings { prompt_ms: u64 }`. In the
  non-streaming generate handler, capture `Instant::now()` before the
  FIRST prefill forward in `decode_start`, compute elapsed after the
  prefill loop; the difference is `prompt_ms`. Tokenization and
  template rendering SHALL be excluded from the measured span
  (llama.cpp timings.prompt_ms equivalence). Streaming responses omit
  `timings`.
- **Verification**: `cargo test -p quantibbler-server -- timings` —
  a non-streaming response carries `timings.prompt_ms > 0`; a streaming
  response does not carry `timings`.

## Task 7 — /v1/models dtype field

- [ ] Add `dtype: String` to the model entry in `GET /v1/models`. Value
  is `"fp32"` or `"bf16"` from `ModelWeights.serving_dtype`.
- **Verification**: `cargo test -p quantibbler-server -- api_models` —
  `GET /v1/models` returns `data[0].dtype` matching the artifact.

## Task 8 — Serving parity test (bf16 vs fp32)

- [ ] Write `tests/parity_bf16.rs`: load the fp32 blob fixture and the
  bf16 blob fixture, run 5 fixed probe prompts through both at
  temperature 0, assert token streams identical (BS-9.1). Assert the
  bf16-leg final prefill logits are finite, NOT bitwise-identical to
  the fp32-leg logits (must-differ guard, BS-9.3), and report
  worst/median elementwise absolute deviation (BS-9.2). No per-linear
  intermediate capture.
- **Verification**: `cargo test -p quantibbler-server -- parity_bf16`
  — green; removing the must-differ assertion or flipping
  bitwise-equality reddens the test.

## Task 9 — Fixture Generation

- [ ] **fp32 and bf16 blob fixtures**: Commit the fp32 blob twin of the
  server-model fixture (produced from the directory by the blob
  exporter) and the bf16 blob repack produced by
  `tools/repack_remainder.py`. The fixture remainder tensors SHALL
  carry values NOT exactly representable in bf16 (e.g. 0.1f32, not
  only 0.25-multiples) — provenance recorded in the commit body so the
  must-differ guard is falsifiable.

- [ ] **Chat template in the fixture tokenizer.json**: Install the
  genuine Qwen3 `chat_template` string (provenance-pinned to the HF
  export; record source model revision). Update
  `tools/gen_server_fixtures.py` to embed the template.

- [ ] **Golden chat files**: Generate and commit `chat_golden.json`
  containing the rendered prompt string and token IDs from the Python
  HF stack with pinned probing messages
  `[{role: "user", content: "hello"}]`, `add_generation_prompt=true`,
  `enable_thinking=false`, against the committed fixture tokenizer.json
  (post chat_template installation).

- **Verification**: `sha256sum` of the fp32 blob matches the committed
  sha256; repack produces a byte-identical bf16 blob every time
  (deterministic bytes, blob spec B5). The golden file's rendered
  string matches the integer reference and the must-differ guard is
  exercisable (weight values like 0.1f32 survive the repack).

## Task 10 — Gate suite

- [ ] Run the full gate suite from the worktree:
  `cargo test -p quantibbler-server`, `cargo test -p quantibbler-kernels`,
  `cargo clippy --all-targets -- -D warnings`,
  `cargo fmt --all -- --check`,
  `uv run pytest`, `uv run maturin build --manifest-path
  crates/quantibbler-kernels/Cargo.toml`.
  Record results. All pre-existing tests pass; new tests pass.
- **Verification**: all gates exit 0.
