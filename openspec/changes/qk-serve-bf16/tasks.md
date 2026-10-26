# Tasks: qk-serve-bf16

## Task 1 — Blob load path (Rust)

- [x] Implement the blob load path in
  `crates/quantibbler-server/src/model/mod.rs`: detect `.qpk` file,
  open it, attempt `memmap2::Mmap::map`, fall back to `std::fs::read` on
  failure with a WARN naming the reason, parse header + section table
  from the OPEN FILE before any map is requested, validate every section
  per the blob spec (fail-closed), and load the remainder at the declared
  section dtype per the resolved `ServingDtype`.
  - Shipped as `model::load_blob(path: &Path) ->
    Result<(ModelDirectory, BlobLoadMethod), String>` plus a shared
    private `compose` that builds the one `LoadedModel` both sources
    produce; `LoadedModel.blob_load_method` carries the byte source as a
    VALUE, not only a log line. The section rules stay in
    `quantibbler_qpk1::loader::build_from_blob` (one visibility change to
    `pub`), so the server does not re-derive them.
- **Verification**: `cargo test -p quantibbler-server` — `tests/serve_bf16.rs`
  (`blob_auto_detected_via_mmap_and_identical_to_the_directory_sr_1_3`) and
  `tests/blob_read_fallback.rs` load the committed blob fixture and assert the
  `LoadedModel` fields and PREFILL LOGITS equal the directory-loaded same model
  (bitwise). PASSED.

## Task 2 — CLI: `--artifact` over `--model-dir`

- [x] Add `--artifact <PATH>` to `Cli`; keep `--model-dir` as a deprecated
  alias with a deprecation notice to stderr. Exactly ONE selector is
  required (clap `ArgGroup`, non-multiple): supplying both `--artifact` and
  `--model-dir`, or neither, is refused at parse with a usage error naming
  both flags (fail-closed, scenario SR-1.5). Route to `model::load`
  (auto-detect).
  - Shipped: the group makes the SELECTION required and exclusive rather
    than making `--artifact` unconditionally required, so the legacy spelling
    keeps working (SR-1.4).
- **Verification**: `tests/serve_bf16.rs`
  (`both_artifact_selectors_are_refused_at_parse_sr_1_5`,
  `directory_is_still_loadable_sr_1_4`), plus the process-level legs in
  `tests/load.rs`. PASSED.

## Task 3 — ServingDtype resolution and kernel dispatch

- [x] Define `ServingDtype` enum (`Fp32`, `Bf16`) in
  `crates/quantibbler-server/src/model/mod.rs`. Parse from blob header
  `remainder_dtype` (required; absent → refusal). Store on `ModelWeights`.
  In `PackedLayer.apply`, select the kernel entry point: `Fp32` → the
  existing f64-accumulating path (unchanged), `Bf16` → the dtype-aware
  entry point with `Dtype::Bf16`. The scales/bias stay f32.
  - Shipped: the dtype-aware call is the rlib face
    `dispatch::packed_matvec_dyn(.., DenseOperand::Bf16(v))`. The PyO3 name
    in the row (`packed_matvec_dtype`) is the extension-module face, which
    the server must not link (`binary_purity`); see design crystallization
    entry 1.
- **Verification**: engine unit tests
  `fp32_matvec_is_byte_identical_to_the_pre_change_path_bs_3_1` and
  `bf16_matvec_within_declared_tolerance_bs_3_2`. PASSED.

## Task 4 — Remainder dtype in engine (norms, logits, attention)

- [x] Read `self.serving_dtype` at the norm/logits/attention call sites:
  when `Bf16`, the remainder tensors ARE their declared bf16 values.
  RMSNorm: operand bf16, computed in f32, result cast to bf16. Logits head:
  dot-product with an f32 accumulator over bf16-valued operands.
  - Shipped: the bf16 values are held in f32 storage (the loader widens
    bf16 → f64 exactly, so quantizing at load changes no bit) and the operand
    rounding happens ONCE at each bf16 call site — see design crystallization
    entry 3. The KV cache stays f32 under both dtypes.
- **Verification**: engine unit tests `bf16_norm_within_declared_bound_bs_4_1`
  and `fp32_norm_is_unchanged_bs_4_2`. PASSED.

## Task 5 — Chat template

- [x] Read the committed fixture tokenizer.json's `chat_template` field via
  serde_json (the `tokenizers` crate 0.23.2 does not expose arbitrary
  metadata). Render the Qwen3 template with `minijinja` 3.0.3 (default
  features, zero mandatory transitive deps, no C/C++ deps) with the pinned
  context `{messages, add_generation_prompt: true, enable_thinking: false}`.
  Tokenize the rendered string. On template failure return 400 (BS-5.2). On
  success, assert the rendered prompt string AND the tokenized prompt IDs
  equal the committed goldens (BS-5.1). Confirm minijinja's runtime dep set
  is empty and `binary_purity` stays green.
  - Shipped: the document carrying the template is the same one the
    `tokenizers` crate parses, and that crate rejects ANY unknown top-level
    key (measured) — `model::tokenizer_from_json` removes the key from the
    crate's view; see design crystallization entry 8. The template's Python
    `str` methods are served through minijinja's `unknown_method_callback`
    (entry 9).
- **Verification**: `cargo test -p quantibbler-server -- chat_template` —
  `tests/chat_template.rs` (BS-5.1 directory + blob legs, BS-5.2 400 legs, the
  served-leg prompt-token count) is green; the module's unit tests and
  `binary_purity` stay green; `cargo tree -p minijinja -e normal --locked`
  shows an EMPTY dependency set. PASSED.

## Task 6 — Prefill timing in responses

- [x] Add `timings: Option<Timings>` to `ChatCompletionResponse` and
  `CompletionResponse`. `Timings { prompt_ms: u64 }`. In the non-streaming
  generate handler, capture `Instant::now()` before the FIRST prefill forward
  in `decode_start`, compute elapsed after the prefill loop; the difference
  is `prompt_ms`. Tokenization and template rendering are excluded from the
  measured span. Streaming responses omit `timings`.
- **Verification**: `cargo test -p quantibbler-server -- timings` —
  `tests/api_timings.rs` (BS-6.1 non-streaming positive `prompt_ms` bounded by
  the request wall time on BOTH surfaces; BS-6.2 the SSE body carries no
  `timings`). PASSED.

## Task 7 — /v1/models dtype field

- [x] Add `dtype` to the model entry in `GET /v1/models`. Value is `"fp32"` or
  `"bf16"` from `ModelWeights.serving_dtype`.
- **Verification**: `tests/api_models.rs::models_reports_the_serving_dtype_bs_7`
  (fp32 blob → `fp32`, bf16 blob → `bf16`). PASSED.

## Task 8 — Serving parity test (bf16 vs fp32)

- [x] Write `tests/parity_bf16.rs`: load the fp32 blob fixture and the bf16
  blob fixture, run 5 fixed probe prompts through both at temperature 0,
  assert token streams identical (BS-9.1). Assert the bf16-leg final prefill
  logits are finite, NOT bitwise-identical to the fp32-leg logits (must-differ
  guard, BS-9.3), and report worst/median elementwise absolute deviation
  (BS-9.2). No per-linear intermediate capture.
- **Verification**: `cargo test -p quantibbler-server --test parity_bf16` —
  green. PASSED.

## Task 9 — Fixture Generation

- [x] **fp32 and bf16 blob fixtures**: the fp32 blob twin of the server-model
  fixture (produced from the directory by the blob exporter) and the bf16 blob
  repack produced by `tools/repack_remainder.py` are committed
  (`tests/fixtures/server-model.qpk` sha256
  `5a72bfb0c90d529650e251104bd5d6c355769d69c9b1a62a56c5336775ed2ba7`;
  `tests/fixtures/server-model-bf16.qpk` sha256
  `9ac15c5e17eb2155ac9b4c9c3f7dd79b06147553abe43164606317e801fcf1e2`). The
  fixture remainder tensors carry values NOT exactly representable in bf16
  (`(i % 7 - 3) * 0.25` embedding, `0.1 * (i + 1)` norms), so the must-differ
  guard is falsifiable. The bf16 blob is now emitted by
  `tools/gen_server_fixtures.py::build_bf16_blob_fixture`, so the parity pair
  cannot drift from the fp32 blob.

- [x] **Chat template in the fixture tokenizer.json**: the genuine Qwen3
  `chat_template` string is installed (provenance pinned in
  `tools/gen_server_fixtures.py`: `Qwen/Qwen3-0.6B` `tokenizer_config.json`,
  revision `c1899de289a04d12100db370d81485cdf75e47ca`).

- [x] **Golden chat files**: `tests/fixtures/chat_golden.json` contains the
  rendered prompt string and token IDs from the Python HF stack
  (`transformers.apply_chat_template`, cross-checked against a direct jinja2
  render; token ids from the Python `tokenizers` on the committed fixture
  tokenizer.json) with the pinned probing messages
  `[{role: "user", content: "hello"}]`, `add_generation_prompt=true`,
  `enable_thinking=false`. Generated by the committed
  `tools/gen_chat_golden.py`; deterministic (no timestamps written).

- **Verification**: both blobs load and serve (serve_bf16 / parity_bf16 /
  chat_template); `tools/repack_remainder.py` reproduces the committed bf16
  bytes (`tests/ptq/test_artifact_dtype.py` repack legs); two consecutive
  `tools/gen_server_fixtures.py` passes produce byte-identical blobs
  (`cmp` identical), and `tools/gen_chat_golden.py` is byte-stable. PASSED.

## Task 10 — Gate suite

- [x] Run the full gate suite from the worktree and record results.
- **Verification** (branch `coder/t_2b7e43f3-serve-bf16`; rustc 1.98.1; all
  commands run from the worktree):
  - `cargo test -p quantibbler-server --locked` → RC=0, 405 passed / 0 failed
    / 10 ignored.
  - `cargo test -p quantibbler-qpk1 --locked` → RC=0, 59 passed / 0 failed.
  - `cargo test -p quantibbler-kernels --locked` → RC=0, 32 passed / 0 failed.
  - `cargo clippy --all-targets --locked -- -D warnings` → RC=0.
  - `cargo fmt --all -- --check` → RC=0.
  - `npx --yes @fission-ai/openspec validate qk-serve-bf16 --strict` → RC=0.
  - `maturin build --manifest-path crates/quantibbler-kernels/Cargo.toml` →
    RC=0 (abi3 wheel built).
  - Python tier: `python -m pytest tests/test_qpk1_blob.py
    tests/test_attention_gate_record.py tests/ptq/test_artifact_dtype.py
    tests/ptq/test_blob_export.py -q` → 33 passed, 2 deselected (the two
    `integration`-marked legs, per the default marker filter). These are the
    Python tests that consume the regenerated fixtures.
