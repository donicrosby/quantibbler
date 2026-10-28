# qk-server delta: qk-serve-bf16

## MODIFIED Requirements

### Requirement: Startup loads a local qpk1 model directory and reports ready only after success

On startup the server SHALL load the qpk1 model artifact supplied on the
command line — a single-file `.qpk` blob or a legacy model directory,
auto-detected — and MUST NOT report ready or serve inference traffic until
every layer stream, remainder tensor, and tokenizer file has loaded
successfully under the fail-closed reader rules of `qk-representation` (no
partial model is ever served). Supplying both `--artifact` and `--model-dir`
SHALL be refused at parse with a usage error.

#### Scenario: SR-1.1 — Ready only after a successful load

- **WHEN** the server is started with `--artifact` pointing at a valid qpk1
  blob or model directory and loading completes
- **THEN** `GET /health` reports ready (200) and the inference endpoints accept
  traffic only from that point forward
- **AND** the served model id corresponds to the loaded artifact

#### Scenario: SR-1.2 — Failed load never reaches ready

- **WHEN** the server is started with a missing, malformed, or
  failed-validation artifact (bad magic, wrong version, unknown dtype
  tag, truncation, trailing bytes, mismatched section table, missing
  `remainder_dtype`, or any other `qk-representation` reader failure)
- **THEN** the server never reports ready, serves no inference traffic, and
  exits non-zero with a diagnostic naming the failing artifact and the
  specific reason

#### Scenario: SR-1.3 — Blob is auto-detected and loaded

- **WHEN** `--artifact` names a regular file whose first four bytes are `QKPB`
- **THEN** the blob loader parses the header and section table, mmap's the
  payloads where possible, and constructs a loaded model identical to a
  directory load of the same artifact

#### Scenario: SR-1.4 — Directory is still loadable

- **WHEN** `--artifact` names a directory containing a valid `manifest.toml`
  and `config.toml`
- **THEN** the legacy directory loader loads the model; the server reports
  ready and serves identically to a pre-change directory load

#### Scenario: SR-1.5 — Both artifact selectors supplied is refused

- **WHEN** the CLI receives both `--artifact` and `--model-dir`
- **THEN** the server refuses with a usage error naming both flags and exits
  non-zero before loading; no model load or inference is attempted

## ADDED Requirements

### Requirement: The blob loader SHALL mmap where alignment permits and fall back to read

The blob loader SHALL open the `.qpk` file and attempt `memmap2::Mmap::map`
for zero-copy section access, leveraging the blob format's 4K page alignment.
When mmap is unavailable (unsupported platform, network filesystem, or OS
error), the loader SHALL fall back to `std::fs::read` for the entire file and
SHALL log the fallback reason at WARN level. The loaded model is identical
under both paths.

#### Scenario: BS-1.1 — Mmap succeeds on a local file

- **WHEN** the blob file resides on a local filesystem that supports mmap
- **THEN** the loader opens it via `memmap2::Mmap`, section bytes are
  accessed as `&[u8]` slices into the mapped region, and the startup log
  records `load_method: "mmap"`

#### Scenario: BS-1.2 — Read fallback when mmap fails

- **WHEN** mmap returns an error (the blob is on a network mount or an
  unsupported platform)
- **THEN** the loader reads the entire file into a `Vec<u8>` via
  `std::fs::read`, loads successfully, and emits a WARN log line naming the
  mmap error and the fallback reason

#### Scenario: BS-1.3 — Both paths produce the identical loaded model

- **WHEN** the same blob fixture is loaded through mmap and through the
  read fallback (force-fallback via the test-only env var
  `QKB_TEST_FORCE_READ_FALLBACK=1` — reachable only through
  `tests/common/mod.rs`; the production loader reads it solely for this
  test harness path)
- **THEN** the two `LoadedModel` instances have identical `id`, `d_model`,
  `vocab_size`, layer count, and bitwise-identical token IDs on a fixed
  probe prompt

### Requirement: The engine SHALL resolve its serving dtype from the artifact declaration

The server SHALL read `remainder_dtype` from the blob header and resolve a
`ServingDtype` (`Fp32` or `Bf16`). It SHALL NOT infer the dtype from byte
sizes, extensions, or defaults. A missing or unrecognized `remainder_dtype`
SHALL refuse startup. The directory path SHALL NOT select `Bf16`: native
bf16 serving is reachable only through the blob loader (directory loading
follows qk-artifact-dtype RD-1/RD-2).

#### Scenario: BS-2.1 — Bf16 artifact resolves to Bf16

- **WHEN** the blob header declares `"remainder_dtype": "bf16"` and every
  remainder section in the section table carries the `bf16_le` dtype tag
- **THEN** `ModelWeights.serving_dtype` is `ServingDtype::Bf16` and the
  engine serves in bf16 operand mode

#### Scenario: BS-2.2 — Fp32 artifact resolves to Fp32

- **WHEN** the blob header declares `"remainder_dtype": "fp32"` and every
  remainder section carries the `fp32_le` tag
- **THEN** `ModelWeights.serving_dtype` is `ServingDtype::Fp32` and the
  engine serves in fp32 mode (the pre-change path)

#### Scenario: BS-2.3 — Missing remainder_dtype is refused

- **WHEN** the blob header JSON parses but `remainder_dtype` is absent
- **THEN** startup refuses with a diagnostic naming the missing field

#### Scenario: BS-2.4 — Header/section-table dtype mismatch is refused

- **WHEN** the header declares `"remainder_dtype": "bf16"` but any
  remainder section's dtype tag is `fp32_le` (or vice versa)
- **THEN** startup refuses with a diagnostic naming the mismatched section
  and the two disagreeing values

#### Scenario: BS-2.5 — Bf16-declared directory serves widened, never native

- **WHEN** a directory artifact declares `[sidecar] dtype = "bf16"`
  (qk-artifact-dtype RD-1.1)
- **THEN** the load succeeds per the sibling's widening contract
  (qk-artifact-dtype RD-2), `ServingDtype` stays `Fp32`, and the engine
  never routes bf16 kernel dispatch for this artifact — native bf16
  serving is blob-only

### Requirement: Packed matvecs SHALL dispatch to the dtype-aware kernel entry points

The engine's `PackedLayer.apply` SHALL route packed matrix-vector products
through the dtype-aware kernel entry points: the existing `f64` default path
for `ServingDtype::Fp32` and `packed_matvec_dtype` / `packed_matmul_dtype`
with `Dtype::Bf16` for `ServingDtype::Bf16`. Scale and bias application SHALL
stay at f32 under both paths. The fp32 path SHALL be byte-identical to the
pre-change path.

#### Scenario: BS-3.1 — Fp32 matvec path is unchanged

- **WHEN** `PackedLayer.apply` runs on a committed fp32 fixture layer with
  `ServingDtype::Fp32`
- **THEN** the output bytes are identical to the pre-change output of
  `apply` (the v0.1 f64-accumulated path)

#### Scenario: BS-3.2 — Bf16 matvec path uses the dtype-aware kernel

- **WHEN** `PackedLayer.apply` runs with `ServingDtype::Bf16` and a bf16
  input vector
- **THEN** the kernel receives `Dtype::Bf16`, accumulates in f32, and
  the per-output-element backward error vs exact arithmetic over the declared-dtype operands (unrounded products, declared accumulator) is within
  1e-2 of the row term scale Σ_k |v_k| (qk-kernels-dtype-generic D6)

### Requirement: Norm application and logits SHALL use the declared serving dtype

When `ServingDtype::Bf16`, RMSNorm SHALL take bf16 operands, compute the
statistics and weighted output in f32, and cast the result back to bf16.
The logits head dot-product SHALL accumulate bf16 hidden × bf16 head-row
products in f32. The KV cache SHALL store f32 regardless of serving dtype.
When `ServingDtype::Fp32`, every float path is the pre-change f32 path.

#### Scenario: BS-4.1 — Bf16 norm produces bf16 output within tolerance

- **WHEN** an RMSNorm with bf16 weight and bf16 input is computed through
  the engine's bf16 path
- **THEN** the output (cast to f32 for comparison) agrees with an fp32
  reference norm on the same values widened to f32 within the change's
  declared serving-level bound |Δ| ≤ 1e-2 · max(|ref|, 1e-6) — a
  combined absolute-relative form chosen so near-zero reference elements
  cannot make the bound unbounded; the kernel-level backward tolerance
  (qk-kernels-dtype-generic D6) governs matvec products only, not norm
  statistics

#### Scenario: BS-4.2 — Fp32 norm path is unchanged

- **WHEN** `ServingDtype::Fp32` is active
- **THEN** RMSNorm and logits head dot-product produce byte-identical
  results to the pre-change code on identical inputs

### Requirement: Chat completions SHALL apply the model's chat template

`POST /v1/chat/completions` SHALL apply the loaded model's Qwen3 chat
template to the `messages` array before tokenization. Rendering SHALL use
`minijinja` with the template string from the fixture's `chat_template`
field; the render context SHALL include `add_generation_prompt=true` and
`enable_thinking=false`. A committed golden file (Python HF stack, pinned
kwargs) SHALL serve as the byte-level render parity contract. Template
failure SHALL return 400.

#### Scenario: BS-5.1 — Chat template is rendered and tokenized against goldens

- **WHEN** a chat completion request carries Qwen3-format messages
  `[{role: "user", content: "hello"}]`
- **THEN** the rendered prompt string equals the committed golden render
  string (generated by Python `tokenizer.apply_chat_template(messages,
  tokenize=False, add_generation_prompt=True)` with the version of
  `enable_thinking` pinned false, provenance recorded in the golden
  directory) AND the tokenized prompt IDs equal the committed golden
  token IDs (generated by Python tokenizers on the committed fixture
  tokenizer.json from that render string)

#### Scenario: BS-5.2 — Template failure returns 400

- **WHEN** the messages array is empty or carries a role the template does
  not accept
- **THEN** the server returns 400 with the invalid_request_error envelope
  and a message naming the template failure; no tokenization or generation
  is attempted

### Requirement: Non-streaming responses SHALL report prefill wall-clock timing

Non-streaming chat and text completion responses SHALL carry a `timings`
block with `prompt_ms` (u64, milliseconds) measuring the elapsed wall-clock
from the start of the FIRST prefill forward through the end of the LAST
prefill forward. Tokenization and template application SHALL be excluded
(llama.cpp timings.prompt_ms equivalence per the t_72781a70 addendum, so PP
t/s is computed identically across legs). Decode steps SHALL be excluded.
Streaming responses SHALL omit `timings`.

#### Scenario: BS-6.1 — Non-streaming response carries prompt_ms

- **WHEN** a non-streaming chat completion is served
- **THEN** the response JSON carries `"timings": {"prompt_ms": <N>}` where
  N is a positive integer and less than the total request wall time

#### Scenario: BS-6.2 — Streaming response omits timings

- **WHEN** a streaming chat completion is served
- **THEN** the SSE chunks do not contain a `timings` field; the stream
  ends with no timing metadata and no chunk carries a timings field

### Requirement: /v1/models SHALL report the serving dtype

The model entry in `GET /v1/models` SHALL carry a `dtype` field whose value
is the artifact's declared `remainder_dtype` (`"fp32"` or `"bf16"`).

#### Scenario: BS-7.1 — Fp32 model reports fp32 dtype

- **WHEN** the loaded artifact declares `remainder_dtype: "fp32"`
- **THEN** `GET /v1/models` returns `data[0].dtype: "fp32"`

#### Scenario: BS-7.2 — Bf16 model reports bf16 dtype

- **WHEN** the loaded artifact declares `remainder_dtype: "bf16"`
- **THEN** `GET /v1/models` returns `data[0].dtype: "bf16"`

### Requirement: Fp32 serving SHALL be regression-free

Every existing server test SHALL pass identically against an fp32 blob
artifact — the same tests, the same assertions, the same outputs. The test
harness SHALL parametrize over both the directory and fp32-blob fixtures.
The fp32 dispatching path SHALL produce the same bytes the pre-change path
produced.

#### Scenario: BS-8.1 — All pre-existing tests pass on an fp32 blob

- **WHEN** the full server test suite runs against both the committed
  directory fixture AND the committed fp32 blob fixture
- **THEN** every test that passed on the directory fixture passes with the
  same assertions on both fixture sources (no tolerance relaxations, no
  skipped tests)

#### Scenario: BS-8.2 — Same request through both backends yields identical
completion content

- **WHEN** the fp32 blob and the directory fixture serve the same
  completion request (temperature 0, identical probe prompt)
- **THEN** the completion content (choices, token IDs, finish reason) is
  byte-identical across the two backends; timing fields excluded from the
  comparison

### Requirement: Bf16 and fp32 artifacts SHALL produce equivalent generations

A bf16 remainder artifact and the fp32 repack of the same model SHALL
produce temperature-0 token streams that are identical on a committed probe
set of at least five prompts of varying lengths. The first token position at
which the two streams diverge SHALL be recorded as a parity failure; a
single-token difference from a "close" logit distribution is a failure.

#### Scenario: BS-9.1 — Token streams are identical on the probe set

- **WHEN** both the fp32 and bf16 blobs of the server-model fixture are
  loaded and served at temperature 0 over the committed prompts
- **THEN** the completion token IDs are identical for every prompt; any
  difference is a test failure with the diverging prompt and position
  named

#### Scenario: BS-9.2 — Both legs' prefill logits are finite; deviation recorded

- **WHEN** the prefill logits of the fp32 and bf16 artifacts are compared
  on the probe set at the final logits layer (not per linear intermediate)
- **THEN** both legs' final prefill logits are finite; the worst and median
  elementwise absolute deviations |Δ| = |L_bf16 − L_fp32| are reported in
  the test output (no threshold; per-linear backward error stays with
  qk-kernels-dtype-generic)

#### Scenario: BS-9.3 — Bf16-leg logits differ from the widened path (must-differ guard)

- **WHEN** the bf16 artifact's final prefill logits are compared to the
  fp32 artifact's on the probe set
- **THEN** the bf16-leg logits are NOT bitwise-identical to the fp32-leg
  logits; bit-identity means native bf16 serving never executed and the
  test fails loudly — the committed fixture remainder tensors SHALL carry
  values not exactly representable in bf16 (provenance recorded in the
  fixture commit) so the native path necessarily differs from a
  widen-and-serve path
