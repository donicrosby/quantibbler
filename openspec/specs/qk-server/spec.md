# qk-server Specification

## Purpose

Define the pure-Rust OpenAI-compatible inference service that fail-closed loads a local qpk1 model directory and serves deterministic completion traffic without Python.

## Requirements

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

### Requirement: GET /v1/models lists the single loaded model

The server SHALL expose `GET /v1/models` listing exactly the one loaded model
per process once ready; v0.1 MUST NOT define a multi-model registry.

#### Scenario: SR-2.1 — Loaded model is listed
- **WHEN** `GET /v1/models` is served on a ready server
- **THEN** the response is an OpenAI-shaped list whose `data` contains exactly
  one entry with `object` `"model"` and an `id` equal to the loaded model's
  served id

#### Scenario: SR-2.2 — No second model can be registered
- **WHEN** a client attempts to serve or register a second model directory on
  the running process
- **THEN** no such operation exists in v0.1 (no registry or hot-swap
  endpoint); the process serves only the single startup model

### Requirement: POST /v1/chat/completions returns an OpenAI-shaped chat completion

`POST /v1/chat/completions` with `"stream": false` SHALL return an
OpenAI-shaped chat completion object (id, `object` `"chat.completion"`,
created, model, `choices[].message.{role, content}`, usage); a request naming
an unknown model SHALL be rejected with 404 and code `model_not_found`.

#### Scenario: SR-3.1 — Non-streaming chat response shape
- **WHEN** a valid chat completion request with `"stream": false` names the
  loaded model
- **THEN** the response JSON parses with `object` `"chat.completion"`, one or
  more choices whose `message` has role `"assistant"` and string `content`, and
  a usage object with prompt, completion, and total token counts

#### Scenario: SR-3.2 — Unknown model id is rejected
- **WHEN** the request's `model` field names anything other than the loaded
  model
- **THEN** the server returns 404 with the OpenAI error envelope and code
  `model_not_found`

### Requirement: POST /v1/chat/completions streams SSE chunks ending at the DONE sentinel

`POST /v1/chat/completions` with `"stream": true` SHALL respond with content
type `text/event-stream`, emitting OpenAI-format `data:` chunk objects (the
first content delta carrying the assistant role) and terminating with the
literal `data: [DONE]` sentinel.

#### Scenario: SR-4.1 — SSE chunk framing
- **WHEN** a streaming chat completion request is served
- **THEN** the response content type is `text/event-stream` and the body is a
  sequence of `data: {json}` chunk events whose first content delta carries the
  assistant role, per the OpenAI chunk format

#### Scenario: SR-4.2 — Stream ends at the DONE sentinel
- **WHEN** the stream completes normally
- **THEN** the last event emitted is the literal sentinel `data: [DONE]`,
  following the final chunk

#### Scenario: SR-4.3 — Client disconnect aborts its stream only
- **WHEN** a client disconnects mid-stream
- **THEN** the server aborts that decode (no further chunks are produced for
  it) and other in-flight requests are unaffected

### Requirement: POST /v1/completions serves legacy text completions

The server SHALL serve `POST /v1/completions` in the legacy completion shape
(`object` `"text_completion"`) when `"stream": false`, and as an SSE chunk
stream ending at the `data: [DONE]` sentinel when `"stream": true`.

#### Scenario: SR-5.1 — Non-streaming completion shape
- **WHEN** a valid completion request with `"stream": false` is served
- **THEN** the response is an OpenAI text completion object with
  `choices[].text` and a usage object

#### Scenario: SR-5.2 — Streaming completion ends at DONE
- **WHEN** a completion request with `"stream": true` is served
- **THEN** text chunks stream as `data: {json}` events and the stream ends with
  the `data: [DONE]` sentinel

### Requirement: GET /health reports model load readiness

The server SHALL expose `GET /health` returning 200 once the model is loaded
and the server is serving, and a failure status (503) while loading has not
succeeded.

#### Scenario: SR-6.1 — Healthy after load
- **WHEN** the model has finished loading
- **THEN** `GET /health` returns 200 with readiness reported as true

#### Scenario: SR-6.2 — Not healthy during load
- **WHILE** loading is still in progress
- **THEN** `GET /health` returns 503, and inference requests are rejected with
  the 503 error envelope (code `model_not_ready`)

### Requirement: Errors use the OpenAI error envelope

Every error response SHALL use the OpenAI error envelope
`{"error": {"message", "type", "param", "code"}}` with the documented status
map: 400 `invalid_request_error`; 404 `invalid_request_error` with code
`model_not_found`; 503 `server_error` with code `model_not_ready`; 500
`server_error`.

#### Scenario: SR-7.1 — Malformed request maps to 400
- **WHEN** a request carries malformed JSON or invalid parameters
- **THEN** the response is 400 with the `invalid_request_error` envelope,
  parseable as `{"error": {...}}`

#### Scenario: SR-7.2 — Not-ready maps to 503
- **WHEN** an inference request arrives before readiness
- **THEN** the response is 503 with the `server_error` envelope and code
  `model_not_ready`

#### Scenario: SR-7.3 — Internal failure maps to 500
- **WHEN** an internal error occurs during generation
- **THEN** the response is 500 with the `server_error` envelope, and no error
  path leaks anything other than the envelope (including body-parse
  rejections)

### Requirement: Serving is CPU-only pure Rust with no Python in the build graph

In v0.1 the server SHALL execute quantized inference on CPU through the rlib
face of `quantibbler-kernels` (capability `qk-kernels`) and MUST NOT link or
embed Python: no interpreter, no PyO3 extension, no compressor dependency,
anywhere in its build graph (config Component decomposition).

#### Scenario: SR-8.1 — Inference runs on CPU via the kernels rlib face
- **WHEN** the ready server serves a completion for the fixture model directory
- **THEN** inference runs on CPU through the `quantibbler-kernels` rlib face
  and the response is produced without any Python process or interpreter
  involvement

#### Scenario: SR-8.2 — Built binary carries no Python linkage
- **WHEN** the built `quantibbler-server` binary is inspected (e.g. `ldd`) and
  its dependency graph is listed (`cargo tree`)
- **THEN** no libpython linkage appears and the crate graph contains no
  Python-facing dependency (the kernels rlib default features, not its PyO3
  extension-module feature)

### Requirement: Concurrent requests execute correctly against the shared loaded model

The server SHALL accept concurrent requests against the single loaded model;
weights are shared immutable state, and each request MUST produce its own
complete, uncorrupted response independent of concurrent traffic.

#### Scenario: SR-9.1 — N concurrent requests all complete
- **WHEN** N concurrent valid requests are issued against the ready server
- **THEN** all N complete with valid, independent responses

#### Scenario: SR-9.2 — One failing request does not corrupt others
- **WHEN** one concurrent request fails (e.g. an invalid body) while others are
  in flight
- **THEN** the other requests complete normally with uncorrupted responses

### Requirement: The model directory SHALL declare its norm inventory

The exported model directory SHALL declare its per-block norm inventory in `config.toml` as an ordered `norm_slots` array. The sidecar SHALL hold, per block, each declared slot's vector in inventory order, then the model-final norm, in the directory's declared remainder dtype — fp32 little-endian when undeclared, bf16 little-endian when the manifest declares `dtype = "bf16"` (RD-1). The total float count SHALL be derived from the declared inventory, never from any model family's formula.

#### Scenario: N-1.0 — Slot vocabulary, widths, and layout
- **WHEN** an inventory is declared
- **THEN** slot names come from the closed set `input`,
  `post_attention`, `q_norm`, `k_norm`, `post_feedforward`, with no
  repeats and at least one slot; `input`, `post_attention`, and
  `post_feedforward` are `d_model` floats each; `q_norm` and `k_norm`
  are `head_dim` floats each (one shared vector per projection,
  broadcast across heads — Qwen3's post-projection Q/K RMSNorm
  convention); per-block vectors follow inventory order, then the final
  norm's `d_model` floats

### Requirement: The loader SHALL validate the norm sidecar fail-closed

The loader SHALL validate the sidecar byte length against the
inventory-derived count times the declared dtype's itemsize and reject
the directory fail-closed (per `qk-server` SR-1.2 semantics) on any
inventory violation (absent, unknown-named, repeated) or sidecar
violation (missing, truncated, miscounted at the declared itemsize).

#### Scenario: N-1.1 — Load succeeds with a complete sidecar
- **WHEN** the server loads a model directory whose sidecar matches
  `n_layers` blocks at `d_model` each, at its declared dtype's itemsize
- **THEN** loading completes and the engine reports ready

#### Scenario: N-1.2 — Missing or malformed sidecar fails closed
- **WHEN** the sidecar is absent, truncated, or wrong-count
- **THEN** the server never reports ready and exits non-zero naming the
  failing artifact

#### Scenario: N-1.3 — A second declared inventory of the same geometry loads
- **WHEN** two directories declare identical geometry but different
  inventories (e.g. `["input", "post_attention"]` vs
  `["input", "post_attention", "q_norm", "k_norm"]` vs a Gemma-style
  `["input", "post_attention", "post_feedforward"]`)
- **THEN** each loads iff its sidecar byte length equals its own
  inventory-derived count times its declared itemsize, and each is
  rejected against the other's count

### Requirement: The model config SHALL declare validated attention geometry

`config.toml` SHALL declare `n_layers`, `n_heads`, `n_kv_heads`, `head_dim`,
`rms_norm_eps`, and `rope_theta`. The loader SHALL reject configurations
where `n_kv_heads * head_dim != d_model` or `n_heads % n_kv_heads != 0`
(GQA group consistency), and any missing field.

#### Scenario: N-2.1 — Consistent geometry loads
- **WHEN** the declared geometry satisfies the GQA consistency equations
- **THEN** the engine reports ready with that geometry

#### Scenario: N-2.2 — Inconsistent geometry is rejected
- **WHEN** a geometry field is missing or violates the consistency
  equations
- **THEN** the server exits non-zero naming the violation

### Requirement: The engine SHALL execute transformer-block forward semantics

Generation SHALL apply, per block in manifest order: each declared norm slot,
packed factorized Q/K/V projections, RoPE, GQA causal self-attention, output
projection, residual adds, gate/up projections with SwiGLU, down projection —
then final norm and LM head, in f32 with the kernels rlib's f64 packed-matvec
accumulation. Each packed projection SHALL add the bias its container
carries; a pre-bias stream contributes the zero map. Slot application points
are normative per Scenario E-2.0.

#### Scenario: E-2.0 — Slot semantics
- **WHEN** a declared inventory drives the forward
- **THEN** `input` = RMSNorm on the block input (pre-attention);
  `post_attention` = RMSNorm on the attention output (pre-MLP);
  `q_norm`/`k_norm` = RMSNorm over `head_dim` on Q/K after projection,
  before RoPE; `post_feedforward` = RMSNorm on the MLP output
  (Gemma-style sandwich, residual around each normed sublayer)

#### Scenario: E-2.1 — Greedy decode is deterministic and EOS-terminated
- **WHEN** a prompt is decoded greedily twice with the same inputs
- **THEN** both runs produce byte-identical token streams and terminate on
  EOS under the existing EOS/stop semantics

#### Scenario: E-2.2 — Block forward matches the HF reference on a tiny fixture
- **WHEN** the engine's forward runs against the HF reference forward on a
  tiny Qwen3 fixture with identical weights and fixed inputs
- **THEN** the maximum absolute logit divergence is within 1e-2

#### Scenario: E-2.3 — Inventory presence drives the forward path
- **WHEN** a model declares `q_norm`/`k_norm` its Q/K are normalized
  before RoPE, and when a model omits them its Q/K are NOT normalized;
  the same binary serves both without code changes
- **THEN** each directory's forward is driven by its declared inventory,
  and the engine rejects no inventory it declares (Qwen3 and
  Llama-class both serve)

#### Scenario: E-2.4 — The forward applies the container bias
- **WHEN** two directories describe the same model, one with bias-carrying
  version-2 streams and one whose streams are pre-bias
- **THEN** the engine loads each bias at full width, a pre-bias stream
  contributes zeros, and the two forwards over identical token ids differ —
  an ignored bias would produce identical results.

### Requirement: Sage attention face SHALL be admitted only on a passing shipped gate record

The server SHALL expose `gpu:sage` as a third attention face (opt-in env,
frozen at load, honest `/health` label) only for models whose shipped gate
record passes the [SA] §4.3 metric clauses (Cosine Sim ≥ 0.999, Relative
L1 ≤ 0.02 vs the exact face). The gate SHALL distinguish evidence classes
by the record's `activations_kind`. A model that fails the gate MUST
refuse the face at startup with a typed error, never serving silently
degraded attention.

#### Scenario: Gate pass enables the face

- **WHEN** the opt-in is set and the model's shipped gate record is within
  the [SA] §4.3 metric bounds (Cosine Sim ≥ 0.999, Relative L1 ≤ 0.02 vs
  the exact face),
- **THEN** the server starts with `"attention_face": "gpu:sage"` and the
  label stays fixed for the process lifetime.

#### Scenario: Gate fail refuses the face

- **WHEN** the opt-in is set and the gate record is missing or out of
  bounds,
- **THEN** startup aborts with a typed error naming the gate (it does NOT
  silently fall back — a silently degraded model is worse than a failed
  start).

#### Scenario: Soft fallback on device failure only

- **WHEN** the gate passes but device init fails,
- **THEN** the face falls back to the exact tiled face (or scalar) exactly
  per the frozen-decision pattern — quality is never the fallback trigger.

#### Scenario: Fixture-battery evidence admits experimental status only

- **WHEN** the gate record's `activations_kind` is `"fixture-battery"` and
  the [SA] §4.3 metric clauses pass,
- **THEN** the face SHALL be admitted as `opt-in-experimental` with a WARN
  log naming the unmeasured real-activation quality, and SHALL NOT be
  promoted to a recommended status on that record alone.

#### Scenario: Real-activation evidence is the enabling class

- **WHEN** the gate record's `activations_kind` is a recognized
  real-activation source and the [SA] §4.3 metric clauses pass,
- **THEN** the face MAY be considered for status beyond experimental, and
  the log SHALL record the activation source and the clause values.

### Requirement: Gate record PPL clause is enforced when declared, admitted when absent

A shipped gate record that declares a PPL clause SHALL have its delta
within the bound recorded at that run; the server SHALL refuse the face on
violation. A record that declares no PPL clause SHALL be admitted on the
[SA] §4.3 metrics alone with the gap logged at WARN. The PPL delta bound
is a policy threshold (0.05, from the change's shipped-test convention),
not a paper-derived constant.

#### Scenario: PPL clause present and within bounds

- **WHEN** the gate record declares a `ppl` clause with `delta` and
  `bound` and `delta <= bound`,
- **THEN** the server admits the face on that clause, and the PPL delta
  value is recorded in the startup log.

#### Scenario: PPL clause present and out of bounds

- **WHEN** the gate record declares a `ppl` clause with `delta` and
  `bound` and `delta > bound`,
- **THEN** the server refuses the face with a typed error naming the PPL
  clause violation.

#### Scenario: PPL clause absent is admitted with a WARN

- **WHEN** the opt-in is set and the model's captured gate record declares no
  PPL clause,
- **THEN** the server admits the face on the [SA] §4.3 metrics and
  logs the absent clause at WARN level; no PPL value is invented (the
  clause's only producer is a real-model eval run requiring an explicit
  GO, and a fabricated number would be worse than the gap).

### Requirement: Quality gate provenance is shipped and reproducible

The gate record (activations source, per-layer metrics, PPL delta, date,
code commit) SHALL be shipped with the model directory and regenerable by a
committed tool against the pinned [SA] §4.3 metric definitions. The
`activations_kind` field SHALL name the evidence class
(`"fixture-battery"` or a recognized real-activation source), so the
face's enabling evidence is auditable rather than asserted.

#### Scenario: Gate record regeneration matches

- **WHEN** the committed gate tool re-runs against the model directory and
  its captured activation fixture,
- **THEN** it reproduces the shipped record's metrics within f32
  round-off, and the server's startup gate accepts it.

### Requirement: Serving parity across the quantized attention face

Generation through the `gpu:sage` face SHALL be indistinguishable from the
exact scalar face at the serving contract: identical token streams on the
fixture pipeline, with the final logits agreeing inside the face's OWN
quality gate ([SA] §4.3 — Cosine Sim ≥ 0.999, Relative L1 ≤ 0.02). The
INT8 face is lossy by design, so the exact face's reassociation budget
(rel ≤ 1e-5) SHALL NOT be applied to it.

#### Scenario: Quantized-face token identity on the fixture pipeline

- **WHEN** the fixture transformer generates greedily through the exact
  scalar face and through the `gpu:sage` face over identical inputs,
- **THEN** the token streams are identical and the final logits agree at
  the [SA] §4.3 gate (Cosine Sim ≥ 0.999, Relative L1 ≤ 0.02).

### Requirement: The loader SHALL resolve the chat template from its canonical location

On load the server SHALL resolve the model's chat template from the artifact's
canonical location `chat_template.jinja` — the directory file, or a blob's
same-named raw section — and MUST fall back to the legacy `chat_template`
string field in `tokenizer_config.json` (then the deprecated `tokenizer.json`
field) only when the canonical file is absent or blank. Resolving no template SHALL
leave it `None`, and `/v1/chat/completions` SHALL then refuse 400 rather than
guess a prompt format.

#### Scenario: CT-1.1 — The canonical `chat_template.jinja` file is read first

- **WHEN** a directory artifact contains a non-empty `chat_template.jinja`
  (the transformers-5.x sidecar no `tokenizer.json` field accompanies)
- **THEN** the loader resolves `chat_template` from that file's text
- **AND** a subsequent chat completion renders the prompt through that template
  without error

#### Scenario: CT-1.2 — Legacy fields are consulted only when the canonical file is absent

- **WHEN** an artifact has no `chat_template.jinja` but its
  `tokenizer_config.json` carries a `chat_template` string field
- **THEN** the loader resolves `chat_template` from `tokenizer_config.json`
- **AND** an artifact with neither a `.jinja` file nor a `tokenizer_config.json`
  field but a `tokenizer.json` field is resolved from that deprecated location
  so the pre-regeneration fixtures keep loading

#### Scenario: CT-1.3 — No template anywhere refuses the chat endpoint

- **WHEN** an artifact has no `chat_template.jinja`, no `tokenizer_config.json`
  field, and no `tokenizer.json` field (all absent, blank, or whitespace-only)
- **THEN** `chat_template` is `None` and `/v1/chat/completions` returns 400 in
  the invalid_request_error envelope naming the absence; no tokenization or
  generation is attempted

#### Scenario: CT-1.4 — A blob's `chat_template.jinja` section resolves identically

- **WHEN** a `.qpk` blob carries a raw `chat_template.jinja` section and the
  same source directory served as a directory artifact
- **THEN** both resolve the identical template string and render identical
  prompts on the same message

### Requirement: The exporter SHALL guarantee a non-empty canonical chat template

The exporter SHALL, after `tokenizer.save_pretrained(out_dir)`, verify the
artifact contains a non-empty `chat_template.jinja` whose trimmed content
equals the trimmed loaded tokenizer's `chat_template` string, and MUST refuse the
export — emitting no artifact — on any missing file, blank content, or
mismatch. It MUST NOT write the `chat_template` string into `tokenizer.json`
or `tokenizer_config.json`.

#### Scenario: CT-2.1 — A matching `/chat_template.jinja` passes the self-check

- **WHEN** an export leaves a `chat_template.jinja` whose trimmed content
  equals the trimmed loaded tokenizer's `chat_template`
- **THEN** the export completes and the artifact is emitted
- **AND** the artifact's `tokenizer.json` carries no `chat_template` field

#### Scenario: CT-2.2 — A missing, blank, or mismatched template refuses the export

- **WHEN** after `save_pretrained` the artifact lacks `chat_template.jinja`, or
  it is empty after trim, or its trimmed content differs from the trimmed tokenizer's
  `chat_template` string
- **THEN** the exporter raises with a diagnostic naming the file and the
  disagreement; no blob is emitted; the failed export's partial output
  directory SHALL NOT be registered as a successful artifact

### Requirement: A real-artifact regression leg SHALL verify chat serves from the canonical location

The suite SHALL include a gated regression leg that loads a REAL exported
artifact directory — activated by `QKB_REAL_ARTIFACT_DIR`, SKIP when unset —
and serves `/v1/chat/completions`, asserting a non-400 response and prompt
parity with the Python HF stack's `apply_chat_template` on the same message.
The committed fixtures SHALL reproduce the real-artifact shape: the template
in `chat_template.jinja` and no `chat_template` field in `tokenizer.json`.

#### Scenario: CT-3.1 — A real artifact serves chat with HF-prompt parity

- **WHEN** the gated leg loads a real exported artifact directory (a
  `chat_template.jinja` present with no `tokenizer.json` field — e.g. the
  v6b artifact) and `QKB_REAL_ARTIFACT_DIR` is set
- **THEN** `/v1/chat/completions` on the pinned probe message
  `[{"role":"user","content":"hello"}]` returns non-400
- **AND** the rendered prompt string equals HF's
  `AutoTokenizer.from_pretrained(<artifact>).apply_chat_template(messages,
  tokenize=False, add_generation_prompt=True)` with thinking off
  (oracle: `tools/gen_chat_golden.py`, `ENABLE_THINKING=False`)

#### Scenario: CT-3.2 — Committed fixtures reproduce the real-artifact shape

- **WHEN** `tools/gen_server_fixtures.py` regenerates the server fixtures and
  `tools/gen_chat_golden.py` regenerates the golden
- **THEN** the fixture's template lives in `chat_template.jinja` with no
  `chat_template` field in `tokenizer.json`
- **AND** the always-run suite asserts that shape loads and renders the
  unchanged golden (same template bytes, relocated)

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

### Requirement: The model directory SHALL declare its remainder dtype

The manifest `[sidecar]` block SHALL declare the remainder tensors'
dtype through an optional `dtype` key whose values are `fp32` or
`bf16`; an absent key SHALL read as `fp32`. The `weight_tying` suffix
SHALL accept `@fp32` or `@bf16` and SHALL equal the declared
`[sidecar]` dtype. Any value outside the declared set SHALL refuse the
load with a diagnostic.

#### Scenario: RD-1.1 — A declared bf16 directory is accepted
- **WHEN** a manifest declares `[sidecar] dtype = "bf16"` and its
  `weight_tying` line carries a matching `@bf16` suffix
- **THEN** the loader resolves bf16 for the embedding, lm_head, and norm
  sidecar and proceeds with loading

#### Scenario: RD-1.2 — An absent declaration reads fp32
- **WHEN** a pre-change-shaped manifest carries no `dtype` key
- **THEN** the loader reads the remainder as fp32 exactly as before the
  change — the directory loads with values identical to the pre-change
  loader

#### Scenario: RD-1.3 — Declaration disagreement refuses
- **WHEN** the `[sidecar]` dtype and the `weight_tying` suffix disagree
  (either direction: `@fp32` with bf16 declared, or `@bf16` with fp32
  declared)
- **THEN** the load refuses with a diagnostic naming both declarations

#### Scenario: RD-1.4 — Unknown tokens refuse
- **WHEN** the `dtype` key holds any value outside `fp32`/`bf16` (e.g.
  `f8`, `fp16`, `b16`, the empty string), or the `weight_tying` suffix
  holds any token outside `@fp32`/`@bf16` — including a value carrying
  no `@`-suffix at all
- **THEN** the load refuses with a diagnostic naming the offending token

#### Scenario: RD-1.5 — An unset weight_tying with a bf16 sidecar is accepted
- **WHEN** the manifest declares `[sidecar] dtype = "bf16"` and
  `config.toml` carries no `weight_tying` line (an untied head)
- **THEN** the loader reads the head file at bf16 like the other two
  remainder files, and loading proceeds

### Requirement: The loader SHALL read remainder tensors at the declared dtype

The loader SHALL resolve each remainder file's dtype from the
declaration only — never from file-size heuristics — and SHALL read the
embedding, lm_head, and norm sidecar at the declared itemsize (4 bytes
for fp32, 2 for bf16), validating the stat length against the
inventory-derived float count times that itemsize before any read.
Loaded values SHALL widen to f64 exactly (bf16 to f64 is exact).

#### Scenario: RD-2.1 — A bf16 fixture loads with pinned values
- **WHEN** the loader loads the committed bf16 twin fixture
- **THEN** every loaded remainder value equals its pinned f64 literal —
  the bf16-to-f64 widening is exact, and no fp32-sized read occurs

#### Scenario: RD-2.2 — A dtype/byte-length mismatch refuses both ways
- **WHEN** a bf16-declared file is actually fp32-sized, or an
  fp32-declared file is actually bf16-sized
- **THEN** the load refuses with a diagnostic naming the file, the
  declared dtype, and the expected byte count

#### Scenario: RD-2.3 — The fp32 fixture is unchanged by the change
- **WHEN** the committed fp32 fixture loads before and after the change
- **THEN** the loaded values are identical — the fp32 path's byte math
  is untouched

### Requirement: A bf16-remainder model SHALL decode within the declared bf16 tolerance

The decode logits of a bf16-remainder artifact SHALL sit within a
tolerance derived from bf16's 8-bit mantissa: per-element half-ulp 2^-8,
accumulated over the artifact's own tensor magnitudes per the design.md
formula — instantiated on the committed fixture as 2^-4 times
max(1, max fp32 logit). The float64 1e-12 parity tolerance SHALL NOT
govern bf16 comparisons.

#### Scenario: RD-3.1 — Fixture decode within the bf16 bound
- **WHEN** the bf16 twin fixture and its fp32 original decode the same
  token ids through the engine
- **THEN** the max absolute logit delta is within 2^-4 times
  max(1, max fp32 logit) — the bound computed from the fixture's own
  loaded tensors — and the max delta is nonzero, proving the bf16
  remainder was actually exercised

### Requirement: The server SHALL accept a single-file blob as model source

The server SHALL accept a `.qpk` single-file blob as its model
source. When the model path names a regular file, the loader SHALL
detect the blob format by its `QKPB` magic and load via the blob
path; a regular file without the magic SHALL refuse; when the path
names a directory, the loader SHALL fall back to the legacy directory
format (dual-read). The server SHALL report readiness only after the
blob passes every validation gate.

#### Scenario: BL-1.1 — A blob file is detected and loaded

- **WHEN** the server is started with a `ModelDirectory::load()` call
  targeting a `.qpk` file whose first four bytes are `QKPB`
- **THEN** the loader parses the blob header, validates the section table,
  constructs the `ModelDirectory`, and reports ready with the model's
  declared identity.

#### Scenario: BL-1.2 — A directory path still loads

- **WHEN** the server is started with a path that is a directory
  containing `manifest.toml` and a valid model
- **THEN** the legacy loader path executes and the server reports ready —
  the blob change does not break existing directories.

#### Scenario: BL-1.3 — A regular file without the magic refuses

- **WHEN** the model path names a regular file whose first four bytes
  are not `QKPB`
- **THEN** the loader refuses with a diagnostic naming the path — it
  never attempts a directory load or a partial parse.

### Requirement: The loader SHALL validate every blob section against the header metadata

The loader SHALL validate every blob section against the header
metadata before constructing any tensor. Layer-section byte lengths
SHALL match the stream's own declared size; remainder-section byte
lengths SHALL equal the inventory-derived float count (declared
norm_slots inventory, N-1.0 — never a family formula) times the
declared itemsize; every required section SHALL be present. Any
violation SHALL yield a named diagnostic naming the section and the
expected value.

#### Scenario: BL-2.1 — All required sections present

- **WHEN** a blob carries section entries for every declared layer, the
  embedding, norms, tokenizer.json, and (when present) an untied
  lm_head, each at the correct byte length for its dtype and geometry
- **THEN** the loader constructs a complete `ModelDirectory` with no
  errors.

#### Scenario: BL-2.2 — A missing required section refuses

- **WHEN** a blob's section table omits any declared layer name, the
  `"embedding"` entry, the `"norms"` entry, or the `"tokenizer.json"`
  entry
- **THEN** the loader refuses with a diagnostic naming the missing section.

#### Scenario: BL-2.3 — A wrong-length remainder section refuses

- **WHEN** a blob's `"embedding"` section has a byte length that is not
  `vocab_size * d_model * itemsize(remainder_dtype)`, or the `"norms"`
  section's length does not match
  `(per-block inventory count * n_layers + d_model) * itemsize(remainder_dtype)`,
  where the per-block count sums the declared `norm_slots` widths
  (`input`/`post_attention`/`post_feedforward` → `d_model`,
  `q_norm`/`k_norm` → `head_dim`) per N-1.0
- **THEN** the loader refuses with a diagnostic naming the section, the
  expected byte count, and the actual byte count.

### Requirement: The remainder dtype SHALL be read from the blob header only

The remainder dtype SHALL resolve from the header's `remainder_dtype`
field only. Remainder section tags SHALL equal the declared dtype
(fp32_le↔fp32, bf16_le↔bf16); layer section tags SHALL agree with the
stream's own version; either disagreement is a named refusal.
`weight_tying` SHALL be required: a `lm_head=<path>@{fp32,bf16}`
string, the sentinel, or null (untied head, no suffix); a carried
suffix SHALL agree with `remainder_dtype`, and null SHALL read the
head at the declared dtype.

#### Scenario: BL-3.1 — The declared dtype and weight_tying suffix agree

- **WHEN** a blob header declares `"remainder_dtype": "bf16"` and its
  `weight_tying` value carries a `@bf16` suffix
- **THEN** the loader resolves bf16 for all remainder sections and loads
  the model.

#### Scenario: BL-3.2 — A suffix disagreement refuses

- **WHEN** the header's `remainder_dtype` and `weight_tying` suffix do
  not match (either direction)
- **THEN** the loader refuses with a diagnostic naming both declarations.

#### Scenario: BL-3.3 — A remainder tag disagreeing with the header refuses

- **WHEN** a remainder section's dtype tag does not match the header's
  `remainder_dtype` (e.g. `remainder_dtype: fp32` with an `embedding`
  tagged `bf16_le`)
- **THEN** the loader refuses with a diagnostic naming the section, its
  tag, and the declared dtype.

#### Scenario: BL-3.4 — A layer tag disagreeing with the stream version refuses

- **WHEN** a layer section's tag names a qpk1 stream version that
  differs from the version its own stream header declares (e.g.
  `qpk1_v2` over a stream whose header declares version 1)
- **THEN** the loader refuses with a diagnostic naming the section, its
  tag, and the stream's declared version.

#### Scenario: BL-3.5 — A null weight_tying reads the head at the declared dtype

- **WHEN** the header's `weight_tying` is null and `remainder_dtype` is
  `"bf16"`
- **THEN** the loader reads the `"lm_head"` section at bf16 like the
  other remainder sections, and loading proceeds.

### Requirement: The loader SHALL mmap the blob with alignment validation

The loader SHALL map the blob file with mmap (or, on platforms without
mmap, a page-aligned read) rather than slurping the whole file into a
buffer. The 4096-byte alignment of every section offset SHALL be
asserted at load before any mmap call. Section payloads SHALL be
accessed through mapped views or page-aligned copies; the section table
itself SHALL be read into memory.

#### Scenario: BL-4.1 — A well-formed blob maps and loads

- **WHEN** a valid blob is presented to the loader
- **THEN** the file is opened, every section offset is verified
  4096-aligned, mmap is requested, and at least the layer-stream
  sections are accessed through the resulting mapped views.

#### Scenario: BL-4.2 — A misaligned blob refuses before mmap

- **WHEN** any section offset in a blob's table is not a multiple of
  4096
- **THEN** the loader refuses with a diagnostic naming the section and
  its offset — it never attempts to mmap the file.

### Requirement: Attention face dispatch (frozen decision, soft fallback)

The server SHALL select the attention face ONCE at model load —
`cpu:scalar` or `gpu:tiled` behind the `gpu` feature and an env opt-in —
using the same frozen-decision pattern as matvec dispatch: the face never
swaps mid-stream, and `/health` SHALL report the active face alongside
`matvec_face`. Device init failure SHALL be a soft fallback to `cpu:scalar`
(both faces compute the same attention within the parity budget).

#### Scenario: Face selection and honest health labeling

- **WHEN** the server starts with the opt-in enabled and a usable device,
- **THEN** `/health` reports `"attention_face": "gpu:tiled"`, the label
  stays fixed for the process lifetime, and with the opt-in unset it reports
  `cpu:scalar` with no device touched.

#### Scenario: Soft fallback on init failure

- **WHEN** the opt-in is set but device initialization fails,
- **THEN** the server serves on `cpu:scalar`, `/health` reports the CPU face
  (never a stale GPU label), and startup does not abort.

### Requirement: Serving parity across attention faces

Generation through the GPU attention face MUST be indistinguishable from the
scalar face at the serving contract: identical token streams on the fixture
pipeline, with logits agreement inside the reassociation budget (rel ≤ 1e-5;
bit-identity is not required — this face reassociates, per the 3.5
crystallization).

#### Scenario: Cross-face token identity on the fixture pipeline

- **WHEN** the fixture transformer generates greedily through each face over
  identical inputs,
- **THEN** the token streams are identical and final logits agree with
  relative error ≤ 1e-5.
