# qk-server Specification

## Purpose
TBD - created by archiving change qk-server. Update Purpose after archive.

## Requirements

### Requirement: Startup loads a local qpk1 model directory and reports ready only after success

On startup the server SHALL load the qpk1 model directory supplied on the
command line and MUST NOT report ready or serve inference traffic until every
layer stream and tokenizer file in the directory has loaded successfully under
the fail-closed reader rules of `qk-representation` (no partial model is ever
served).

#### Scenario: SR-1.1 — Ready only after a successful load
- **WHEN** the server is started with `--model-dir` pointing at a valid qpk1
  model directory and loading completes
- **THEN** `GET /health` reports ready (200) and the inference endpoints accept
  traffic only from that point forward
- **AND** the served model id corresponds to the loaded directory

#### Scenario: SR-1.2 — Failed load never reaches ready
- **WHEN** the server is started with a missing, malformed, or
  failed-validation model directory (bad magic, wrong version, unknown dtype
  tag, truncation, or trailing bytes, per the `qk-representation` reader rules)
- **THEN** the server never reports ready, serves no inference traffic, and
  exits non-zero with a diagnostic naming the failing file

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

The exported model directory SHALL declare its per-block norm inventory
in `config.toml` as an ordered `norm_slots` array. The sidecar SHALL
hold, per block, each declared slot's vector in inventory order, then
the model-final norm, as fp32 little-endian. The total float count SHALL
be derived from the declared inventory, never from any model family's
formula.

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
inventory-derived count and reject the directory fail-closed (per
`qk-server` SR-1.2 semantics) on any inventory violation (absent,
unknown-named, repeated) or sidecar violation (missing, truncated,
miscounted).

#### Scenario: N-1.1 — Load succeeds with a complete sidecar
- **WHEN** the server loads a model directory whose sidecar matches
  `n_layers` blocks at `d_model` each
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
  inventory-derived count, and each is rejected against the other's
  count

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

Generation SHALL apply, per block in manifest order: each declared norm
slot applied at its semantics-defined point, packed factorized Q/K/V
projections, RoPE, GQA causal self-attention over a KV cache, output
projection, residual adds, gate/up projections with SwiGLU, down
projection — then final norm and LM head, in f32 with the kernels rlib's
f64 packed-matvec accumulation. Slot application points are normative
per Scenario E-2.0.

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
