# design.md — qk-server

Follows `openspec/config.yaml` (binding): Component decomposition and GPU
policy sections govern this change; evidence hierarchy, C/C++ build-graph ban,
dependency freshness rule, and verification gates apply as written.

## 1. Context

The repo today has a Python compressor scaffold (`src/quantibbler/`), a kernels
crate scaffold (`crates/quantibbler-kernels/`, `cdylib`+`rlib`, pyo3 0.30 with
abi3-py311, edition 2024, MSRV 1.85), and three sibling changes in flight:
`qk-representation` (pins the qpk1 container), `qk-kernels` (pins the kernel
contract and the Rust/NumPy parity gates; its rlib face is what this server
links), and `qk-ptq` (the compressor pipeline that produces qpk1 artifacts).
There is no root Cargo workspace: the kernels crate currently builds standalone
under uv/maturin as a pyo3 extension. The server is the third leg — pure Rust,
tokio + axum, tokenizers crate for BPE — consuming qpk1 model directories and
serving OpenAI-compatible HTTP, with no Python anywhere in its build graph
(config Component decomposition).

Evidence discipline for this document: design decisions cite the pinned papers
([NQ] = nanoquant-2602.06694v3, [LB] = littlebit-2506.13771v5,
[LB2] = littlebit2-2603.00042v2) where the papers speak; everything else is
flagged "informed assumption" with rationale and a failure-surface, per config
Evidence hierarchy. Registry floors below were verified today (2026-10-17)
against crates.io: tokio 1.53.3, axum 0.8.9, tokenizers 0.23.2, serde 1.0.229,
tracing 0.1.44, clap 4.6.7, candle-core 0.11.0 (updated 2026-06-26), mistralrs
0.8.1 (updated 2026-04-02), cubecl 0.11.0 (released 2026-10-06, repo pushed
2026-10-16). The dependency freshness rule (within one major; within one minor
if 0.x) applies to every floor the implementation pins — re-verify at
implementation time, never trust a floor from memory.

## 2. Decisions

### D1. Crate layout: `crates/quantibbler-server/` workspace member; binary `quantibbler-server`

The server lives at `crates/quantibbler-server/` (config fixes this path), a
new workspace member of a root `Cargo.toml` workspace that also admits
`crates/quantibbler-kernels` as a path dependency. The package produces one
binary, `quantibbler-server`, taking the model directory path and port via
clap:

```
quantibbler-server --model-dir <DIR> --port <PORT> [--host <HOST>]
                   [--max-concurrency <N>] [--serve-api-key <KEY>]
```

`--host` defaults to loopback, `--max-concurrency` defaults to a small N
(informed assumption IA3), and `--serve-api-key` turns on bearer checking when
set (unset = open server; D7). Internal layout:

- `src/main.rs` — clap parse, load orchestration, shutdown handling;
- `src/cli.rs` — clap definitions (derive API);
- `src/model/mod.rs` — model-directory acquisition: manifest scan, qpk1 layer
  load, tokenizer load;
- `src/api/mod.rs` + `src/api/{models,chat,completions,health,error}.rs` —
  axum routes, request/response types, SSE, error mapping;
- `src/engine/mod.rs` — the generation loop over kernels-rlib primitives;
- `src/sampler.rs` — greedy default (D6).

Rationale: config names both the crate path and the binary shape (model dir
path + port via clap); a workspace member keeps the kernels rlib a path
dependency with zero publishing ceremony, and matches how the kernels crate
already sits under `crates/`. One binary, no library target in v0.1 — tests
exercise the binary's modules directly; an extracted lib target can come later
without spec change.

### D2. Async runtime: tokio + axum, verified floors

Stack: `tokio` (full runtime: rt-multi-thread, net, signal, macros; floor
1.53.3), `axum` (floor 0.8.9, features as needed: json, tokio, http1), `serde`
+ `serde_json` (floor 1.0.229) for the OpenAI wire types, `tracing` +
`tracing-subscriber` (floors 0.1.44 / latest 0.3.x at implementation, re-verify)
for structured logs, `clap` (floor 4.6.7, derive) for the CLI,
`tokenizers` (floor 0.23.2) for BPE. Tower defaults for timeouts come with
axum; nothing exotic. All floors verified today against the live registry and
must be re-verified at implementation time (config freshness rule).

Rationale: config mandates "pure Rust: tokio + axum; tokenizers crate for BPE"
for this component. tokio 1.x and axum 0.8 are the current stable line of
both; pinning `major.minor` floors at implementation keeps the freshness rule
mechanical.

### D3. Model acquisition: directory walk + qpk1 layer load + tokenizer load (fail-closed)

A qpk1 model directory for v0.1 is:

```
<model-dir>/
  manifest.toml        # layer name -> qpk1 file, ordering, non-quantized parts
  tokenizer.json       # tokenizers (HF) JSON, loaded by the tokenizers crate
  config.toml          # eos/bos token ids, model-level hyperparameters
  layers/<name>.qpk1   # one qpk1 stream per factorized linear layer
```

Loading is fail-closed end to end: any missing file, any qpk1 reader rejection
(bad magic, version != 1, unknown dtype tag, truncation, trailing bytes — the
`qk-representation` reader rules), any tokenizer parse failure aborts startup
before the listener starts accepting; there is no partial model and no
best-effort load. The server never writes to the model directory.

Rationale: `qk-representation` pins the per-layer stream only (its
single-layer extent is a strict non-goal there) and explicitly defers
model-level packaging (its IA2: "users tar/key them by layer name
themselves"). Serving needs the composition, so this change defines the
*acquisition* convention above — manifest format kept deliberately minimal —
while the durable home for that contract is the representation capability
(OQ1). Fixture model directories for tests are generated byte-level against
qpk1's pinned layout (header `'QPK1'`, u32 LE fields, fp16/fp32 scale vectors,
row-major LSB-first packed bitplanes), so the server test suite does not
depend on the compressor lane landing first; correctness of the bytes is
already owned by the representation spec's exact-size and known-layer
scenarios.

### D4. Request lifecycle (non-streaming and streaming)

Non-streaming chat/completion: tokio task per connection (axum) →
authenticate (D7) → validate request body → acquire engine access (D5) →
tokenize prompt(s) → prefill (packed matmul over prompt tokens) → decode
until stop condition (max_tokens, EOS, stop sequence) → detokenize → usage
accounting (prompt + completion tokens via the tokenizers crate) → JSON
response.

Streaming: identical until the first token, then the response future switches
to an SSE body (`axum::response::sse::Sse` + a `tokio::sync::mpsc` channel);
each generated token is serialized as an OpenAI chunk object and sent as a
`data:` event; on completion the channel emits the final chunk (with finish
reason and, for chat, the role on the first content delta) then the literal
`data: [DONE]` sentinel. Client disconnect is observed through the SSE body's
drop/close and cancels that decode via token dropped — no orphaned generation
work continues beyond the current step.

### D5. Concurrency model: immutable shared weights behind `Arc`, engine access through a permit (informed assumption on lock-free reads)

The loaded model's weights are immutable after load (the server never writes
qpk1 or derived weight state), so weights live in `Arc<ModelWeights>` — shared
read access with no lock at all. The mutable per-process state (the generation
engine: KV cache allocation, sampler state, concurrency budget) sits behind a
`tokio::sync::Mutex<Engine>` or — the v0.1 choice — a fixed-size
`tokio::sync::Semaphore` of permits sized by `--max-concurrency`, one permit
per in-flight generation; tokenization, sampling plumbing, and SSE writing run
outside the permit. Queueing beyond the permit count backpressures naturally.

The task brief's "Arc RwLock / lock-free reads" is an informed assumption
about the concurrency shape, adopted here with one sharpening: `RwLock` is the
general answer when interior state is mutated; for v0.1 the weights side is
mutation-free after load, so the cheaper and harder-to-misuse shape is
lock-free *shared* reads (`Arc`) plus a semaphore-bounded engine. If profiling
or a future feature (weight hot-swap, adapter loading) introduces weight-side
mutation, that is the trigger to revisit toward `Arc<RwLock<...>>`; the
scenario contract (concurrent requests independent and uncorrupted) does not
change either way.

Rationale: correctness under concurrency is a spec scenario; immutability
after load is the strongest and simplest way to guarantee it (data races are
structurally excluded rather than locked around). The semaphore keeps memory
(OOM via unbounded KV cache growth) bounded, which is the real v0.1 failure
mode.

### D6. Generation path v0.1: greedy decode, plain KV cache, no batch scheduler

v0.1 decodes greedily (argmax over logits) with a per-request KV cache held
for the request's duration; no continuous batching, no paged attention, no
sampling parameters beyond presence in the request schema (accepted and
ignored with the response honestly reflecting greedy behavior — informed
assumption IA4).

Rationale: the papers' serving claims are about storage ([NQ] App. F; [LB]
App. D), not scheduler design; no paper evidence bears on batching policy, so
v0.1 takes the simplest correct scheduler and defers sophistication until a
profile demands it (config GPU policy's "profile showing material performance
unclaimed" pattern, applied to scheduling).

### D7. Error mapping: the OpenAI error envelope, single choke point

Every error response is produced by one axum IntoResponse type that renders
`{"error": {"message", "type", "param", "code"}}`:

| condition | status | type | code |
|---|---|---|---|
| malformed JSON body | 400 | invalid_request_error | (null) |
| invalid parameters (bad role, negative max_tokens, …) | 400 | invalid_request_error | (null) |
| unknown model id | 404 | invalid_request_error | model_not_found |
| request before readiness | 503 | server_error | model_not_ready |
| over concurrency budget / shutdown | 503 | server_error | (null) |
| internal failure | 500 | server_error | (null) |

Body-parse rejections also flow through this type (an invalid body never
leaks a non-envelope response). No auth failure path exists unless
`--serve-api-key` is set (then 401, `invalid_request_error`, code
`invalid_api_key`); plaintext HTTP without TLS in v0.1 is a non-goal boundary
(proposal), and no TLS stack is added — which also keeps C-building crypto
crates out of the build graph per the config ban.

### D8. GPU substrate: CPU-first decision record, three candidates evaluated, none adopted (config GPU policy)

Config's GPU policy is binding for v0.1: "Rust kernels: CPU-only in v0.1 on
BOTH faces (server and PyO3)... evaluate candle or mistral.rs as the GPU
substrate ONLY with registry evidence and a profile showing CPU leaves
material performance unclaimed. Abandoned crates are never adopted." This
decision record exists so the evaluation criteria and candidates are pinned
*before* any profile exists; it adopts nothing.

Registry evidence today (verified 2026-10-17):

| candidate | version | updated | recent downloads | category |
|---|---|---|---|---|
| candle-core | 0.11.0 | 2026-06-26 | 3.87M | tensor framework (CPU/CUDA backends, Rust-native ops) |
| mistralrs | 0.8.1 | 2026-04-02 | 144k | full inference engine (scheduler, sampling, OpenAI API built in) |
| cubecl | 0.11.0 | 2026-10-06 (repo 2026-10-16) | 681k | kernel-authoring framework (write kernels in Rust; compile to CUDA / HIP-ROCm / wgpu) |

The three are not interchangeable — category differs, and that drives the
trade:

- **cubecl** is the leading candidate for a *custom packed-matmul GPU kernel
  path*: quantibbler's hot loop is its own packed bitplane GEMV/GEMM, and
  cubecl would let `quantibbler-kernels` grow a GPU face of the same kernels
  from a single Rust source covering NVIDIA (CUDA), AMD (ROCm/HIP), and
  wgpu — matching multi-vendor deployment, and landing the GPU code in the
  kernels crate (where the parity gates live) rather than in the server. The
  server would consume it as just another kernels backend; the server's own
  dependency surface would not grow.
- **candle** is the faster-to-integrate *tensor-framework* path: standard ops
  already implemented, less kernel authoring, at the cost of expressing
  packed-bitplane math inside its op model and taking its backend set
  (CUDA-first) as given.
- **mistralrs** is a *full engine* — adopting it means adopting its scheduler,
  its model zoo assumptions, and its OpenAI server, replacing rather than
  complementing everything this change specifies. It is evaluated for
  completeness and kept only as a "what would we do if we stopped building a
  server" alternative; it is not a substrate for this design.

**Decision: v0.1 is CPU-native packed kernels (kernels rlib face), no GPU
substrate dependency, regardless of candidate.** The gate to revisit is
explicit: (a) a serving profile on real hardware showing CPU decode leaves
material performance unclaimed (the config GPU-policy trigger), and (b) at
that point, cubecl is the first candidate to prototype for a custom
packed-matmul kernel face in `quantibbler-kernels`, with candle as the
pragmatic fallback if framework-level integration wins on evidence. Freshness
rule applies at revisit time: re-verify versions/updates on the live registry
then; an updated-but-inactive candidate (no releases, dead repo) is
disqualified per "abandoned crates are never adopted".

This is a design record, not a dependency: no `candle-core`, `mistralrs`, or
`cubecl` entry appears in the server's (or kernels') Cargo manifest under this
change.

## 3. Discarded alternatives

- **DA1. Embedded Python (cpython crates, PyO3 interpreter embed) for
  tokenizer fallback or config.** Discarded: config bans it absolutely ("No
  Python dependency, no embedded interpreter, ever") and it would drag
  libpython into the build graph.
- **DA2. Calling the Python compressor subprocess to parse model dirs.**
  Discarded for the same reason plus runtime-process fragility; the server
  reads qpk1 bytes directly per `qk-representation`.
- **DA3. actix-web / hyper-only (no axum) / warp.** Discarded: config names
  axum for this component; hyper-only would hand-roll routing/SSE that axum
  provides; actix adds no capability and splits the stack story.
- **DA4. Multi-model registry keyed by model id (paths in a config file).**
  Discarded for v0.1 per proposal non-goals; single model per process keeps
  the concurrency and memory stories simple; registry is additive later.
- **DA5. Hot model reload (SIGHUP / watch).** Discarded: mutates the
  "weights immutable after load" invariant that D5's concurrency safety rests
  on; restart-under-supervisor is the v0.1 answer.
- **DA6. Sampling server-side (temperature/top-p/top-k implementation in
  v0.1).** Discarded: no paper evidence pins sampler semantics for this
  stack; greedy is honest, testable, and unblocks the API surface; sampling
  parameters are accepted but the response reflects greedy behavior until a
  change specifies sampler semantics (IA4 records the assumption).
- **DA7. TLS termination in-process (rustls or native-tls).** Discarded:
  deployment concern (reverse proxy is standard for OpenAI-compatible
  servers), and the config build-graph ban makes several common TLS stacks
  non-options anyway; plaintext loopback/vPC + proxy covers v0.1.
- **DA8. `Arc<RwLock<ModelWeights>>` from day one.** Discarded as the default:
  read-lock acquisition is pure overhead when nothing ever writes; D5 records
  the revisit trigger instead.
- **DA9. Per-request thread-per-core engine with no concurrency bound.**
  Discarded: unbounded concurrent generations OOM on KV cache before CPU
  saturates; the semaphore bound is the cheap safety rail (IA3).
- **DA10. Serving layers by mmap'ing qpk1 files lazily at request time.**
  Discarded for v0.1: load-time full materialization keeps the fail-closed
  story total (all bytes validated at startup) and memory layout under our
  control; lazy mmap is a future optimization gated on startup-time profiles
  for large models.

## 4. Open questions

- **OQ1. Model-directory manifest (tokenizer files, layer naming/ordering,
  non-quantized components) as a `qk-representation` requirement.**
  Status: open — this change flags the representation capability as needing
  it; owner: qk-representation lane. qpk1 today is strictly single-layer and
  explicitly stores no tokenizer or config; a server cannot acquire a model
  from a directory without a composition contract. This change proposes the
  representation lane gain: tokenizer files carried in the model dir, a
  manifest mapping layer names to qpk1 streams with serving order, and
  storage for non-quantized components (embedding/lm_head/residual weights).
  Until it lands, this lane's fixtures define the de-facto layout (D3).
  Trigger to resolve: first implementation task 2.x of this change, or the
  representation lane's next revision, whichever comes first.
- **OQ2. Non-quantized component representation (embeddings, lm_head,
  residual/norm weights) in the model dir.**
  Status: open — owner: qk-representation lane (with this lane as consumer).
  The qpk1 container stores factorized linear layers only; a transformer
  block needs more. Whether these ship as a sidecar npz-equivalent defined by
  the manifest, float16 raw binaries, or a future qpk1 dtype tag is the
  representation lane's call; this change consumes whatever that capability
  specifies and keeps its own fixtures minimal (config: interchange is qpk1
  only, so the addition must live in that capability, not here).
- **OQ3. Serving-scale KV cache dtype and layout (f16 vs f32 cache,
  per-layer ring buffer vs flat slab).**
  Status: open — owner: qk-server lane (implementation-time, profile-guided).
  No paper evidence pins it ([LB]/[NQ] do not specify serving caches);
  fixtures run fp32 cache for parity-testability, real sizing deferred to the
  first profile. Does not change any scenario in this delta.
- **OQ4. Sampler semantics (do temperature/top-p become specified server
  behavior?).**
  Status: open — owner: qk-server lane, next serving change. v0.1 documents
  greedy behavior explicitly; if callers require sampler parity with a
  reference stack, that is a specified-behavior change, not a silent default.

## 5. Informed assumptions

Each assumption states its rationale and how a wrong assumption would surface.

- **IA1. Tokenizer acquisition via `tokenizers` 0.23.2 (HF `tokenizer.json`)
  covers the qpk1 model dirs that matter in v0.1.** Rationale: config
  mandates the tokenizers crate for BPE for this component, and compressor
  outputs target HF-style BPE vocabularies in the pinned papers' setups. *If
  wrong* (a model family ships sentencepiece-only), server startup fails at
  tokenizer load with a clear diagnostic — fail-closed by design — and the
  representation lane's manifest (OQ1) is where a converter or alternative
  format would be specified. It cannot surface as a wrong-response bug; it
  surfaces as a refused load.
- **IA2. Model weights are immutable after load.** Rationale: the server has
  no writer path for qpk1 (non-goal), and no v0.1 feature mutates weights
  (DA5 discards hot reload). *If wrong*, the concurrency safety of D5
  degrades to whatever locking the mutation introduces — which is exactly
  D5's recorded revisit trigger (`Arc<RwLock<...>>`), applied before the
  mutation ships.
- **IA3. A small default concurrency bound (order 4-16) is adequate for v0.1
  CPU serving.** Rationale: CPU decode is the bottleneck; more concurrent
  generations than cores mostly grows KV cache memory, not throughput. *If
  wrong*, throughput reports show the queue saturating before cores do;
  `--max-concurrency` is already a flag, and retuning is operational, not a
  redesign.
- **IA4. Callers accept greedy-only generation in v0.1 with sampling params
  acknowledged-but-greedy.** Rationale: the API surface is the deliverable;
  sampler semantics are unspecified by the papers and deferred (OQ4). *If
  wrong*, it surfaces as user-visible response mismatch on sampled requests
  — the honest fix is specifying samplers (OQ4), not silently pretending the
  parameters took effect; the docs must state the greedy behavior plainly.
- **IA5. Fixture model dirs generated against qpk1's pinned byte layout are a
  sound test stand-in for compressor-produced artifacts.** Rationale: the
  representation capability pins the exact byte layout (header, offsets,
  sizes, padding) with property scenarios; any byte-valid fixture is
  indistinguishable from a compressor artifact at the reader level by
  construction. *If wrong* (the compressor emits something byte-legal but
  semantically different, e.g. layer ordering semantics), it surfaces at the
  first end-to-end integration test against real compressor output — a
  planned task in this change's tasks (integration gate) — not in production.
- **IA6. axum 0.8's SSE support (per-connection event streams, disconnect
  observation via body drop) is sufficient for the streaming scenarios.**
  Rationale: SSE-with-`[DONE]` is plain event framing; axum's `Sse` type plus
  an mpsc channel covers it, and disconnect cancellation rides tokio's
  structured concurrency. *If wrong*, it surfaces as the SR-4.3 scenario
  failing under load tests (orphaned generations), which would push the
  cancelation token deeper into the engine loop — contained work, no API
  change.
- **IA7. Verified registry floors (tokio 1.53.3, axum 0.8.9, tokenizers
  0.23.2, serde 1.0.229, tracing 0.1.44, clap 4.6.7) remain fresh through
  implementation.** Rationale: all verified today (2026-10-17) against the
  live registry; the freshness rule re-checks at bump time regardless. *If
  wrong* (a new major/minor lands mid-implementation), the rule's own
  re-verify step catches it at dependency-addition time — the failure mode
  is a task-time re-pin, not a design change.

## 6. Implementation crystallization (2026-10-17, qk-server lane)

Entries per AGENTS.md spec discipline: choices the implementation had to
make that the design was silent on, recorded alongside the code that
embodies them.

- **C1. The qpk1 container bitplane layout and the kernels rlib bitplane
  layout are DIFFERENT, and the server converts at load.** The container
  (qk-representation SF-3.1) packs each matrix as ONE continuous bit
  stream (element `(i, j)` at stream bit `i*r + j`, LSB-first within
  bytes); the kernels rlib packs PER-ROW bitplanes (each row starts a new
  byte, row width `ceil(r/8)`). They coincide exactly when `r % 8 == 0`.
  The server repacks at load (`engine::container_to_rows`) and also builds
  the Vb TRANSPOSE (`container_to_rows_transposed`, `r x din`), because
  the kernels matvec is row-oriented and `Vb^T z` contracts over the din
  axis — the transpose is built once at load, never per request. *Wrong-if*
  surface: parity fixtures with non-byte-aligned ranks would catch any
  layout drift; `r % 8 == 0` fixtures alone would not.
- **C2. `tokenizers` must be used with `default-features = false,
  features = ["fancy-regex"]`.** Verified against the published crate
  0.23.2: the default feature set is `["progressbar", "onig",
  "esaxx_fast"]`, where `onig` builds C Oniguruma and `esaxx_fast` routes
  `esaxx-rs` to its C++ backend — both violate the config C/C++/Cython
  build-graph ban. `fancy-regex` (pure Rust) is the sanctioned SysRegex
  backend (`src/utils/regex.rs` hard-requires one of onig/fancy-regex);
  tokenizers itself declares `esaxx-rs` with `default_features = false`,
  so its `cpp` feature stays off. BPE loading works identically under
  fancy-regex for the BPE-only v0.1 scope (IA1).
- **C3. Design D2's "serde_json 1.0.229" floor was a transcription slip:
  1.0.229 is serde's latest; serde_json's latest stable is 1.0.154** (the
  same version the kernels crate already dev-pins). The server pins
  serde_json 1.0.154. All other D2 floors re-verified live 2026-10-17 via
  the crates.io sparse index: tokio 1.53.3, axum 0.8.9, serde 1.0.229,
  tracing 0.1.44, tracing-subscriber 0.3.23 (env-filter feature added for
  `EnvFilter`), clap 4.6.7, tokenizers 0.23.2, toml 1.1.8, tokio-stream
  0.1.19.
- **C4. The v0.1 fixture model has no attention; the "KV cache" of D6 is
  deferred with it.** OQ2 (non-quantized components) is unresolved, so the
  fixture model defines the minimal serving stack that exercises every
  scenario: fp32 embedding lookup (last prompt token) -> factorized qpk1
  layer -> (tied) fp32 lm_head. Chat prompts join message contents with
  spaces; greedy decode walks `embedding -> W -> lm_head` per step. There
  is no attention and hence no KV cache to cache; when the representation
  lane lands real transformer sidecars (OQ2), the per-request cache lands
  with them — D6's scenario contracts (stop conditions, usage counting,
  independence) are already shaped for it. Sampling parameters remain
  accepted-but-greedy (IA4, OQ4).
- **C5. The fixture manifest's `[sidecar]` block is this lane's de-facto
  OQ2 answer** (as D3 anticipated): `embedding` and `lm_head` name raw
  little-endian float32 binaries of `vocab_size x d_model` row-major
  matrices under the model dir; equal paths mean weight tying. The
  representation lane owns the durable contract; until then the fixtures
  pin it.
- **C6. Gates after the workspace manifest: `uv run maturin build` from
  the repo root must now pass `-m crates/quantibbler-kernels/Cargo.toml`**
  (maturin reads a Cargo.toml, and the root one is a virtual workspace
  manifest with no `[package]`). `uv sync` (which builds the extension via
  the uv workspace member) is unaffected.
- **C7. Test-infrastructure note: integration tests embed the binary's
  modules via `#[path]` (D1's no-lib-target rule) with
  `#![allow(clippy::duplicate_mod)]`**, and in-process router fixtures run
  axum on a dedicated OS thread so blocking test HTTP clients cannot starve
  the server's reactor.

### Landing notes (orchestrator, 2026-10-17)

- Task 2.1's checkbox is unticked: the child disclosed that test-first was
  compressed (tests and implementation landed together after task 1.1's
  scaffolding existed). Same evidence standard as qk-kernels 8.1 — the
  RED state was never literally observed, so the box stays open.
- Registry evidence method: `api.crates.io` was not resolvable from the
  container (DNS); freshness was verified against the crates.io sparse
  index (index.crates.io) plus static.crates.io tarball inspection —
  including confirming `esaxx-rs`'s C++ path is gated behind a `cpp`
  feature that `tokenizers` leaves off. Same registry source cargo uses.
