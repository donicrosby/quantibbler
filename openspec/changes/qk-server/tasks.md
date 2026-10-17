# tasks.md — qk-server

Every task is independently verifiable with a command and expected outcome.
Tests precede implementation per capability; test names reference scenario IDs
from `specs/qk-server/spec.md`. Commands run from repo root
(`/opt/data/quantibbler`) unless noted; Rust server commands run from
`crates/quantibbler-server/` as noted. Registry floors named below were
verified 2026-10-17 (config freshness rule) and MUST be re-verified against
the live registry when pinned: tokio 1.53.3, axum 0.8.9, tokenizers 0.23.2,
serde 1.0.229, tracing 0.1.44, clap 4.6.7.

## 1. Workspace and harness scaffold

- [ ] 1.1 Create root `Cargo.toml` workspace admitting `crates/quantibbler-kernels`
      and `crates/quantibbler-server` as members (kernels keeps its maturin
      pyproject; the workspace is additive and must not change how uv builds
      the extension), and scaffold `crates/quantibbler-server/` (edition 2024,
      rust-version 1.85, Apache-2.0, binary `quantibbler-server`) with
      dependency floors from the verified set (tokio, axum, serde, serde_json,
      tracing, tracing-subscriber, clap, tokenizers) plus a dev-dependency on
      the kernels rlib path and an HTTP test client. Verify: from
      `crates/quantibbler-server/`, `cargo build` compiles a stub main
      (exit 0), and from `crates/quantibbler-kernels/`, `cargo build` still
      succeeds unchanged (exit 0).
- [ ] 1.2 Fixture generator `tools/gen_server_fixtures.py` writing a minimal
      valid qpk1 model directory (manifest.toml, tokenizer.json for a tiny BPE
      vocab, config.toml, layers/*.qpk1 byte streams per the pinned qpk1
      layout: 24-byte header magic 'QPK1' version 1 dtype tag dout din r,
      fp16 scale vectors, row-major LSB-first packed bitplanes) into
      `tests/fixtures/server-model/`, plus one corrupted variant per
      SR-1.2 rejection class. Verify: `uv run python
      tools/gen_server_fixtures.py` writes the directory and each qpk1 file's
      length matches the qk-representation size formula
      `24 + (dout + din) * w + ceil(dout*r/8) + ceil(din*r/8)` (exit 0).

## 2. CLI, model loading, readiness (SR-1.*)

- [ ] 2.1 Test-first: `crates/quantibbler-server/tests/load.rs` with
      `load_ready_only_after_success` (SR-1.1) and
      `load_failure_never_ready` (SR-1.2, parameterized over the corrupted
      variants from 1.2), asserting via `GET /health` against a spawned
      server process that ready is reported only after successful load and
      failed loads exit non-zero with a diagnostic naming the failing file.
      Verify (from `crates/quantibbler-server/`): `cargo test load_ -- --nocase`
      — tests compile and fail (RED) with "not implemented" panics; exit code
      101 is the expected outcome at this stage.
- [ ] 2.2 Implement clap CLI (`--model-dir <DIR>` required, `--port <PORT>`
      required, `--host` default loopback, `--max-concurrency` default small,
      `--serve-api-key` optional) and fail-closed directory loading
      (manifest scan, qpk1 layer load via the kernels rlib face, tokenizers
      load) in `src/cli.rs` + `src/model/`; startup binds the listener only
      after the load completes. Verify (from `crates/quantibbler-server/`):
      `cargo test load_ -- --nocase` — both tests pass (exit 0).
- [ ] 2.3 Routing and health endpoint: `GET /health` 200 with readiness true
      once serving (SR-6.1) and 503 while not ready (SR-6.2). Verify (from
      `crates/quantibbler-server/`): `cargo test health_ -- --nocase` —
      passes (exit 0).

## 3. Models and non-streaming inference endpoints (SR-2.*, SR-3.*, SR-5.1, SR-9.*)

- [ ] 3.1 Test-first: `tests/api_models.rs` — `models_lists_single_loaded_model`
      (SR-2.1: OpenAI-shaped list, exactly one entry, object "model", id equal
      to the served id) and `no_second_model_can_be_registered` (SR-2.2: no
      registry/hot-swap route exists; probing returns 404 through the error
      envelope). Verify (from `crates/quantibbler-server/`): `cargo test
      models_ -- --nocase` — compile-and-fail RED is expected at this stage
      (exit 101).
- [ ] 3.2 Implement `GET /v1/models` in `src/api/models.rs` (single hardcoded
      entry from the loaded model; no registration surface). Verify (from
      `crates/quantibbler-server/`): `cargo test models_ -- --nocase` —
      passes (exit 0).
- [ ] 3.3 Test-first: `tests/api_chat.rs` — `chat_nonstreaming_openai_shape`
      (SR-3.1: object "chat.completion", choices[].message.{role, content},
      usage tokens), `chat_unknown_model_404` (SR-3.2: error envelope, code
      model_not_found), and `chat_rejects_malformed_body_400` (SR-7.1 leg).
      Verify (from `crates/quantibbler-server/`): `cargo test chat_ -- --nocase`
      — compile-and-fail RED expected (exit 101).
- [ ] 3.4 Implement `POST /v1/chat/completions` (`stream: false`) with the
      OpenAI response types in `src/api/chat.rs`, greedy decode via the
      engine (task 4), unknown-model and malformed-body error mapping through
      the single envelope type (design D7). Verify (from
      `crates/quantibbler-server/`): `cargo test chat_ -- --nocase` — passes
      (exit 0).
- [ ] 3.5 Test-first + implement `POST /v1/completions` (`stream: false`,
      object "text_completion", choices[].text, usage — SR-5.1; unknown-model
      404 shared with 3.4's mapping) as `tests/api_completions.rs::`
      `completions_nonstreaming_openai_shape`. Verify (from
      `crates/quantibbler-server/`): `cargo test completions_ -- --nocase` —
      passes (exit 0).
- [ ] 3.6 Concurrency probes `tests/api_concurrency.rs`:
      `n_concurrent_requests_all_complete` (SR-9.1, N over the semaphore
      bound and under it) and `one_failing_request_does_not_corrupt_others`
      (SR-9.2, one malformed body among in-flight generations). Verify (from
      `crates/quantibbler-server/`): `cargo test` — all green including these
      (exit 0).

## 4. Engine and concurrency core (backs SR-3.1, SR-9.*; informed-assumption checks)

- [ ] 4.1 Implement `src/engine/`: weights as `Arc<ModelWeights>` shared
      lock-free (immutable after load, design D5), engine access via
      `tokio::sync::Semaphore` permits sized by `--max-concurrency`, greedy
      decode loop (prefill + decode with per-request KV cache, design D6)
      over kernels-rlib packed matvec/matmul primitives, token counting via
      the tokenizers crate. Verify (from `crates/quantibbler-server/`):
      `cargo test` — engine unit tests (prompt tokenization, greedy
      termination on EOS and max_tokens, usage counts) pass (exit 0).
- [ ] 4.2 `ldd`/`cargo tree` purity check as an executable assertion:
      `tests/binary_purity.rs` (or an xtask check) asserting the built binary
      links no libpython (SR-8.2) and `cargo tree -e normal` contains no
      pyo3-embedded-interpreter path (kernels rlib with default features,
      extension-module feature off). Verify (from
      `crates/quantibbler-server/`): `cargo run --bin quantibbler-server --
      --help` works and the purity check passes (exit 0).

## 5. Streaming SSE (SR-4.*, SR-5.2) and error envelope (SR-7.*)

- [ ] 5.1 Test-first: `tests/api_streaming.rs` — `chat_streaming_sse_framing`
      (SR-4.1: content-type text/event-stream, data: {json} chunks, assistant
      role on first content delta), `chat_streaming_ends_at_done_sentinel`
      (SR-4.2: literal final `data: [DONE]`),
      `completions_streaming_ends_at_done_sentinel` (SR-5.2), and
      `client_disconnect_aborts_only_its_stream` (SR-4.3: disconnect mid-
      stream; concurrent streams unaffected). Verify (from
      `crates/quantibbler-server/`): `cargo test streaming_ -- --nocase` —
      compile-and-fail RED expected (exit 101).
- [ ] 5.2 Implement SSE bodies via axum Sse + mpsc channel in
      `src/api/chat.rs` / `src/api/completions.rs`: chunk emission per token,
      final chunk with finish reason, literal `data: [DONE]` sentinel,
      disconnect-observed cancellation of that decode only (design D4, IA6).
      Verify (from `crates/quantibbler-server/`): `cargo test streaming_ --
      --nocase` — passes (exit 0).
- [ ] 5.3 Implement the single error-envelope IntoResponse type (design D7)
      and route every error path through it: 400 invalid_request_error
      (malformed JSON, invalid params), 404 model_not_found, 503
      model_not_ready before readiness (SR-7.2), 503 on concurrency/shutdown
      rejection, 500 server_error (SR-7.3, including body-parse
      rejections). Test-first where not already covered:
      `tests/api_errors.rs::errors_notready_503_envelope` (SR-7.2) and
      `errors_internal_500_envelope` (SR-7.3, injected engine failure).
      Verify (from `crates/quantibbler-server/`): `cargo test errors_ --
      --nocase` — passes (exit 0).

## 6. Integration against real artifacts and docs

- [ ] 6.1 End-to-end smoke against a compressor-produced qpk1 model directory
      (the qk-ptq lane's tiny-model artifact when it exists, else the fixture
      dir plus a byte-diff review against a real compressor layer stream):
      quantize -> serialize -> serve -> complete, asserting the fixture
      stand-in assumption (design IA5) holds at the reader level. Verify:
      from `crates/quantibbler-server/`, `cargo test e2e_ -- --nocase` —
      passes (exit 0); any byte-layout divergence is a qk-representation
      finding, raised to that lane, not patched around here.
- [ ] 6.2 README section (or `crates/quantibbler-server/README.md`) covering
      build/run (`cargo run -p quantibbler-server -- --model-dir ... --port
      ...`), endpoints, greedy-only generation note (design OQ4/IA4), and
      the no-Python / no-download / single-model constraints. Verify:
      `cargo run -p quantibbler-server -- --help` exits 0 and the documented
      flags appear in its output (exit 0).

## 7. Full gate suite (final task)

- [ ] 7.1 Run the full gate suite and record results — every gate must pass
      with the new crate included and the existing lanes untouched:
      1. `cargo test` (in `crates/quantibbler-server/`) — PASS/FAIL: ____
      2. `cargo clippy --all-targets -- -D warnings` (in
         `crates/quantibbler-server/`) — PASS/FAIL: ____
      3. `cargo fmt --check` (in `crates/quantibbler-server/`) — PASS/FAIL: ____
      4. `cargo test` (in `crates/quantibbler-kernels/`) — unchanged-and-green —
         PASS/FAIL: ____
      5. `cargo clippy --all-targets -- -D warnings` +
         `cargo fmt --check` (in `crates/quantibbler-kernels/`) —
         PASS/FAIL: ____
      6. `uv run pytest` — untouched-and-green — PASS/FAIL: ____
      7. `uv run maturin build` — still green — PASS/FAIL: ____
      Verify: all seven PASS recorded above with command versions/dates in
      the change PR description; this change does not land with any gate
      failing.
