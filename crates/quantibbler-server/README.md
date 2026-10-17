# quantibbler-server

Pure-Rust OpenAI-compatible inference server for qpk1 models
(the `qk-server` capability, v0.1). CPU-only, no Python anywhere in its
build graph, no network downloads, one model per process.

## Build and run

```sh
# from the repo root (workspace member; kernels rlib is a path dependency)
cargo build -p quantibbler-server

# serve a qpk1 model directory
cargo run -p quantibbler-server -- --model-dir /path/to/model --port 8080
```

Flags (see `--help`):

| flag | meaning | default |
|---|---|---|
| `--model-dir <DIR>` | qpk1 model directory (required) | — |
| `--port <PORT>` | TCP port (required) | — |
| `--host <HOST>` | bind address | `127.0.0.1` |
| `--max-concurrency <N>` | max in-flight generations | `4` |
| `--serve-api-key <KEY>` | require `Authorization: Bearer <KEY>` | open server |

## Model directory layout (v0.1 acquisition convention)

```
<model-dir>/
  manifest.toml        # model_id, ordered layer list, [layer_files], [sidecar]
  tokenizer.json       # HF tokenizers JSON (BPE), loaded by the tokenizers crate
  config.toml          # vocab_size, d_model, eos_token_id, bos_token_id
  layers/<name>.qpk1   # one qpk1 stream per factorized linear layer
  <sidecar>.f32.bin    # fp32 embedding / lm_head raw LE float32 matrices
```

Loading is fail-closed: any missing file, malformed qpk1 stream (bad magic,
wrong version, unknown dtype tag, truncation, trailing bytes, infeasible
rank), or tokenizer parse failure aborts startup with a diagnostic naming
the file — the listener never starts on a partial model.

## Endpoints

- `GET /health` — `200 {"ready": true, ...}` once serving; `503` while not ready.
- `GET /v1/models` — OpenAI list with exactly the one loaded model.
- `POST /v1/chat/completions` — chat completions; `"stream": true` switches to SSE.
- `POST /v1/completions` — legacy text completions; `"stream": true` switches to SSE.

Streaming uses server-sent events in the OpenAI chunk format and terminates
with the literal `data: [DONE]` sentinel. A client disconnect aborts only
that decode.

## Errors

Every error response uses the OpenAI envelope
`{"error": {"message", "type", "param", "code"}}`: 400
`invalid_request_error` (malformed JSON, invalid parameters), 404
`invalid_request_error` code `model_not_found`, 503 `server_error` code
`model_not_ready` before readiness, 500 `server_error` for internal
failures.

## v0.1 constraints (by design)

- **Greedy-only generation.** Decoding is argmax; sampling parameters
  (`temperature`, `top_p`) are accepted but the response honestly reflects
  greedy behavior (design OQ4/IA4) until a change specifies sampler
  semantics.
- **No Python.** No embedded interpreter, no PyO3 extension-module feature,
  no compressor dependency anywhere in the build graph (asserted by
  `tests/binary_purity.rs`).
- **No downloads, no model zoo.** Models come from local paths or network
  mounts; downloading is the caller's job. No runtime network access.
- **One model per process.** No registry, no hot-swap; restart under a
  supervisor to serve a different model.
- **No TLS, no auth by default.** Plaintext HTTP on loopback by default;
  TLS termination is the deployment's job (reverse proxy). Pass
  `--serve-api-key` to require a bearer token.
- **CPU-only serving** through the `quantibbler-kernels` rlib face (packed
  bitplane kernels); the GPU substrate is a recorded design decision
  (qk-server design D8), not a dependency.
