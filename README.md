# quantibbler

Sub-1-bit weight quantization for large language models via latent binary
factorization — a Python compressor and a pure-Rust OpenAI-compatible
inference server.

A weight matrix `W` is represented as

```
W ≈ diag(s₁) · (Uᵇ Vᵇᵀ) · diag(s₂)
```

where `Uᵇ, Vᵇ ∈ {−1,+1}` are packed at one bit per entry and `s₁, s₂` are
full-precision scale vectors — driving effective storage below one bit per
weight, counted from serialized bytes.

## Components

- **`quantibbler`** (Python) — the compressor. Post-training quantization
  (ADMM initialization, block-wise reconstruction, scale calibration) and
  quantization-aware training (SVD/ITQ initialization, straight-through
  sign estimation, distillation recovery). Emits `qpk1` model directories.
- **`quantibbler-server`** (Rust) — serving in the llama.cpp / vLLM shape:
  an OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/completions`,
  `/v1/models`) that loads compressed models from local paths or network
  volumes and serves token streaming. Pure Rust — tokio + axum, no embedded
  interpreter.
- **`quantibbler-kernels`** (Rust) — one kernel implementation, two
  consumers: the server links it directly; Python binds the same code via
  PyO3. NumPy reference implementations are the verification fixtures at
  the declared dtype; Rust must match them within the per-dtype tolerances
  the qk-kernels capability declares on every probe.

## Workflow

```bash
# compress (Python, GPU-optional)
uv run quantibbler compress --model ./my-model --calibration ./calib.json --out ./my-model-qpk1

# serve (Rust)
quantibbler-server ./my-model-qpk1 --port 8080

# use
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "my-model-qpk1", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Status

Spec-first. Requirements and task state live under `openspec/`;
implementation changes land against them. See `openspec/config.yaml`
for the architecture invariants.

## References

- LittleBit: Ultra Low-Bit Quantization via Latent Factorization
  (NeurIPS 2025) — arXiv:2506.13771
- LittleBit-2: Latent Geometry Alignment (ICML 2026) — arXiv:2603.00042
- NanoQuant: Efficient Sub-1-Bit Quantization (ICML 2026) — arXiv:2602.06694

Reference copies for citation live in `docs/papers/`.

## License

Apache-2.0
