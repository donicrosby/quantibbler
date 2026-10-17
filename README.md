# quantibbler

Sub-1-bit weight quantization for large language models via latent binary
factorization. Python-first, with optional Rust kernels.

A weight matrix `W` is represented as

```
W ≈ diag(s₁) · (Uᵇ Vᵇᵀ) · diag(s₂)
```

where `Uᵇ, Vᵇ ∈ {−1,+1}` are packed at one bit per entry and `s₁, s₂` are
full-precision scale vectors — driving effective storage below one bit per
weight, counted from serialized bytes.

Two paths:

- **PTQ** — ADMM initialization, block-wise reconstruction, scale
  calibration. No training required.
- **QAT** — SVD or ITQ initialization, straight-through sign estimation,
  distillation-recovered training.

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
