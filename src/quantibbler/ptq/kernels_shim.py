"""Parity-gated kernel dispatch re-export for the PTQ package.

The qk-kernels capability owns the dispatch contract
(``quantibbler.kernels``); this shim re-exports it under ``quantibbler.ptq``
so the pipeline consumes the same parity-gated surface (NumPy reference is
the verification fixture at the declared dtype; Rust must match within the
per-dtype tolerances — bf16 backward 1e-2, f32 backward 1e-6, float64
rel 1e-12 — per config.yaml) without reimplementing or shadowing any kernel.
"""

from ..kernels import HAVE_KERNELS, backend, pack_bits, unpack_bits

__all__ = ["HAVE_KERNELS", "backend", "pack_bits", "unpack_bits"]
