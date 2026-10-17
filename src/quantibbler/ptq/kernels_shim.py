"""Parity-gated kernel dispatch re-export for the PTQ package.

The qk-kernels capability owns the dispatch contract
(``quantibbler.kernels``); this shim re-exports it under ``quantibbler.ptq``
so the pipeline consumes the same parity-gated surface (NumPy reference is
the executable specification; Rust matches at 1e-12 rel on float64 per
config.yaml) without reimplementing or shadowing any kernel.
"""

from ..kernels import HAVE_KERNELS, backend, pack_bits, unpack_bits

__all__ = ["HAVE_KERNELS", "backend", "pack_bits", "unpack_bits"]
