"""Kernel dispatch: Rust extension when available, NumPy reference otherwise.

The NumPy implementations are the executable specification of every kernel;
the Rust extension must match within the tolerances pinned in
``openspec/config.yaml`` (verification-gates section).
"""

try:  # pragma: no cover - depends on install extras
    import quantibbler_kernels as _kernels  # type: ignore[import-not-found]

    HAVE_KERNELS = True
except ImportError:
    _kernels = None  # type: ignore[assignment]
    HAVE_KERNELS = False


def backend() -> str:
    """Name of the active kernel backend: ``rust`` or ``numpy-reference``."""
    return "rust" if HAVE_KERNELS else "numpy-reference"
