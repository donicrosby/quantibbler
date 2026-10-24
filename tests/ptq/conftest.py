"""Tier gate for the ptq tests (torch/hf extras).

The ptq tier tests the PRODUCT's torch lane (PTQ driver, calibration, KD):
torch + transformers are runtime deps of that lane (extras `torch`/`hf`),
not dev deps — a user quantizing a model needs them with no test runner
in sight. This conftest mirrors the kernels crate's hardware-gate pattern:
without the extras the tier is IGNORED cleanly (lean suite stays green)
instead of erroring at collection. Run the full tier with:
    uv sync --all-groups --extra kernels --extra torch --extra hf
"""

from __future__ import annotations

import os

# BLAS threading is a reproduction parameter, not a default. OpenBLAS picks its
# reduction order from the thread count, and THIS variable must be set before
# the library loads, so it lives at module scope -- an assignment inside a
# pytest hook runs after numpy may already be imported and would not re-thread
# an already-loaded OpenBLAS. Companion to _pin_determinism()'s
# torch.set_num_threads(1).
#
# Known, deliberate gap: the production wrapper sets OPENBLAS_NUM_THREADS=16
# (tools/run-ptq-container.sh) as a perf choice, so test numerics and
# production numerics use different BLAS reduction orders by construction.
# Recording that in the run fingerprint is a separate schema change.
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def pytest_ignore_collect(collection_path, config):  # noqa: ANN001, ARG001
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return True
    _pin_determinism()
    return None


def _pin_determinism() -> None:
    """One thread, one reduction order.

    BLAS picks its reduction order from the thread count, so a multi-threaded
    run can give a different sum for the same inputs. The divergence gate reads
    that sum, and this fixture family sits close enough to the band that
    reduction order can decide which side it lands on. Pinning the thread count
    is the same determinism discipline the rest of the box already runs, and it
    is what lets a recorded seed-band measurement mean anything.
    """
    import torch

    torch.set_num_threads(1)
