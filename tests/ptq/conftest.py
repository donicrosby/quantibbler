"""Determinism pins for the ptq tier (hard dev dependencies, never extras).

The ptq tier tests the PRODUCT's torch lane (PTQ driver, calibration, KD).
torch, transformers, tokenizers, safetensors and pyarrow are HARD dev
dependencies of this repo (``[dependency-groups].dev``), so there is
deliberately NO skip-when-absent hook here: a missing one of them is a sync
error, and the tier must fail loudly at collection. Ignoring the directory
would report a green suite with a hole in it — the exact false green this
tier's dependencies stopped being extras to prevent. Run the full tier with:

    uv sync && uv run pytest
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


# Import-time, like the OPENBLAS pin above: the conftest is loaded before any
# test module in this directory, so the thread count is fixed before the first
# reduction the divergence gate reads.
_pin_determinism()
