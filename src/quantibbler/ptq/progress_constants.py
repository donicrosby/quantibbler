"""Measured per-phase rate constants, for ETA sanity-checking.

These exist so a progress bar cannot lie. The failure they prevent is concrete: a
bar that predicts 20 minutes for a stage measured at 21 hours reads as
information and is actually misdirection, and this campaign has already paid for
misleading instrumentation more than once.

Constants are measured on the target hardware at the protocol geometry. Re-measure
after any change to the solver backend, the attention implementation, or the GPU
driver version -- each of those moves the pace by more than the constants are
worth. `measured_env` records what they were measured against so a mismatch is
detectable rather than assumed away.
"""

from __future__ import annotations

import torch

# One entry per phase name, one value per hardware target.
# `seconds_per_unit` is wall-clock seconds for one `unit` of work in that phase.
RATE_CONSTANTS: dict[str, dict[str, dict[str, object]]] = {
    "admm": {
        "gfx1151": {
            "seconds_per_unit": 89.0,
            "unit": "linear",
            "measured_on": "2026-10-23",
            "geometry": "seq_len=512 rank=512 iters=400",
        },
        "cpu-fp64": {
            "seconds_per_unit": 89.0,
            "unit": "linear",
            "measured_on": "2026-10-23",
            "geometry": "seq_len=512 rank=512 iters=400",
        },
    },
    "svd": {
        "gfx1151": {
            "seconds_per_unit": 19.4,
            "unit": "linear",
            "measured_on": "2026-10-23",
            "geometry": "seq_len=512 rank=512",
        },
    },
    "ste": {
        "gfx1151": {
            "seconds_per_unit": 154.0,
            "unit": "linear",
            "measured_on": "2026-10-23",
            "geometry": "seq_len=2048 rank=512",
        },
        "cpu-fp64": {
            "seconds_per_unit": 154.0,
            "unit": "linear",
            "measured_on": "2026-10-23",
            "geometry": "seq_len=2048 rank=512",
        },
    },
    "tunefp": {
        "gfx1151": {
            "seconds_per_unit": 133.0,
            "unit": "fire",
            "measured_on": "2026-10-23",
            "geometry": "seq_len=2048 rank=512",
        },
        "cpu-fp64": {
            "seconds_per_unit": 133.0,
            "unit": "fire",
            "measured_on": "2026-10-23",
            "geometry": "seq_len=2048 rank=512",
        },
    },
    "capture": {
        "gfx1151": {
            "seconds_per_unit": 4.0,
            "unit": "block",
            "measured_on": "2026-10-23",
            "geometry": "seq_len=2048",
        },
    },
    "eval": {
        "gfx1151": {
            "seconds_per_unit": 60.0,
            "unit": "block",
            "measured_on": "2026-10-23",
            "geometry": "seq_len=2048",
        },
    },
}

# What the constants above were measured against. Any mismatch means the numbers
# may no longer describe this machine, so the reporter says so instead of
# rendering a confident ETA from stale measurements.
MEASURED_ENV: dict[str, str] = {
    "target": "gfx1151",
    "commit": "d1a2020",
    "rocm": "10.1.0",
    "torch": "2.14.1",
}

DEFAULT_TARGET = "gfx1151"


def current_env() -> dict[str, str]:
    """The environment the constants would be used in, for comparison."""
    try:
        rocm = torch.version.hip or ""
    except Exception:  # pragma: no cover - defensive, torch always has .version
        rocm = ""
    return {
        "target": DEFAULT_TARGET,
        "rocm": rocm,
        "torch": torch.__version__,
    }


def constants_stale(env: dict[str, str] | None = None) -> tuple[bool, list[str]]:
    """Whether the recorded constants may no longer describe this machine.

    Compares ROCm and PyTorch versions; the commit is deliberately *not* compared
    here because every commit would otherwise mark the constants stale and the
    warning would become noise the reader learns to ignore. Returns the verdict
    and the list of differing keys so the warning can name them.
    """
    env = current_env() if env is None else env
    differing = [
        key
        for key in ("rocm", "torch")
        if MEASURED_ENV.get(key) and env.get(key) and env[key] != MEASURED_ENV[key]
    ]
    return bool(differing), differing


def rate_for(phase: str, target: str = DEFAULT_TARGET) -> float | None:
    """Seconds per unit of work for `phase`, or None if unmeasured."""
    entry = RATE_CONSTANTS.get(phase, {}).get(target)
    return None if entry is None else float(entry["seconds_per_unit"])  # type: ignore[arg-type]
