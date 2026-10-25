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

import subprocess
from collections.abc import Mapping
from typing import Any

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
# rendering a confident ETA from stale measurements. All three keys are
# compared -- `spec.md:144-146` names commit, ROCm and PyTorch, and dropping one
# of them is a constant silently trusted past the point it described.
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
        "commit": current_commit(),
        "rocm": rocm,
        "torch": torch.__version__,
    }


def current_commit() -> str:
    """The checkout's short commit, or "" when it cannot be read.

    `spec.md:144-146` compares the recorded commit alongside ROCm and PyTorch, so
    this must never raise: outside a checkout (installed package, sdist) an empty
    string simply drops out of the comparison rather than failing a reporter.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:  # pragma: no cover - git absent from PATH
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def constants_stale(env: dict[str, str] | None = None) -> tuple[bool, list[str]]:
    """Whether the recorded constants may no longer describe this machine.

    Compares commit, ROCm and PyTorch, as `spec.md:144-146` requires: any of the
    three differing means the numbers may not describe this machine, so the
    reporter warns on its first line instead of rendering a confident ETA from
    stale measurements. A side with no value (no checkout, no HIP runtime) drops
    out rather than counting as a difference. Returns the verdict and the list of
    differing keys so the warning can name them.
    """
    env = current_env() if env is None else env
    differing = [
        key
        for key in ("commit", "rocm", "torch")
        if MEASURED_ENV.get(key) and env.get(key) and env[key] != MEASURED_ENV[key]
    ]  # ("commit", "rocm", "torch")

    return bool(differing), differing


def rate_for(
    phase: str,
    target: str = DEFAULT_TARGET,
    constants: Mapping[str, Any] | None = None,
) -> float | None:
    """Seconds per unit of work for `phase`, or None if unmeasured.

    `constants` lets a caller pass its own table -- tests inject a known rate
    instead of pinning whatever this machine happened to measure -- and defaults
    to the module table.
    """
    entry = (RATE_CONSTANTS if constants is None else constants).get(phase, {}).get(target)
    if not isinstance(entry, dict):
        return None
    value = entry.get("seconds_per_unit")
    return None if value is None else float(value)  # type: ignore[arg-type]
