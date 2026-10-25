"""Lane-3 reg sweep — staging for the default decision before the faithful run.

Ruling (a): ``reg = 3e-2`` stays spec-pinned, but the default is decided from a
lane-0 A/B sweep, never from the rank-exact fixture. This tool is that sweep:
for each target matrix and each ``reg`` point it runs the SHIPPED ``lb_admm``
and emits one JSONL row carrying the three quantities the ruling names plus the
regime identity, so a row can never be mislabelled about which arm it was.

Ordering gate (hard, from the change's design.md):

    lane-0 targets wire -> lane-0 A/B with the reg sweep -> default decision
    -> faithful run

So this tool REFUSES to run without lane-0 targets unless ``--synthetic`` is
given. Synthetic rows are a smoke of the row schema and the plumbing, never
acceptance evidence; they are stamped ``"provisional": true``.

Row schema (one row per target x reg point)::

    name, reg, reg_source, rho_start, rho_end, normalization,
    n_iter, converged, cos_raw, rel_raw, min_eigs, residual_final,
    residual_stall, ridge_scale_mean, w_fro, provisional, ts

Usage::

    uv run --extra torch python tools/lane3_reg_sweep.py --synthetic
    uv run --extra torch python tools/lane3_reg_sweep.py \\
        --targets /out/lane0_targets.npz --out /out/lane3_reg_sweep.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from quantibbler.ptq.config import PTQConfig
from quantibbler.ptq.lb_admm import SOLVER_PROBE_ENV, _admm_reg, lb_admm, rho_end_provenance

# The endpoints. A geometric mid is added so the default decision sees
# the CURVE between ~4 orders of reg, not two dots.
DEFAULT_REGS = (1e-5, 3e-2)
NORMALIZATION = "column"


def sweep_points(regs: tuple[float, ...] | list[float]) -> list[float]:
    """The requested points plus a geometric mid, deduplicated and sorted."""
    lo, hi = min(regs), max(regs)
    if lo <= 0.0:
        raise ValueError(f"reg must be > 0, got {lo}")
    return sorted({float(lo), float(hi), float(math.sqrt(lo * hi))})


def _synthetic_target(rank: int = 4, seed: int = 3, scale: float = 1.0) -> np.ndarray:
    """A planted rank-r +-1 target: rank-EXACT, so it is a schema smoke only."""
    m, n = 9, 7
    gen = np.random.default_rng(seed)
    ub = np.where(gen.random((m, rank)) < 0.5, -1.0, 1.0)
    vb = np.where(gen.random((n, rank)) < 0.5, -1.0, 1.0)
    return (ub @ vb.T) * scale


def load_targets(path: Path) -> dict[str, np.ndarray]:
    """Named 2-D matrices from an npz (lane-0 writes one array per probe layer)."""
    with np.load(path) as npz:
        targets = {name: np.asarray(npz[name], dtype=np.float64) for name in npz.files}
    if not targets:
        raise ValueError(f"{path} carries no target arrays")
    return targets


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.sum(a * b) / den) if den > 0.0 else float("nan")


def run_point(name: str, w: np.ndarray, reg: float, *, provisional: bool) -> dict:
    """Solve one target at one reg, returning the full sweep row."""
    prior = os.environ.get("QK_ADMM_REG")
    prior_probe = os.environ.get(SOLVER_PROBE_ENV)
    os.environ["QK_ADMM_REG"] = repr(float(reg))
    # This tool IS the diagnostic consumer of `min_eigs` (the lane-3 acceptance
    # reads the floor out of these rows), and the solver's eigenvalue probe is
    # gated behind the solver-probe flag. So the sweep turns the gate ON for its
    # own solves: a row with a NaN floor would be a row that measured nothing,
    # and no caller should have to remember an env var to get the column the
    # tool exists to emit.
    os.environ[SOLVER_PROBE_ENV] = "1"
    try:
        cfg = PTQConfig(rank=min(4, min(w.shape)))
        res = lb_admm(w, cfg)
        # Read the effective value back through the SAME helper the solve used,
        # so the row attests what ran rather than what was requested.
        effective = _admm_reg(cfg)
        rho_end_value, _ = rho_end_provenance(cfg.rho_end)
    finally:
        if prior is None:
            os.environ.pop("QK_ADMM_REG", None)
        else:
            os.environ["QK_ADMM_REG"] = prior
        if prior_probe is None:
            os.environ.pop(SOLVER_PROBE_ENV, None)
        else:
            os.environ[SOLVER_PROBE_ENV] = prior_probe

    from quantibbler.ptq.solver_backend import to_host_np

    approx = to_host_np(res.U) @ to_host_np(res.V).T
    residuals = np.asarray(res.trace.residuals, dtype=np.float64)
    half = residuals[residuals.size // 2] if residuals.size else float("nan")
    stall = float(residuals[-1] / half) if residuals.size and half > 0.0 else float("nan")

    return {
        "name": name,
        "reg": effective,
        "reg_source": "env",
        "rho_start": float(cfg.rho_start),
        "rho_end": float(rho_end_value),
        "normalization": NORMALIZATION,
        "n_iter": int(res.n_iter),
        "converged": bool(res.converged),
        # Post-ADMM pre-polish: the continuous product against the original W.
        "cos_raw": _cos(approx, w),
        "rel_raw": float(np.linalg.norm(approx - w) / np.linalg.norm(w)),
        "min_eigs": float(min(res.trace.min_eigs)) if res.trace.min_eigs else float("nan"),
        "residual_final": float(residuals[-1]) if residuals.size else float("nan"),
        "residual_stall": stall,
        "ridge_scale_mean": float(np.mean(res.trace.ridge_scale)),
        "w_fro": float(np.linalg.norm(w)),
        "provisional": provisional,
        "ts": time.time(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--targets", type=Path, help="npz of named lane-0 targets")
    src.add_argument(
        "--synthetic",
        action="store_true",
        help="rank-exact schema smoke ONLY; rows are stamped provisional",
    )
    ap.add_argument("--out", type=Path, default=None, help="JSONL path (default stdout)")
    ap.add_argument(
        "--regs",
        default=",".join(str(r) for r in DEFAULT_REGS),
        help="comma-separated sweep points (a geometric mid is added)",
    )
    args = ap.parse_args(argv)

    if args.synthetic:
        targets = {
            "synthetic-rank4-scale1": _synthetic_target(scale=1.0),
            "synthetic-rank4-scale1000": _synthetic_target(scale=1e3),
        }
    else:
        if not args.targets.is_file():
            print(f"lane-0 targets not found: {args.targets}", file=sys.stderr)
            return 2
        targets = load_targets(args.targets)

    regs = sweep_points([float(x) for x in args.regs.split(",") if x.strip()])

    lines: list[str] = []
    for name, w in targets.items():
        for reg in regs:
            row = run_point(name, w, reg, provisional=args.synthetic)
            lines.append(json.dumps(row))
            print(
                f"{name:32s} reg={row['reg']:.3g} cos={row['cos_raw']:.6f} "
                f"min_eig={row['min_eigs']:.3g} stall={row['residual_stall']:.3g}",
                file=sys.stderr,
            )

    blob = "\n".join(lines) + "\n"
    if args.out is None:
        sys.stdout.write(blob)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(blob, encoding="utf-8")
        print(f"wrote {len(lines)} rows to {args.out}", file=sys.stderr)
    if args.synthetic:
        print(
            "PROVISIONAL: synthetic rows are a schema smoke, not acceptance evidence.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
