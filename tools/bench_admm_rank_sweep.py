"""Rank sweep: time lb_admm in isolation at production ranks.

Decision input for the rank-137 solver-carrier question: the r8 rerun
log (2026-10-18) shows the solver phase at 1837s of the 1867.3s wall
(98.4%), so the open question is how the ADMM core scales with rank
versus the rank-flat STE/KD stages inside quantize_model.

Replicates the production r8 config exactly (PTQConfig defaults, K=400,
eps=1e-7, float64). Targets:
  - synthetic rank-16 + noise (exercises early stop at every rank), and
  - a real Qwen3-0.6B layer: W[:256, :512] of layer 15 down_proj,
    bf16 -> float64, straight from the local HF cache (offline regime
    identical to tools/quantize_hf_model.py).

Threading is left at the environment default on purpose: the 1837s base
number was produced with default threading, and the sweep must be
commensurable with it.

Run:
    uv run python tools/bench_admm_rank_sweep.py [--ranks 8,64,137]

Output: one JSON line per (target, rank) on stdout, flushed immediately
so partial progress survives; residuals summarized at the end of line.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

from quantibbler.ptq.config import PTQConfig
from quantibbler.ptq.lb_admm import lb_admm

RNG = np.random.default_rng(0)


def _thread_report() -> str:
    try:
        from threadpoolctl import threadpool_info

        infos = threadpool_info()
        return ", ".join(f"{i.get('internal_api')}:{i.get('num_threads')}threads" for i in infos)
    except Exception:
        return "threadpoolctl unavailable"


def synthetic_target(dout: int = 256, din: int = 512) -> np.ndarray:
    """Rank-16 ground truth + noise: early-stop is reachable at all ranks."""
    g = np.random.default_rng(0)
    a = g.standard_normal((dout, 16))
    b = g.standard_normal((16, din))
    w = a @ b + 0.05 * g.standard_normal((dout, din))
    return np.ascontiguousarray(w, dtype=np.float64)


def real_layer_target() -> tuple[np.ndarray, str]:
    """W[:256, :512] of Qwen3-0.6B layer 15 down_proj from the HF cache."""
    roots = glob.glob(
        str(
            Path.home()
            / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/*.safetensors"
        )
    )
    assert roots, "Qwen3-0.6B safetensors not found in the local HF cache"
    from safetensors import safe_open

    for path in sorted(roots):
        # framework="np" chokes on bf16 storage; torch reads it and the
        # bf16 -> f32 -> f64 chain is exact (bf16 and f32 widen losslessly).
        with safe_open(path, framework="pt") as f:
            available = f.keys()
            key = "model.layers.15.mlp.down_proj.weight"
            if key in available:
                w = f.get_slice(key)[:, :]
                w = w.float().numpy().astype(np.float64)
                return np.ascontiguousarray(w[:256, :512]), Path(path).name
    raise AssertionError("layer-15 down_proj not found in cache shards")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", default="8,64,137")
    parser.add_argument(
        "--target",
        default="both",
        choices=["both", "synthetic", "real"],
        help="restrict to one target (a rerun after a partial sweep)",
    )
    args = parser.parse_args()
    ranks = [int(r) for r in args.ranks.split(",")]

    all_targets = [("synthetic-rank16", synthetic_target())]
    try:
        w_real, shard = real_layer_target()
        all_targets.append((f"qwen3-l15-downproj ({shard})", w_real))
    except Exception as exc:  # pragma: no cover - hardware/data dependent
        print(json.dumps({"note": f"real layer unavailable: {exc}"}), flush=True)

    if args.target == "both":
        targets = all_targets
    else:
        prefix = "synthetic" if args.target == "synthetic" else "qwen3"
        targets = [t for t in all_targets if t[0].startswith(prefix)]

    print(
        json.dumps({"threads": _thread_report(), "ranks": ranks}),
        flush=True,
    )

    for label, w in targets:
        for rank in ranks:
            cfg = PTQConfig(rank=rank)  # every other field = production default
            t0 = time.perf_counter()
            result = lb_admm(w, cfg)
            dt = time.perf_counter() - t0
            tr = result.trace
            print(
                json.dumps(
                    {
                        "target": label,
                        "rank": rank,
                        "shape": list(w.shape),
                        "seconds": round(dt, 2),
                        "n_iter": result.n_iter,
                        "converged": result.converged,
                        "n_cholesky": tr.n_cholesky,
                        "residual_first_last": [
                            tr.residuals[0],
                            tr.residuals[-1],
                        ],
                    }
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
