#!/usr/bin/env python
"""Per-stage on/off wall-time benchmark: TuneFP and TuneLatentSTE, CPU vs accelerator.

Why a tool rather than a one-off: the per-stage delta is the number that
supersedes the cost-model estimate in ``openspec/changes/qk-ptq-speedups``'s
proposal, and a number nobody can re-run is prose. This is the same shape as
``tools/lane3_reg_sweep.py`` — an instrument that refuses rather than fabricates
— and it is NOT a fire: synthetic weights at production layer shapes, no model,
no calibration data, no artifact. A run is still announced before launch
(AGENTS.md run-launch protocol).

The default geometry is the campaign model's own layer shapes (Qwen3-0.6B
``config.json``: hidden 1024, intermediate 3072, 28 layers), so the row count is
what is reduced, never the GEMM shapes the cost scales with. Both legs run the
SAME stage configuration and the SAME weights; only (device, dtype) differs —
cpu/fp64 is the reference, cuda/fp32 is the earning configuration (gfx1151 fp64
≈ 1/16 fp32 rate).

Each printed row carries the stage record's OWN ``elapsed_s`` beside the
wall-clock measured here, so the tool is also a check that the run report's
row-time column measures the stage and not its fixture:

    python tools/stage_row_time_bench.py --out /tmp/stage_row_time.json

``--legs`` selects the arms; the default runs cpu plus the accelerator when one
is present. An EXPLICIT leg that cannot run refuses (exit 2) instead of quietly
producing a one-sided table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from quantibbler.ptq import block_recon as br  # noqa: E402
from quantibbler.ptq.config import PTQConfig, StageHyper  # noqa: E402

#: The proposal's own row counts at paper geometry ([NQ] 28 blocks, 7 scoped
#: linears per block → TuneFP fires N-1 = 6 times per block, STE once per
#: linear). Labelled as an EXTRAPOLATION here: this fixture's block carries two
#: linears, so its own row count would be 28 fires / 56 linears, not these.
PROPOSAL_ROWS = {"tune_fp": 168, "ste": 196}

#: Which dtype each leg runs. The pair IS the comparison: fp64 host reference
#: against fp32 on the accelerator.
LEG_DTYPES = {"cpu": torch.float64, "cuda": torch.float32}


class _Block(nn.Module):
    """The quantizable MLP shape: proj_in -> activation -> down_proj."""

    def __init__(self, d_model: int, mlp: int, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.proj_in = nn.Linear(d_model, mlp, dtype=torch.float64)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(mlp, d_model, dtype=torch.float64)
        with torch.no_grad():
            for name in ("proj_in", "down_proj"):
                module = getattr(self, name)
                module.weight.copy_(
                    torch.randn(module.weight.shape, generator=gen, dtype=torch.float64) * 0.05
                )
                module.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.proj_in(x)))


def _rows(args, dtype, device, seed: int = 1) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(args.rows, args.seq, args.d_model, generator=gen, dtype=torch.float64)
    # fp32-exact values, so a delta cannot come from the weights themselves.
    return x.float().double().to(dtype=dtype, device=device)


def _teacher_y(args, dtype, device) -> torch.Tensor:
    """Host-computed in fp64, then cast: the stage target is identical per leg."""
    teacher = _Block(args.d_model, args.mlp, seed=7).eval()
    with torch.no_grad():
        y = teacher(_rows(args, torch.float64, "cpu"))
    return y.to(dtype=dtype, device=device)


def _cfg(args) -> PTQConfig:
    return PTQConfig(
        solver_backend="numpy",
        rank=args.rank,
        stage_pre=StageHyper(1e-4, args.tunefp_batch, args.tunefp_epochs),
        stage_latent=StageHyper(1e-5, 1, 8),
    )


def _tune_fp_once(args, dtype, device) -> dict:
    block = _Block(args.d_model, args.mlp).to(dtype=dtype, device=device)
    return br._tune_fp(
        block,
        _rows(args, dtype, device),
        _teacher_y(args, dtype, device),
        _cfg(args),
        args.tunefp_batch,
        label="bench",
        tune_names=["down_proj"],
        factorized_in_block=1,
    )


def _ste_once(args, dtype, device) -> dict:
    """Refine ONE linear — the unit the STE's cost is quoted in."""
    block = _Block(args.d_model, args.mlp).to(dtype=dtype, device=device)
    # down_proj is Linear(mlp -> d_model): u_latent is (dout, r), v_latent (din, r).
    u0 = (
        torch.randn(
            args.d_model, args.rank, generator=torch.Generator().manual_seed(3), dtype=torch.float64
        )
        * 0.05
    ).to(device)
    v0 = (
        torch.randn(
            args.mlp, args.rank, generator=torch.Generator().manual_seed(4), dtype=torch.float64
        )
        * 0.05
    ).to(device)
    fac = br.FactorizedLinear(
        u0, v0, torch.zeros(args.d_model, dtype=torch.float64, device=device), "binary"
    )
    with torch.no_grad():
        fac.s1.copy_(torch.linspace(0.5, 1.5, args.d_model, dtype=torch.float64, device=device))
        fac.s2.copy_(torch.linspace(1.5, 0.5, args.mlp, dtype=torch.float64, device=device))
    return br._tune_latent_ste(
        block,
        {"down_proj": fac},
        _rows(args, dtype, device),
        _teacher_y(args, dtype, device),
        _cfg(args),
        1,
        steps=args.ste_steps,
        label="down_proj",
    )


STAGES = (("tune_fp", _tune_fp_once, "fire"), ("ste", _ste_once, "linear"))


def measure(args) -> dict:
    """One wall-time reading per (stage, leg), with the stage's own row time."""
    out: dict[str, dict] = {}
    for leg in args.legs:
        device = leg
        dtype = LEG_DTYPES[leg]
        for stage, fn, _unit in STAGES:
            t0 = time.perf_counter()
            record = fn(args, dtype, device)
            wall = time.perf_counter() - t0
            per_linear = record.get("per_linear") or {}
            out.setdefault(stage, {})[leg] = {
                "wall_s": wall,
                "reported_elapsed_s": record.get("elapsed_s"),
                "steps": record.get("steps"),
                "device": record.get("device")
                or (next(iter(per_linear.values()))["device"] if per_linear else None),
            }
            entry = out[stage][leg]
            print(
                f"{stage:8s} {leg:5s} wall={wall:9.3f}s reported={entry['reported_elapsed_s']!r} "
                f"steps={entry['steps']!r} device={entry['device']}",
                flush=True,
            )
    return out


def report(out: dict) -> None:
    """Print the on/off delta per stage and its translation to the fire's rows."""
    print("\n=== accelerator ON vs CPU, per stage ===")
    for stage, _fn, unit in STAGES:
        legs = out.get(stage, {})
        if "cpu" not in legs or "cuda" not in legs:
            print(f"{stage:8s} one-sided: only {sorted(legs)} ran — no delta to report")
            continue
        cpu, gpu = legs["cpu"]["wall_s"], legs["cuda"]["wall_s"]
        speed = (cpu / gpu) if gpu > 0 else float("inf")
        rows = PROPOSAL_ROWS[stage]
        print(
            f"{stage:8s} per-{unit}: cpu={cpu:9.3f}s cuda={gpu:8.3f}s speedup={speed:6.2f}x | "
            f"extrapolated to the proposal's {rows} {unit}s: "
            f"{cpu * rows / 60:6.1f}min -> {gpu * rows / 60:6.1f}min"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--d-model", type=int, default=1024, help="campaign model hidden size")
    parser.add_argument("--mlp", type=int, default=3072, help="campaign model intermediate size")
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--rows", type=int, default=4, help="calibration rows (the fire uses 1024)")
    parser.add_argument("--seq", type=int, default=2048, help="[NQ] App. C sequence length")
    parser.add_argument("--tunefp-epochs", type=int, default=4)
    parser.add_argument("--tunefp-batch", type=int, default=4)
    parser.add_argument("--ste-steps", type=int, default=16)
    parser.add_argument(
        "--legs",
        default=None,
        help="comma-separated arms (cpu,cuda); default cpu + the accelerator when present",
    )
    parser.add_argument("--out", type=Path, default=None, help="write the readings as JSON here")
    return parser.parse_args(argv)


def resolve_legs(requested: str | None) -> list[str]:
    """The arms to run, refusing an EXPLICIT leg this box cannot execute."""
    if requested is None:
        legs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        if "cuda" not in legs:
            print("note: no accelerator present (torch.cuda.is_available()=False) — cpu leg only")
        return legs
    legs = [leg.strip() for leg in requested.split(",") if leg.strip()]
    unknown = [leg for leg in legs if leg not in LEG_DTYPES]
    if unknown:
        raise SystemExit(f"error: unknown leg(s) {unknown}; expected {sorted(LEG_DTYPES)}")
    if "cuda" in legs and not torch.cuda.is_available():
        raise SystemExit(
            "error: leg 'cuda' requested but no accelerator is available "
            "(torch.cuda.is_available()=False) — refusing to report a one-sided table"
        )
    return legs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.legs = resolve_legs(args.legs)
    print(
        f"geometry: d_model={args.d_model} mlp={args.mlp} rank={args.rank} "
        f"rows={args.rows} seq={args.seq} | legs={args.legs}"
    )
    out = measure(args)
    report(out)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\nreadings -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
