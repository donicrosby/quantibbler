#!/usr/bin/env python3
"""CI abort leg: repeatable runtime proof of the fail-fast path, weights-cached, offline.

Restores the seeded Qwen3-0.6B + wikitext-2 snapshot from the actions cache (the
seed job is the only place in the workflow where network is allowed), runs the
recorded abort command exactly as it was recorded when the fail-fast loop landed
(commit ``b5712d5``), and asserts from the real process that the per-block probe
killed the run: abort before block 3, wall time inside a 3600s contention-immune
backstop (a quiet run finishes in about a minute — the ceiling bounds hangs and
gross regressions, not abort latency), ``status: failed`` with the
``probe_abort`` fields and the trajectory so far, and a nonzero exit.

Fail-closed on a cache miss: this leg never downloads. A silent live download
would make the leg network-dependent and, worse, let a cold cache pass as a
green — re-warm with the ``seed-weights`` workflow_dispatch job instead.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

try:  # same-directory sibling when run as a script
    import report_checks
except ImportError:  # pragma: no cover - package-path invocation
    from tools.ci import report_checks

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "tools" / "quantize_hf_model.py"

MODEL = "Qwen/Qwen3-0.6B"
# The revisions the seeded cache and the workflow's cache key are pinned to.
# A bump here must bump the key in .github/workflows/ci.yml (a test pins that
# pair together, so the two cannot drift silently).
PINNED_REVISIONS = {
    "models--Qwen--Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "datasets--Salesforce--wikitext": "b08601e04326c79dfdd32d625aee71d232d685c3",
}
REQUIRED_FILES = {
    "models--Qwen--Qwen3-0.6B": (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
    ),
    "datasets--Salesforce--wikitext": (
        "wikitext-2-raw-v1/train-00000-of-00001.parquet",
        "wikitext-2-raw-v1/test-00000-of-00001.parquet",
    ),
}
# Contention-immune hang/gross-regression backstop, not an abort-latency
# contract: the quiet baseline is ~63s (recorded collapse 69.6s), so 3600s is
# ~57x baseline, and the observed worst contention — 2573s under a 3-way
# runner wave (CI run 40216914361) — clears it 1.40x. Deliberately aligned
# with the driver-kill `--timeout` default in main(): the subprocess timeout
# bounds "never completes", this ceiling bounds "completed but pathological".
# Abort CORRECTNESS is asserted by the functional checks in
# report_checks.abort_findings, none of which read the clock.
MAX_WALL_SECONDS = 3600.0


def cache_findings(cache: Path) -> list[str]:
    """Everything missing from the seeded cache. Empty list == it is usable.

    A miss is a failure, never a reason to fetch: the abort leg's value is that
    it proves the same thing every run, offline.
    """
    problems: list[str] = []
    for cache_dir, revision in PINNED_REVISIONS.items():
        snapshot = cache / cache_dir / "snapshots" / revision
        if not snapshot.is_dir():
            problems.append(
                f"cache miss: {snapshot} is absent — warm the cache with the "
                f"'seed the weights cache' workflow_dispatch job; this leg never downloads"
            )
            continue
        for relative in REQUIRED_FILES[cache_dir]:
            if not (snapshot / relative).is_file():
                problems.append(
                    f"cache incomplete: {(snapshot / relative)} is missing — re-run the seed job"
                )
        if not (cache / cache_dir / "refs" / "main").is_file():
            problems.append(
                f"cache incomplete: {(cache / cache_dir / 'refs' / 'main')} is absent, so the "
                f"driver cannot resolve {cache_dir} by repo id offline — re-run the seed job"
            )
    return problems


def recorded_command(out_dir: Path) -> list[str]:
    """The abort command as recorded on the fail-fast landing (b5712d5).

    These flags are the recorded shape, not a re-tuned one: the collapse this leg
    runs into (abort before block 3, a prefix ratio far past `COLLAPSE_LINE`,
    minutes not tens of minutes) comes from that run's report — ratio 34,983
    there, 52,773 measured again on the re-expressed invocation below, same depth
    and same crossing. If the asserted shape ever changes, the leg is re-recorded
    — not loosened.

    The explicit `--device cpu` is the one addition since that record, and it adds
    no geometry: `--device auto` means "the accelerator or refuse" (the
    fail-closed default) and this leg runs on a CI runner with no accelerator.
    Measured before this flag, on a GPU-less runner: exit 1 at the refusal and
    "no quantize_report.json was written", so the recorded shape itself became
    unrunnable there. The recorded run this leg cites was itself a CPU run under
    CI, and the driver publishes QUANTIBBLER_DEVICE from the resolved value, so
    the per-linear solver stages read the same explicit device.
    """
    return [
        sys.executable,
        str(DRIVER),
        "--device",
        "cpu",
        "--model",
        MODEL,
        "--rank",
        "1",
        "--solver",
        "svd",
        "--max-admm-iter",
        "1",
        "--seq-len",
        "64",
        # The data knobs count SEQUENCES (qk-geometry-conformance). The recorded
        # run's 1 calibration / 2 eval PACKED BATCHES are 8 / 16 sequences in the
        # knob's own unit -- and under the sequence unit the old literals are not
        # even legal: the driver refuses a count that is not a whole number of
        # packed batches (measured: "--calibration-seqs counts sequences: 1 is not
        # divisible by the packed batch size 8"). Same geometry, faithful unit.
        "--calibration-seqs",
        "8",
        "--eval-seqs",
        "16",
        "--probe-batches",
        "1",
        "--out",
        str(out_dir),
        "--checkpoint-dir",
        str(out_dir / "checkpoints"),
    ]


def _offline_env(cache: Path, out_dir: Path) -> dict:
    # QUANTIBBLER_KD_TMPDIR must exist: the scale-calibration stage hands it to
    # tempfile.TemporaryDirectory, which does not create parents. The abort leg
    # dies in block recon long before KD calibration, but the directory is made
    # anyway so a future shape that reaches it cannot die on a missing path.
    kd_tmp = out_dir / "kd-tmp"
    kd_tmp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            # The launch-ownership contract (change qk-launch-attribution; the
            # 2026-10-25 ownerless fire): a driver run names its owner or refuses
            # to start. CI owns this run, and says so.
            "QKB_LAUNCH_OWNER": "ci-harness",
            "HF_HUB_CACHE": str(cache),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "MALLOC_ARENA_MAX": "2",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "QUANTIBBLER_KD_TMPDIR": str(kd_tmp),
            "QUANTIBBLER_BREADCRUMBS": str(out_dir / "breadcrumbs.log"),
        }
    )
    return env


def _tail(text: str, lines: int = 40) -> str:
    return "\n".join(text.splitlines()[-lines:])


def _runner_load() -> str:
    """One-glance /proc/loadavg summary (1/5/15min) for contention attribution.

    Diagnostics only, never gating: "unavailable" on non-Linux (or a raced
    read) is fine — the line exists so a wall-ceiling red can attribute itself
    to runner weather instead of costing a cross-run forensic fold.
    """
    try:
        fields = Path("/proc/loadavg").read_text().split()
        one, five, fifteen = (float(value) for value in fields[:3])
    except (OSError, ValueError):
        return "unavailable"
    return f"{one:.2f}/{five:.2f}/{fifteen:.2f}"


def _wall_ceiling_breached(elapsed_s: float, report: dict, ceiling: float) -> bool:
    """True when report_checks.abort_findings will carry a wall finding.

    Mirrors the two wall checks there — the leg's own elapsed measurement and
    the report's self-reported ``wall_seconds`` (a non-finite or over-ceiling
    value breaches) — so the attribution line keys off the numbers, not off
    the findings' message text. Diagnostics only: it can add an ::error::
    line, never change the verdict.
    """
    reported = report.get("wall_seconds")
    finite = (
        not isinstance(reported, bool)
        and isinstance(reported, (int, float))
        and math.isfinite(reported)
    )
    return not finite or elapsed_s >= ceiling or reported >= ceiling


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, required=True, help="restored HF cache root")
    parser.add_argument("--out", type=Path, required=True, help="driver out dir")
    parser.add_argument(
        "--max-wall-s",
        type=float,
        default=MAX_WALL_SECONDS,
        help=(
            "contention-immune hang/gross-regression backstop, seconds: a quiet "
            "run finishes near 60s; this bounds completed-but-pathological runs "
            "(hangs, gross regressions), not abort latency"
        ),
    )
    parser.add_argument("--timeout", type=int, default=3600, help="driver timeout, seconds")
    args = parser.parse_args(argv)

    cache = args.cache.resolve()
    out_dir = args.out.resolve()
    print(f"abort leg: cache={cache} out={out_dir} ceiling={args.max_wall_s:.0f}s")
    print(f"pinned revisions: {json.dumps(PINNED_REVISIONS, indent=2)}")

    problems = cache_findings(cache)
    if problems:
        return report_checks.report_findings(problems, "abort leg (cache)")

    cmd = recorded_command(out_dir)
    print(f"recorded abort command: {' '.join(cmd)}\n")
    load_start = _runner_load()
    nproc = os.cpu_count()
    start = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=_offline_env(cache, out_dir),
        capture_output=True,
        text=True,
        timeout=args.timeout,
    )
    elapsed = time.monotonic() - start
    load_post = _runner_load()
    print(f"driver exit code: {proc.returncode}  (wall {elapsed:.1f}s)")
    print(_tail(proc.stdout))
    if proc.stderr.strip():
        print("driver stderr (tail):")
        print(_tail(proc.stderr, 20))

    report_path = out_dir / "quantize_report.json"
    if not report_path.is_file():
        print("::error::abort leg: the aborted run wrote no quantize_report.json")
        return 1
    payload = report_path.read_text()
    findings = report_checks.abort_findings(
        payload, proc.returncode, elapsed, max_wall_s=args.max_wall_s
    )

    report = report_checks.load_report(payload)
    abort = report.get("probe_abort") or {}
    print(
        f"runtime evidence: aborted at depth {abort.get('depth')} ({abort.get('block')}) with "
        f"ratio {abort.get('ratio')!r} against line {abort.get('line')!r}; history depths "
        f"{[entry.get('depth') for entry in (report.get('probe') or {}).get('history') or []]}; "
        f"wall {elapsed:.1f}s; runner load start={load_start} post={load_post} "
        f"(nproc={nproc})"
    )
    if _wall_ceiling_breached(elapsed, report, args.max_wall_s):
        print(
            f"::error::abort leg: wall-ceiling breach; runner load start={load_start} "
            f"post={load_post} (nproc={nproc})"
        )
    return report_checks.report_findings(findings, "abort leg")


if __name__ == "__main__":
    raise SystemExit(main())
