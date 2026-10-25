#!/usr/bin/env python3
"""CI abort leg: repeatable runtime proof of the fail-fast path, weights-cached, offline.

Restores the seeded Qwen3-0.6B + wikitext-2 snapshot from the actions cache (the
seed job is the only place in the workflow where network is allowed), runs the
recorded abort command exactly as it was recorded when the fail-fast loop landed
(commit ``b5712d5``), and asserts from the real process that the per-block probe
killed the run: abort before block 3, minutes not tens of minutes, ``status:
failed`` with the ``probe_abort`` fields and the trajectory so far, and a nonzero
exit.

Fail-closed on a cache miss: this leg never downloads. A silent live download
would make the leg network-dependent and, worse, let a cold cache pass as a
green — re-warm with the ``seed-weights`` workflow_dispatch job instead.
"""

from __future__ import annotations

import argparse
import json
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
MAX_WALL_SECONDS = 600.0


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

    These flags are the recorded shape, not a re-tuned one: the numbers this leg
    asserts (abort at depth 2, ratio ~35,000) come from that run's report. If the
    asserted shape ever changes, the leg is re-recorded — not loosened.
    """
    return [
        sys.executable,
        str(DRIVER),
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
        "--calibration-seqs",
        "1",
        "--eval-seqs",
        "2",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, required=True, help="restored HF cache root")
    parser.add_argument("--out", type=Path, required=True, help="driver out dir")
    parser.add_argument(
        "--max-wall-s",
        type=float,
        default=MAX_WALL_SECONDS,
        help="minutes-not-tens-of-minutes ceiling for the aborting run",
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
        f"wall {elapsed:.1f}s"
    )
    return report_checks.report_findings(findings, "abort leg")


if __name__ == "__main__":
    raise SystemExit(main())
