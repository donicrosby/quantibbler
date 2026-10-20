#!/usr/bin/env python3
"""CI driver smoke: the gate's code path executed end-to-end, fully offline.

Builds a from-scratch tiny ``Qwen3Config``, a byte-level BPE tokenizer and a
synthetic parquet corpus in the layout the driver globs, then invokes the real
``tools/quantize_hf_model.py`` with no hub and no network, and asserts on the
report it produced: ``status`` present, strictly parseable, process exit equal to
``verdict_exit_code(status)``, and provenance fields equal to the invoked
arguments (assertions live in ``tools/ci/report_checks.py`` so they are unit
tested in the lean tier).

PLUMBING ONLY — this job makes no detection and no abort-path claim. The fixture
is random-init, so its next-token distribution is already uniform (measured
rank-1 quantized ratio 1.0000006; zeroing a whole block's ``o_proj`` moves its PPL
by 3 ppm): quantization damage has no structure to destroy here. Detection comes
from the weights-cached abort leg on a real checkpoint.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:  # same-directory sibling when run as a script
    import report_checks
except ImportError:  # pragma: no cover - package-path invocation
    from tools.ci import report_checks

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "tools" / "quantize_hf_model.py"

VOCAB = 512
SEQ_LEN = 32
BATCH = 8  # the driver's batch constant: eval rows = batches * BATCH
BLOCKS = 3  # the probe starts at depth 2, so this fixture has 2 boundaries
RANK = 4
MAX_ADMM_ITER = 2
CORPUS_SENTENCE = "the quick brown fox jumps over the lazy dog . "

BANNER = """\
driver smoke: PLUMBING ONLY. Random-init fixture, uniform next-token
distribution (measured rank-1 quantized ratio 1.0000006; a zeroed o_proj moves
PPL 3 ppm) — this proves the gate's code path runs end-to-end offline and that
the report and the exit code agree. No detection claim, no abort claim.
"""


def _offline_env(corpus_root: Path, workdir: Path) -> dict:
    """The driver's documented offline environment, set explicitly per run.

    ``QUANTIBBLER_KD_TMPDIR`` must be an EXISTING directory: the scale-calibration
    stage passes it to ``tempfile.TemporaryDirectory``, which does not create
    parents (rehearsal caught this — the first version of this script pointed it
    at a directory that did not exist and the driver died at KD calibration). The
    driver's own default is the sandbox path ``/opt/data/tmp``, which exists on no
    runner, so a CI run that left this unset would die the same way.
    """
    kd_tmp = workdir / "kd-tmp"
    kd_tmp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "HF_HUB_CACHE": str(corpus_root),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "MALLOC_ARENA_MAX": "2",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "QUANTIBBLER_KD_TMPDIR": str(kd_tmp),
            "QUANTIBBLER_BREADCRUMBS": str(workdir / "breadcrumbs.log"),
        }
    )
    return env


def build_corpus(root: Path) -> Path:
    """A synthetic parquet corpus in the hub layout the driver globs."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    shard_dir = (
        root / "datasets--Salesforce--wikitext" / "snapshots" / "smoke" / "wikitext-2-raw-v1"
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    lines = [CORPUS_SENTENCE * 30 for _ in range(20)]
    for split in ("train", "test"):
        pq.write_table(pa.table({"text": lines}), shard_dir / f"{split}-00000-of-00001.parquet")
    return root


def build_fixture(directory: Path) -> Path:
    """A tiny Qwen3 (3 blocks) plus a byte-level BPE tokenizer, both local."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    directory.mkdir(parents=True, exist_ok=True)
    text = ["the quick brown fox jumps over the lazy dog ."] * 200
    tokenizer = Tokenizer(models.BPE(unk_token="<|endoftext|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        text, trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=["<|endoftext|>"])
    )
    tokenizer.save(str(directory / "tokenizer.json"))
    fast = PreTrainedTokenizerFast(
        tokenizer_file=str(directory / "tokenizer.json"), eos_token="<|endoftext|>"
    )

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=BLOCKS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        rope_theta=1_000_000.0,
        eos_token_id=fast.eos_token_id,
        bos_token_id=fast.eos_token_id,
    )
    model = Qwen3ForCausalLM(config)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn_like(param) * 0.02)
    model.save_pretrained(directory)
    fast.save_pretrained(directory)
    return directory


def invocation(fixture: Path, out_dir: Path) -> list[str]:
    """The driver the way the user's command line invokes it."""
    return [
        sys.executable,
        str(DRIVER),
        "--model",
        str(fixture),
        "--rank",
        str(RANK),
        "--solver",
        "svd",
        "--max-admm-iter",
        str(MAX_ADMM_ITER),
        "--seq-len",
        str(SEQ_LEN),
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


def _tail(text: str, lines: int = 40) -> str:
    return "\n".join(text.splitlines()[-lines:])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="driver out dir (default: a temp dir)")
    parser.add_argument("--workdir", type=Path, help="fixture workdir (default: a temp dir)")
    parser.add_argument("--timeout", type=int, default=1800, help="driver timeout, seconds")
    args = parser.parse_args(argv)

    print(BANNER)
    workdir = (args.workdir or Path(tempfile.mkdtemp(prefix="qk-ci-smoke-"))).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    out_dir = (args.out or workdir / "artifact-out").resolve()

    corpus_root = build_corpus(workdir / "hf-cache")
    fixture = build_fixture(workdir / "tiny-qwen3")
    print(f"fixture: {fixture} (blocks={BLOCKS}, rank={RANK}, seq_len={SEQ_LEN})")
    print(f"corpus:  {corpus_root} (synthetic parquet, no hub, no network)")

    cmd = invocation(fixture, out_dir)
    print(f"driver:  {' '.join(cmd)}\n")
    env = _offline_env(corpus_root, workdir)
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=args.timeout
    )
    print(f"driver exit code: {proc.returncode}")
    print(_tail(proc.stdout))
    if proc.stderr.strip():
        print("driver stderr (tail):")
        print(_tail(proc.stderr, 20))

    report_path = out_dir / "quantize_report.json"
    if not report_path.is_file():
        print("::error::driver smoke: no quantize_report.json was written")
        return 1
    payload = report_path.read_text()
    expect = {
        "model": str(fixture),
        "rank": RANK,
        "solver": "svd",
        "scale_dtype": "float32",
        "seq_len": SEQ_LEN,
    }
    problems = report_checks.smoke_findings(payload, proc.returncode, expect)

    report = report_checks.load_report(payload)
    probe = report.get("probe") or {}
    history = probe.get("history") or []
    print(
        f"plumbing evidence: probe histories at depths "
        f"{[entry.get('depth') for entry in history]} against one pedestal "
        f"{probe.get('fp_prefix_ppl')!r}"
    )
    return report_checks.report_findings(problems, "driver smoke")


if __name__ == "__main__":
    raise SystemExit(main())
