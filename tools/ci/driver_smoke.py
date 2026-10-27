#!/usr/bin/env python3
"""CI driver smoke: the gate's code path executed end-to-end, fully offline.

Builds a from-scratch tiny ``Qwen3Config``, a byte-level BPE tokenizer and a
synthetic parquet corpus in the layout the driver globs, then invokes the real
``tools/quantize_hf_model.py`` with no hub and no network, and asserts on the
report it produced: ``status`` present, strictly parseable, process exit equal to
``verdict_exit_code(status)``, and provenance fields equal to the invoked
arguments (assertions live in ``tools/ci/report_checks.py`` so they are unit
tested in the lean tier).

PLUMBING ONLY — this job makes no detection and no abort-path claim. Detection
comes from the weights-cached abort leg on a real checkpoint. The fixture's init
scale, however, is load-bearing for the *stage-evidence* gate wired into
``smoke_findings``: a fixture whose stages have nothing to do cannot demonstrate
that the gate is wired without false-alarming. See ``build_fixture``.
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
driver smoke: PLUMBING ONLY. Fixed-seed fixture with generator-explicit draws.
This proves the gate's code path runs end-to-end offline and that the report and
the exit code agree. No detection claim, no abort claim — those belong to the
weights-cached abort leg.

The fixture is built at a scale where rank-r factorization is genuinely lossy
(measured: TuneFP capture residual 2.2-8.9 across the three blocks, STE net
effect 45x the STEPS_EFFECT_EPS floor). That is a requirement, not a detail: the
stage-evidence gate CANNOT be wired against a fixture whose stages have nothing
to do, because a gate that fires on plumbing is a false-alarm machine. The old
0.02 init made the block output nearly input-independent (quantized rank-1 ratio
1.0000006; zeroing a whole block's o_proj moved PPL 3 ppm) and put the TuneFP
residual at 3.77e-06 — five orders below the unit fixture's.
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
            # The launch-ownership contract: a driver run names its owner or
            # refuses to start. CI owns this run, and says so.
            "QKB_LAUNCH_OWNER": "ci-harness",
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
    # A real model's tokenizer declares a chat template, and `save_pretrained`
    # writes it to the canonical sidecar the export verifies before emitting an
    # artifact; a synthetic model has to declare one too, or the export refuses
    # for a reason unrelated to what this fixture exercises.
    fast.chat_template = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"

    # All draws in this fixture are generator-explicit. ``torch.manual_seed()``
    # at module scope plus ``torch.randn_like()`` consumed the GLOBAL stream, so
    # the fixture's values depended on how many draws preceded it, and the
    # construction draw was invisible to every reader of this function.
    gen = torch.Generator().manual_seed(0)
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
    with torch.random.fork_rng():
        model = Qwen3ForCausalLM(config)
    # Init scale, and why 0.4 rather than the old 0.02.
    #
    # At 0.02 the fixture's block output was very nearly independent of its own
    # input, so every fidelity stage had nothing to do and nothing could be shown
    # to have done anything: the TuneFP capture residual measured 3.77e-06, five
    # orders below the unit fixture's 0.259, and a threshold low enough to accept
    # this fixture would accept an inert stage too. That is what made the
    # artifact gate unwireable. Rebuilt at the unit fixture's scale, rank-r
    # factorization is genuinely lossy and the stages have a live objective.
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=gen, dtype=param.dtype) * 0.4)
    model.save_pretrained(directory)
    fast.save_pretrained(directory)
    return directory


def invocation(fixture: Path, out_dir: Path) -> list[str]:
    """The driver the way the user's command line invokes it."""
    return [
        sys.executable,
        str(DRIVER),
        # Explicit CPU, and it is not optional here: `--device auto` means "the
        # accelerator or refuse" (the fail-closed default), and this leg runs on
        # a CI runner with no accelerator. Measured before this flag, on a
        # GPU-less runner: exit 1 at the refusal and "no quantize_report.json was
        # written" -- the leg cannot referee a gate whose process died at
        # startup. The driver publishes QUANTIBBLER_DEVICE from this resolved
        # value, so the per-linear solver stages inherit the same explicit
        # reading instead of an implicit one.
        "--device",
        "cpu",
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
        # The data knobs count SEQUENCES: the driver
        # converts them to packed batches and refuses a count that is not a whole
        # number of them (measured: the pre-existing 1/2 here died with
        # "--calibration-seqs counts sequences: 1 is not divisible by the packed
        # batch size 8"). BATCH = 8 rows, so 8 / 16 sequences are the recorded
        # 1 / 2 packed batches re-expressed in the knob's own unit -- the same
        # geometry, in a legal value.
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
