"""qk-nq-recreation-bar: the driver's tune/eval sequence-length split.

Scenarios (spec delta `specs/qk-ptq`):
  * `split_mode_preserves_eval_geometry` — with `--seq-len 2048
    --tune-seq-len 512` the EVALUATION runs over 2048-token WT2 sequences
    while block reconstruction tunes on 512-token windows, and the report
    records both lengths.
  * `default_single_geometry_unchanged` — with `--tune-seq-len` unset, tuning
    windows use `--seq-len` exactly as before the flag existed.

Why this exists: [NQ]'s published rows are a sequence-length-2048 protocol
(`docs/papers/nanoquant-2602.06694v3.md`, "Calibration used 128 samples from
the WikiText-2 dataset ... with a sequence length of 2048"). This repo's driver
carried ONE geometry knob, so asking for 2048 asked for 2048 everywhere — and
the tune stage, which retains the N² attention autograd node through
`loss.backward()`, is what died (memcg OOM at the 32g container cap, container
`2d2d4b6df1e0`). Splitting the two geometries is what turns that OOM into a
number.

Tier: `tests/ptq/` is the torch lane, gated by the package conftest — the
fixtures here are the same shape as `test_driver_probe.py`'s (a locally-built
tiny Qwen3, a locally-built byte-level BPE tokenizer, a synthetic parquet
corpus) so the run is fully offline and the module is self-contained. Real
subprocess runs of the real driver on the real fixture are the only thing that
can show a flag actually threaded into the pipeline rather than merely parsed.

What these tests CANNOT show, said plainly: that the tuning windows were
genuinely shortened at runtime. The report records the geometry it was given;
"the windows were used" is a memory fact, and this change assigns it to the
32g-container microbench in review rather than to an assertion here. Cf.
`test_driver_probe.py`'s docstring for the same discipline.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "tools" / "quantize_hf_model.py"

VOCAB = 512
SEQ_LEN = 32  # evaluation geometry
TUNE_SEQ_LEN = 16  # tuning-window geometry, strictly shorter
BATCH = 8
BLOCKS = 3
DRIVER_TIMEOUT = 900


def _reject_constant(name: str):  # noqa: ANN202
    raise ValueError(f"non-finite constant {name!r} reached the report")


@pytest.fixture(scope="module")
def offline_corpus(tmp_path_factory):  # noqa: ANN001, ANN201
    """A parquet corpus in the hub layout the driver globs, built locally."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path_factory.mktemp("hf-cache")
    shard_dir = (
        root / "datasets--Salesforce--wikitext" / "snapshots" / "fixture" / "wikitext-2-raw-v1"
    )
    shard_dir.mkdir(parents=True)
    sentence = "the quick brown fox jumps over the lazy dog . "
    lines = [sentence * 30 for _ in range(20)]
    for split in ("train", "test"):
        pq.write_table(pa.table({"text": lines}), shard_dir / f"{split}-00000-of-00001.parquet")
    return root


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):  # noqa: ANN001, ANN201
    """A tiny Qwen3 (3 blocks) plus a byte-level BPE tokenizer, both local."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    directory = tmp_path_factory.mktemp("tiny-qwen3")
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


def _run_driver(
    corpus_root: Path,
    model_dir: Path,
    out_dir: Path,
    seq_len: int = SEQ_LEN,
    extra: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Invoke the real driver the way the user's command line invokes it."""
    env = dict(os.environ)
    # The launch-ownership contract: a driver run names its owner or refuses to
    # start. This suite owns the runs it spawns, so it says so -- the refused
    # half of that contract is pinned in tests/ptq/test_launch_attribution.py.
    env["QKB_LAUNCH_OWNER"] = "pytest"
    env["HF_HUB_CACHE"] = str(corpus_root)
    env["QUANTIBBLER_BREADCRUMBS"] = str(out_dir / "breadcrumbs.log")
    kd_tmp = out_dir / "kd-tmp"
    kd_tmp.mkdir(parents=True, exist_ok=True)
    env["QUANTIBBLER_KD_TMPDIR"] = str(kd_tmp)
    cmd = [
        sys.executable,
        str(DRIVER),
        # 'auto' now refuses without an accelerator, and this suite has none:
        # asking for CPU by name is the point of the change. The silence it
        # replaces is what let a production fire run 21 hours on one socket.
        "--device",
        "cpu",
        "--model",
        str(model_dir),
        "--rank",
        "4",
        "--solver",
        "svd",
        "--max-admm-iter",
        "2",
        "--seq-len",
        str(seq_len),
        "--calibration-seqs",
        "16",
        "--eval-seqs",
        "16",
        "--probe-batches",
        "1",
        "--out",
        str(out_dir),
        "--checkpoint-dir",
        str(out_dir / "checkpoints"),
        *(extra or []),
    ]
    return subprocess.run(
        cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=DRIVER_TIMEOUT
    )


def _report(out_dir: Path) -> dict:
    path = out_dir / "quantize_report.json"
    assert path.is_file(), f"no report at {path}"
    return json.loads(path.read_text(), parse_constant=_reject_constant)


def test_the_flag_is_documented_and_separate_from_the_eval_geometry():
    """`--tune-seq-len` is its own knob; `--seq-len` stays the eval geometry.

    Parser surface only — no model is touched — so a missing flag fails here
    for the right reason rather than as a confusing downstream KeyError.
    """
    proc = subprocess.run(
        [sys.executable, str(DRIVER), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "--tune-seq-len" in proc.stdout
    # The help text has to say which geometry each flag owns: a reader who
    # cannot tell them apart will set the wrong one and silently widen the
    # windows back to the eval length (the exact OOM this flag exists to fix).
    assert "--seq-len" in proc.stdout
    tail = proc.stdout.split("--tune-seq-len", 1)[1][:400].lower()
    assert "window" in tail or "tune" in tail, proc.stdout[-1500:]


def test_default_mode_records_a_single_geometry(
    offline_corpus,
    tiny_model,
    tmp_path,  # noqa: ANN001
):
    """`default_single_geometry_unchanged`: unset means tune == eval, exactly.

    An unset flag must not read as "some other geometry" — the run's tuning
    geometry IS `--seq-len`, and the report says so in two fields that agree.
    """
    out_dir = tmp_path / "default-artifact"
    proc = _run_driver(offline_corpus, tiny_model, out_dir)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    report = _report(out_dir)

    assert report["seq_len"] == SEQ_LEN
    assert report["tune_seq_len"] == SEQ_LEN, (
        "with the flag unset the tuning geometry must default to --seq-len; "
        f"got {report.get('tune_seq_len')!r}"
    )
    # Single-geometry behavior is unchanged in the eval direction too: the
    # held-out set is still built at --seq-len, so the row count is the same
    # as it has always been and the probe slice still sits on it.
    assert report["eval_rows"] == 2 * BATCH
    assert report["probe"]["batches"] == 1


def test_split_mode_keeps_eval_at_seq_len_and_records_both_geometries(
    offline_corpus,
    tiny_model,
    tmp_path,  # noqa: ANN001
):
    """`split_mode_preserves_eval_geometry`: eval at 32, tuning windows at 16.

    Non-vacuity is the point of running BOTH modes: a driver that simply echoes
    whatever was passed cannot satisfy the default case and this one at once.
    """
    out_dir = tmp_path / "split-artifact"
    proc = _run_driver(
        offline_corpus,
        tiny_model,
        out_dir,
        seq_len=SEQ_LEN,
        extra=["--tune-seq-len", str(TUNE_SEQ_LEN)],
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    report = _report(out_dir)

    # The evaluation geometry is `--seq-len` and only `--seq-len`.
    assert report["seq_len"] == SEQ_LEN
    assert report["eval_rows"] == 2 * BATCH
    assert report["probe"]["batches"] == 1
    # The tuning geometry is the new flag, and it is genuinely different.
    assert report["tune_seq_len"] == TUNE_SEQ_LEN
    assert report["tune_seq_len"] != report["seq_len"]
    # Non-vacuity: the driver sliced the calibration rows into windows of
    # the tune geometry rather than echoing the flag into the report.
    # 2 batches of 32-token rows -> 4 windows of 16 tokens.
    assert "calib sliced to 4 windows of 16" in proc.stdout, proc.stdout[-2000:]
    # Token preservation: the number of calibration SEQUENCES is unchanged
    # pre-slice, so no calibration data was dropped to shorten the windows.
    assert "calib 2 batches, eval 2 batches" in proc.stdout, proc.stdout[-2000:]
