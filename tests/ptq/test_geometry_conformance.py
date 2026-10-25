"""qk-geometry-conformance: the data knobs count sequences, not packed batches.

RED-FIRST RECORD (pre-fix, HEAD ``47661cc``). The driver handed the knob value
straight to the loader, and the loader's ``n_chunks`` parameter indexes *packed
batches* of ``BATCH = 8`` rows, so ``--calibration-seqs 128`` produced
``len(calib) * BATCH == 1024`` — eight times the [NQ] protocol's 128 — and the
report's own deviation flag fired:

    runtime_deviations: [{"kind": "calibration_rows",
                          "detail": "1024 calibration rows against the
                                     protocol's 128 ([NQ] Sec 3.4/App. C)"}]

The v2 run carried exactly that line. ``EVAL_SEQS=18`` carried the same defect
on the eval axis: 18 batches = 144 sequences, which is the fingerprint in the
run's own ``eval coverage: 144/145 blocks`` health line.

These tests pin the corrected contract at the knob boundary: the conversion to
packed batches, the fail-closed divisibility rule, and the eval-window
re-expression in sequences. The wiring test runs the real driver as the user
invokes it — its ``calib N batches`` line is where the knob's unit becomes
observable, and pre-fix it read ``calib 128 batches`` for a 128-sequence knob.

Geometry note: the wiring run passes ``--paper-faithful`` with a deliberately
non-maximal ``--eval-seqs`` so the driver's own preflight gate exits it right
after the line under test, before any quantization work.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
# The RED records are taken against the PRE-FIX driver revision, so the path is
# overridable: `QHFM_DRIVER=<pre-fix driver> uv run pytest ...` reproduces the
# red-first failure without touching the working tree.
DRIVER = pathlib.Path(os.environ.get("QHFM_DRIVER", REPO_ROOT / "tools" / "quantize_hf_model.py"))

VOCAB = 512
SEQ_LEN = 32
BLOCKS = 3
DRIVER_TIMEOUT = 900

CALIBRATION_SEQS = 128  # the [NQ] protocol value, in sequences
PAPER_SEQ_LEN = 2048  # the protocol sequence length ([NQ] Sec 3.4/App. C)
PAPER_BLOCKS = 145  # wikitext-2 test at 2048: the maximal local eval window


def _load_driver():
    """The driver as a module. It is ``main()``-guarded, so importing is inert."""
    spec = importlib.util.spec_from_file_location("qhfm_geometry", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load_driver()


@pytest.fixture(scope="module")
def offline_corpus(tmp_path_factory):
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
def tiny_model(tmp_path_factory):
    """A tiny Qwen3 (3 blocks) plus a byte-level BPE tokenizer, both local."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

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


# --- the knob boundary -------------------------------------------------------


def test_calibration_knob_counts_sequences(driver):
    """``--calibration-seqs N`` loads N sequences, i.e. N // BATCH packed batches.

    The end-to-end invariant in the spec's own form: ``len(calib) * BATCH ==
    CALIBRATION_SEQS``. Pre-fix the knob value itself was the chunk count, so
    the same expression evaluated to 128 * 8 == 1024.
    """
    n_chunks = driver.packed_chunks(CALIBRATION_SEQS, "--calibration-seqs")
    assert n_chunks == CALIBRATION_SEQS // driver.BATCH == 16
    assert n_chunks * driver.BATCH == CALIBRATION_SEQS


def test_calibration_knob_fails_closed_on_odd_count(driver):
    """A sequence count that is not a whole number of packed batches is refused."""
    with pytest.raises(ValueError) as excinfo:
        driver.packed_chunks(CALIBRATION_SEQS + 2, "--calibration-seqs")
    message = str(excinfo.value)
    assert "--calibration-seqs" in message
    assert "divisible" in message
    assert str(driver.BATCH) in message


def test_help_text_names_the_knobs_unit():
    """Both knobs document sequences, carrying the protocol's own numbers.

    Pre-fix the help read "16 batches of 8 sequences = 128 calibration rows"
    and "5 batches of 8 = 40 eval rows": the paper's numbers were expressed in
    a unit the CLI never named. The default is the protocol value in the knob's
    own unit (128 calibration / 40 eval sequences).
    """
    proc = subprocess.run(
        [sys.executable, str(DRIVER), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    help_text = proc.stdout
    assert "128 calibration SEQUENCES" in help_text
    assert "16 batches" in help_text
    assert "40 held-out evaluation SEQUENCES" in help_text
    assert "5 batches" in help_text


# --- the eval axis -----------------------------------------------------------


def test_eval_knob_counts_sequences(driver):
    """Spec scenario ``eval_knob_counts_sequences``: 18 batches -> 144 sequences.

    The derivation is the driver's own, fed the run record it is actually fed:
    ``evaluation_record`` at the paper geometry (wikitext-2 test, seq 2048)
    gives 145 blocks and keeps ``full_local_window_batches == 18`` — a harness
    fact in BATCHES, pinned as such by the report's own tests — while
    ``--eval-seqs`` counts SEQUENCES, so the paper-faithful gate's demanded
    value is ``eval_window_sequences(record) == 18 * BATCH == 144``. A gate
    comparing the knob against the raw batch count would call an 18-sequence
    window (2 batches) maximal.

    RED-FIRST RECORD (pre-fix driver, ``84b4615^``): this test fails with
    ``AttributeError: module 'qhfm_geometry' has no attribute
    'eval_window_sequences'`` — the sequence-unit derivation does not exist
    there. The unit defect it replaces was observable at the CLI: the pre-fix
    ``--help`` read ``--eval-seqs 5 batches of 8 = 40 eval rows`` and the gate
    demanded ``--eval-seqs {batches}``, so ``18`` named 18 packed batches (144
    sequences) and 144 was not a legal value of the knob at all.
    """
    evaluation = driver.evaluation_record(
        evaluator=driver.EVAL_EVALUATOR,
        source_tokens=PAPER_BLOCKS * PAPER_SEQ_LEN,
        seq_len=PAPER_SEQ_LEN,
        selected_blocks=0,
        batch=driver.BATCH,
        selection_seed=0,
        selection_fingerprint="0" * 16,
    )
    assert evaluation["full_blocks"] == PAPER_BLOCKS == 145
    assert evaluation["full_local_window_batches"] == 18
    window = driver.eval_window_sequences(evaluation)
    assert window == 144 == 18 * driver.BATCH
    # The unit is not cosmetic: the raw batch count is a different number, and
    # the knob's own conversion is the exact inverse of the derivation.
    assert window != evaluation["full_local_window_batches"]
    assert driver.packed_chunks(window, "--eval-seqs") == evaluation["full_local_window_batches"]


# --- wiring: the driver actually uses the boundary ---------------------------


def test_driver_reports_sequences_for_the_calibration_knob(offline_corpus, tiny_model, tmp_path):
    """The real driver, invoked as the user invokes it, reports N sequences.

    ``calib {len(calib)} batches`` is where the knob's unit becomes observable.
    Pre-fix, ``--calibration-seqs 128`` printed ``calib 128 batches`` (1024
    sequences); post-fix it must print ``calib 16 batches`` (128 sequences).

    The run is bounded by the driver's own preflight: ``--paper-faithful`` with
    a non-maximal ``--eval-seqs`` raises right after the line under test.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    env = dict(os.environ)
    # The launch-ownership contract: a driver run names its owner or refuses to
    # start.
    env["QKB_LAUNCH_OWNER"] = "pytest"
    env["HF_HUB_CACHE"] = str(offline_corpus)
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
        str(tiny_model),
        "--rank",
        "4",
        "--solver",
        "svd",
        "--max-admm-iter",
        "2",
        "--seq-len",
        str(SEQ_LEN),
        "--calibration-seqs",
        str(CALIBRATION_SEQS),
        # Deliberately non-maximal but batch-aligned (8 sequences = 1 packed
        # batch): trips the paper-faithful preflight gate immediately after the
        # calib line, so no quantization work runs. A non-multiple would be
        # refused by the divisibility rule before the gate is reached.
        "--eval-seqs",
        "8",
        "--paper-faithful",
        "--out",
        str(out_dir / "artifact"),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=DRIVER_TIMEOUT, check=False
    )
    combined = proc.stdout + proc.stderr
    expected = f"calib {CALIBRATION_SEQS // 8} batches"
    assert expected in combined, (
        f"driver did not report the knob in sequences; expected {expected!r}.\n"
        f"stdout+stderr tail:\n{combined[-2000:]}"
    )
    assert "maximal local evaluation window" in combined, (
        "the wiring run was meant to stop at the paper-faithful preflight gate"
    )
