"""qk-fail-fast-loop: the driver-side probe, executed rather than asserted.

The lean tier pins the driver's *shape* (`_driver_wiring_findings` + the
poison battery in `tests/test_qk_report_driver_pin.py`) and the probe's decision
maths (`tests/test_qk_report_probe.py::test_probe_*`). This tier pins the *runtime*: the
parser's knobs, the help surface, and one real driver run — the actual
`tools/quantize_hf_model.py` as the user invokes it, on a locally-built tiny
Qwen3 with a locally-built tokenizer and a synthetic parquet corpus, fully
offline (no HF hub, no network, no cache dependency).

What that run proves: the probe is invoked once per block boundary on the
live host model, the fp pedestal is measured once on the head of the same
held-out batches and reused at every depth, the per-depth trajectory lands in
`quantize_report.json` through `dump_report`, and the process exit code agrees
with the report's status.

What it cannot prove, stated plainly so nobody reads it as more: the fixture
model is random-init, so its next-token distribution is near-uniform and the
prefix ratio sits at ~1.0 no matter how much of the model has been quantized.
This smoke covers the *plumbing*; the *detection* threshold is covered by the
lean comparator tests (`probe_aborts` over the shared `COLLAPSE_LINE`), and a
runtime collapse needs a model whose predictions are sharp enough to be
damaged — see the fixture notes in the change's design.md.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from quantibbler.ptq.calibration import collect_stats
from quantibbler.qk_report import FAILED, MIN_PROBE_DEPTH, exit_code, verdict

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "tools" / "quantize_hf_model.py"

VOCAB = 512
SEQ_LEN = 32
BATCH = 8
BLOCKS = 3  # tiny Qwen3 depth; the probe starts at depth 2, so 2 boundaries
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
    """A tiny Qwen3 (3 blocks) plus a byte-level BPE tokenizer, both local.

    Random init is enough for the plumbing this smoke covers; it is also
    exactly why the ratios here sit at ~1.0 (see the module docstring).
    """
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
    # A real model's tokenizer declares a chat template, and `save_pretrained`
    # writes it to the canonical sidecar the export verifies before emitting an
    # artifact; a synthetic model has to declare one too, or the export refuses
    # for a reason unrelated to what this fixture exercises.
    fast.chat_template = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"

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
    # The probe's batch count is lane instrumentation and env-owned: this suite
    # wants the probe cheap, which is a choice the driver no longer exposes on
    # its operator surface.
    env["QK_PROBE_BATCHES"] = "1"
    # The KD teacher-logit cache needs a directory that EXISTS
    # (`tempfile.TemporaryDirectory` does not create parents), and this test must
    # not inherit the host's scratch dir: the driver's default was the sandbox
    # path `/opt/data/tmp`, which exists here and on no CI runner — that is what
    # turned the first full-python run red. Owner the directory, explicitly.
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
        str(SEQ_LEN),
        "--calibration-seqs",
        "8",
        "--eval-seqs",
        "16",
        "--out",
        str(out_dir),
        "--checkpoint-dir",
        str(out_dir / "checkpoints"),
        *(extra or []),
    ]
    return subprocess.run(
        cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=DRIVER_TIMEOUT
    )


def test_compute_dtype_flag_is_live_and_reaches_the_solve(
    offline_corpus, tiny_model, tmp_path, monkeypatch
):  # noqa: ANN001
    """fp32 must be launchable, not merely plumbed.

    A knob is worthless if the flag is accepted and then ignored -- which is
    precisely what a hardcoded ``compute_dtype="float64"`` in the config builder
    looks like from the outside, and why the driver flag alone proves nothing.
    So this fires the real driver on the tiny model at float32 with the solver
    probe on, and reads the dtype back off a per-linear row: flag -> config ->
    solve -> artifact. The probe row is the only place the solve's own dtype is
    observable, which is the reason that field exists.
    """
    out_dir = tmp_path / "out-fp32"
    monkeypatch.setenv("QK_SOLVER_PROBE", "1")
    # The probe's default path is the CONTAINER path (/out/solver_probe.jsonl) and
    # it raises rather than silently measuring nothing when that is unwritable --
    # correct fail-closed behaviour, and why a local driver invocation must
    # redirect it.
    monkeypatch.setenv("QK_SOLVER_PROBE_PATH", str(out_dir / "solver_probe.jsonl"))
    proc = _run_driver(offline_corpus, tiny_model, out_dir, extra=["--compute-dtype", "float32"])
    assert proc.returncode == 0, (proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:])

    probe = out_dir / "solver_probe.jsonl"
    assert probe.exists(), "the probe was enabled but no row was written"
    rows = [json.loads(x) for x in probe.read_text().strip().splitlines()]
    assert rows, "a run that solved nothing cannot evidence the knob"
    assert {r["compute_dtype"] for r in rows} == {"float32"}, sorted(
        {r["compute_dtype"] for r in rows}
    )


@pytest.mark.parametrize(
    ("knob", "value"),
    [
        ("QK_PROBE_INTERVAL", "0"),
        ("QK_PROBE_INTERVAL", "-2"),
        ("QK_PROBE_BATCHES", "0"),
        ("QK_PROBE_INTERVAL", "abc"),
        ("QK_PROBE_BATCHES", "1.5"),
    ],
)
def test_scn_probe_non_integer_rejected(knob, value, tmp_path):  # noqa: ANN001
    """`probe_non_integer_rejected` + `probe_knob_defaults_and_validation`.

    Runtime half: a non-positive or non-integer probe knob refuses at STARTUP
    naming the value, rather than being clamped into "the probe just did not
    run". It happens before the model is touched, so this needs no fixture.
    """
    proc = subprocess.run(
        [sys.executable, str(DRIVER), "--out", str(tmp_path / "out")],
        cwd=REPO_ROOT,
        env={**os.environ, "QKB_LAUNCH_OWNER": "pytest", knob: value},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode != 0, (proc.returncode, proc.stdout[-2000:])
    assert knob in proc.stderr and value in proc.stderr, proc.stderr[-2000:]
    assert "positive integer" in proc.stderr, proc.stderr[-2000:]


def test_scn_probe_knobs_absent_from_help():
    """The probe knobs are lane instrumentation: no operator surface exposes them."""
    proc = subprocess.run(
        [sys.executable, str(DRIVER), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    for knob in ("--probe-interval", "--probe-batches"):
        assert knob not in proc.stdout, f"{knob} is back on the operator surface"
    for banned in ("probe-line", "probe-threshold", "abort-line", "collapse-line", "probe-limit"):
        assert banned not in proc.stdout, f"{banned} is a threshold override in disguise"


def test_driver_run_records_the_probe_trajectory_and_agrees_with_its_exit_code(
    offline_corpus,
    tiny_model,
    tmp_path,  # noqa: ANN001
):
    """The end-to-end contract: history written, status parseable, exit code
    equal to the report's verdict, provenance equal to the invoked args."""
    out_dir = tmp_path / "artifact"
    proc = _run_driver(offline_corpus, tiny_model, out_dir)
    report_path = out_dir / "quantize_report.json"
    assert report_path.is_file(), proc.stdout[-4000:] + proc.stderr[-2000:]
    payload = report_path.read_text()

    # Strict parse: JSON that carries NaN/Infinity is not a report we accept.
    report = json.loads(payload, parse_constant=_reject_constant)
    assert report["probe_abort"] is None, "the healthy fixture must not abort"
    status = report["status"]
    assert status in ("ok", "degraded", "failed")
    assert proc.returncode == exit_code(status), (status, proc.returncode)
    assert proc.returncode == 0, report  # this fixture is the healthy leg
    assert report["artifact_exported"] is True  # one shape, both paths
    assert report["ppl_quant_native"] is not None

    # Provenance is the invoked arguments, not a literal.
    assert report["solver"] == "svd"
    assert report["scale_dtype"] == "float32"
    assert report["rank"] == 4
    assert report["seq_len"] == SEQ_LEN

    probe = report["probe"]
    assert probe["interval"] == 1
    assert probe["batches"] == 1
    assert report["eval_rows"] == 2 * BATCH  # the slice is a strict subset
    assert math.isfinite(probe["fp_prefix_ppl"]) and probe["fp_prefix_ppl"] > 0

    history = probe["history"]
    assert [entry["depth"] for entry in history] == [2, 3]
    assert [entry["block"] for entry in history] == ["layers.1", "layers.2"]
    assert len(history) == BLOCKS - MIN_PROBE_DEPTH + 1
    for entry in history:
        # One pedestal, every depth: the baseline is the fp model on the same
        # slice, not a per-depth re-draw (fp_prefix_baselines_same_depth).
        assert entry["fp_prefix_ppl"] == probe["fp_prefix_ppl"]
        assert math.isfinite(entry["prefix_ppl"]) and entry["prefix_ppl"] > 0
        assert entry["ratio"] == pytest.approx(entry["prefix_ppl"] / entry["fp_prefix_ppl"])
        assert verdict(entry["ratio"]) != FAILED


def test_device_auto_refuses_without_an_accelerator(monkeypatch):
    """`auto` is the accelerator or refuse -- never a silent CPU fallback.

    This replaces a source-text pin (`tests/test_qk_report_driver_pin.py`) that
    asserted the driver's *source* contained no `SystemExit` outside the verdict
    path. It rotted the moment the device gate landed, because it was a
    statement about text rather than about behaviour -- and the behaviour it
    should have been checking is the one that cost 21 hours on one socket while
    the run's own launch record printed `device=cpu`.

    The gate is a pre-flight refusal, deliberately outside `main()`: `main()` has
    exactly one exit, the verdict path, and that invariant stays pinned.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("qk_driver_under_test", DRIVER)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    monkeypatch.setattr(driver.torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as excinfo:
        driver._resolve_device("auto")
    message = str(excinfo.value)
    assert "GPU=1" in message, message
    assert "rocm-official" in message, message
    # The non-accelerator path is reachable -- by explicit name only.
    assert driver._resolve_device("cpu") == "cpu"

    monkeypatch.setattr(driver.torch.cuda, "is_available", lambda: True)
    assert driver._resolve_device("auto") == "cuda"
    assert driver._resolve_device("cpu") == "cpu"


def test_scn_wrapper_instance_serves_collect_stats(tiny_model):
    """The driver wrapper instance feeds collect_stats's whole-model CE route.

    Locks the trunk-routing fix (Lane 1, qk-nanoquant-alignment-lane1):
    ``CausalLMLogits`` composes trunk and head handles exactly as the wrapper's
    forward does, and collect_stats must build its CE from those handles
    WITHOUT applying the head twice. A reintroduced always-route would crash
    here with ``lm_head`` applied to vocab-wide logits — hidden-wide Linear —
    which is the exact shape that killed the first cut of this fix.
    """
    import importlib.util

    qhm_path = Path(__file__).resolve().parents[2] / "tools" / "quantize_hf_model.py"
    spec = importlib.util.spec_from_file_location("quantize_hf_model_under_test", qhm_path)
    qhm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qhm)

    causal = Qwen3ForCausalLM.from_pretrained(tiny_model)
    wrapper = qhm.CausalLMLogits(causal)
    torch.manual_seed(0)
    batches = [torch.randint(0, VOCAB, (2, 16)) for _ in range(3)]
    book = collect_stats(wrapper, batches)
    assert len(book) > 0
    for entry in book:
        assert np.isfinite(entry.d_out_raw).all()
        assert float(entry.d_out_raw.max()) > 0.0  # the CE cotangent has real magnitude
