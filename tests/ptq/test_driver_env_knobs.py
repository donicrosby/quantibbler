"""qk-cli-sane: the lane-internal env knobs, executed through the real driver.

Scenarios (spec delta ``specs/qk-ptq``): ``env_var_empty_treated_as_unset``,
``probe_empty_env_var_treated_as_unset``, ``env_var_wires_to_config``,
``probe_env_var_override``, ``env_var_validation_rejects_invalid`` and
``env_var_not_a_cli_flag``.

Why these legs have to be subprocess runs of the real driver: the parser no
longer exposes any of these knobs, so the operator surface cannot pin them, and
a helper-level call pins the helper rather than the wiring. The driver is the
only thing that resolves the environment and builds the config from it.

The empty-string arms are the load-bearing ones. ``tools/run-ptq-container.sh``
passes ``-e QK_FACTOR_MODE="$FACTOR_MODE"`` positionally, so a launcher that
simply does not set the knob still hands the driver an EMPTY value — and an
empty value read as a value refuses every default launch at startup. An earlier
revision of this change had exactly that defect out of sight: 54 owning tests
(wrapper, driver-wiring AST pin, probe tier) stayed green while an empty
``QK_FACTOR_MODE=`` refused. These legs run the driver with the knobs on the
command *environment*, so the refusal (or the default) is observed where it
happens.

Tier: ``tests/ptq/`` (torch lane). The fixtures are the tiny Qwen3 + synthetic
parquet corpus from ``test_driver_tune_seq_len`` — imported, not re-declared, so
the two modules cannot drift on the fixture shape.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_driver_tune_seq_len import (  # noqa: F401  (pytest fixtures)
    offline_corpus,
    tiny_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "tools" / "quantize_hf_model.py"

SEQ_LEN = 32
BATCH = 8
DRIVER_TIMEOUT = 900

# The five knobs the CLI cut migrated to the environment. Listed once so a leg
# can clear them all before setting the value it means to test.
_QK_KNOBS = (
    "QK_PROBE_INTERVAL",
    "QK_PROBE_BATCHES",
    "QK_FACTOR_MODE",
    "QK_ABLATION",
    "QK_LAYER_SCOPE",
)

# The seven flags the parser cut. Kept here as the negative list for the
# parser-level leg (``env_var_not_a_cli_flag`` and its siblings): each was a
# real argument before this change, so a caller's old command line must now be
# REJECTED rather than silently accepted-and-ignored.
_CUT_FLAGS = (
    ("--paper-faithful",),
    ("--ste-flip-pass-budget", "8"),
    ("--probe-interval", "4"),
    ("--probe-batches", "1"),
    ("--factor-mode", "fp"),
    ("--ablation", "none"),
    ("--layer-scope", "all"),
)


def _reject_constant(name: str):  # noqa: ANN202
    raise ValueError(f"non-finite constant {name!r} reached the report")


def _driver_env(out_dir: Path, **knobs: str) -> dict[str, str]:
    """The environment one driver invocation runs under.

    Every QK_ knob is POPPED from the inherited environment first: these legs
    are about what the driver resolves, and a knob a co-tenant exported must not
    decide the answer.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for name in _QK_KNOBS:
        env.pop(name, None)
    env.update(knobs)
    env["QKB_LAUNCH_OWNER"] = "pytest"
    env["HF_HUB_CACHE"] = str(out_dir.parent / "hf-cache")
    env["QUANTIBBLER_BREADCRUMBS"] = str(out_dir / "breadcrumbs.log")
    kd_tmp = out_dir / "kd-tmp"
    kd_tmp.mkdir(parents=True, exist_ok=True)
    env["QUANTIBBLER_KD_TMPDIR"] = str(kd_tmp)
    return env


def _run_driver(
    corpus_root: Path,
    model_dir: Path,
    out_dir: Path,
    extra: list[str] | None = None,
    **knobs: str,
) -> subprocess.CompletedProcess:
    """Invoke the real driver with the given QK_ knobs in its environment."""
    env = _driver_env(out_dir, **knobs)
    env["HF_HUB_CACHE"] = str(corpus_root)
    cmd = [
        sys.executable,
        str(DRIVER),
        # 'auto' refuses without an accelerator; this suite has none. It is also
        # the DECLARED device the regime verdict reads.
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
        "16",
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


def _report(out_dir: Path) -> dict:
    path = out_dir / "quantize_report.json"
    assert path.is_file(), f"no report at {path}"
    return json.loads(path.read_text(), parse_constant=_reject_constant)


@pytest.fixture(scope="module")
def empty_knob_run(offline_corpus, tiny_model, tmp_path_factory):  # noqa: ANN001, ANN201, F811
    """ONE driver leg with every QK_ knob explicitly EMPTY.

    Shared by the two empty-string scenarios because they are the same launch
    condition (the wrapper's ``-e QK_X="$UNSET_VAR"`` pass-through) asserted on
    two different knobs; running the model twice would buy nothing.
    """
    out_dir = tmp_path_factory.mktemp("empty-knobs") / "artifact"
    proc = _run_driver(
        offline_corpus,
        tiny_model,
        out_dir,
        **dict.fromkeys(_QK_KNOBS, ""),
    )
    return proc, out_dir


def test_scn_env_var_empty_treated_as_unset(empty_knob_run):
    """`env_var_empty_treated_as_unset`: `QK_FACTOR_MODE=` is not a value.

    The default ``"binary"`` applies, the pipeline runs, and the declaration the
    report records says so — an empty variable is the shell's way of saying
    "never exported", and reading it as a value refuses every default launch.
    """
    proc, out_dir = empty_knob_run
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-3000:]
    report = _report(out_dir)

    declared = report["fidelity_config"]
    assert declared["factor_mode"] == "binary", declared
    assert declared["ablation"] == "none", declared
    assert declared["layer_scope"] == "all", declared
    # The identity record carries the same resolved value: the env var feeds the
    # ONE config path, not a report-side substitution.
    assert report["factor_mode"] == "binary", report["factor_mode"]
    assert report["ablation"] == "none", report["ablation"]
    assert report["layer_scope"] == "all", report["layer_scope"]


def test_scn_probe_empty_env_var_treated_as_unset(empty_knob_run):
    """`probe_empty_env_var_treated_as_unset`: `QK_PROBE_INTERVAL=` -> default 1.

    Non-vacuity: the probe must actually have RUN at every depth-2+ boundary,
    which is what distinguishes "the default interval applied" from "the probe
    was disabled". With the raise-on-empty defect this leg refuses at startup.
    """
    proc, out_dir = empty_knob_run
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-3000:]
    report = _report(out_dir)

    probe = report["probe"]
    assert probe["interval"] == 1, probe
    assert probe["batches"] == 2, probe
    # 3 blocks, probe starts at depth 2, interval 1 -> a reading at 2 AND 3.
    assert [entry["depth"] for entry in probe["history"]] == [2, 3], probe["history"]


def test_scn_env_var_wires_to_config(offline_corpus, tiny_model, tmp_path):  # noqa: ANN001, F811
    """`env_var_wires_to_config`: `QK_FACTOR_MODE=fp` reaches `PTQConfig`.

    The report's ``fidelity_config`` is built by the production
    ``declared_regime_config(quantized_config(args))``, so a value read back
    there is the value the config carried into the pipeline — the env var feeds
    the config field through the driver, not through a second config path.
    """
    out_dir = tmp_path / "fp-artifact"
    # An fp measurement quantizes nothing, so a block checkpoint would archive an
    # empty solve — the checkpoint writer refuses that by design. The measurement
    # is what this leg is about, so it runs with checkpointing off.
    proc = _run_driver(
        offline_corpus, tiny_model, out_dir, extra=["--no-checkpoints"], QK_FACTOR_MODE="fp"
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-3000:]
    report = _report(out_dir)

    assert report["fidelity_config"]["factor_mode"] == "fp", report["fidelity_config"]
    assert report["factor_mode"] == "fp", report["factor_mode"]
    # An fp measurement packs nothing; the report says so rather than claiming
    # an artifact (the export and BPW arms key on the same resolved field).
    assert report["artifact_exported"] is False, report["artifact_exported"]
    assert report["effective_bpw"] is None, report["effective_bpw"]


def test_scn_probe_env_var_override(offline_corpus, tiny_model, tmp_path):  # noqa: ANN001, F811
    """`probe_env_var_override`: `QK_PROBE_INTERVAL=4 QK_PROBE_BATCHES=1`.

    Both knobs are asserted against their DEFAULTS (1 and 2) in the same report,
    so the leg cannot pass by echoing a value the driver would have used anyway:
    interval 4 fires the probe every fourth depth-2+ boundary (only depth 2 on a
    3-block model) and the slice is a single batch.
    """
    out_dir = tmp_path / "override-artifact"
    proc = _run_driver(
        offline_corpus, tiny_model, out_dir, QK_PROBE_INTERVAL="4", QK_PROBE_BATCHES="1"
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-3000:]
    report = _report(out_dir)

    probe = report["probe"]
    assert probe["interval"] == 4, probe
    assert probe["batches"] == 1, probe
    assert probe["interval"] != 1  # non-vacuity against the default leg
    assert probe["batches"] != 2
    assert [entry["depth"] for entry in probe["history"]] == [2], probe["history"]


@pytest.mark.parametrize(
    ("knob", "value"),
    [
        ("QK_FACTOR_MODE", "invalid"),
        ("QK_ABLATION", "invalid"),
        ("QK_LAYER_SCOPE", "invalid"),
    ],
)
def test_scn_env_var_validation_rejects_invalid(knob, value, tmp_path):  # noqa: ANN001
    """`env_var_validation_rejects_invalid`: a choice knob refuses by name.

    Startup refusal, before the model is touched: the message names the variable,
    the offending value, and the accepted choices — the same named-refusal shape
    the probe knobs use. Runtime half of the choice-knob vocabulary, the probe
    knobs' twin living in ``test_driver_probe.py``.
    """
    out_dir = tmp_path / "out"
    env = _driver_env(out_dir, **{knob: value})
    proc = subprocess.run(
        [sys.executable, str(DRIVER), "--out", str(out_dir)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode != 0, (proc.returncode, proc.stdout[-2000:])
    assert knob in proc.stderr, proc.stderr[-2000:]
    assert repr(value) in proc.stderr, proc.stderr[-2000:]
    assert "binary" in proc.stderr or "none" in proc.stderr or "all" in proc.stderr, proc.stderr[
        -2000:
    ]


@pytest.mark.parametrize("argv", _CUT_FLAGS, ids=lambda argv: argv[0])
def test_scn_env_var_not_a_cli_flag(argv, tmp_path):  # noqa: ANN001
    """`env_var_not_a_cli_flag`: the cut flags are not flags any more.

    Each of the seven cut names was a real argument before this change, so a
    caller's old command line must fail at the PARSER (argparse's own rc=2)
    rather than being accepted and ignored. `--factor-mode fp` is the named case;
    the other six ride the same leg so the cut cannot half-land.
    """
    out_dir = tmp_path / "out"
    env = _driver_env(out_dir)
    proc = subprocess.run(
        [sys.executable, str(DRIVER), "--out", str(out_dir), *argv],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 2, (proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:])
    assert "unrecognized arguments" in proc.stderr, proc.stderr[-2000:]
    assert argv[0] in proc.stderr, proc.stderr[-2000:]
