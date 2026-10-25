"""`tools/run-ptq-container.sh`: the launch interface, pinned.

The wrapper is how real runs are launched, and it just bit us twice in one
sitting: positional arguments were **silently ignored** (so `SEQ_LEN=2048` in
argument position produced a healthy-looking seq-512 run), and there was no way
to inspect the launch without executing it — which is why a "check the command
line" probe started a real 32 GiB container run. `DRY_RUN=1` is the seam that
makes this testable without a daemon, and these tests use it.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

_REPO = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "tools" / "run-ptq-container.sh"
_KNOBS = (
    "MODEL",
    "OUT",
    "RANK",
    "SOLVER",
    "SEQ_LEN",
    "TUNE_SEQ_LEN",
    "CALIBRATION_SEQS",
    "EVAL_SEQS",
    "FACTOR_MODE",
    "ABLATION",
    "LAYER_SCOPE",
    "PAPER_FAITHFUL",
    "STE_FLIP_PASS_BUDGET",
    "COMPUTE_DTYPE",
    "IMAGE",
    "GPU",
    "DRY_RUN",
    "MODE",
)


def _launch(**env: str) -> subprocess.CompletedProcess:
    """Run the wrapper with DRY_RUN on, so no container is ever started.

    The ambient environment's knobs are stripped first: an exported `SOLVER` or
    `SEQ_LEN` (empty or otherwise) would otherwise decide what these tests
    observe, which is exactly the kind of hidden input the wrapper is being
    pinned against.
    """
    environment = {key: value for key, value in os.environ.items() if key not in _KNOBS}
    environment.update({"DRY_RUN": "1", **env})
    return subprocess.run(
        ["sh", str(_SCRIPT)],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _launch_past_dry_run(tmp_path: pathlib.Path, **env: str) -> subprocess.CompletedProcess:
    """Run the wrapper through its REAL launch path, with no daemon in the loop.

    `DRY_RUN` exits before the container starts, so everything that happens
    after the launch (the container witness, the exit-code plumbing) is
    unreachable from `_launch`. A silent `docker` stub on the head of PATH
    stands in for the daemon: the run completes, exit 0, having printed
    nothing -- which is exactly the shape of a fire whose account was not
    captured.

    The stub is a SYMLINK to `/bin/true`, not a written script, and that is
    load-bearing: `/tmp` is mounted `noexec` on this host (as in CI sandboxes
    generally), and a shell does not stop at a PATH entry it cannot exec --
    dash prints `Permission denied` for the stub, moves to the next PATH
    entry, and runs the REAL `/usr/bin/docker`. Measured, not reasoned: the
    first cut of this leg started a real 32 GiB container. The symlink
    resolves to an exec-able inode, so the stub wins wherever it is placed.

    `infrastructure_interception_proof` (openspec/changes/qk-wrapper-contract)
    makes the in-band proof below non-optional: a test that overrides `PATH`
    to intercept a command which launches real infrastructure SHALL assert the
    resolved binary is the shim, in the same environment handed to the code
    under test, BEFORE that code runs. `command -v` alone does not discharge
    that -- it happily reports the stub the shell later refuses to exec, which
    is precisely how the noexec fall-through above survived a green suite. The
    exec probe is the assertion that actually binds.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "docker"
    stub.symlink_to("/bin/true")

    environment = {key: value for key, value in os.environ.items() if key not in _KNOBS}
    environment.update(
        {
            "WATCHDOG": "0",  # keep the host-side watchdog out of a unit test
            "PATH": str(bindir) + os.pathsep + os.environ.get("PATH", ""),
            **env,
        }
    )

    # Proven in-band, in this very environment, before the wrapper runs.
    resolved = subprocess.run(
        ["sh", "-c", "command -v docker"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert resolved.stdout.strip() == str(stub), (
        f"the shell resolves docker to {resolved.stdout.strip()!r}, not the stub {stub}"
    )
    probe = subprocess.run(
        ["docker", "inspect", "--format", "{{.Id}}", "qkb-interception-probe"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (probe.returncode, probe.stdout, probe.stderr) == (0, "", ""), (
        "the stub did not answer -- the real docker answers this inspect (no such object, or a "
        "daemon refusal) and would put this leg on real infrastructure under the test's timeout",
        probe.returncode,
        probe.stdout,
        probe.stderr,
    )

    return subprocess.run(
        ["sh", str(_SCRIPT)],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_arguments_are_refused_not_ignored():
    """The defect that motivated this file: a dropped argument looks healthy."""
    completed = subprocess.run(
        ["sh", str(_SCRIPT), "SEQ_LEN=2048"],
        cwd=_REPO,
        env=dict(os.environ, DRY_RUN="1"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2, (completed.returncode, completed.stdout)
    assert "takes no arguments" in completed.stderr
    assert "SEQ_LEN=2048" in completed.stderr
    assert "docker" not in completed.stdout


def test_env_knobs_reach_the_command_line():
    completed = _launch(RANK="512", SOLVER="svd", SEQ_LEN="2048", OUT="/tmp/nq-run")
    assert completed.returncode == 0, completed.stderr
    # The config echo comes first by design; the launch itself is the last line.
    assert completed.stdout.strip().splitlines()[-1].startswith("'docker' 'run'")
    assert "'--rank' '512'" in completed.stdout
    assert "'--solver' 'svd'" in completed.stdout
    assert "'--seq-len' '2048'" in completed.stdout
    assert "'/tmp/nq-run:/out'" in completed.stdout


def test_tune_knob_is_passthrough_and_never_wrapper_chosen():
    """`TUNE_SEQ_LEN` (qk-nq-recreation-bar): set -> on the command line; absent.

    Unset must mean UNSET, not "whatever the wrapper likes": the driver's own
    default (tuning windows == `--seq-len`) is the pre-flag behavior, and a
    wrapper that always passed a number would be choosing the tuning geometry -
    the same silent decision this script exists to stop making.
    """
    unset = _launch(RANK="512", SOLVER="svd", SEQ_LEN="2048")
    assert unset.returncode == 0, unset.stderr
    assert "'--tune-seq-len'" not in unset.stdout
    assert "tune_seq_len=unset" in unset.stderr
    # The stream contract, pinned from both sides: stdout is the machine channel
    # (pure argv), stderr carries the announce and the advisories. A future
    # accidental redirect reversal lands here, not in a human's eyeballs.
    assert "run-ptq-container: model=" not in unset.stdout
    assert unset.stdout.strip().splitlines()[-1].startswith("'docker' 'run'")

    split = _launch(RANK="512", SOLVER="svd", SEQ_LEN="2048", TUNE_SEQ_LEN="512")
    assert split.returncode == 0, split.stderr
    assert "'--tune-seq-len' '512'" in split.stdout
    assert "'--seq-len' '2048'" in split.stdout  # eval geometry untouched
    assert "tune_seq_len=512" in split.stderr


def test_compute_dtype_knob_is_passthrough_and_recorded():
    """Precision is run identity, not a wrapper default.

    Unset must mean UNSET on the command line -- the driver's own float64
    default is the reference path -- while the announce, the launch.env and the
    breadcrumb trail all state the effective value, because two arms whose
    records differ only in precision cannot be told apart afterwards.
    """
    unset = _launch(RANK="512", SOLVER="svd", SEQ_LEN="2048", OUT="/tmp/nq-dtype")
    assert unset.returncode == 0, unset.stderr
    assert "'--compute-dtype'" not in unset.stdout
    assert "compute_dtype=float64" in unset.stderr

    fp32 = _launch(
        RANK="512", SOLVER="svd", SEQ_LEN="2048", OUT="/tmp/nq-dtype", COMPUTE_DTYPE="float32"
    )
    assert fp32.returncode == 0, fp32.stderr
    assert "'--compute-dtype' 'float32'" in fp32.stdout
    assert "compute_dtype=float32" in fp32.stderr


def test_gpu_requires_a_rocm_tagged_image():
    """GPU=1 on the default tag is the collision the rocm tag exists to prevent.

    `quantibbler-ptq:rocm` and `quantibbler-ptq:latest` are different images
    living under two names, so passing ROCm devices to the default tag hands the
    hardware a CUDA-wheel torch -- the same failure the rocm Dockerfile's build
    assertion makes unbuildable. Refusing is the only safe direction: a run's
    record cannot repair a wrong binary after the fact.

    This leg is device-independent by construction, and that is the point:
    DRY_RUN inspects a launch without executing it, so an absent /dev/kfd +
    /dev/dri is an advisory for the dry run (whose argv is the plan for a GPU
    host) and a refusal for the real launch. Measured on a GPU-less ubuntu
    runner: with the presence check binding the dry run too, this leg and its
    siblings died at "GPU=1 but /dev/kfd is absent on this host" before printing
    any argv -- the seam unable to inspect a launch it never runs.
    """
    refused = _launch(RANK="512", SOLVER="svd", SEQ_LEN="2048", GPU="1")
    assert refused.returncode == 2, refused.stderr
    assert "requires a rocm-tagged image" in refused.stderr
    assert "'docker'" not in refused.stdout, "a refused launch must not resolve an argv"

    ok = _launch(RANK="512", SOLVER="svd", SEQ_LEN="2048", GPU="1", IMAGE="quantibbler-ptq:rocm")
    assert ok.returncode == 0, ok.stderr
    assert "'quantibbler-ptq:rocm'" in ok.stdout
    assert "'--device' '/dev/kfd'" in ok.stdout


def test_data_knob_defaults_are_protocol_values_in_sequences():
    """The data knobs count sequences, so the defaults are the protocol's numbers.

    Pre-fix the wrapper defaulted to 16 *batches* (128 sequences) and 5 batches
    (40 sequences) — the protocol's values in a unit the CLI never named. With
    the knob re-expressed in sequences (qk-geometry-conformance), a default that
    still said 16 would silently calibrate on an eighth of the paper's data.
    """
    default = _launch(RANK="512", SOLVER="svd", SEQ_LEN="2048")
    assert default.returncode == 0, default.stderr
    assert "calibration_seqs=128" in default.stderr
    assert "eval_seqs=40" in default.stderr


def test_image_identity_and_digest_are_stated():
    """A tag is a mutable name; the digest is the binary.

    Both arms of a comparison can carry the same tag while holding different
    bytes, which is the compute_dtype/rho_end gap in its quietest form. The
    default is stated rather than implied, and an uninspectable image says
    `unknown` instead of silently omitting the field.
    """
    default = _launch(RANK="512", SOLVER="svd", SEQ_LEN="2048")
    assert default.returncode == 0, default.stderr
    assert "'quantibbler-ptq:latest'" in default.stdout
    assert "image=quantibbler-ptq:latest" in default.stderr
    assert "image_digest=" in default.stderr, "the digest is never left implied"


def test_dry_run_announces_nothing_it_need_not_and_the_advisories_it_must():
    """The advisories are the difference between a diagnostic and a claim."""
    default = _launch()
    assert default.returncode == 0
    assert "SOLVER unset -> admm" in default.stderr
    assert "SEQ_LEN=512 is a diagnostic geometry" in default.stderr
    assert "rank=8 solver=admm seq_len=512" in default.stderr
    assert "run-ptq-container: model=" not in default.stdout
    # A launch that does not claim paper fidelity is not running the paper's
    # method (gate-shaped stage budgets, TuneFP at one step): the wrapper says
    # so, and the flag never reaches the command line unasked.
    assert "PAPER_FAITHFUL unset" in default.stderr
    assert "'--paper-faithful'" not in default.stdout
    assert "paper_faithful=0 ste_flip_pass_budget=0" in default.stderr

    # The claimable split run: eval geometry, tune window AND the paper-faithful
    # claim all chosen -> silent.
    split = _launch(
        RANK="512", SOLVER="svd", SEQ_LEN="2048", TUNE_SEQ_LEN="512", PAPER_FAITHFUL="1"
    )
    assert split.returncode == 0
    assert "note:" not in split.stderr, split.stderr
    assert "rank=512 solver=svd seq_len=2048" in split.stderr
    assert "paper_faithful=1" in split.stderr
    assert "'--paper-faithful'" in split.stdout

    # The discrete flip pass is a STRICT PASSTHROUGH: unset means the driver's
    # own default, set means the flag and its value reach the driver.
    flip = _launch(STE_FLIP_PASS_BUDGET="4")
    assert flip.returncode == 0, flip.stderr
    assert "'--ste-flip-pass-budget' '4'" in flip.stdout
    assert "ste_flip_pass_budget=4" in flip.stderr
    assert "'--ste-flip-pass-budget'" not in default.stdout

    # The OOM-shaped launch is ANNOUNCED, not refused: seq 2048 with the tune
    # window left unset is the geometry that died at the ceiling, so the wrapper
    # says so and still starts the run (returncode 0, launch line printed).
    oom = _launch(RANK="512", SOLVER="svd", SEQ_LEN="2048")
    assert oom.returncode == 0, oom.stderr
    assert "SEQ_LEN=2048 with TUNE_SEQ_LEN unset" in oom.stderr
    assert "rank=512 solver=svd seq_len=2048" in oom.stderr


def test_wrapper_never_truncates_the_breadcrumb_trail():
    """The trail is the post-mortem of the run that died, not of the next one."""
    src = _SCRIPT.read_text()
    assert ": > /opt/data/tmp/quantibbler-breadcrumbs.log" not in src
    assert ">> /opt/data/tmp/quantibbler-breadcrumbs.log" in src
    assert "=== run %s rank=%s solver=%s" in src


def test_retired_lane0_modes_are_refused_by_name():
    """The lane0 modes were retired with their harness (`e5c6939`).

    A deleted probe behind a live `MODE` arm is a silent dud: the launch looks
    valid and dies inside the container. The wrapper refuses both retired mode
    names up front instead, so a stale launcher learns at the wrapper rather
    than at a missing file.
    """
    for mode in ("lane0-rehearsal", "lane0-fire"):
        completed = _launch(MODE=mode)
        assert completed.returncode == 2, (mode, completed.returncode, completed.stderr)
        assert "MODE retired" in completed.stderr
        assert mode in completed.stderr
        assert "'docker'" not in completed.stdout, "a refused MODE must not resolve an argv"


def test_only_ptq_is_a_valid_mode():
    """`ptq` is the only surviving mode; anything else is refused by name."""
    completed = _launch(MODE="bogus")
    assert completed.returncode == 2, (completed.returncode, completed.stderr)
    assert "unknown" in completed.stderr
    assert "only ptq" in completed.stderr
    assert "'docker'" not in completed.stdout


def test_launch_env_records_the_mode_and_no_retired_lane0_knobs(tmp_path):
    """The provenance block names the mode and carries no retired lane0 fields.

    The lane0 knobs (`rehearsal_batches`, `chunk_tokens`, `wall_min`,
    `mem_floor_gb`) were written unconditionally into launch.env before the
    modes were retired; they are now gone with the modes, and `mode` remains.
    """
    completed = _launch(OUT=str(tmp_path))
    assert completed.returncode == 0, completed.stderr
    text = (tmp_path / "launch.env").read_text()
    assert "mode=ptq" in text
    lines = text.splitlines()
    for retired in ("rehearsal_batches=", "chunk_tokens=", "wall_min=", "mem_floor_gb="):
        assert retired not in text, f"retired lane0 field in launch.env: {retired}"
    # The splice defect that put a bare `?` line in the green-1 record.
    assert "?" not in lines, f"stray splice line in launch.env: {lines}"
    assert any(
        line.startswith("tree_dirty_files=") and line.split("=", 1)[1].isdigit() for line in lines
    ), lines


def test_absent_container_log_witness_fails_closed(tmp_path):
    """The witness is fail-closed for ANY mode, and `ptq` is now every mode.

    `container_log_witness` law: a fire whose `container.log` is absent or
    empty SHALL fail closed, because a run with no captured account cannot be
    read after the fact -- "it died with no reason" is the failure this
    wrapper exists to stop paying for. The gate was written for the lane0
    probing path and carried a `MODE != ptq` qualifier; with the lane0 modes
    retired, `MODE` can only be `ptq`, so that qualifier made the gate
    permanently dead. It is dropped here, and this is the pin.

    The `ptq` happy path writes `container.log` through `tee`, so in normal
    operation the gate never fires; only a genuinely-uncaptured run reaches
    it. A container that exits cleanly and prints nothing leaves a zero-byte
    witness -- the smallest real instance of that case.

    The law now lives in `openspec/changes/qk-wrapper-contract/specs/qk-wraper/
    spec.md` (`container_log_witness`, extracted from the archived lane0 delta),
    whose sibling requirement `wrapper_failclosed_branches_hermetic_tests` names
    `witness fail-closed` in the branch list that SHALL be covered by a hermetic
    test. This is that test.
    """
    out = tmp_path / "out"
    completed = _launch_past_dry_run(
        tmp_path, RANK="512", SOLVER="svd", SEQ_LEN="2048", OUT=str(out)
    )

    witness = out / "container.log"
    assert witness.exists(), "the tee must have produced the witness for this leg to mean anything"
    assert witness.stat().st_size == 0, "fixture precondition: the witness is empty"

    assert completed.returncode == 2, (completed.returncode, completed.stdout, completed.stderr)
    assert "container_log_witness missing or empty" in completed.stderr
    assert str(witness) in completed.stderr
