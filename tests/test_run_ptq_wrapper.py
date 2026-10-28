"""`tools/run-ptq-container.sh`: the launch interface, pinned.

The wrapper is how real runs are launched, and it just bit us twice in one
sitting: positional arguments were **silently ignored** (so `SEQ_LEN=2048` in
argument position produced a healthy-looking seq-512 run), and there was no way
to inspect the launch without executing it — which is why a "check the command
line" probe started a real 32 GiB container run. `DRY_RUN=1` is the seam that
lets these legs run without launching anything.

`DRY_RUN` alone is not daemon-free, and that was a real defect here: the seam
still probes the image digest and guards the fixed container name before it
prints argv, so run against the real daemon the file's rc became a property of
the HOST — a live fire holding `quantibbler-ptq-run` turned eight `rc == 0`
assertions into `name is already held` refusals. Every leg therefore runs
against the form-selective `docker` double in `tests/stubs/docker`, placed at
the head of PATH and proven in-band before the wrapper runs.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "tools" / "run-ptq-container.sh"
# The double is a real file on an exec-able filesystem, reached by SYMLINK from
# the (noexec) tmpdir; see `_hermetic_docker`.
_DOCKER_STUB = _REPO / "tests" / "stubs" / "docker"
#: The launch-ownership contract. Every leg declares its owner exactly as a real
#: fire must, so `rc == 0` keeps meaning "the whole gate chain passed" -- a leg
#: that wants the refusal passes the empty string explicitly.
_OWNER_ENV = {"QKB_LAUNCH_OWNER": "wrapper-test", "QKB_LAUNCH_CARD": "t_29f52eb6"}
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
    "COMPUTE_DTYPE",
    "IMAGE",
    "GPU",
    "DRY_RUN",
    "MODE",
)


@pytest.fixture(autouse=True)
def _hermetic_docker(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Head PATH with the form-selective `docker` double, for EVERY leg.

    `infrastructure_interception_proof` (openspec/changes/qk-wrapper-contract):
    the wrapper launches real infrastructure, so its `docker` calls have to be
    intercepted, not merely hoped away. The double is SYMLINKED into
    `tmp_path/bin` from its tracked source rather than written there: `/tmp` is
    mounted `noexec` on this host (as in CI sandboxes generally), so a written
    script would print `Permission denied`, the shell would move to the next
    PATH entry and the REAL `/usr/bin/docker` would run — measured, not
    reasoned: a first cut of a sibling leg started a real 32 GiB container.

    The double's knobs are cleared here so no ambient export can decide a leg's
    answer; a leg that wants the held state sets them explicitly.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "docker").symlink_to(_DOCKER_STUB)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.delenv("QKB_TEST_DOCKER_DIGEST", raising=False)
    monkeypatch.delenv("QKB_TEST_DOCKER_HOLDER", raising=False)


def _assert_interception(environment: dict[str, str]) -> None:
    """Prove the double answers, in this very environment, before the wrapper runs.

    `command -v` alone does not discharge `infrastructure_interception_proof`:
    it happily reports a stub the shell later refuses to exec, which is exactly
    how the noexec fall-through survived a green suite. The exec probe is the
    assertion that binds — a real docker answers this inspect with a refusal on
    stderr, and would put the leg on real infrastructure.
    """
    stub = os.path.join(environment["PATH"].split(os.pathsep)[0], "docker")
    resolved = subprocess.run(
        ["sh", "-c", "command -v docker"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert resolved.stdout.strip() == stub, (
        f"the shell resolves docker to {resolved.stdout.strip()!r}, not the stub {stub}"
    )
    probe = subprocess.run(
        ["docker", "inspect", "--format", "{{.Id}}", "qkb-interception-probe"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (probe.returncode, probe.stdout, probe.stderr) == (
        0,
        environment.get("QKB_TEST_DOCKER_DIGEST", ""),
        "",
    ), (
        "the stub did not answer -- the real docker answers this inspect (no such object, or a "
        "daemon refusal) and would put this leg on real infrastructure under the test's timeout",
        probe.returncode,
        probe.stdout,
        probe.stderr,
    )


def _launch(**env: str) -> subprocess.CompletedProcess:
    """Run the wrapper with DRY_RUN on, against the hermetic `docker` double.

    The ambient environment's knobs are stripped first: an exported `SOLVER` or
    `SEQ_LEN` (empty or otherwise) would otherwise decide what these tests
    observe, which is exactly the kind of hidden input the wrapper is being
    pinned against. Interception is proven in-band first, because DRY_RUN still
    probes the digest and the name guard before it prints argv.
    """
    environment = {key: value for key, value in os.environ.items() if key not in _KNOBS}
    environment.update({**_OWNER_ENV, "DRY_RUN": "1", **env})
    _assert_interception(environment)
    return subprocess.run(
        ["sh", str(_SCRIPT)],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _launch_past_dry_run(**env: str) -> subprocess.CompletedProcess:
    """Run the wrapper through its REAL launch path, with no daemon in the loop.

    `DRY_RUN` exits before the container starts, so everything that happens
    after the launch (the container witness, the exit-code plumbing) is
    unreachable from `_launch`. The same `docker` double the other legs use
    stands in for the daemon (`docker run` and friends answer nothing, exit 0):
    the run completes, exit 0, having printed nothing -- which is exactly the
    shape of a fire whose account was not captured.
    """
    environment = {key: value for key, value in os.environ.items() if key not in _KNOBS}
    environment.update(
        {
            **_OWNER_ENV,
            "WATCHDOG": "0",  # keep the host-side watchdog out of a unit test
            **env,
        }
    )
    _assert_interception(environment)
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

    Unset must mean UNSET on the command line -- the driver's own bf16 default
    is the pipeline's one-declared-dtype path -- while the announce, the
    launch.env and the breadcrumb trail all state the effective value, because
    two arms whose records differ only in precision cannot be told apart
    afterwards.
    """
    unset = _launch(RANK="512", SOLVER="svd", SEQ_LEN="2048", OUT="/tmp/nq-dtype")
    assert unset.returncode == 0, unset.stderr
    assert "'--compute-dtype'" not in unset.stdout
    assert "compute_dtype=bf16" in unset.stderr

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
    # The fidelity claim left the launcher entirely: the report computes the
    # verdict from the declared config, so there is nothing here to announce,
    # record, or pass through.
    assert "PAPER_FAITHFUL" not in default.stderr
    assert "'--paper-faithful'" not in default.stdout
    assert "paper_faithful" not in default.stderr
    assert "ste_flip_pass_budget" not in default.stderr

    # The resolvable split run: eval geometry AND tune window chosen -> silent.
    split = _launch(RANK="512", SOLVER="svd", SEQ_LEN="2048", TUNE_SEQ_LEN="512")
    assert split.returncode == 0
    assert "note:" not in split.stderr, split.stderr
    assert "rank=512 solver=svd seq_len=2048" in split.stderr

    # The OOM-shaped launch is ANNOUNCED, not refused: seq 2048 with the tune
    # window left unset is the geometry that died at the ceiling, so the wrapper
    # says so and still starts the run (returncode 0, launch line printed).
    oom = _launch(RANK="512", SOLVER="svd", SEQ_LEN="2048")
    assert oom.returncode == 0, oom.stderr
    assert "SEQ_LEN=2048 with TUNE_SEQ_LEN unset" in oom.stderr
    assert "rank=512 solver=svd seq_len=2048" in oom.stderr


def test_the_measurement_knobs_reach_the_container_as_environment():
    """`FACTOR_MODE`/`ABLATION`/`LAYER_SCOPE` are env pass-throughs, not argv.

    The driver parser no longer accepts them: they are measurement surface a
    human never writes, so they travel as QK_* container environment. Set means
    the value reaches the container; unset means an EMPTY value, which the driver
    reads as unset — never a wrapper-chosen geometry.
    """
    set_run = _launch(RANK="512", SOLVER="svd", SEQ_LEN="2048", FACTOR_MODE="fp")
    assert set_run.returncode == 0, set_run.stderr
    assert "'-e' 'QK_FACTOR_MODE=fp'" in set_run.stdout
    assert "'--factor-mode'" not in set_run.stdout

    default = _launch(RANK="512", SOLVER="svd", SEQ_LEN="2048")
    assert default.returncode == 0, default.stderr
    # Empty is how "unset" is spelled, and it is spelled for all three: the
    # launcher never authors an ablation mode or a scope.
    for knob in ("QK_FACTOR_MODE", "QK_ABLATION", "QK_LAYER_SCOPE"):
        assert f"'-e' '{knob}='" in default.stdout, (knob, default.stdout)
    assert "'--ablation'" not in default.stdout
    assert "'--layer-scope'" not in default.stdout


def test_scn_seam_oracle_is_dry_run_rc(monkeypatch):
    """`seam_oracle_is_dry_run_rc`: the guard chain, with the double as the daemon.

    `openspec/specs/qk-wrapper/spec.md`, `guards_shall_be_tri_state`: held SHALL
    refuse, free SHALL proceed, unverifiable SHALL refuse. The refusal is
    correct and this test does not relax it. What was wrong is that the whole
    file's verdict depended on what a live fire happened to be doing on the
    host: every DRY_RUN `rc == 0` assertion here read as this refusal (8 reds
    at rc 2) with no code defect behind them.

    The scenario: the name-inspect answers with a held payload, so `rc == 2`
    means the interception is alive AND the guard refused through the full
    chain — the held payload is the double's answer, so the branch is covered
    by construction on any host, daemon or no daemon, fire or no fire.
    """
    monkeypatch.setenv("QKB_TEST_DOCKER_DIGEST", "sha256:" + "0" * 64)
    monkeypatch.setenv(
        "QKB_TEST_DOCKER_HOLDER",
        "quantibbler-ptq:rocm-official running=true launch_id=qk-pertarget-v4",
    )
    refused = _launch(RANK="512", SOLVER="svd", SEQ_LEN="2048")
    assert refused.returncode == 2, (refused.returncode, refused.stdout, refused.stderr)
    assert "is already held" in refused.stderr
    assert "qk-pertarget-v4" in refused.stderr, "the held payload reaches the refusal verbatim"
    assert "'docker'" not in refused.stdout, "a refused launch must not resolve an argv"


def test_free_name_proceeds_whatever_the_stub_is_asked(monkeypatch):
    """The FREE state is the double's default, so `rc == 0` cannot be ambient.

    Pinned beside the held leg so the pair is discriminating: same helper, same
    wrapper, same host — only the double's answer differs. Without this, a stub
    that always answered `held` would look like coverage.
    """
    monkeypatch.setenv("QKB_TEST_DOCKER_DIGEST", "sha256:" + "0" * 64)
    completed = _launch(RANK="512", SOLVER="svd", SEQ_LEN="2048")
    assert completed.returncode == 0, (completed.returncode, completed.stderr)
    assert "is already held" not in completed.stderr
    assert completed.stdout.strip().splitlines()[-1].startswith("'docker' 'run'")
    assert "image_digest=sha256:" in completed.stderr


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
    completed = _launch_past_dry_run(RANK="512", SOLVER="svd", SEQ_LEN="2048", OUT=str(out))

    witness = out / "container.log"
    assert witness.exists(), "the tee must have produced the witness for this leg to mean anything"
    assert witness.stat().st_size == 0, "fixture precondition: the witness is empty"

    assert completed.returncode == 2, (completed.returncode, completed.stdout, completed.stderr)
    assert "container_log_witness missing or empty" in completed.stderr
    assert str(witness) in completed.stderr


def test_fence_hook_install_resolves_the_hooks_dir_in_a_linked_worktree(tmp_path):
    """The fence installer must ASK git where hooks live, not assume `.git/`.

    The push fence re-installs `tools/git-hooks/pre-push` on every launch, and it
    used to write to the literal `$REPO_ROOT/.git/hooks`. In a git LINKED
    WORKTREE -- which is how every lane on this board is told to work -- `.git`
    is a FILE (a gitdir pointer), so `mkdir -p` on that literal path failed with
    `Not a directory`, `set -eu` took the whole launch down, and nine legs of
    THIS file read red in every worktree while a full clone -- where `.git` IS a
    directory -- stayed green. Every reviewer burned a differential cycle
    chasing reds that had no other defect behind them.

    `git rev-parse --git-path hooks` is the worktree-aware form: it resolves to
    the COMMON dir in a linked worktree, which is where git actually reads hooks
    from, so the fence lands where it is honoured -- once, for the clone and all
    its linked worktrees. This leg runs the shipped wrapper BYTES from a real
    linked worktree to pin it, inside a throwaway repo so the install never
    touches this checkout's own hooks directory. The hooks dir is deleted first,
    so the `mkdir -p` branch is the one that has to work.
    """
    source = _REPO / "tools" / "git-hooks" / "pre-push"
    repo = tmp_path / "repo"
    (repo / "tools" / "git-hooks").mkdir(parents=True)
    # The wrapper derives REPO_ROOT from `dirname $0`, so it has to physically
    # live in the worktree for this shape to be the real one: a copy of the
    # shipped bytes, not a stub.
    (repo / "tools" / "run-ptq-container.sh").write_bytes(_SCRIPT.read_bytes())
    (repo / "tools" / "git-hooks" / "pre-push").write_bytes(source.read_bytes())

    # The host's git config is kept out of the throwaway repo: a global
    # `core.hooksPath` or signing config would decide a leg that is about the
    # repo shape, not about this box.
    config_env = dict(
        os.environ,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_SYSTEM=os.devnull,
    )

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(repo), *args],
            env=config_env,
            capture_output=True,
            text=True,
            check=True,
        )

    git("init", "-q")
    git("add", "tools")
    git(
        "-c",
        "user.email=wrapper-test@example.invalid",
        "-c",
        "user.name=wrapper-test",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "seed",
    )
    worktree = tmp_path / "wt"
    git("worktree", "add", "--detach", str(worktree))
    assert (worktree / ".git").is_file(), (
        "fixture precondition: a linked worktree's .git is a FILE, which is the whole point"
    )

    hooks_dir = repo / ".git" / "hooks"
    for stale in hooks_dir.iterdir():
        stale.unlink()
    hooks_dir.rmdir()
    assert not hooks_dir.exists(), "fixture precondition: the installer has to create it"

    environment = {key: value for key, value in os.environ.items() if key not in _KNOBS}
    environment.update({**_OWNER_ENV, "DRY_RUN": "1", "OUT": str(tmp_path / "out")})
    _assert_interception(environment)
    completed = subprocess.run(
        ["sh", str(worktree / "tools" / "run-ptq-container.sh")],
        cwd=worktree,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (completed.returncode, completed.stdout, completed.stderr)
    installed = hooks_dir / "pre-push"
    assert installed.is_file(), (
        "the fence did not land in the COMMON hooks dir git resolves for the worktree: "
        f"stderr={completed.stderr!r}"
    )
    assert installed.read_bytes() == source.read_bytes(), (
        "the installed hook is not the tracked source byte-for-byte"
    )
    # The MODE, not os.access: `/tmp` is mounted noexec on this host (the same
    # fact the docker-double symlink dance above exists for), so X_OK reads
    # False for a perfectly 755 file. `chmod 755` is what the installer does and
    # what the mode pin checks.
    mode = installed.stat().st_mode & 0o777
    assert mode == 0o755, f"installed hook mode is {oct(mode)}, not 0o755"


# --- launch ownership: an unattributable fire is refused ----------------------
# A fire that names no owner leaves the launch path unable to say whose run it
# was. These legs pin the refusal and the attribution it lacked, through the
# same DRY_RUN seam every other gate in this file is proved through.

_TRAIL = pathlib.Path("/opt/data/tmp/quantibbler-breadcrumbs.log")


def test_scn_wrapper_refuses_unattributed_launch():
    """Spec scenario ``wrapper_refuses_unattributed_launch``.

    With no owner stamp the wrapper MUST refuse before the container name is
    taken: exit 2, an error naming the variables, and no argv resolved -- the
    refusal is a refusal, not a plan for a run it will start anyway.
    """
    refused = _launch(QKB_LAUNCH_OWNER="", QKB_LAUNCH_CARD="")
    assert refused.returncode == 2, (refused.returncode, refused.stdout, refused.stderr)
    assert "launch ownership missing" in refused.stderr
    assert "QKB_LAUNCH_OWNER" in refused.stderr, "the refusal names the variable to set"
    assert "QKB_LAUNCH_CARD" in refused.stderr
    assert "'docker'" not in refused.stdout, "a refused launch must not resolve an argv"


def test_scn_launch_ownership_label_spellings_carry_the_same_evidence():
    """Spec scenario ``launch_ownership_label_spellings_accepted``.

    The container labels (`qkb_launch_owner` / `qkb_launch_card`) are the
    contract's spelling, and docker never shows a container its own labels, so a
    launcher holding only that vocabulary has to hand it over as env. Those
    spellings MUST be read as the same evidence -- never as a second, weaker
    contract -- and the launch MUST proceed and stamp the contract labels.
    """
    completed = _launch(
        QKB_LAUNCH_OWNER="",
        QKB_LAUNCH_CARD="",
        qkb_launch_owner="ruling-owner",
        qkb_launch_card="t_ruling",
    )
    assert completed.returncode == 0, (completed.returncode, completed.stderr)
    assert "'--label' 'qkb_launch_owner=ruling-owner'" in completed.stdout
    assert "'--label' 'qkb_launch_card=t_ruling'" in completed.stdout


def test_scn_attributed_launch_stamps_owner_labels_and_owner_line(tmp_path):
    """Spec scenario ``attributed_launch_stamps_owner_labels``.

    An attributed launch MUST stamp `qkb_launch_owner` and `qkb_launch_card`
    beside `qkb_launch_id`, mirror both into the container (the driver's own
    gate reads env -- labels are invisible from inside), and name the owner in
    the announce and in the artifact's `launch.env`: the owner of a fire has to
    be readable from the artifact, not from somebody's scrollback.
    """
    out = tmp_path / "out"
    completed = _launch(OUT=str(out))
    assert completed.returncode == 0, (completed.returncode, completed.stderr)
    assert "'--label' 'qkb_launch_id=" in completed.stdout
    assert "'--label' 'qkb_launch_owner=wrapper-test'" in completed.stdout
    assert "'--label' 'qkb_launch_card=t_29f52eb6'" in completed.stdout
    assert "'-e' 'QKB_LAUNCH_OWNER=wrapper-test'" in completed.stdout
    assert "'-e' 'QKB_LAUNCH_CARD=t_29f52eb6'" in completed.stdout
    assert "owner=wrapper-test card=t_29f52eb6" in completed.stderr

    launch_env = (out / "launch.env").read_text()
    assert "owner=wrapper-test\n" in launch_env
    assert "card=t_29f52eb6\n" in launch_env


def test_scn_attributed_fire_writes_the_owner_into_the_trail_header():
    """Spec scenario ``attributed_launch_stamps_owner_labels`` (owner line).

    The breadcrumb tail is what a post-mortem reads; the fire's owner MUST be
    written there by the launcher at fire time, not remembered by the person
    who fired it. The header keeps its leading `=== run <ISO> rank=...` shape
    and gains `owner=... card=...` before the closing `===`.

    This leg walks the REAL launch path (`_launch_past_dry_run`) rather than
    reading the script's source, because a header nothing executes is exactly
    the "verified by construction" gap the wrapper spec names. The trail is the
    host's shared file, so the assertion is scoped to the tail window my own
    launch just wrote.
    """
    trailer = subprocess.run(
        ["tail", "-n", "40", str(_TRAIL)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    completed = _launch_past_dry_run(RANK="512", SOLVER="svd", SEQ_LEN="2048")

    # The witness gate fails this leg closed (the double prints nothing), which
    # is downstream of the header write -- the header is written first.
    assert completed.returncode == 2, (completed.returncode, completed.stderr)

    written = subprocess.run(
        ["tail", "-n", "40", str(_TRAIL)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    added = [line for line in written.splitlines() if line not in trailer.splitlines()]
    mine = [line for line in added if "owner=wrapper-test card=t_29f52eb6 ===" in line]
    assert mine, (
        f"the attributed fire wrote no owner line into the trail header (new lines: {added!r})"
    )
    assert mine[-1].startswith("=== run "), mine[-1]
