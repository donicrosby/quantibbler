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
    "REHEARSAL_BATCHES",
    "CHUNK_TOKENS",
    "WALL_MIN",
    "LANE0_IMAGE_DIGEST_ALLOW",
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


_LANE0 = {
    "MODE": "lane0-rehearsal",
    "GPU": "1",
    "REHEARSAL_BATCHES": "1",
    "CHUNK_TOKENS": "256",
    "WALL_MIN": "30",
}
_BROKEN_DIY = "quantibbler-ptq:rocm"
_BROKEN_DIY_DIGEST = "sha256:42649ada93e0bd4221075999ba85a40d9142d3e644281fdaaa20f7afafad0a6b"


def _resolves(image: str) -> bool:
    """Can docker tell us this image's digest on this machine?"""
    try:
        done = subprocess.run(
            ["docker", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    return done.returncode == 0 and done.stdout.strip() != ""


def test_lane0_pins_its_image_rather_than_inheriting_the_ambient_one():
    """Lane 0 must not launch on whatever `IMAGE` the shell happens to carry.

    The green-1 failure: an `IMAGE` exported from the retired-DIY arc flowed
    straight into the launch, and the rocm-tag guard waved it through because
    `quantibbler-ptq:rocm` satisfies `*rocm*` exactly as well as the real one.
    """
    completed = _launch(**_LANE0)
    assert completed.returncode == 0, completed.stderr
    assert "quantibbler-ptq:rocm-official" in completed.stdout


def test_lane0_refuses_the_unpinned_diy_digest():
    """The image that actually killed green-1 is refused, by digest not by tag."""
    if not _resolves(_BROKEN_DIY):
        import pytest

        pytest.skip(f"{_BROKEN_DIY} not present, cannot resolve a digest to refuse")
    completed = _launch(IMAGE=_BROKEN_DIY, **_LANE0)
    assert completed.returncode == 2, (completed.returncode, completed.stdout)
    assert "lane0 image identity is pinned" in completed.stderr


def test_lane0_digest_override_names_the_digest_it_authorizes():
    """The door exists, and it opens only for a digest someone wrote down."""
    if not _resolves(_BROKEN_DIY):
        import pytest

        pytest.skip(f"{_BROKEN_DIY} not present, cannot authorize its digest")
    completed = _launch(IMAGE=_BROKEN_DIY, LANE0_IMAGE_DIGEST_ALLOW=_BROKEN_DIY_DIGEST, **_LANE0)
    assert completed.returncode == 0, completed.stderr


def test_lane0_launch_env_records_harness_knobs_not_driver_geometry(tmp_path):
    """The record must describe the run that happened, not the other lane's."""
    completed = _launch(OUT=str(tmp_path), **_LANE0)
    assert completed.returncode == 0, completed.stderr
    lines = (tmp_path / "launch.env").read_text().splitlines()
    text = "\n".join(lines)
    assert "mode=lane0-rehearsal" in text
    assert "rehearsal_batches=1" in text and "chunk_tokens=256" in text
    for foreign in ("rank=", "seq_len=", "compute_dtype=", "eval_seqs=", "calibration_seqs="):
        assert foreign not in text, f"lane0 launch.env carries ptq geometry: {foreign}"
    # The splice defect that put a bare `?` line in the green-1 record.
    assert "?" not in lines, f"stray splice line in launch.env: {lines}"
    assert any(
        line.startswith("tree_dirty_files=") and line.split("=", 1)[1].isdigit() for line in lines
    ), lines


def test_lane0_disables_the_caching_allocator_and_ptq_does_not():
    """The rehearsal releases as it runs; the campaign keeps its normal allocator.

    With caching on, freed blocks stay in the pool and the host keeps seeing the
    high-water mark -- the difference between "the allocator is holding it" and
    "the process is using it". The ptq peak is a number the campaign is compared
    against, so its allocator configuration must not move.
    """
    lane0 = _launch(**_LANE0)
    assert lane0.returncode == 0, lane0.stderr
    assert "PYTORCH_NO_CUDA_MEMORY_CACHING=1" in lane0.stdout

    ptq = _launch(MODE="ptq")
    assert ptq.returncode == 0, ptq.stderr
    assert "PYTORCH_NO_CUDA_MEMORY_CACHING" not in ptq.stdout


def test_lane0_launch_env_records_the_release_configuration(tmp_path):
    completed = _launch(OUT=str(tmp_path), **_LANE0)
    assert completed.returncode == 0, completed.stderr
    text = (tmp_path / "launch.env").read_text()
    assert "no_cuda_memory_caching=1" in text
    assert "attn_impl=sdpa" in text
    assert "mem_csv=rehearsal-mem.csv" in text
