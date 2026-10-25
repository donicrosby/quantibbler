"""qk-launch-attribution: the crumb trail names its own launch.

Spec: ``openspec/changes/qk-launch-attribution/specs/qk-ptq/spec.md``, and the
change's ``design.md`` decision 1 (the field ORDER is pinned there — one line,
first in every trail: ``start pid=… argv=… rev=<git rev|unknown> out=<out dir>``).

Scenarios covered (IDs verbatim from the spec delta):

- ``launch_identity_line_inside_a_checkout``
- ``launch_identity_line_outside_a_checkout``
- ``default_path_disambiguates_concurrent_launches``
- ``configured_path_honored_exactly``

and the launch-ownership scenarios:

- ``driver_refuses_unattributed_launch``
- ``label_spelled_owner_accepted``
- ``owner_without_card_records_unset``

Why the driver is invoked as a doomed start rather than a full run: the
survivor attributes its run even when the launch protocol is violated, so a
process that aborts where it lands still exercises the attribution.
Every process here starts the driver for real and lets it abort where it lands —
at model load, on a model path that does not exist — WITHOUT ever reaching
quantization. ``--device cpu`` is passed by name (the driver's ``auto`` refuses
on a box with no accelerator, which would abort before the trail under test is
read on some hosts and proceed on others; the explicit value is the same on
both).

RED-FIRST RECORD — observed before any implementation, against the working tree
at ``075e7ec78e4773fb0a68c31907398d5fd02d55b3`` (``tools/quantize_hf_model.py``
still carrying the pre-change launch line), with:

    uv run --extra torch --extra hf pytest tests/ptq/test_launch_attribution.py -q
    # (the command as it was run. The torch/hf extras were deleted by the
    # qk-dependency-floor change — torch is a product dependency now — so
    # reproduce it with `uv run pytest tests/ptq/test_launch_attribution.py -q`.)

``4 failed in 49.92s`` (RC=1); the same four re-failed on the post-``ruff format``
file (same command, ``4 failed in 63.48s``, RC=1 — wall time varies with the
torch import, the failure set does not). Every failure is the intended one — the
attribution is absent:

- inside a checkout: the first line reads
  ``… :: start pid=553345 argv=--device cpu --model …/absent-model-A --out …/out-A``
  → ``AssertionError: launch line carries no/incorrect rev
  ('075e7ec78e4773fb0a68c31907398d5fd02d55b3' expected)``
- outside a checkout: same line shape → ``no/incorrect rev ('unknown' expected)``
- concurrent launches: ``AssertionError: expected one per-launch trail per
  process, got [] in …/shared-kd (shared default still in use:
  ['quantibbler-breadcrumbs.log'])`` → ``assert 0 == 2``
- configured path: same missing ``rev`` (and ``out``) on the identity line

No other field was missing: pre-fix the line already carried ``pid`` and
``argv`` (the change's design.md "facts checked" note), so each failure isolates
exactly the attribution the requirement adds.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "tools" / "quantize_hf_model.py"

#: The launch-ownership contract: a fire names its owner or does not start. The
#: three owner spellings are the SAME evidence -- the container labels can only
#: reach the process as env (docker never shows a container its own labels).
_OWNER_VARS = ("QKB_LAUNCH_OWNER", "QKB_LAUNCH_LABEL_OWNER", "qkb_launch_owner")
_CARD_VARS = ("QKB_LAUNCH_CARD", "QKB_LAUNCH_LABEL_CARD", "qkb_launch_card")
OWNER = "launch-attribution-test"
CARD = "t_29f52eb6"

#: A model-load abort costs the torch/transformers import (~10 s measured); the
#: timeout is a hang guard, not a budget.
DRIVER_TIMEOUT = 300


def _git_head(repo: pathlib.Path) -> str:
    """The checkout's HEAD revision — the value the launch line must carry."""
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _env(kd_tmp: pathlib.Path, breadcrumbs: pathlib.Path | None) -> dict[str, str]:
    """A driver environment with the trail knobs explicit.

    ``QUANTIBBLER_BREADCRUMBS`` is POPPED (not merely left alone) when the
    scenario wants the defaulted path: an inherited value would silently turn
    the default-path scenario into the configured-path one.
    """
    env = dict(os.environ)
    env["QUANTIBBLER_KD_TMPDIR"] = str(kd_tmp)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    # Every run names its owner, exactly as a real fire must: the scenarios below
    # are about the launch line, and a refused launch writes none. The refusal
    # scenario removes all three accepted spellings deliberately -- an inherited
    # owner (or one leaked by an earlier leg) would silently turn it into the
    # attributed case and the leg would pass for the wrong reason.
    env["QKB_LAUNCH_OWNER"] = OWNER
    env["QKB_LAUNCH_CARD"] = CARD
    if breadcrumbs is None:
        env.pop("QUANTIBBLER_BREADCRUMBS", None)
    else:
        env["QUANTIBBLER_BREADCRUMBS"] = str(breadcrumbs)
    return env


def _start_cmd(driver: pathlib.Path, scratch: pathlib.Path, tag: str) -> list[str]:
    """Start the driver, distinguishably, on a model that cannot exist.

    The fake argv is the scenario's own: a nonexistent absolute model path is
    a legal argv that aborts at tokenizer load (``Repo id must be in the form…``)
    before any quantization work, so the trail's first line is the only thing
    the run gets to write — which is exactly the corpse case.
    """
    return [
        sys.executable,
        str(driver),
        "--device",
        "cpu",
        "--model",
        str(scratch / f"absent-model-{tag}"),
        "--out",
        str((scratch / f"out-{tag}").resolve()),
    ]


def _spawn(cmd: list[str], env: dict[str, str], cwd: pathlib.Path) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        env=env,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _launch_line(trail: pathlib.Path) -> str:
    """The trail's FIRST line — the launch line, per the requirement."""
    assert trail.exists(), f"no crumb trail at {trail}"
    lines = trail.read_text().splitlines()
    assert lines, f"crumb trail {trail} is empty"
    return lines[0]


def _assert_launch_identity(
    line: str,
    *,
    pid: int,
    out_dir: pathlib.Path,
    rev: str,
    argv_token: str,
    owner: str = OWNER,
    card: str = CARD,
) -> None:
    """The full identity contract for one launch line.

    The epoch crumb field leads the line (every crumb's first field is the
    watchdog's own timestamp) and the identity fields follow in the order
    ``design.md`` decision 1 pins, with the launch-ownership fields appended
    after ``out``: pid, argv, rev, out, owner, card.
    """
    assert line.split()[0].isdigit(), f"launch line does not lead with the epoch field: {line!r}"
    assert "start pid=" in line, line
    assert f"pid={pid}" in line, f"launch line does not name its own pid {pid}: {line!r}"
    assert " argv=" in line, f"launch line carries no argv: {line!r}"
    assert argv_token in line, f"launch line does not carry the fake argv {argv_token!r}: {line!r}"
    assert f"rev={rev}" in line, (
        f"launch line carries no/incorrect rev ({rev!r} expected): {line!r}"
    )
    assert f"out={out_dir}" in line, f"launch line carries no resolved out dir: {line!r}"
    assert f"owner={owner}" in line, (
        f"launch line does not name its owner ({owner!r} expected): {line!r}"
    )
    assert f"card={card}" in line, f"launch line does not name its card ({card!r}): {line!r}"
    fields = (" pid=", " argv=", " rev=", " out=", " owner=", " card=")
    order = [line.find(token) for token in fields]
    assert -1 not in order and order == sorted(order), (
        f"launch identity fields are not in the pinned order {fields}: {line!r}"
    )


# --- scenario: launch identity line inside a checkout ------------------------


def test_scn_launch_identity_line_inside_a_checkout(tmp_path):
    """Spec scenario ``launch_identity_line_inside_a_checkout``.

    Starting the driver from the checkout itself, the trail's first line MUST
    contain the pid, the fake argv, the repo's HEAD revision, and the resolved
    output directory. Pre-fix the line reads
    ``… :: start pid=<pid> argv=<argv>`` — no ``rev``, no ``out``.
    """
    scratch = tmp_path / "inside"
    scratch.mkdir()
    kd_tmp = scratch / "kd"
    kd_tmp.mkdir()
    trail = scratch / "breadcrumbs.log"
    out_dir = (scratch / "out-A").resolve()
    head = _git_head(REPO_ROOT)

    proc = _spawn(
        _start_cmd(DRIVER, scratch, "A"),
        _env(kd_tmp, trail),
        cwd=REPO_ROOT,
    )
    pid = proc.pid
    _, stderr = proc.communicate(timeout=DRIVER_TIMEOUT)
    assert proc.returncode != 0, (
        "the launch under test was meant to abort at model load; a zero exit "
        f"means the run went further than the trail reading\n{stderr[-2000:]}"
    )

    line = _launch_line(trail)
    _assert_launch_identity(
        line, pid=pid, out_dir=out_dir, rev=head, argv_token=str(scratch / "absent-model-A")
    )


# --- scenario: launch identity line outside a checkout -----------------------


def test_scn_launch_identity_line_outside_a_checkout(tmp_path):
    """Spec scenario ``launch_identity_line_outside_a_checkout``.

    With no ``.git`` reachable from the driver's own location, the revision
    field MUST be literally ``unknown`` — never a crash, never a fabricated
    value — while pid, argv and the resolved out dir are still recorded.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    copied = outside / "quantize_hf_model.py"
    shutil.copyfile(DRIVER, copied)

    precondition = subprocess.run(
        ["git", "-C", str(outside), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert precondition.returncode != 0, (
        f"precondition failed: {outside} is inside a git work tree, so the "
        "outside-a-checkout reading cannot be taken there"
    )

    kd_tmp = outside / "kd"
    kd_tmp.mkdir()
    trail = outside / "breadcrumbs.log"
    out_dir = (outside / "out-A").resolve()

    proc = _spawn(_start_cmd(copied, outside, "A"), _env(kd_tmp, trail), cwd=outside)
    pid = proc.pid
    _, stderr = proc.communicate(timeout=DRIVER_TIMEOUT)
    assert proc.returncode != 0, (
        f"the launch under test was meant to abort at model load\n{stderr[-2000:]}"
    )

    line = _launch_line(trail)
    _assert_launch_identity(
        line, pid=pid, out_dir=out_dir, rev="unknown", argv_token=str(outside / "absent-model-A")
    )


# --- scenario: default path disambiguates concurrent launches ----------------


def test_scn_default_path_disambiguates_concurrent_launches(tmp_path):
    """Spec scenario ``default_path_disambiguates_concurrent_launches``.

    With ``QUANTIBBLER_BREADCRUMBS`` unset and a SHARED KD scratch dir, two
    concurrent launches MUST write two files — ``breadcrumbs-<pid>.log`` — and
    no line of one may appear in the other (no interleave). Pre-fix both append
    into the one shared ``quantibbler-breadcrumbs.log``.
    """
    kd_tmp = tmp_path / "shared-kd"
    kd_tmp.mkdir()
    env = _env(kd_tmp, None)

    launched = [
        (tag, _spawn(_start_cmd(DRIVER, tmp_path, tag), env, cwd=REPO_ROOT)) for tag in ("A", "B")
    ]
    results = {}
    for tag, proc in launched:
        out, err = proc.communicate(timeout=DRIVER_TIMEOUT)
        results[tag] = (proc.pid, proc.returncode, err)

    trails = sorted(kd_tmp.glob("breadcrumbs-*.log"))
    assert len(trails) == 2, (
        "expected one per-launch trail per process, got "
        f"{[p.name for p in trails]} in {kd_tmp} "
        f"(shared default still in use: {[p.name for p in kd_tmp.iterdir()]})"
    )
    assert not (kd_tmp / "quantibbler-breadcrumbs.log").exists(), (
        "the shared single-file default was still written"
    )

    for tag, (pid, returncode, stderr) in results.items():
        assert returncode != 0, f"launch {tag} was meant to abort at model load\n{stderr[-2000:]}"
        own = kd_tmp / f"breadcrumbs-{pid}.log"
        assert own.exists(), f"launch {tag} (pid {pid}) wrote no {own.name}"
        body = own.read_text()
        line = _launch_line(own)
        _assert_launch_identity(
            line,
            pid=pid,
            out_dir=(tmp_path / f"out-{tag}").resolve(),
            # Inside the checkout: the real HEAD, read from the repo.
            rev=_git_head(REPO_ROOT),
            argv_token=str(tmp_path / f"absent-model-{tag}"),
        )
        other = "B" if tag == "A" else "A"
        assert f"absent-model-{other}" not in body, (
            f"launch {tag}'s trail carries launch {other}'s lines: the trails interleaved"
        )


# --- scenario: configured path honored exactly -------------------------------


def test_scn_configured_path_honored_exactly(tmp_path):
    """Spec scenario ``configured_path_honored_exactly``.

    With ``QUANTIBBLER_BREADCRUMBS`` set the driver appends ONLY to that path
    (no pid suffix), and its launch identity line still comes first — this is
    the single-run container contract the wrapper's run headers and the
    forensics read path depend on.
    """
    kd_tmp = tmp_path / "kd"
    kd_tmp.mkdir()
    configured = tmp_path / "configured" / "quantibbler-breadcrumbs.log"
    configured.parent.mkdir()
    out_dir = (tmp_path / "out-A").resolve()

    proc = _spawn(
        _start_cmd(DRIVER, tmp_path, "A"),
        _env(kd_tmp, configured),
        cwd=REPO_ROOT,
    )
    pid = proc.pid
    _, stderr = proc.communicate(timeout=DRIVER_TIMEOUT)
    assert proc.returncode != 0, (
        f"the launch under test was meant to abort at model load\n{stderr[-2000:]}"
    )

    line = _launch_line(configured)
    _assert_launch_identity(
        line,
        pid=pid,
        out_dir=out_dir,
        rev=_git_head(REPO_ROOT),
        argv_token=str(tmp_path / "absent-model-A"),
    )
    # The configured path is honored EXACTLY: no per-launch sibling anywhere.
    assert not list(tmp_path.glob("**/breadcrumbs-*.log")), (
        "a pid-suffixed trail was written beside an explicitly configured path"
    )


# --- launch ownership: no owner, no run ---------------------------------------
# A fire that names no owner leaves nothing in the launch path that could
# say whose run it was. A run that cannot name its owner is refused before it
# writes anything.
#
# RED-FIRST RECORD -- observed before this cut, against the pre-change driver
# bytes at `HEAD` (`git show HEAD:tools/quantize_hf_model.py`, copied outside the
# checkout):
#
#     env -u QKB_LAUNCH_OWNER -u QKB_LAUNCH_LABEL_OWNER -u qkb_launch_owner \
#       QUANTIBBLER_KD_TMPDIR=<kd> python <pre-change driver> --device cpu \
#       --model /nonexistent-model-xyz --out <out>
#
# started for real (exit 1 only because the model does not exist) and wrote the
# trail's first line with NO owner field:
#
#     1792955343 cgroup=3.69G :: start pid=837399 argv=--device cpu --model
#     /nonexistent-model-xyz --out ... rev=unknown out=/opt/data/tmp/prefix-red/out
#
# which is the defect: an unattributable run was allowed to launch, and its trail
# could not name its owner. Post-fix the same command exits 1 with
# `error: launch ownership missing` and writes no trail at all.


def test_scn_driver_refuses_unattributed_launch(tmp_path):
    """Spec scenario ``driver_refuses_unattributed_launch``.

    With none of the accepted owner spellings present the driver MUST refuse:
    nonzero exit, an error naming the variable to set, and no trail written --
    an unattributable run has nothing to record itself into.
    """
    scratch = tmp_path / "unattributed"
    scratch.mkdir()
    kd_tmp = scratch / "kd"
    kd_tmp.mkdir()
    trail = scratch / "breadcrumbs.log"

    env = _env(kd_tmp, trail)
    for var in (*_OWNER_VARS, *_CARD_VARS):
        env.pop(var, None)
    assert not any(var in env for var in _OWNER_VARS), "the refusal case still carries an owner"

    proc = _spawn(_start_cmd(DRIVER, scratch, "A"), env, cwd=REPO_ROOT)
    _, stderr = proc.communicate(timeout=DRIVER_TIMEOUT)

    assert proc.returncode != 0, f"an ownerless launch was allowed to start\n{stderr[-2000:]}"
    assert "launch ownership missing" in stderr, stderr[-2000:]
    assert "QKB_LAUNCH_OWNER" in stderr, "the refusal names the variable to set"
    assert not trail.exists(), (
        f"an unattributable run wrote a trail at {trail}: {trail.read_text()[:200]!r}"
    )


def test_scn_label_spelled_owner_accepted(tmp_path):
    """Spec scenario ``label_spelled_owner_accepted``.

    The container labels are the contract's spelling, and docker never shows a
    container its own labels, so a launcher holding only that vocabulary hands it
    over as env. Those spellings MUST be read as the same evidence: the run
    proceeds and the launch line records that owner and card.
    """
    scratch = tmp_path / "label-spelled"
    scratch.mkdir()
    kd_tmp = scratch / "kd"
    kd_tmp.mkdir()
    trail = scratch / "breadcrumbs.log"
    out_dir = (scratch / "out-A").resolve()

    env = _env(kd_tmp, trail)
    for var in (*_OWNER_VARS, *_CARD_VARS):
        env.pop(var, None)
    env["qkb_launch_owner"] = "ruling-owner"
    env["qkb_launch_card"] = "t_ruling"

    proc = _spawn(_start_cmd(DRIVER, scratch, "A"), env, cwd=REPO_ROOT)
    pid = proc.pid
    _, stderr = proc.communicate(timeout=DRIVER_TIMEOUT)
    assert proc.returncode != 0, (
        f"the launch under test was meant to abort at model load\n{stderr[-2000:]}"
    )

    _assert_launch_identity(
        _launch_line(trail),
        pid=pid,
        out_dir=out_dir,
        rev=_git_head(REPO_ROOT),
        argv_token=str(scratch / "absent-model-A"),
        owner="ruling-owner",
        card="t_ruling",
    )


def test_scn_owner_without_card_records_unset(tmp_path):
    """Spec scenario ``owner_without_card_records_unset``.

    The owner is required; the owning card id is a board fact a CI smoke leg or a
    local diagnostic legitimately has not got. The launch line MUST say so
    literally rather than omit the field -- a missing marker reads as "not
    measured", which is the shape this change exists to keep out of the trail.
    """
    scratch = tmp_path / "owner-only"
    scratch.mkdir()
    kd_tmp = scratch / "kd"
    kd_tmp.mkdir()
    trail = scratch / "breadcrumbs.log"
    out_dir = (scratch / "out-A").resolve()

    env = _env(kd_tmp, trail)
    for var in (*_OWNER_VARS, *_CARD_VARS):
        env.pop(var, None)
    env["QKB_LAUNCH_OWNER"] = "owner-only"

    proc = _spawn(_start_cmd(DRIVER, scratch, "A"), env, cwd=REPO_ROOT)
    pid = proc.pid
    _, stderr = proc.communicate(timeout=DRIVER_TIMEOUT)
    assert proc.returncode != 0, (
        f"the launch under test was meant to abort at model load\n{stderr[-2000:]}"
    )

    _assert_launch_identity(
        _launch_line(trail),
        pid=pid,
        out_dir=out_dir,
        rev=_git_head(REPO_ROOT),
        argv_token=str(scratch / "absent-model-A"),
        owner="owner-only",
        card="unset",
    )
