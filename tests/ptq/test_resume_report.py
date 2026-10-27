"""The restart declaration through the REAL driver (torch lane).

Scenarios (spec delta ``specs/qk-ptq``): ``resume_record_on_resumed_run``,
``resume_record_fresh_run_consistent_shape``, ``resume_true_only_when_admitted``,
``resume_record_on_failure_paths`` and ``resume_identity_is_evidence_not_an_assertion``.

The lean tier (``tests/test_qk_report_resume.py``) pins the builder's shape and
the fact that nothing reads it. This tier pins the WIRING against the shipped
driver on a real — if tiny — model:

* a fresh launch carries the record with nothing restored, and a launch that
  finds a prior launch's block archives declares the restart, the directory, the
  launch environment's image identity, and the admission the restore was
  checked against;
* the admission members are the restored ``fingerprint.json``'s own, read off
  disk rather than re-derived;
* a refused restore leaves NO report behind, so ``resumed: true`` cannot be
  claimed by a launch whose ``check_fingerprint`` refused — shown red-first
  against a mutant driver that neuters the guard and then does land the claim.

The one thing this module does not do is force a probe abort or a vacuous
teacher on the fixture: both need a model whose predictions collapse, which is a
separate and expensive fixture. The failure shapes' key set is pinned where it
belongs instead — the abort payload's keys are DERIVED from the driver source by
the lean driver pin, and the two failure builders are driven directly with the
driver's own identity payload here.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_driver_tune_seq_len as _driver_module
from test_driver_tune_seq_len import (  # noqa: F401  (pytest fixtures)
    DRIVER,
    _report,
    _run_driver,
    offline_corpus,
    tiny_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import quantize_hf_model as qhf  # noqa: E402

from quantibbler.ptq.checkpoints import (  # noqa: E402
    _FINGERPRINT_PROVENANCE,
    config_fingerprint,
)
from quantibbler.ptq.config import PTQConfig  # noqa: E402
from quantibbler.qk_report import (  # noqa: E402
    abort_report,
    provenance,
    resume_record,
    stage_failure_report,
)

_IMAGE = "quantibbler-ptq:test-resume"
_DIGEST = "sha256:" + "abcdef0123456789" * 4
_RECORD_KEYS = {
    "resumed",
    "restored_blocks",
    "checkpoint_dir",
    "image",
    "image_digest",
    "admitted_by_fingerprint",
}
_ADMISSION_KEYS = {"fingerprint", "model_identity", "solver_device", *_FINGERPRINT_PROVENANCE}


def _fingerprint_path(out_dir: Path) -> Path:
    return out_dir / "checkpoints" / "fingerprint.json"


@pytest.fixture(scope="module")
def launch_pair(offline_corpus, tiny_model, tmp_path_factory):  # noqa: ANN001, ANN201, F811
    """One fresh launch, then one that resumes from what the first left.

    Module-scoped so the two scenarios share the two real driver runs rather
    than each paying for its own pair.
    """
    out_dir = tmp_path_factory.mktemp("resume-pair") / "artifact"
    env = {
        "QUANTIBBLER_IMAGE": _IMAGE,
        "QUANTIBBLER_IMAGE_DIGEST": _DIGEST,
    }
    saved = {name: os.environ.get(name) for name in env}
    os.environ.update(env)
    try:
        first = _run_driver(offline_corpus, tiny_model, out_dir)
        if first.returncode != 0:
            raise AssertionError(first.stdout[-4000:] + first.stderr[-2000:])
        fresh = _report(out_dir)
        second = _run_driver(offline_corpus, tiny_model, out_dir)
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    assert second.returncode == 0, second.stdout[-4000:] + second.stderr[-2000:]
    return SimpleNamespace(
        out_dir=out_dir,
        fresh=fresh,
        resumed=_report(out_dir),
        fingerprint=json.loads(_fingerprint_path(out_dir).read_text()),
    )


@pytest.mark.integration
def test_scn_resume_record_fresh_run_consistent_shape(launch_pair):  # noqa: ANN001
    """``resume_record_fresh_run_consistent_shape``: present, empty, null directory.

    The key exists on a launch that resumed nothing — the three report shapes
    stay single-shaped — and ``checkpoint_dir`` is null rather than the
    directory this launch wrote, because the field names where blocks were
    RESTORED FROM.
    """
    record = launch_pair.fresh["resume"]
    assert set(record) == _RECORD_KEYS
    assert record["resumed"] is False
    assert record["restored_blocks"] == []
    assert record["checkpoint_dir"] is None
    assert record["admitted_by_fingerprint"] is None
    # The launch environment is read, not invented: the wrapper's two variables
    # travel into the record even on a run that restored nothing.
    assert record["image"] == _IMAGE
    assert record["image_digest"] == _DIGEST


@pytest.mark.integration
def test_scn_resume_record_on_resumed_run(launch_pair):  # noqa: ANN001
    """``resume_record_on_resumed_run``: the restart is declared, from launch state.

    The second launch found the first launch's block archives and restored them.
    The record says so, names the blocks the ``completed_blocks`` scan saw at
    launch, names the directory, and carries the launch environment's image
    identity — the surface that makes a resumed fire's ``wall_seconds``
    readable as a segment rather than a whole solve.
    """
    record = launch_pair.resumed["resume"]
    assert set(record) == _RECORD_KEYS
    assert record["resumed"] is True
    # Every block of the tiny fixture, and nothing invented: the scan at launch.
    # Block indices are 1-based (`block-1.npz` is the first).
    assert record["restored_blocks"] == [1, 2, 3]
    assert record["checkpoint_dir"] == str(launch_pair.out_dir / "checkpoints")
    assert record["image"] == _IMAGE
    assert record["image_digest"] == _DIGEST


@pytest.mark.integration
def test_scn_resume_identity_is_evidence_not_an_assertion_wired(launch_pair):  # noqa: ANN001
    """``resume_identity_is_evidence_not_an_assertion``: the record's own admission.

    The members are the restored ``fingerprint.json``'s, copied verbatim: the two
    digests a reader compares against the launch trail, the recorded provenance
    fields, and the resolved device. No ``identical_bytes`` claim is emitted —
    source-byte identity across the reboot is a host-side git diff.
    """
    admission = launch_pair.resumed["resume"]["admitted_by_fingerprint"]
    fp = launch_pair.fingerprint
    assert set(admission) == _ADMISSION_KEYS
    assert admission["fingerprint"] == fp["fingerprint"]
    assert admission["model_identity"] == fp["model_identity"]
    # The provenance members come off the checkpoint's own record, not off a
    # second list written here.
    for name in (*_FINGERPRINT_PROVENANCE, "solver_device"):
        assert name in fp, f"{name} is not in the fingerprint record at all"
        assert admission[name] == fp[name], name
    assert "identical_bytes" not in json.dumps(launch_pair.resumed["resume"])


def test_scn_resume_marker_never_folds_the_fingerprint():
    """``resume_marker_never_folds_the_fingerprint``: a resumed run admits itself.

    The config digest is what refuses a resume, and it hashes the whole config.
    A resume marker inside ``PTQConfig`` — or inside the recorded provenance
    tuple — would move it, and a resumed launch would then hash differently from
    the blocks it is resuming and refuse its own checkpoint. Red-first: a config
    that DOES carry such a field hashes differently, so the green below is the
    absence of the marker and not a digest that ignores its inputs.
    """
    names = {field.name for field in dataclasses.fields(PTQConfig)}
    assert "resume" not in names and "resumed" not in names, sorted(names)
    assert not any("resume" in name for name in _FINGERPRINT_PROVENANCE)

    cfg = PTQConfig(rank=4, solver="svd", admm_max_iter=2)
    before = config_fingerprint(cfg)
    # Emitting the record is a report-time act: nothing about it reaches the
    # config or the digest.
    resume_record(
        resumed=True,
        restored_blocks=[0, 1, 2],
        checkpoint_dir="/out/checkpoints",
        image=_IMAGE,
        image_digest=_DIGEST,
        admitted_by_fingerprint={"fingerprint": "f", "model_identity": {"digest": "m"}},
    )
    assert config_fingerprint(cfg) == before

    @dataclasses.dataclass(frozen=True)
    class _Marked(PTQConfig):
        resumed: bool = False

    marked = _Marked(**{field.name: getattr(cfg, field.name) for field in dataclasses.fields(cfg)})
    assert config_fingerprint(marked) != before, (
        "a config carrying a resume marker hashed identically — the digest no "
        "longer separates the resumed launch from its own pre-resume blocks"
    )
    # ... and the digest really is sensitive to solver state, so the equality
    # above is not a constant.
    assert config_fingerprint(dataclasses.replace(cfg, rank=cfg.rank + 1)) != before


def test_scn_resume_record_on_failure_paths():
    """``resume_record_on_failure_paths``: the driver's own identity carries it.

    Both failure builders copy the identity payload they are handed, and the
    driver hands them the payload ``_run_identity`` builds — so the record is
    present on the abort and stage-failure shapes too. Driven through the real
    identity builder here, because that is the half the lean tier cannot reach
    (it cannot import the torch-importing driver).
    """
    from test_artifact_dtype import _identity_args

    record = resume_record(
        resumed=True,
        restored_blocks=[0, 1, 2],
        checkpoint_dir="/out/checkpoints",
        image=_IMAGE,
        image_digest=_DIGEST,
        admitted_by_fingerprint={"fingerprint": "f", "model_identity": {"digest": "m"}},
    )
    identity = qhf._run_identity(
        _identity_args("fp32"),
        provenance("svd", "float16"),
        tune_seq_len=2048,
        calibration_rows=128,
        eval_rows=1,
        ppl_baseline_bf16=1.0,
        evaluation={},
        resume=record,
    )
    assert identity["resume"] is record
    aborted = abort_report(
        SimpleNamespace(
            block="layers.1",
            depth=2,
            prefix_ppl=2.9e6,
            fp_prefix_ppl=83.0,
            ratio=34983.7,
            line=40.0,
        ),
        {
            **identity,
            "probe": {"interval": 1, "batches": 1, "fp_prefix_ppl": 83.0},
            "stages": [],
            "stage_map": {},
            "wall_seconds": 69.6,
        },
        [],
    )
    failed = stage_failure_report(
        SimpleNamespace(
            stage="tune_fp",
            block="layers.3",
            reason="zero_first_step_gradient",
            gradient_norm=0.0,
            initial_residual_norm=0.25,
        ),
        {**identity, "probe_abort": None},
        [],
    )
    for shape in (aborted, failed):
        assert shape["resume"] is record
        assert set(shape["resume"]) == _RECORD_KEYS


@pytest.mark.integration
def test_scn_resume_true_only_when_admitted(
    offline_corpus,  # noqa: ANN001, F811
    tiny_model,  # noqa: ANN001, F811
    tmp_path,  # noqa: ANN001
    monkeypatch,  # noqa: ANN001
):
    """``resume_true_only_when_admitted``: a refused restore leaves no claim.

    Poison the restored fingerprint record's config digest and relaunch: the
    guard's ``RuntimeError`` must surface unchecked — neither relabelled a
    ``PrefixProbeAbort`` nor a stage failure — and the report the refused launch
    would have overwritten must still be the FIRST launch's bytes. Red-first:
    the same poisoned directory driven by a mutant that neuters the guard DOES
    land ``resumed: true``, so the absence above is the guard's refusal and not
    an artifact of the fixture.
    """
    out_dir = tmp_path / "refused-artifact"
    first = _run_driver(offline_corpus, tiny_model, out_dir)
    assert first.returncode == 0, first.stdout[-4000:] + first.stderr[-2000:]
    report_path = out_dir / "quantize_report.json"
    honest_bytes = report_path.read_bytes()
    assert _report(out_dir)["resume"]["resumed"] is False

    fp_path = _fingerprint_path(out_dir)
    record = json.loads(fp_path.read_text())
    record["fingerprint"] = "0" * 64
    fp_path.write_text(json.dumps(record))

    refused = _run_driver(offline_corpus, tiny_model, out_dir)
    assert refused.returncode != 0, refused.stdout[-4000:]
    combined = refused.stdout + refused.stderr
    assert "checkpoint fingerprint mismatch" in combined, combined[-3000:]
    assert "RuntimeError" in combined, combined[-3000:]
    # Unchecked: it is not the probe's abort and not the fail-closed stage gate.
    assert "PROBE ABORT" not in refused.stdout
    assert "TUNEFP VACUOUS TEACHER" not in refused.stdout
    # Nothing was written on the refused launch's behalf.
    assert report_path.read_bytes() == honest_bytes

    # --- the mutant: same directory, guard neutered, claim lands -----------
    mutant = tmp_path / "mutant_quantize_hf_model.py"
    source = DRIVER.read_text()
    tail = 'if __name__ == "__main__":\n    raise SystemExit(main())\n'
    assert source.endswith(tail), "the driver's entry point moved; re-anchor the mutant"
    mutant.write_text(
        source[: -len(tail)] + 'if __name__ == "__main__":\n'
        "    from quantibbler.ptq import checkpoints as _mutant_ckpt\n"
        "    _mutant_ckpt.check_fingerprint = lambda *a, **k: None\n"
        "    raise SystemExit(main())\n"
    )
    monkeypatch.setattr(_driver_module, "DRIVER", mutant)
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT / "src"))
    claimed = _run_driver(offline_corpus, tiny_model, out_dir)
    assert claimed.returncode == 0, claimed.stdout[-4000:] + claimed.stderr[-2000:]
    launched = json.loads(report_path.read_text())
    assert launched["resume"]["resumed"] is True, (
        "the mutant did not land the false claim the refusal leg excludes — the "
        "leg is not discriminating"
    )
    assert launched["resume"]["restored_blocks"]
    assert report_path.read_bytes() != honest_bytes
