"""``tools/gen_attention_gate_record.py``: the captured-activation (npz) path.

Change ``qk-sage-attention``, task 4.3 (planner ruling t_fccd1ced, Decision 4).
The gate tool consumed only the committed JSON *fixture battery* until now
(i.i.d. Gaussian inputs, `activations_kind: "fixture-battery"`). A real-shape
case is ~82 KB of JSON per unit of seq, so a real battery is orders past the
repo's 500 KB large-file gate — hence the compact npz capture path, owned by
``tools/attention_activations.py`` and produced by
``tools/capture_attention_activations.py``.

What these tests pin, and why each one earns its place:

* a capture mints a record whose ``activations_kind`` names the evidence class
  AND its source (the planner's ratified distinction keys on that value);
* ``--check`` reproduces the record, and refuses one whose declared class the
  battery no longer produces (the drift guard, extended to the new field);
* an EDITED capture is refused by its checksum — the capture stores operands
  only, so unlike the JSON battery there is no second stored computation to
  cross-check, and the checksum is what makes its contents trustworthy;
* a foreign ``format_version`` and a truncated file are refused, not guessed at;
* the legacy JSON battery still regenerates (development parity is a design
  decision of the card, not a nice-to-have).

The tools are loaded by path (they are scripts, not package modules), the same
idiom ``tests/test_ci_referee.py`` uses.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
SHIPPED_FIXTURE_MODEL = REPO_ROOT / "tests" / "fixtures" / "server-model"


def _load(name: str, path: Path):  # noqa: ANN202
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

FORMAT = _load("attention_activations", TOOLS / "attention_activations.py")
TOOL = _load("gen_attention_gate_record", TOOLS / "gen_attention_gate_record.py")

CAPTURE_MODEL = "Qwen/Qwen3-0.6B"
MODEL_ID = "qwen3-0.6b-qpk1-gate-record-test"


def _synthetic_capture(path: Path, *, cases: int = 2, seq: int = 16, d: int = 32) -> Path:
    """A small capture in the committed format (GQA 2:1, causal), for speed."""
    rng = np.random.default_rng(4242)
    built = [
        FORMAT.CapturedCase(
            q=rng.standard_normal((1, 4, seq, d)).astype(np.float32),
            k=rng.standard_normal((1, 2, seq, d)).astype(np.float32),
            v=rng.standard_normal((1, 2, seq, d)).astype(np.float32),
            causal=True,
            block_size=8,
            layer=index,
            window=0,
        )
        for index in range(cases)
    ]
    FORMAT.write_capture(
        path,
        built,
        capture_model=CAPTURE_MODEL,
        metadata={"instrument": "tests/test_attention_gate_record.py"},
    )
    return path


def _model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "manifest.toml").write_text(f'model_id = "{MODEL_ID}"\n')
    return model_dir


def _record(model_dir: Path) -> dict:
    return json.loads((model_dir / "attention_gate.json").read_text())


def test_a_capture_mints_a_record_naming_its_evidence_class(tmp_path: Path):
    """The npz path is accepted and the record names the class AND the source."""
    model_dir = _model_dir(tmp_path)
    capture = _synthetic_capture(tmp_path / "real.npz", cases=3)

    record = TOOL.build_record(model_dir, capture, str(capture), None)

    assert record["schema"] == TOOL.SCHEMA
    assert record["model_id"] == MODEL_ID
    assert record["activations_kind"] == f"real-activation:{CAPTURE_MODEL}"
    assert record["activations_source"] == str(capture)
    assert len(record["calls"]) == 3
    assert record["clauses"] == {
        "cosine_min": TOOL.COSINE_MIN,
        "relative_l1_max": TOOL.RELATIVE_L1_MAX,
    }
    assert record["ppl"] is None
    # The aggregate is the calls' own aggregate, computed here from the record.
    assert record["worst"]["cosine"] == min(call["cosine"] for call in record["calls"])
    assert record["worst"]["relative_l1"] == max(call["relative_l1"] for call in record["calls"])


def test_the_capture_path_regenerates_and_the_check_detects_drift(tmp_path: Path):
    """`--check` reproduces the record, and refuses a record whose class moved."""
    model_dir = _model_dir(tmp_path)
    capture = _synthetic_capture(tmp_path / "real.npz")
    argv = ["--model-dir", str(model_dir), "--activations", str(capture)]

    assert TOOL.main(argv) == 0
    shipped = _record(model_dir)
    assert shipped["activations_kind"] == f"real-activation:{CAPTURE_MODEL}"
    assert TOOL.main([*argv, "--check"]) == 0

    # A shipped record that claims the development class for a real capture is
    # drift, not a detail the check may ignore.
    shipped["activations_kind"] = FORMAT.FIXTURE_BATTERY_KIND
    (model_dir / "attention_gate.json").write_text(json.dumps(shipped))
    assert TOOL.main([*argv, "--check"]) == 1


def test_a_tampered_capture_is_refused_by_its_checksum(tmp_path: Path):
    """A capture's operands cannot be edited under its stored checksum.

    The JSON battery cross-checks a second stored computation; a capture has no
    such second witness, so the checksum is what makes it trustworthy — and this
    is the test that shows it refusing an edit rather than measuring it.
    """
    capture = _synthetic_capture(tmp_path / "real.npz")
    with np.load(capture) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    arrays["case_0_q"] = arrays["case_0_q"] + np.float32(0.25)
    np.savez(tmp_path / "edited.npz", **arrays)

    with pytest.raises(SystemExit) as excinfo:
        TOOL.load_cases(tmp_path / "edited.npz")
    assert "checksum mismatch" in str(excinfo.value)


def test_a_foreign_format_version_is_refused(tmp_path: Path):
    """A version this reader does not speak is refused, never guessed at."""
    capture = _synthetic_capture(tmp_path / "real.npz")
    with np.load(capture) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    arrays["format_version"] = np.int64(FORMAT.FORMAT_VERSION + 1)
    arrays["checksum"] = np.array(FORMAT.checksum(arrays))
    np.savez(tmp_path / "future.npz", **arrays)

    with pytest.raises(SystemExit) as excinfo:
        TOOL.load_cases(tmp_path / "future.npz")
    assert f"this reader speaks {FORMAT.FORMAT_VERSION}" in str(excinfo.value)


def test_a_truncated_capture_is_refused(tmp_path: Path):
    """A damaged file names itself, rather than reporting a bogus battery."""
    capture = _synthetic_capture(tmp_path / "real.npz")
    truncated = tmp_path / "truncated.npz"
    truncated.write_bytes(capture.read_bytes()[:200])

    with pytest.raises(SystemExit) as excinfo:
        TOOL.load_cases(truncated)
    assert "not a readable npz capture" in str(excinfo.value)


def test_the_json_fixture_battery_still_regenerates():
    """Development parity: the committed JSON battery path is untouched."""
    assert SHIPPED_FIXTURE_MODEL.is_dir(), f"missing {SHIPPED_FIXTURE_MODEL}"
    cases, kind = TOOL.load_cases(
        REPO_ROOT / "crates/quantibbler-kernels/tests/fixtures/attention_sage.json"
    )
    assert kind == FORMAT.FIXTURE_BATTERY_KIND
    assert cases, "the committed battery must not be empty"
    assert all(case["stored"] for case in cases), "the JSON battery stores its own metrics"
    assert TOOL.main(["--model-dir", str(SHIPPED_FIXTURE_MODEL), "--check"]) == 0
