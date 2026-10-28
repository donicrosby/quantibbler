"""qk-artifact-blob: the exporter's blob path (BW-1/BW-2).

The exporter's blob byte path is `quantibbler.blob` + the Rust loader; the
cross-face byte pins live in tests/test_qpk1_blob.py. What THIS file pins is
the driver surface: the `--container-format` knob exists, and a real driver run
emits a `.qpk` the Rust loader accepts and records its sha256 in the report
(BW-1.1/BW-1.3, design B6 — the hash stays OUTSIDE the container).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import quantibbler_qpk1
from test_driver_tune_seq_len import (  # noqa: F401  (pytest fixtures)
    _run_driver,
    offline_corpus,
    tiny_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "tools" / "quantize_hf_model.py"


def test_driver_exposes_the_container_format_knob_bw1_1():
    """BW-1: the export surface names both containers, blob first."""
    proc = subprocess.run(
        [sys.executable, str(DRIVER), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--container-format" in proc.stdout
    assert "blob" in proc.stdout and "dir" in proc.stdout


@pytest.mark.integration
def test_blob_export_loads_in_rust_bw1_1_bw1_3(
    offline_corpus,  # noqa: ANN001, F811
    tiny_model,  # noqa: ANN001, F811
    tmp_path,  # noqa: ANN001
):
    """BW-1.1/BW-1.3: a real run emits a loadable blob + records its sha256."""
    out_dir = tmp_path / "blob-export"
    proc = _run_driver(offline_corpus, tiny_model, out_dir, extra=[])
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    blob_path = out_dir.with_suffix(".qpk")
    assert blob_path.is_file(), "the default container is a single-file blob"
    report = json.loads((out_dir / "quantize_report.json").read_text())
    assert report["container"]["format"] == "blob"
    assert report["container"]["path"] == str(blob_path)
    assert report["container"]["sha256"] == hashlib.sha256(blob_path.read_bytes()).hexdigest()
    # The full BL-1.1 gate is exercised on the COMMITTED fixture blob
    # (tests/test_qpk1_blob.py::test_loads_blob_file_bl1_1_bl2_1). It cannot be
    # exercised here: this tiny model's own geometry violates the loader's N-2
    # GQA rule (`n_kv_heads * head_dim = 32 != d_model 64`), so the LOADER
    # refuses any artifact exported from it — a pre-existing property of the
    # fixture, not of the container. What this run proves is the container
    # surface: header round-trip and the report's external hash.
    parsed = quantibbler_qpk1.read_blob(str(blob_path))
    assert parsed["header"]["model_id"] == report["model"].replace("/", "-")
    assert parsed["header"]["layers"] == report_layers(report)
    assert any(s["name"] == "tokenizer.json" for s in parsed["sections"])


def report_layers(report: dict) -> list[str]:
    """The layer names the exported directory declared (manifest order)."""
    import tomllib

    manifest = report["container"]["path"].replace(".qpk", "") + "/manifest.toml"
    with open(manifest, "rb") as fh:
        return tomllib.load(fh)["layers"]


@pytest.mark.integration
def test_directory_export_still_accepted_bw1_2(
    offline_corpus,  # noqa: ANN001, F811
    tiny_model,  # noqa: ANN001, F811
    tmp_path,  # noqa: ANN001
):
    """BW-1.2: `--container-format dir` keeps the pre-change directory export."""
    out_dir = tmp_path / "dir-export"
    proc = _run_driver(
        offline_corpus,
        tiny_model,
        out_dir,
        extra=["--container-format", "dir"],
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert (out_dir / "manifest.toml").is_file()
    assert (out_dir / "config.toml").is_file()
    assert list(out_dir.glob("layers/**/*.qpk1")), "the layer streams are still written"
    assert not out_dir.with_suffix(".qpk").exists(), "dir mode emits no container"
    report = json.loads((out_dir / "quantize_report.json").read_text())
    assert report["container"] == {"format": "dir", "path": None, "sha256": None}
