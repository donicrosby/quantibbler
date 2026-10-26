"""qpk1 single-file blob: cross-face parity (change qk-artifact-blob).

The blob READER exercised here is the qpk1 crate's PyO3 face (design B10):
there is exactly ONE blob parser in the repo, so a Python test that read the
container with a second implementation would be proving nothing about the
bytes the server reads. The WRITER is ``quantibbler.blob`` (design B11).

Scenario IDs in the test names come from
``openspec/changes/qk-artifact-blob/specs/*/spec.md``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import quantibbler_qpk1

from quantibbler import blob as blob_writer

REPO_ROOT = Path(__file__).resolve().parents[1]
BLOB_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "server-model.qpk"
DIR_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "server-model"
LAYER_FILE = {
    "layers.0.inner.self_attn.q_proj": "layers/q_proj.qpk1",
    "layers.0.inner.self_attn.k_proj": "layers/k_proj.qpk1",
    "layers.0.inner.self_attn.v_proj": "layers/v_proj.qpk1",
    "layers.0.inner.self_attn.o_proj": "layers/o_proj.qpk1",
    "layers.0.inner.mlp.gate_proj": "layers/gate_proj.qpk1",
    "layers.0.inner.mlp.up_proj": "layers/up_proj.qpk1",
    "layers.0.inner.mlp.down_proj": "layers/down_proj.qpk1",
}


def _sections(parsed: dict) -> dict[str, dict]:
    return {section["name"]: section for section in parsed["sections"]}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def test_loads_blob_file_bl1_1_bl2_1():
    """BL-1.1 + BL-2.1: the committed blob passes every loader gate."""
    model = quantibbler_qpk1.load_blob(str(BLOB_FIXTURE))
    assert model["model_id"] == "server-model-fixture"
    assert model["layers"] == list(LAYER_FILE)
    assert model["vocab_size"] == 32
    assert model["d_model"] == 4
    assert model["remainder_dtype"] == "fp32"
    assert model["has_tokenizer"] is True
    assert model["has_lm_head"] is False, "the fixture head is weight-tied"


def test_blob_entry_point_is_file_only_bl1_2():
    """BL-1.2's dual-read lives in Rust (`ModelDirectory::load`); the PyO3
    entry point reads a BLOB, so a directory path is refused rather than
    silently reinterpreted."""
    with pytest.raises((OSError, ValueError)):
        quantibbler_qpk1.load_blob(str(DIR_FIXTURE))


def test_layer_streams_are_byte_identical_to_the_directory_bw3_2():
    """BW-3.2: every layer stream section equals its `.qpk1` file's bytes."""
    parsed = quantibbler_qpk1.read_blob(str(BLOB_FIXTURE))
    sections = _sections(parsed)
    for name, rel in LAYER_FILE.items():
        assert name in sections, name
        assert _sha(sections[name]["data"]) == _sha((DIR_FIXTURE / rel).read_bytes()), name
        assert sections[name]["tag"] in (0, 1), "a layer section carries a qpk1 stream"


def test_tokenizer_section_is_the_raw_file_b9():
    """B9: `tokenizer.json` is a RAW section carrying the file's exact bytes."""
    sections = _sections(quantibbler_qpk1.read_blob(str(BLOB_FIXTURE)))
    tokenizer = sections["tokenizer.json"]
    assert tokenizer["tag"] == blob_writer.TAG_RAW
    assert tokenizer["data"] == (DIR_FIXTURE / "tokenizer.json").read_bytes()


def test_header_roundtrips_through_the_rust_reader_bf2_1():
    """BF-2.1: every declared field comes back at its written value."""
    header = quantibbler_qpk1.read_blob(str(BLOB_FIXTURE))["header"]
    raw = BLOB_FIXTURE.read_bytes()
    header_len = int.from_bytes(raw[8:12], "little")
    written = json.loads(raw[12 : 12 + header_len])
    assert written["created_by"] == "quantibbler-gen-server-fixtures"
    for field in (
        "model_id",
        "vocab_size",
        "d_model",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "head_dim",
        "eos_token_id",
        "bos_token_id",
        "rms_norm_eps",
        "rope_theta",
        "norm_slots",
        "layers",
        "weight_tying",
        "remainder_dtype",
    ):
        assert header[field] == written[field], field


def test_pack_twice_is_byte_identical_bf6_1(tmp_path):
    """BF-6.1 / BW-2.1: the same inputs produce identical bytes."""
    first = tmp_path / "one.qpk"
    second = tmp_path / "two.qpk"
    sha_one = blob_writer.write_blob_from_directory(
        DIR_FIXTURE, first, created_by="quantibbler-test"
    )
    sha_two = blob_writer.write_blob_from_directory(
        DIR_FIXTURE, second, created_by="quantibbler-test"
    )
    assert sha_one == sha_two == _sha(first.read_bytes())
    assert first.read_bytes() == second.read_bytes()


def test_malformed_blob_refuses_in_the_reader_bf1_2(tmp_path):
    """BF-1.2 through the PyO3 face: a bad magic names the observed bytes."""
    broken = tmp_path / "broken.qpk"
    broken.write_bytes(b"BLOB" + BLOB_FIXTURE.read_bytes()[4:])
    with pytest.raises(ValueError, match="bad blob magic"):
        quantibbler_qpk1.read_blob(str(broken))


def test_repack_dir_to_blob_loads_bw3_1(tmp_path):
    """BW-3.1: `tools/repack_remainder.py --container-format blob` loads."""
    out = tmp_path / "repacked.qpk"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "repack_remainder.py"),
            "--in",
            str(DIR_FIXTURE),
            "--out",
            str(out),
            "--container-format",
            "blob",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    model = quantibbler_qpk1.load_blob(str(out))
    assert model["model_id"] == "server-model-fixture"
    assert model["remainder_dtype"] == "fp32", "a dir->blob leg carries the declared dtype"
    # And the layer sections are the source's bytes.
    sections = _sections(quantibbler_qpk1.read_blob(str(out)))
    assert _sha(sections["embedding"]["data"]) == _sha((DIR_FIXTURE / "e1.f32.bin").read_bytes())


def test_repack_blob_to_blob_conversion_bw4_1_bw4_2(tmp_path):
    """BW-4.1/4.2: fp32 -> bf16 loads, and non-remainder sections survive."""
    out = tmp_path / "bf16.qpk"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "repack_remainder.py"),
            "--in",
            str(BLOB_FIXTURE),
            "--out",
            str(out),
            "--remainder-dtype",
            "bf16",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    model = quantibbler_qpk1.load_blob(str(out))
    assert model["remainder_dtype"] == "bf16"
    after = _sections(quantibbler_qpk1.read_blob(str(out)))
    before = _sections(quantibbler_qpk1.read_blob(str(BLOB_FIXTURE)))
    for name, section in before.items():
        if section["tag"] in (
            blob_writer.TAG_QPK1_V1,
            blob_writer.TAG_QPK1_V2,
            blob_writer.TAG_RAW,
        ):
            assert _sha(after[name]["data"]) == _sha(section["data"]), name
    for name in ("embedding", "norms"):
        assert after[name]["tag"] == blob_writer.TAG_BF16_LE, name
        assert len(after[name]["data"]) * 2 == len(before[name]["data"]), name


def test_repack_refuses_a_second_conversion_to_the_same_dtype(tmp_path):
    """BW-4's complement: converting to the declared dtype is refused, not a no-op."""
    out = tmp_path / "same.qpk"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "repack_remainder.py"),
            "--in",
            str(BLOB_FIXTURE),
            "--out",
            str(out),
            "--remainder-dtype",
            "fp32",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2, proc.stdout
    assert "already declares" in proc.stderr
    assert not out.exists()
