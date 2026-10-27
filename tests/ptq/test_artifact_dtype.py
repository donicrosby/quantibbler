"""qk-artifact-dtype: the declared remainder dtype, host side.

The Rust half of this contract is pinned in
``crates/quantibbler-qpk1/src/loader.rs`` (the RD-1/RD-2 battery, which loads
the committed bf16 twin) and ``crates/quantibbler-server/tests/remainder_dtype.rs``
(RD-3.1, the decode tolerance). This module pins the host half — the codec, the
export option, the report fields, and the repack utility — and proves the
committed bf16 fixture IS the utility's output on the committed fp32 one, which
is what makes those Rust pins load the bytes this code writes.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import quantize_hf_model as qhf  # noqa: E402
import repack_remainder  # noqa: E402

from quantibbler.qk_report import provenance, resume_record  # noqa: E402
from quantibbler.qpk1 import serialize  # noqa: E402
from quantibbler.remainder_dtype import (  # noqa: E402
    REMAINDER_DTYPES,
    RemainderDtypeError,
    cast_to_bf16,
    decode,
    encode,
    itemsize,
)

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "server-model"
BF16_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "server-model-bf16"
SIDECARS = ("embedding", "lm_head", "norms")


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _fp32(path: Path) -> np.ndarray:
    return decode(path.read_bytes(), "fp32")


# --- the codec (RDP-2.1) ----------------------------------------------------


def test_scn_cast_elementwise_half_ulp_rdp2_1():
    # Every element of a deterministic fp32 matrix converted to bf16 satisfies
    # the half-ulp bound, with the subnormal floor term: 2^-8|x| + 2^-134.
    rng = np.random.default_rng(20261025)
    matrix = np.concatenate(
        [
            rng.standard_normal(20_000),
            rng.standard_normal(5_000) * 1e-30,  # bf16's subnormal range
            np.array([0.0, 1.0, -1.0, 1.0 + 2.0**-8, 1.0 + 2.0**-9, 2.0**-133], dtype=np.float64),
        ]
    )
    widened = decode(encode(matrix, "bf16"), "bf16")
    error = np.abs(widened - matrix)
    bound = (2.0**-8) * np.abs(matrix) + 2.0**-134
    assert np.all(error <= bound), f"max error {error.max()} exceeds the half-ulp bound"
    # And the conversions are genuinely lossy (a pure widening would prove nothing).
    assert np.any(error > 0.0)


def test_bf16_codec_refuses_nonfinite_and_unknown_dtypes():
    with pytest.raises(RemainderDtypeError, match="non-finite"):
        encode(np.array([np.inf]), "bf16")
    with pytest.raises(RemainderDtypeError, match="finite range"):
        encode(np.array([3.4e38]), "bf16")  # finite f32 that overflows bf16 to inf
    with pytest.raises(RemainderDtypeError, match="unknown remainder dtype"):
        itemsize("fp8")
    assert REMAINDER_DTYPES == ("fp32", "bf16")
    # torch's round-to-nearest-even is the shared cast's only rounding rule.
    assert int(cast_to_bf16(np.array([1.0 + 2.0**-9])).view(_torch_uint16())[0].item()) == 0x3F80


def _torch_uint16():
    import torch

    return torch.uint16


# --- the export (RDW-1) -----------------------------------------------------

_GEOMETRY = {
    "n_layers": 1,
    "n_heads": 2,
    "n_kv_heads": 2,
    "head_dim": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10_000.0,
}
_INVENTORY = ["input", "post_attention"]  # 4 + 4 floats/block, +4 final = 12
_EMBED = np.arange(32, dtype=np.float64).reshape(8, 4) / 7.0
_NORMS = np.linspace(-1.0, 1.0, 12)


class _FakeQuantizedModel:
    """Only the surface ``export_model_directory`` consumes."""

    def __init__(self, blobs: dict[str, bytes]) -> None:
        self._blobs = blobs

    def serialize(self) -> dict[str, bytes]:
        return self._blobs


class _FakeTokenizer:
    """Only the surface ``export_model_directory`` consumes.

    A real ``save_pretrained`` writes the model's chat template to the canonical
    sidecar beside the tokenizer document, and the export verifies that sidecar
    before it emits anything. The fake does the same: these legs are about the
    remainder encoding, and a tokenizer that emits no template would stop the
    export for a reason unrelated to what they pin.
    """

    chat_template = "{{ messages[0].content }}"

    def save_pretrained(self, out_dir) -> None:
        out = Path(out_dir)
        (out / "tokenizer.json").write_text("{}", encoding="utf-8")
        (out / qhf.CHAT_TEMPLATE_FILE).write_text(self.chat_template, encoding="utf-8")


def _export(tmp_path: Path, remainder_dtype: str, **overrides) -> Path:
    blobs = {
        "layers.0.inner.self_attn.q_proj": serialize(
            np.ones((4, 4), dtype=np.int8),
            np.ones((4, 4), dtype=np.int8),
            np.ones(4),
            np.ones(4),
            dtype=np.float16,
        )
    }
    out_dir = tmp_path / f"export-{remainder_dtype}"
    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "embedding_f64": _EMBED,
        "model_id": "export-remainder-fixture",
        "eos_token_id": 3,
        "bos_token_id": 1,
        "tokenizer": _FakeTokenizer(),
        "geometry": _GEOMETRY,
        "inventory": _INVENTORY,
        "norm_weights": _NORMS,
        "lm_head_f64": None,
        "tied": True,
        "remainder_dtype": remainder_dtype,
    }
    kwargs.update(overrides)
    qhf.export_model_directory(out_dir, _FakeQuantizedModel(blobs), **kwargs)
    return out_dir


def test_scn_fp32_default_export_byte_identical_rdw1_1(tmp_path):
    out = _export(tmp_path, "fp32")
    # Same file names and same bytes as the pre-change writer, which was
    # `np.ascontiguousarray(x, dtype=np.float32).tobytes(order="C")`.
    assert (out / "e1.f32.bin").read_bytes() == np.ascontiguousarray(
        _EMBED, dtype=np.float32
    ).tobytes(order="C")
    assert (out / "norms.bin").read_bytes() == np.ascontiguousarray(
        _NORMS, dtype=np.float32
    ).tobytes(order="C")
    manifest = (out / "manifest.toml").read_text(encoding="utf-8")
    assert 'dtype = "fp32"' in manifest, "the writer always declares its dtype"
    assert 'embedding = "e1.f32.bin"' in manifest
    assert 'norms = "norms.bin"' in manifest
    # The declaration and weight_tying's own suffix agree.
    assert 'weight_tying = "lm_head=e1@fp32"' in (out / "config.toml").read_text(encoding="utf-8")


def test_scn_bf16_export_loads_in_rust_rdw1_2(tmp_path):
    # The exported shape is exactly what qk-server RD-1.1/RD-2.1 accept, and the
    # writer's own bf16 bytes are the repack utility's bytes for the same
    # inputs — the two host faces cannot disagree.
    out = _export(tmp_path, "bf16")
    manifest = (out / "manifest.toml").read_text(encoding="utf-8")
    assert 'dtype = "bf16"' in manifest
    assert 'embedding = "e1.bf16.bin"' in manifest
    assert 'lm_head = "e1.bf16.bin"' in manifest
    assert 'norms = "norms.bf16.bin"' in manifest
    assert 'weight_tying = "lm_head=e1@bf16"' in (out / "config.toml").read_text(encoding="utf-8")
    assert (out / "e1.bf16.bin").stat().st_size == _EMBED.size * 2
    assert (out / "norms.bf16.bin").stat().st_size == _NORMS.size * 2
    assert decode((out / "norms.bf16.bin").read_bytes(), "bf16") == pytest.approx(
        decode(encode(_NORMS, "bf16"), "bf16")
    )

    # The repack of the SAME fp32 export produces the same remainder bytes.
    fp32_out = _export(tmp_path, "fp32")
    repacked = tmp_path / "repacked"
    repack_remainder.repack(fp32_out, repacked)
    for name in ("e1.bf16.bin", "norms.bf16.bin"):
        assert (out / name).read_bytes() == (repacked / name).read_bytes(), name


def test_scn_bf16_path_refuses_nonfinite_rdw1_3(tmp_path):
    bad = _NORMS.copy()
    bad[3] = np.nan
    with pytest.raises(RemainderDtypeError, match="non-finite"):
        _export(tmp_path, "bf16", norm_weights=bad)
    # The fp32 path is unchanged: NaN is a legitimate fp32 byte pattern.
    out = _export(tmp_path, "fp32", norm_weights=bad)
    assert np.isnan(_fp32(out / "norms.bin")[3])


def test_scn_export_selfcheck_itemsize_rdw1_4(tmp_path):
    fp32 = _export(tmp_path, "fp32")
    bf16 = _export(tmp_path, "bf16")
    count = _NORMS.size
    assert (fp32 / "norms.bin").stat().st_size == count * itemsize("fp32")
    assert (bf16 / "norms.bf16.bin").stat().st_size == count * itemsize("bf16")
    # A miscount fails at export, before any server ever sees the directory.
    with pytest.raises(AssertionError):
        _export(tmp_path / "bad", "bf16", norm_weights=_NORMS[:-1])


# --- the report (RDW-2) -----------------------------------------------------


def _identity_args(remainder_dtype: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model="m",
        rank=1,
        solver="svd",
        scale_dtype="float16",
        remainder_dtype=remainder_dtype,
        factor_mode="svd",
        ablation="none",
        layer_scope="all",
        seq_len=2048,
        max_admm_iter=1,
        seed=0,
        paper_faithful=False,
        ste_flip_pass_budget=0,
        # The emitted container (qk-artifact-blob BW-1): the identity record
        # carries the requested format on every exit path.
        container_format="blob",
    )


def _identity(remainder_dtype: str) -> dict:
    prov = provenance("svd", "float16", remainder_dtype=remainder_dtype)
    return qhf._run_identity(
        _identity_args(remainder_dtype),
        prov=prov,
        tune_seq_len=2048,
        calibration_rows=128,
        eval_rows=1,
        ppl_baseline_bf16=1.0,
        evaluation={},
        # The restart declaration rides the shared identity payload on every
        # path, so a caller that builds the identity states the launch's
        # restart state.
        resume=resume_record(
            resumed=False,
            restored_blocks=[],
            checkpoint_dir=None,
            image="unknown",
            image_digest="unknown",
            admitted_by_fingerprint=None,
        ),
    )


def test_scn_report_names_remainder_dtype_from_args_rdw2_1():
    identity = _identity("bf16")
    assert identity["remainder_dtype"] == "bf16"
    assert "bf16" in identity["regime"]["artifact"]
    prov = provenance("svd", "float16", remainder_dtype="bf16")
    assert prov["remainder_dtype"] == "bf16"
    assert "bf16" in prov["regime_artifact"]
    assert "bf16" in prov["dtype_note"]


def test_scn_report_fp32_default_not_hardcoded_rdw2_2():
    identity = _identity("fp32")
    assert identity["remainder_dtype"] == "fp32"
    # The bf16 literal never appears for an fp32 run: the field follows the
    # argument, it is not a constant.
    blob = repr(identity)
    assert "bf16" not in blob.replace("ppl_baseline_bf16", "").replace("baseline_bf16", "")
    prov = provenance("svd", "float16")
    assert prov["remainder_dtype"] == "fp32"
    assert "bf16" not in prov["regime_artifact"]
    assert "bf16" not in prov["dtype_note"]


# --- the repack utility (RDP-1) ---------------------------------------------


def test_scn_repack_roundtrip_loads_rdp1_1(tmp_path):
    # The output is this tool's own bytes on the committed fp32 fixture; the
    # Rust loader acceptance of exactly these bytes is pinned by
    # crates/quantibbler-qpk1/src/loader.rs::loads_bf16_fixture_exact_values_rd2_1.
    out = tmp_path / "bf16"
    record = repack_remainder.repack(FIXTURE, out)
    assert record["to"] == "bf16"
    manifest = (out / "manifest.toml").read_text(encoding="utf-8")
    assert 'dtype = "bf16"' in manifest
    assert "[repack]" in manifest
    assert 'source_model_id = "server-model-fixture"' in manifest
    assert 'conversion = "remainder fp32 -> bf16"' in manifest
    src_sha = hashlib.sha256()
    for key in SIDECARS:
        src_sha.update((FIXTURE / _source_sidecar(key)).read_bytes())
    assert f'source_remainder_sha256 = "{src_sha.hexdigest()}"' in manifest
    # The declared files are the bf16 re-encoding of the fp32 source.
    assert decode((out / "norms.bf16.bin").read_bytes(), "bf16") == pytest.approx(
        decode(encode(_fp32(FIXTURE / "norms.f32.bin"), "bf16"), "bf16")
    )
    assert (out / "norms.bf16.bin").stat().st_size * 2 == (FIXTURE / "norms.f32.bin").stat().st_size


def _source_sidecar(key: str) -> str:
    import tomllib

    with (FIXTURE / "manifest.toml").open("rb") as fh:
        return tomllib.load(fh)["sidecar"][key]


def test_committed_bf16_fixture_is_the_repack_utility_output(tmp_path):
    # The drift alarm: the committed twin is generated BY the tool, so this
    # equality is what keeps the committed bytes from being hand-edited evidence.
    out = tmp_path / "bf16"
    repack_remainder.repack(FIXTURE, out)
    assert _tree_digest(out) == _tree_digest(BF16_FIXTURE)


def test_scn_repack_refuses_already_bf16_rdp1_2(tmp_path):
    with pytest.raises(repack_remainder.RepackError, match="already declares"):
        repack_remainder.repack(BF16_FIXTURE, tmp_path / "again")


def test_scn_repack_refuses_nonfinite_rdp1_3(tmp_path):
    poisoned = tmp_path / "poisoned"
    poisoned.mkdir()
    for entry in FIXTURE.iterdir():
        if entry.is_dir():
            shutil.copytree(entry, poisoned / entry.name)
        else:
            shutil.copy2(entry, poisoned / entry.name)
    norms = bytearray((poisoned / "norms.f32.bin").read_bytes())
    norms[0:4] = np.float32(np.inf).tobytes()
    (poisoned / "norms.f32.bin").write_bytes(bytes(norms))

    out = tmp_path / "out"
    with pytest.raises(repack_remainder.RepackError, match="non-finite"):
        repack_remainder.repack(poisoned, out)
    assert not out.exists(), "a refusal must write no output directory"


def test_scn_repack_streams_byte_identical_rdp1_4(tmp_path):
    out = tmp_path / "bf16"
    repack_remainder.repack(FIXTURE, out)
    before = _tree_digest(FIXTURE)
    # Byte-identity is scoped to what the repack COPIES (layer streams,
    # tokenizer, ...). The two manifests are rewritten by the same requirement —
    # manifest.toml gains the declaration and [repack], and config.toml's
    # weight_tying suffix must follow the bytes or the loader refuses the
    # disagreement (RD-1.3), which is what makes RDP-1.1 loadable — so they are
    # excluded from the copy comparison and pinned exactly below instead.
    for rel, digest in _tree_digest(out).items():
        if rel in ("manifest.toml", "config.toml") or rel.endswith(".bin"):
            continue  # manifests rewritten; remainders re-encoded
        assert digest == before[rel], rel
    # The layer streams specifically are untouched.
    assert (out / "layers/q_proj.qpk1").read_bytes() == (
        FIXTURE / "layers/q_proj.qpk1"
    ).read_bytes()
    # config.toml differs in exactly the one place the conversion owns.
    src_cfg = (FIXTURE / "config.toml").read_text(encoding="utf-8")
    out_cfg = (out / "config.toml").read_text(encoding="utf-8")
    assert out_cfg == src_cfg.replace("lm_head=e1@fp32", "lm_head=e1@bf16")


def test_scn_repack_deterministic_rdp1_5(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    repack_remainder.repack(FIXTURE, first)
    repack_remainder.repack(FIXTURE, second)
    assert _tree_digest(first) == _tree_digest(second)
    # And the source was never written to.
    assert not (FIXTURE / "e1.bf16.bin").exists()


def test_repack_refuses_nested_out_without_touching_source(tmp_path):
    # `--out` inside `--in` is operator error: the copy would land in the source
    # tree and mutate it, against RDP-1's "never mutating the source". The guard
    # refuses before any write, so the source tree digest is byte-for-byte
    # unchanged and the output subtree is never created (H1 hardening).
    src = tmp_path / "model"
    shutil.copytree(FIXTURE, src)
    nested = src / "nested-bf16"
    before = _tree_digest(src)
    with pytest.raises(repack_remainder.RepackError, match="inside the source artifact"):
        repack_remainder.repack(src, nested)
    assert not nested.exists(), "a nested-out refusal must create no output subtree"
    assert _tree_digest(src) == before, "the source tree must be byte-for-byte unchanged"


def test_repack_refuses_malformed_weight_tying_before_write(tmp_path):
    # A suffix-less weight_tying token (`lm_head=e1`) is a source the loader
    # itself refuses (RD-1.4). It must be caught BEFORE the copy: the old path
    # copied the whole tree, rewrote the manifest, and only then refused,
    # leaving a complete but unloadable output dir behind (H2 hardening).
    src = tmp_path / "model"
    shutil.copytree(FIXTURE, src)
    cfg = src / "config.toml"
    cfg.write_text(
        cfg.read_text(encoding="utf-8").replace("lm_head=e1@fp32", "lm_head=e1"),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    with pytest.raises(repack_remainder.RepackError, match="carries no"):
        repack_remainder.repack(src, out)
    assert not out.exists(), "a refusal must write no output directory"
