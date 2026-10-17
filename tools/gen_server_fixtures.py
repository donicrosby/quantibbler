#!/usr/bin/env python
"""Generate qpk1 model-directory fixtures for the pure-Rust server tests.

Writes a minimal valid qpk1 model directory (the ``qk-server`` acquisition
layout, design D3 of openspec/changes/qk-server) into
``tests/fixtures/server-model/`` plus one corrupted variant per SR-1.2
rejection class into ``tests/fixtures/server-model-bad/<name>/``.

Everything is byte-level against the pinned qpk1 layout (qk-representation,
design D1): 24-byte u32 LE header (magic b"QPK1", version 1, dtype tag,
dout, din, r), then the scale vectors in the tagged dtype (float16 tag 1 /
float32 tag 2), then packed uint8 Ub/Vb bitplanes -- row-major, LSB-first
within bytes, +1 -> bit 1, -1 -> bit 0, zero pad to the byte boundary.

Pure Python stdlib (struct + manual bit packing); the Rust crate is never
imported. Deterministic (fixed seeds); regenerating is idempotent, and
fixture drift is a reviewable diff.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "server-model"
BAD_DIR = REPO_ROOT / "tests" / "fixtures" / "server-model-bad"

QPK1_MAGIC = b"QPK1"
REPRESENTATION_VERSION = 1
DTYPE_TAG_FLOAT16 = 1
DTYPE_TAG_FLOAT32 = 2
_HEADER = struct.Struct("<4sIIIII")

# Model-level hyperparameters written to config.toml.
CONFIG = {
    "vocab_size": 32,
    "d_model": 4,
    "eos_token_id": 3,
    "bos_token_id": 1,
}

TOKENIZER_JSON = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {
            "id": 0,
            "content": "<unk>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
        {
            "id": 1,
            "content": "<bos>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
        {
            "id": 2,
            "content": "<eos>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
    ],
    "normalizer": None,
    "pre_tokenizer": {
        "type": "Whitespace",
    },
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "BPE",
        "dropout": None,
        "unk_token": "<unk>",
        "continuing_subword_prefix": None,
        "end_of_word_suffix": None,
        "fuse_unk": True,
        "byte_fallback": False,
        "vocab": {
            "<unk>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "a": 3,
            "b": 4,
            "c": 5,
            "d": 6,
            "e": 7,
            "f": 8,
            "g": 9,
            "h": 10,
            "i": 11,
            "j": 12,
            "k": 13,
            "l": 14,
            "m": 15,
            "n": 16,
            "o": 17,
            "p": 18,
            "q": 19,
            "r": 20,
            "s": 21,
            "t": 22,
            "u": 23,
            "v": 24,
            "w": 25,
            "x": 26,
            "y": 27,
            "z": 28,
            "ab": 29,
            "ba": 30,
        },
        "merges": [
            "a b",
            "b a",
        ],
    },
}


def f16_bytes(value: float) -> bytes:
    """Float16 (IEEE 754 binary16) little-endian encoding, stdlib only."""
    return struct.pack("<e", float(value))


def pack_bits(rows: list[list[int]]) -> bytes:
    """Pack a +/-1 matrix per the pinned qpk1 canonical layout (SF-3.1).

    Row-major with ONE continuous bit stream across the whole matrix:
    element ``(i, j)`` lands at bit ``i * r + j`` of the bitplane, i.e.
    byte ``(i*r + j) // 8``, bit ``(i*r + j) % 8`` from the LSB.
    ``+1 -> bit 1``, ``-1 -> bit 0``, pad bits zero. (Note: this is the
    container layout; the kernels rlib's per-row bitplanes are a different,
    working layout the server converts to at load.)
    """
    flat = [entry for row in rows for entry in row]
    out = bytearray((len(flat) + 7) // 8)
    for k, entry in enumerate(flat):
        if entry == 1:
            out[k >> 3] |= 1 << (k & 7)
        elif entry != -1:
            raise ValueError(f"factor entries must be +/-1, got {entry}")
    return bytes(out)


def deterministic_signs(n_rows: int, n_cols: int, seed: int) -> list[list[int]]:
    """splitmix64-driven +/-1 matrix (fixed seed -> fixed fixture bytes)."""
    state = seed & 0xFFFFFFFFFFFFFFFF
    GOLD = 0x9E3779B97F4A7C15
    MIX1 = 0xBF58476D1CE4E5B9
    MIX2 = 0x94D049BB133111EB
    rows: list[list[int]] = []
    for _ in range(n_rows):
        row: list[int] = []
        for _ in range(n_cols):
            state = (state + GOLD) & 0xFFFFFFFFFFFFFFFF
            z = state
            z = ((z ^ (z >> 30)) * MIX1) & 0xFFFFFFFFFFFFFFFF
            z = ((z ^ (z >> 27)) * MIX2) & 0xFFFFFFFFFFFFFFFF
            z ^= z >> 31
            row.append(1 if (z >> 11) % 2 == 0 else -1)
        rows.append(row)
    return rows


def qpk1_stream(
    dout: int,
    din: int,
    r: int,
    seed: int,
    dtype_tag: int = DTYPE_TAG_FLOAT16,
    s1: list[float] | None = None,
    s2: list[float] | None = None,
) -> bytes:
    """Serialize one factorized linear layer to the canonical qpk1 stream."""
    ub = deterministic_signs(dout, r, seed)
    vb = deterministic_signs(din, r, seed + 1)
    if s1 is None:
        s1 = [round(0.5 + 0.25 * ((seed + k) % 5) / 5.0, 4) for k in range(dout)]
    if s2 is None:
        s2 = [round(0.4 + 0.2 * ((seed + k) % 3) / 3.0, 4) for k in range(din)]
    if dtype_tag == DTYPE_TAG_FLOAT16:
        s1_bytes = b"".join(f16_bytes(v) for v in s1)
        s2_bytes = b"".join(f16_bytes(v) for v in s2)
    elif dtype_tag == DTYPE_TAG_FLOAT32:
        s1_bytes = struct.pack(f"<{dout}f", *map(float, s1))
        s2_bytes = struct.pack(f"<{din}f", *map(float, s2))
    else:
        raise ValueError(f"unknown dtype tag {dtype_tag}")
    header = _HEADER.pack(QPK1_MAGIC, REPRESENTATION_VERSION, dtype_tag, dout, din, r)
    return header + s1_bytes + s2_bytes + pack_bits(ub) + pack_bits(vb), (ub, vb, s1, s2)


def expected_size(dout: int, din: int, r: int, w: int) -> int:
    """qk-representation SF-1.1 size formula."""
    return 24 + (dout + din) * w + (dout * r + 7) // 8 + (din * r + 7) // 8


def build_valid_model(root: Path) -> None:
    """Write the valid fixture model directory (single factorized layer)."""
    (root / "layers").mkdir(parents=True, exist_ok=True)
    dout, din, r = 4, 4, 4
    stream, _ = qpk1_stream(dout, din, r, seed=20261017)
    layer_path = root / "layers" / "block0.mlp.fc.qpk1"
    layer_path.write_bytes(stream)
    expected = expected_size(dout, din, r, 2)
    actual = layer_path.stat().st_size
    if actual != expected:
        raise AssertionError(f"{layer_path}: {actual} bytes, formula says {expected}")
    (root / "manifest.toml").write_text(
        "# qk-server fixture manifest (design D3)\n"
        'model_id = "server-model-fixture"\n'
        'layers = [ "block0.mlp.fc" ]\n'
        "\n"
        "[layer_files]\n"
        "# quoted keys: TOML bare keys with dots would nest tables\n"
        '"block0.mlp.fc" = "layers/block0.mlp.fc.qpk1"\n'
        "\n"
        "[sidecar]\n"
        "# fp32 embedding / lm_head raw LE float32 (OQ2 minimal sidecar)\n"
        'embedding = "e1.f32.bin"\n'
        'lm_head = "e1.f32.bin"\n',
        encoding="utf-8",
    )
    (root / "config.toml").write_text(
        "# qk-server fixture config (design D3)\n"
        f"vocab_size = {CONFIG['vocab_size']}\n"
        f"d_model = {CONFIG['d_model']}\n"
        f"eos_token_id = {CONFIG['eos_token_id']}\n"
        f"bos_token_id = {CONFIG['bos_token_id']}\n"
        'weight_tying = "lm_head=e1@fp32"\n',
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        json.dumps(TOKENIZER_JSON, indent=1) + "\n", encoding="utf-8"
    )
    # Sidecar: fp32 embedding/lm_head weights, weight-tied, byte layout in
    # the manifest (OQ2 minimally: raw LE f32, row-major).
    import array

    emb = array.array("f", range(CONFIG["vocab_size"] * CONFIG["d_model"]))
    # deterministic small values, incl. negatives
    for i in range(len(emb)):
        emb[i] = ((i % 7) - 3) * 0.25
    (root / "e1.f32.bin").write_bytes(emb.tobytes())


def build_bad_variants(base: Path) -> dict[str, str]:
    """One corrupted model dir per SR-1.2 rejection class.

    Returns {variant_name: expected diagnostic substring}.
    """
    import shutil

    if base.exists():
        shutil.rmtree(base)
    variants: dict[str, str] = {}

    def derived(name: str, mutate) -> None:
        dest = base / name
        shutil.copytree(FIXTURE_DIR, dest)
        mutate(dest)
        variants[name] = dest

    # 1. missing manifest.toml
    derived("missing_manifest", lambda d: (d / "manifest.toml").unlink())
    # 2. missing tokenizer.json
    derived("missing_tokenizer", lambda d: (d / "tokenizer.json").unlink())
    # 3. malformed tokenizer.json
    derived("malformed_tokenizer", lambda d: (d / "tokenizer.json").write_text("{not json"))
    # 4. manifest names a layer whose file is missing
    derived("missing_layer_file", lambda d: (d / "layers" / "block0.mlp.fc.qpk1").unlink())

    # 5. bad magic
    def _bad_magic(d: Path) -> None:
        p = d / "layers" / "block0.mlp.fc.qpk1"
        p.write_bytes(b"XPK1" + p.read_bytes()[4:])

    derived("bad_magic", _bad_magic)

    # 6. wrong version
    def _bad_version(d: Path) -> None:
        p = d / "layers" / "block0.mlp.fc.qpk1"
        b = bytearray(p.read_bytes())
        struct.pack_into("<I", b, 4, 7)
        p.write_bytes(bytes(b))

    derived("wrong_version", _bad_version)

    # 7. unknown dtype tag
    def _bad_tag(d: Path) -> None:
        p = d / "layers" / "block0.mlp.fc.qpk1"
        b = bytearray(p.read_bytes())
        struct.pack_into("<I", b, 8, 9)
        p.write_bytes(bytes(b))

    derived("unknown_dtype_tag", _bad_tag)

    # 8. rank 0
    def _rank0(d: Path) -> None:
        p = d / "layers" / "block0.mlp.fc.qpk1"
        b = bytearray(p.read_bytes())
        struct.pack_into("<I", b, 20, 0)
        p.write_bytes(bytes(b))

    derived("rank_zero", _rank0)

    # 9. r > min(dout, din)
    def _rank_too_big(d: Path) -> None:
        p = d / "layers" / "block0.mlp.fc.qpk1"
        b = bytearray(p.read_bytes())
        struct.pack_into("<I", b, 20, 9)
        p.write_bytes(bytes(b))

    derived("rank_infeasible", _rank_too_big)

    # 10. truncated stream (chop the last byte)
    def _truncated(d: Path) -> None:
        p = d / "layers" / "block0.mlp.fc.qpk1"
        raw = p.read_bytes()
        p.write_bytes(raw[:-1])

    derived("truncated", _truncated)

    # 11. trailing garbage
    def _trailing(d: Path) -> None:
        p = d / "layers" / "block0.mlp.fc.qpk1"
        p.write_bytes(p.read_bytes() + b"\x00")

    derived("trailing_bytes", _trailing)

    # 12. manifest layer path escapes the model dir (review finding:
    #     Path::join absolute/..-replacement made this an arbitrary read)
    def _path_escape(d: Path) -> None:
        m = d / "manifest.toml"
        m.write_text(
            "# qk-server fixture manifest (design D3)\n"
            'model_id = "server-model-fixture"\n'
            'layers = [ "block0.mlp.fc" ]\n'
            "\n"
            "[layer_files]\n"
            '"block0.mlp.fc" = "../evil.qpk1"\n'
            "\n"
            "[sidecar]\n"
            'embedding = "e1.f32.bin"\n'
            'lm_head = "e1.f32.bin"\n',
            encoding="utf-8",
        )

    derived("path_escape", _path_escape)

    # 13. layer file is a symlink (escape hatch; loader demands regular files)
    def _symlink_layer(d: Path) -> None:
        target = d / "layers" / "block0.mlp.fc.qpk1"
        raw = target.read_bytes()
        target.unlink()
        outside = d / "outside.qpk1"
        outside.write_bytes(raw)
        target.symlink_to(outside)

    derived("symlink_layer", _symlink_layer)

    # 14. sidecar matrix with a wrong (too small) byte length: the
    #     stat-before-read gate must reject on length, pre-read
    def _sidecar_too_big(d: Path) -> None:
        (d / "e1.f32.bin").write_bytes(b"\x00" * 8)

    derived("sidecar_too_big", _sidecar_too_big)

    # 15. undocumented weight_tying value: fail-closed config validation
    def _bad_weight_tying(d: Path) -> None:
        c = d / "config.toml"
        c.write_text(
            "# qk-server fixture config (design D3)\n"
            "vocab_size = 32\n"
            "d_model = 4\n"
            "eos_token_id = 3\n"
            "bos_token_id = 1\n"
            'weight_tying = "banana"\n',
            encoding="utf-8",
        )

    derived("bad_weight_tying", _bad_weight_tying)

    # 16. token id at/past vocab_size: unrepresentable row reference
    def _eos_out_of_range(d: Path) -> None:
        c = d / "config.toml"
        c.write_text(
            "# qk-server fixture config (design D3)\n"
            "vocab_size = 32\n"
            "d_model = 4\n"
            "eos_token_id = 99\n"
            "bos_token_id = 1\n"
            'weight_tying = "lm_head=e1@fp32"\n',
            encoding="utf-8",
        )

    derived("eos_out_of_range", _eos_out_of_range)
    return variants


def main() -> None:
    build_valid_model(FIXTURE_DIR)
    print(f"wrote {FIXTURE_DIR}")
    variants = build_bad_variants(BAD_DIR)
    for _name, path in sorted(variants.items()):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
