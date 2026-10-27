"""qpk1 single-file blob WRITER (change ``qk-artifact-blob``, design B11).

The blob is the canonical qpk1 model container: one ``QKPB`` file carrying the
declared metadata, a self-delimiting section table and page-aligned payloads
(B1-B5). This module is the HOST writer — the exporter and the repack utility
both compose it — and it is deliberately the only Python code that touches the
blob layout: READING a blob goes through the qpk1 crate's PyO3 face (design
B10), so the Rust reader stays the single canonical implementation.

Layout written here, all integers little-endian:

    magic b"QKPB" | u32 version (1) | u32 header_len | header JSON
    zero padding to the next 4096-byte boundary
    u32 section count | per section: u8 name_len | name | u8 tag | u64 offset | u64 length
    zero padding to the next 4096-byte boundary
    payloads, each at a 4096-aligned offset, in canonical order

Determinism (BF-6/BW-2): the header JSON is serialized with sorted keys and no
whitespace, the section order is canonical, every pad byte is zero and no field
carries a wall-clock time — so the same inputs produce byte-identical files.
"""

from __future__ import annotations

import hashlib
import json
import struct
import tomllib
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "BLOB_MAGIC",
    "BLOB_VERSION",
    "CHAT_TEMPLATE_FILE",
    "PAGE_SIZE",
    "TAG_BF16_LE",
    "TAG_FP32_LE",
    "TAG_QPK1_V1",
    "TAG_QPK1_V2",
    "TAG_RAW",
    "BlobError",
    "Section",
    "blob_sha256",
    "pack",
    "write_blob_from_directory",
]

BLOB_MAGIC = b"QKPB"
BLOB_VERSION = 1
PAGE_SIZE = 4096

#: The canonical chat-template sidecar. It is neither a layer nor a
#: `tokenizer.*` file, so it is packed as its own raw section, in the canonical
#: order the reader enforces (after every `tokenizer.*` section).
CHAT_TEMPLATE_FILE = "chat_template.jinja"

# The closed dtype-tag vocabulary (design B3), identical to BlobTag in
# crates/quantibbler-qpk1/src/blob.rs.
TAG_QPK1_V1 = 0
TAG_QPK1_V2 = 1
TAG_FP32_LE = 2
TAG_BF16_LE = 3
TAG_RAW = 4

_FLOAT_TAG = {"fp32": TAG_FP32_LE, "bf16": TAG_BF16_LE}
#: The remaining sidecar key -> its fixed blob section name and output stem.
_SIDECAR_KEYS = ("embedding", "lm_head", "norms")
_REMAINDER_NAME = {"embedding": "embedding", "lm_head": "lm_head", "norms": "norms"}


class BlobError(RuntimeError):
    """The requested blob cannot be written from the given inputs."""


# The dtype codec (and its numpy/torch load) is imported INSIDE the functions
# that need it: the fixture generator packs a directory whose remainder is
# already at its declared dtype — verbatim bytes, no numpy, no torch — so
# importing this module must not drag the numerical stack in.
def _dtype_itemsize(dtype: str) -> int:
    from quantibbler.remainder_dtype import itemsize

    return itemsize(dtype)


@dataclass(frozen=True)
class Section:
    """One blob payload: name, dtype tag, bytes."""

    name: str
    tag: int
    data: bytes


def _pad_to_page(out: bytearray) -> None:
    out.extend(b"\x00" * (-len(out) % PAGE_SIZE))


def pack(sections: list[Section], header: dict, *, created_by: str | None = None) -> bytes:
    """Serialize ``sections`` + ``header`` (canonical order) into blob bytes.

    The caller passes sections already in the canonical order (BF-5): layers in
    the header's ``layers`` order, then ``embedding`` / ``lm_head`` / ``norms``,
    then ``tokenizer.*`` sorted, then the chat-template sidecar. This function
    enforces the layout, not the order — a mis-ordered table is refused by the
    reader, never silently fixed.
    """
    for section in sections:
        name = section.name.encode("utf-8")
        if not name or len(name) > 255:
            raise BlobError(f"section name {section.name!r} does not fit a u8 length prefix")
        if section.tag not in (TAG_QPK1_V1, TAG_QPK1_V2, TAG_FP32_LE, TAG_BF16_LE, TAG_RAW):
            raise BlobError(f"section {section.name!r} has an unknown dtype tag {section.tag}")

    declared = dict(header)
    if created_by is not None:
        declared["created_by"] = created_by
    # sort_keys + compact separators: the JSON bytes depend only on the values,
    # never on insertion order (BF-6 determinism).
    header_json = json.dumps(
        declared, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")

    out = bytearray()
    out += BLOB_MAGIC
    out += struct.pack("<I", BLOB_VERSION)
    out += struct.pack("<I", len(header_json))
    out += header_json
    _pad_to_page(out)

    # The table's own length is fixed once the names are known, so the first
    # payload offset is computable before the table is written.
    table_len = 4 + sum(1 + len(s.name.encode("utf-8")) + 1 + 8 + 8 for s in sections)
    first_offset = len(out) + table_len
    first_offset += -first_offset % PAGE_SIZE

    offsets: list[int] = []
    cursor = first_offset
    for index, section in enumerate(sections):
        offsets.append(cursor)
        cursor += len(section.data)
        if index + 1 < len(sections):
            cursor += -cursor % PAGE_SIZE  # zero pad only BETWEEN payloads

    out += struct.pack("<I", len(sections))
    for section, offset in zip(sections, offsets, strict=True):
        name = section.name.encode("utf-8")
        out += bytes([len(name)])
        out += name
        out += bytes([section.tag])
        out += struct.pack("<QQ", offset, len(section.data))
    _pad_to_page(out)
    assert len(out) == first_offset, "section table length disagrees with the computed offset"
    for index, section in enumerate(sections):
        out += section.data
        if index + 1 < len(sections):
            _pad_to_page(out)
    return bytes(out)


def blob_sha256(blob: bytes | bytearray | Path) -> str:
    """The external content hash (B6): recorded in the report, never inside."""
    data = blob.read_bytes() if isinstance(blob, Path) else bytes(blob)
    return hashlib.sha256(data).hexdigest()


def _stream_tag(stream: bytes, source: str) -> int:
    """The section tag a layer stream must carry: its OWN declared version."""
    if len(stream) < 8 or stream[0:4] != b"QPK1":
        raise BlobError(f"{source}: not a qpk1 layer stream (bad magic)")
    version = struct.unpack("<I", stream[4:8])[0]
    if version == 1:
        return TAG_QPK1_V1
    if version == 2:
        return TAG_QPK1_V2
    raise BlobError(f"{source}: unsupported stream version {version}")


def _retarget_weight_tying(weight_tying: str | None, dtype: str) -> str | None:
    """Rewrite a ``lm_head=<path>@<dtype>`` suffix to ``dtype``, or refuse."""
    if weight_tying is None:
        return None
    if "@" not in weight_tying or not weight_tying.startswith("lm_head="):
        raise BlobError(
            f"unknown weight_tying {weight_tying!r}; documented forms: absent or "
            f'"lm_head=<path>@<dtype>" with dtype fp32 or bf16'
        )
    path, _ = weight_tying.rsplit("@", 1)
    return f"{path}@{dtype}"


def write_blob_from_directory(
    source: Path,
    out_path: Path,
    *,
    remainder_dtype: str | None = None,
    created_by: str | None = None,
) -> str:
    """Pack a qpk1 model DIRECTORY into a single-file blob.

    Layer streams and tokenizer files are carried verbatim; the remainder
    sections are re-encoded at ``remainder_dtype`` (default: the dtype the
    source declares), and the header's ``remainder_dtype`` + ``weight_tying``
    suffix are rewritten to agree. Returns the blob's sha256.
    """
    manifest_path = source / "manifest.toml"
    config_path = source / "config.toml"
    if not manifest_path.is_file() or not config_path.is_file():
        raise BlobError(f"{source}: not a qpk1 model directory (no manifest.toml/config.toml)")
    with manifest_path.open("rb") as fh:
        manifest = tomllib.load(fh)
    with config_path.open("rb") as fh:
        config = tomllib.load(fh)

    sidecar = manifest.get("sidecar")
    if not isinstance(sidecar, dict):
        raise BlobError(f"{manifest_path}: no [sidecar] block; nothing to carry")
    source_dtype = sidecar.get("dtype", "fp32")  # absent = the historical fp32 reading
    if not isinstance(source_dtype, str):
        raise BlobError(f"{manifest_path}: [sidecar] dtype = {source_dtype!r} is not a dtype")
    target_dtype = remainder_dtype or source_dtype
    _dtype_itemsize(target_dtype)  # undeclared token refuses by name

    layers = manifest.get("layers")
    layer_files = manifest.get("layer_files")
    if not isinstance(layers, list) or not isinstance(layer_files, dict):
        raise BlobError(f"{manifest_path}: manifest lists no layers / [layer_files] block")

    sections: list[Section] = []
    for name in layers:
        rel = layer_files.get(name)
        if not isinstance(rel, str):
            raise BlobError(f"{manifest_path}: layer {name!r} has no [layer_files] entry")
        raw = (source / rel).read_bytes()
        sections.append(Section(name=name, tag=_stream_tag(raw, str(source / rel)), data=raw))

    embedding_path = sidecar.get("embedding")
    head_path = sidecar.get("lm_head")
    norms_path = sidecar.get("norms")
    for key, value in (
        ("embedding", embedding_path),
        ("lm_head", head_path),
        ("norms", norms_path),
    ):
        if not isinstance(value, str):
            raise BlobError(f"{manifest_path}: [sidecar] has no {key!r} entry")

    for key, rel in (("embedding", embedding_path), ("norms", norms_path)):
        raw = (source / rel).read_bytes()
        data = _reencode(raw, source_dtype, target_dtype, str(source / rel))
        sections.append(Section(name=_REMAINDER_NAME[key], tag=_FLOAT_TAG[target_dtype], data=data))
    # An untied head ships its own section; a tied head IS the embedding.
    if head_path != embedding_path:
        raw = (source / head_path).read_bytes()
        data = _reencode(raw, source_dtype, target_dtype, str(source / head_path))
        # canonical order: embedding, lm_head, norms -- so insert before norms
        norms_section = sections.pop()
        sections.append(Section(name="lm_head", tag=_FLOAT_TAG[target_dtype], data=data))
        sections.append(norms_section)

    tokenizer_files = sorted(
        p.name for p in source.iterdir() if p.is_file() and p.name.startswith("tokenizer.")
    )
    for name in tokenizer_files:
        sections.append(Section(name=name, tag=TAG_RAW, data=(source / name).read_bytes()))

    # The canonical chat-template sidecar is a plain text file, not a
    # `tokenizer.*` entry, so it rides as its own raw section after them — the
    # position the reader's canonical order puts it in. A directory without one
    # packs exactly as before.
    sidecar = source / CHAT_TEMPLATE_FILE
    if sidecar.is_file():
        sections.append(Section(name=CHAT_TEMPLATE_FILE, tag=TAG_RAW, data=sidecar.read_bytes()))

    header = {
        "model_id": manifest.get("model_id"),
        "vocab_size": config.get("vocab_size"),
        "d_model": config.get("d_model"),
        "n_layers": config.get("n_layers"),
        "n_heads": config.get("n_heads"),
        "n_kv_heads": config.get("n_kv_heads"),
        "head_dim": config.get("head_dim"),
        "eos_token_id": config.get("eos_token_id"),
        "bos_token_id": config.get("bos_token_id"),
        "rms_norm_eps": config.get("rms_norm_eps"),
        "rope_theta": config.get("rope_theta"),
        "norm_slots": config.get("norm_slots"),
        "layers": list(layers),
        "weight_tying": _retarget_weight_tying(config.get("weight_tying"), target_dtype),
        "remainder_dtype": target_dtype,
    }
    missing = [key for key, value in header.items() if value is None]
    if missing:
        raise BlobError(f"{config_path}: missing required field(s) {missing}")

    blob = pack(sections, header, created_by=created_by)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)
    return blob_sha256(blob)


def _reencode(raw: bytes, source_dtype: str, target_dtype: str, where: str) -> bytes:
    """Carry remainder bytes verbatim, or re-encode them at ``target_dtype``."""
    if source_dtype == target_dtype:
        return raw
    from quantibbler.remainder_dtype import decode, encode

    values = decode(raw, source_dtype)
    expected = values.size
    encoded = encode(values, target_dtype)
    if decode(encoded, target_dtype).size != expected:
        raise BlobError(f"{where}: remainder re-encode changed the element count")
    return encoded
