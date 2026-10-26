#!/usr/bin/env python
"""Repack a qpk1 model directory's FP remainder from fp32 to bf16, no re-training.

The remainder sidecars (embedding, an untied lm_head, the norm weights) are the
FP tensors the per-layer qpk1 streams do not carry, so their storage dtype is an
artifact-level choice that costs no solver work to change. This tool converts
one directory by re-encoding those files and re-declaring them in
``manifest.toml``; the factorized ``.qpk1`` streams, the tokenizer and every
other file are copied byte-for-byte.

Rules the artifact contract imposes, mirrored from
``crates/quantibbler-qpk1/src/loader.rs``:

- the source dtype is the one the source manifest DECLARES — never inferred
  from a byte length — and a source that is already bf16 is refused (the tool's
  direction is fp32 -> bf16 only);
- the output is a NEW directory OUTSIDE the source: in-place mutation, nesting
  the output inside the source tree, and overwriting are all refused, so the
  source is never mutated;
- the conversion is lossy by declaration (bf16 carries 8 mantissa bits), so the
  output manifest records its source artifact and the conversion under
  ``[repack]`` and carries no wall-clock time — identical sources produce
  identical output bytes.

Usage (the container format follows the INPUT unless ``--container-format``
names one; a blob input is never written back as a directory):

    python tools/repack_remainder.py --in <model-dir> --out <new-dir>
    python tools/repack_remainder.py --in <model-dir> --out <new.qpk> --container-format blob
    python tools/repack_remainder.py --in <model.qpk> --out <new.qpk> --remainder-dtype bf16

``--remainder-dtype`` defaults to the tool's historical directory conversion
(``fp32 -> bf16``) for a directory output, and to the source's declared dtype
for a blob output — a dir->blob repack carries the remainder as declared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quantibbler import blob as blob_writer  # noqa: E402
from quantibbler.remainder_dtype import (  # noqa: E402
    DEFAULT_REMAINDER_DTYPE,
    RemainderDtypeError,
    decode,
    encode,
)

#: Blob dtype tags (design B3), mirrored in crates/quantibbler-qpk1/src/blob.rs.
_FLOAT_TAGS = {"fp32": blob_writer.TAG_FP32_LE, "bf16": blob_writer.TAG_BF16_LE}
_TAG_DTYPES = {blob_writer.TAG_FP32_LE: "fp32", blob_writer.TAG_BF16_LE: "bf16"}

#: Sidecar key -> the file stem the repacked output uses. Keyed by ROLE, so the
#: output name depends only on what the file is, not on the source's spelling.
_STEM = {"embedding": "e1", "lm_head": "head", "norms": "norms"}
_SIDECAR_KEYS = ("embedding", "lm_head", "norms")
_TARGET_DTYPE = "bf16"


class RepackError(RuntimeError):
    """The source directory cannot be repacked as asked."""


def _target_name(key: str) -> str:
    return f"{_STEM[key]}.{_TARGET_DTYPE}.bin"


def _read_source(source: Path) -> tuple[dict, str, str]:
    """Return the source ``[sidecar]`` block, its model id and declared dtype."""
    manifest = source / "manifest.toml"
    if not manifest.is_file():
        raise RepackError(f"{source}: not a qpk1 model directory (no manifest.toml)")
    with manifest.open("rb") as fh:
        parsed = tomllib.load(fh)
    sidecar = parsed.get("sidecar")
    if not isinstance(sidecar, dict):
        raise RepackError(f"{manifest}: no [sidecar] block; nothing to repack")
    for key in _SIDECAR_KEYS:
        if not isinstance(sidecar.get(key), str):
            raise RepackError(f"{manifest}: [sidecar] has no {key!r} entry")
    declared = sidecar.get("dtype", DEFAULT_REMAINDER_DTYPE)
    if not isinstance(declared, str):
        raise RepackError(f"{manifest}: [sidecar] dtype = {declared!r} is not a declared dtype")
    model_id = parsed.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RepackError(f"{manifest}: no model_id; the repack cannot name its source")
    return sidecar, model_id, declared


def _retarget_weight_tying(text: str) -> str:
    """Point the source's own ``weight_tying`` suffix at the converted dtype.

    The field is declarative metadata — the loader resolves the head from
    ``[sidecar]`` — but a stale token is a config line that no longer describes
    the bytes beside it, and the loader refuses a mismatch. Absent, the file is
    left alone; present without a dtype token, the conversion refuses to guess.
    """
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("weight_tying") and "=" in stripped:
            value = stripped.partition("=")[2].strip().strip('"')
            if value.startswith("lm_head="):
                stem, sep, _token = value[len("lm_head=") :].rpartition("@")
                if not sep or not stem:
                    raise RepackError(
                        f"config.toml weight_tying {value!r} carries no `<path>@<dtype>` form"
                    )
                line = f'weight_tying = "lm_head={stem}@{_TARGET_DTYPE}"'
        lines.append(line)
    return "\n".join(lines) + "\n"


def _rewrite_sidecar(text: str, entries: dict[str, str], preamble: str) -> str:
    """Replace the `[sidecar]` block's dtype and the three remainder paths.

    Everything outside the block survives byte-for-byte (key order, keys this
    tool does not model, comments elsewhere). The block's own comments are
    replaced by ``preamble`` because they describe the dtype being converted —
    carrying last week's dtype beside this week's bytes is the one kind of prose
    this tool owns.
    """
    lines: list[str] = []
    in_sidecar = False
    seen: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_sidecar = stripped == "[sidecar]"
            lines.append(line)
            if in_sidecar:
                lines.append(preamble)
                lines.append(f'dtype = "{_TARGET_DTYPE}"')
            continue
        if in_sidecar:
            if not stripped or stripped.startswith("#"):
                continue
            if "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                if key == "dtype":
                    continue
                if key in entries:
                    lines.append(f'{key} = "{entries[key]}"')
                    seen.add(key)
                    continue
        lines.append(line)
    missing = [key for key in entries if key not in seen]
    if missing:
        raise RepackError(f"manifest [sidecar] is missing {missing}; refusing a partial rewrite")
    return "\n".join(lines) + "\n"


def _repack_directory(source: Path, out: Path) -> dict:
    """Write a bf16-remainder copy of a DIRECTORY source; ``source`` is never
    written to. Raises :class:`RepackError` (nothing written) on any refusal."""
    source = Path(source).resolve()
    out = Path(out).resolve()
    if out == source:
        raise RepackError(f"{out}: refusing to repack in place; name a new directory")
    if source in out.parents:
        raise RepackError(
            f"{out}: refusing to write inside the source artifact {source}; "
            f"the output must be a new directory outside it"
        )
    if out.exists():
        raise RepackError(f"{out}: already exists; this tool never overwrites an artifact")

    sidecar, model_id, declared = _read_source(source)
    if declared != DEFAULT_REMAINDER_DTYPE:
        raise RepackError(
            f"{source}: remainder already declares dtype = {declared!r}; the conversion "
            f"is {DEFAULT_REMAINDER_DTYPE} -> {_TARGET_DTYPE} only"
        )

    # Parse + validate the source's declarative ``weight_tying`` token BEFORE any
    # write: a suffix-less token is a source this tool cannot faithfully convert
    # (the loader refuses it), and the refusal must leave no output directory
    # behind — the same no-output-on-refusal discipline the non-finite check
    # follows (RDP-1.3). The retargeted text is reused post-copy.
    source_config = source / "config.toml"
    retargeted_config: str | None = None
    if source_config.is_file():
        retargeted_config = _retarget_weight_tying(source_config.read_text(encoding="utf-8"))

    # Resolve each distinct source file to its role-based target name (a tied
    # head shares the embedding's file, and the two entries must keep naming ONE
    # file after the conversion, so the first key — embedding — wins).
    targets: dict[str, str] = {}
    for key in _SIDECAR_KEYS:
        targets.setdefault(sidecar[key], _target_name(key))

    # Read + convert BEFORE touching the filesystem, so a non-finite source
    # refuses with no output directory written.
    digest = hashlib.sha256()
    converted: dict[str, bytes] = {}
    for key in _SIDECAR_KEYS:
        rel = sidecar[key]
        src_file = source / rel
        if not src_file.is_file():
            raise RepackError(f"{source}: {rel} is declared but absent")
        raw = src_file.read_bytes()
        digest.update(raw)
        if rel in converted:
            continue  # a tied head shares the embedding's bytes: one conversion
        try:
            converted[rel] = encode(decode(raw, DEFAULT_REMAINDER_DTYPE), _TARGET_DTYPE)
        except RemainderDtypeError as exc:
            raise RepackError(f"{src_file}: {exc}") from None

    # Copy everything, then replace the converted sidecars (a rename, not an
    # overwrite: a directory holding both dtypes is a trap for whoever reads it
    # next).
    shutil.copytree(source, out)
    for rel, target in targets.items():
        (out / rel).unlink()
        (out / target).write_bytes(converted[rel])

    entries = {key: targets[sidecar[key]] for key in _SIDECAR_KEYS}
    manifest = out / "manifest.toml"
    preamble = "# remainder tensors, raw LE at the declared dtype (see [repack] below)"
    text = _rewrite_sidecar(manifest.read_text(encoding="utf-8"), entries, preamble)
    text += (
        "\n[repack]\n"
        f"source_model_id = {json.dumps(model_id)}\n"
        f'conversion = "remainder {DEFAULT_REMAINDER_DTYPE} -> {_TARGET_DTYPE}"\n'
        f'source_remainder_sha256 = "{digest.hexdigest()}"\n'
    )
    manifest.write_text(text, encoding="utf-8")

    # ``retargeted_config`` is None exactly when the source carried no
    # config.toml (validated pre-write above), so this writes only when the copy
    # placed one, and never re-parses a token the refusal path already checked.
    if retargeted_config is not None:
        (out / "config.toml").write_text(retargeted_config, encoding="utf-8")

    return {
        "source": str(source),
        "source_model_id": model_id,
        "out": str(out),
        "to": _TARGET_DTYPE,
        "source_remainder_sha256": digest.hexdigest(),
        "sidecars": entries,
    }


def _is_blob_file(path: Path) -> bool:
    """True for a regular file whose first four bytes are the blob magic."""
    if not path.is_file():
        return False
    with path.open("rb") as fh:
        return fh.read(4) == blob_writer.BLOB_MAGIC


def _blob_reader():
    """The qpk1 crate's PyO3 blob reader (design B10) — the ONLY blob reader."""
    try:
        import quantibbler_qpk1
    except ImportError as exc:  # the extension is a dev dependency; name it
        raise RepackError(
            "the qpk1 blob reader is not importable; run `uv sync` so the "
            "quantibbler-qpk1 extension is built"
        ) from exc
    return quantibbler_qpk1


def _repack_blob(
    source: Path, out: Path, *, source_is_blob: bool, remainder_dtype: str | None
) -> dict:
    """dir->blob (carry verbatim) or blob->blob (dtype conversion) legs."""
    if out.exists():
        raise RepackError(f"{out}: already exists; this tool never overwrites an artifact")
    if remainder_dtype not in (None, "fp32", "bf16"):
        raise RepackError(
            f"unknown remainder dtype {remainder_dtype!r}; declared dtypes are fp32 and bf16"
        )
    if not source_is_blob:
        sha = blob_writer.write_blob_from_directory(
            source, out, remainder_dtype=remainder_dtype, created_by="quantibbler-repack"
        )
        return {
            "source": str(source),
            "out": str(out),
            "to": remainder_dtype or "declared",
            "sha256": sha,
        }

    parsed = _blob_reader().read_blob(str(source))
    header = dict(parsed["header"])
    source_dtype = header["remainder_dtype"]
    target = remainder_dtype or source_dtype
    if target not in ("fp32", "bf16"):
        raise RepackError(f"unknown remainder dtype {target!r}; declared dtypes are fp32 and bf16")
    if target == source_dtype:
        raise RepackError(
            f"{source}: the remainder already declares {source_dtype!r}; ask for the other dtype"
        )
    sections = []
    for entry in parsed["sections"]:
        tag = entry["tag"]
        data = entry["data"]
        if tag in _TAG_DTYPES:
            data = encode(decode(data, _TAG_DTYPES[tag]), target)
            tag = _FLOAT_TAGS[target]
        sections.append(blob_writer.Section(name=entry["name"], tag=tag, data=data))
    header["remainder_dtype"] = target
    weight_tying = header.get("weight_tying")
    if weight_tying is not None:
        if not isinstance(weight_tying, str) or "@" not in weight_tying:
            raise RepackError(f"unknown weight_tying {weight_tying!r}")
        header["weight_tying"] = f"{weight_tying.rsplit('@', 1)[0]}@{target}"
    header.pop("created_by", None)
    blob = blob_writer.pack(sections, header, created_by="quantibbler-repack")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    return {
        "source": str(source),
        "out": str(out),
        "to": target,
        "sha256": blob_writer.blob_sha256(blob),
    }


def repack(
    source: Path,
    out: Path,
    *,
    container_format: str | None = None,
    remainder_dtype: str | None = None,
) -> dict:
    """Repack ``source`` at ``out``; ``source`` is never touched.

    ``container_format`` selects the OUTPUT container (``"blob"`` single-file /
    ``"dir"`` legacy directory); absent, it follows the INPUT — a blob source is
    never written back as a directory. ``remainder_dtype`` selects the output
    remainder dtype; absent, a directory output keeps the tool's historical
    ``fp32 -> bf16`` conversion and a blob output carries the source's declared
    dtype (or converts, when named).
    """
    source = Path(source).resolve()
    out = Path(out).resolve()
    if out == source:
        raise RepackError(f"{out}: refusing to repack in place; name a new artifact")
    if container_format not in (None, "blob", "dir"):
        raise RepackError(f"unknown container format {container_format!r}; blob or dir")
    source_is_blob = _is_blob_file(source)
    fmt = container_format or ("blob" if source_is_blob else "dir")
    if fmt == "blob":
        return _repack_blob(
            source, out, source_is_blob=source_is_blob, remainder_dtype=remainder_dtype
        )
    if source_is_blob:
        raise RepackError(
            f"{source}: a blob source is not written back as a directory; ask for "
            f"--container-format blob"
        )
    if remainder_dtype not in (None, "bf16"):
        raise RepackError(
            f"a directory output only performs the historical fp32 -> bf16 conversion; got "
            f"--remainder-dtype {remainder_dtype!r}"
        )
    return _repack_directory(source, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--in", dest="source", type=Path, required=True, help="the artifact to convert"
    )
    parser.add_argument("--out", type=Path, required=True, help="the new artifact path")
    parser.add_argument(
        "--container-format",
        choices=("blob", "dir"),
        default=None,
        help="output container: blob (single .qpk file) or dir (legacy directory); "
        "default follows the input (a blob stays a blob)",
    )
    parser.add_argument(
        "--remainder-dtype",
        choices=("fp32", "bf16"),
        default=None,
        help="output remainder dtype; default is the historical fp32 -> bf16 for a "
        "directory output and the source's declared dtype for a blob output",
    )
    args = parser.parse_args(argv)
    try:
        record = repack(
            args.source,
            args.out,
            container_format=args.container_format,
            remainder_dtype=args.remainder_dtype,
        )
    except RepackError as exc:
        print(f"repack REFUSED: {exc}", file=sys.stderr)
        return 2
    print(f"repacked {record['source']} -> {record['out']} (remainder -> {record['to']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
