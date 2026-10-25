#!/usr/bin/env python
"""Captured-attention-activation records: the npz format, versioned and checksummed.

Change ``qk-sage-attention``, task 4.3 (planner ruling t_fccd1ced, Decision 4).
The gate tool's JSON path consumes the committed *fixture battery* (i.i.d.
Gaussian inputs); a real-model record consumes captured forward-pass operands,
which are far too large for JSON. This module owns the one format both the
capture producer (``tools/capture_attention_activations.py``), the record
producer (``tools/gen_attention_gate_record.py``) and the test hook
(``tests/test_attention_sage.py``) read and write, so the three cannot drift.

The file
--------
An ``.npz`` (``numpy.savez_compressed``) of f32 arrays plus scalar metadata:

- ``format_version``   — the format's version (see :data:`FORMAT_VERSION`);
- ``activations_kind`` — the record's evidence class AND its source, as
  ``"<class>:<source>"``: ``"real-activation:Qwen/Qwen3-0.6B"`` names the
  enabling class ``real-activation`` and the model it was captured from. The
  reserved value ``"fixture-battery"`` (:data:`FIXTURE_BATTERY_KIND`) is NOT a
  capture and is written by the JSON path, never here.
- ``capture_model``    — the base model the activations came from. This is the
  CAPTURE's provenance, **not** the gate record's served ``model_id`` (which the
  gate tool reads from the model directory's ``manifest.toml``);
- ``checksum``         — sha256 over every other key (names, dtypes, shapes and
  raw array bytes, in sorted-key order), so a truncated, edited or re-encoded
  capture is refused rather than silently measured;
- ``case_count``       — the number of captured cases;
- ``metadata``         — a JSON string of provenance the caller chose to record
  (interpreter, split, layer/window selection, seed, ...). Nothing reads it
  structurally; it is carried for the operator;
- ``case_<i>_q`` / ``case_<i>_k`` / ``case_<i>_v`` — the captured operands,
  f32, in the layout the faces document (``(B, n_heads, S, head_dim)`` /
  ``(B, n_kv_heads, S, head_dim)``), GQA un-repeated;
- ``case_<i>_causal`` / ``case_<i>_block_size`` / ``case_<i>_layer`` /
  ``case_<i>_window`` — per-case scalars.

f32 is the storage dtype because that is what the device face consumes: the
reference faces re-derive their f64 outputs from these f32 operands, exactly as
the JSON fixture battery does.

Large-file gate: a real-activation capture is **deliberately outside the repo's
500 KB ``check-added-large-files`` gate** — one seq-256 case at 16 heads /
128 dim is ~4 MB of f32 operands, so a real battery is orders past it, and the
exemption is deliberate rather than a reason to shrink the capture. The
exemption is named in the landing commit for this path; the capture itself is
minted by a real-model run, not committed with the tool.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "FIXTURE_BATTERY_KIND",
    "FORMAT_VERSION",
    "REAL_ACTIVATION_KIND",
    "CaptureFormatError",
    "CapturedCase",
    "CaptureFile",
    "read_capture",
    "write_capture",
]

#: The npz format's version. A reader refuses anything else rather than
#: guessing at an unknown layout.
FORMAT_VERSION = 1

#: The reserved evidence class for the committed JSON fixture battery (i.i.d.
#: Gaussian inputs) — produced by the gate tool's JSON path, never by a capture.
FIXTURE_BATTERY_KIND = "fixture-battery"

#: The evidence class of a captured real forward pass. The stored
#: ``activations_kind`` is ``f"{REAL_ACTIVATION_KIND}:{capture_model}"``.
REAL_ACTIVATION_KIND = "real-activation"

_CASE_FIELDS = ("q", "k", "v", "causal", "block_size", "layer", "window")
_SCALARS = (
    "format_version",
    "activations_kind",
    "capture_model",
    "checksum",
    "case_count",
    "metadata",
)


class CaptureFormatError(ValueError):
    """A capture file is missing, unreadable, or not what its metadata claims."""


@dataclass(eq=False)
class CapturedCase:
    """One captured ``(q, k, v)`` triple plus its capture coordinates."""

    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    causal: bool = False
    block_size: int = 64
    layer: int = -1
    window: int = -1

    @property
    def shape(self) -> dict[str, int]:
        """The layout, in the JSON fixture battery's ``shape`` vocabulary."""
        return {
            "batch": int(self.q.shape[0]),
            "n_heads": int(self.q.shape[1]),
            "seq": int(self.q.shape[2]),
            "head_dim": int(self.q.shape[3]),
            "n_kv_heads": int(self.k.shape[1]),
            "val_dim": int(self.v.shape[3]),
        }


@dataclass(frozen=True)
class CaptureFile:
    """A parsed capture: where it came from, and the cases it holds."""

    path: Path
    format_version: int
    activations_kind: str
    capture_model: str
    checksum: str
    metadata: dict
    cases: tuple[CapturedCase, ...]

    @property
    def kind_class(self) -> str:
        """The evidence class: everything before the first ``:`` in the kind."""
        return self.activations_kind.split(":", 1)[0]

    @property
    def source(self) -> str:
        """The capture source: everything after the first ``:`` (or the kind)."""
        head, sep, tail = self.activations_kind.partition(":")
        return tail if sep else head


def _canonical(name: str, value: object) -> bytes:
    """One array's contribution to the checksum: name, kind/shape, raw bytes.

    Strings hash as UTF-8 of their value rather than as ``tobytes`` (NumPy's
    ``U`` dtype is UCS-4 in native byte order), so the digest is stable across
    platforms for the metadata scalars.
    """
    arr = np.asarray(value)
    out = bytearray()
    out += name.encode("utf-8")
    out += b"\0"
    if arr.dtype.kind in ("U", "S"):
        out += b"str"
        out += b"\0"
        out += str(arr.item()).encode("utf-8")
    else:
        arr = np.ascontiguousarray(arr)
        out += arr.dtype.str.encode("ascii")
        out += b"\0"
        out += json.dumps(list(arr.shape)).encode("ascii")
        out += b"\0"
        out += arr.tobytes()
    out += b"\0"
    return bytes(out)


def checksum(arrays: dict) -> str:
    """sha256 over every key except ``checksum`` itself, in sorted-key order."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        if name == "checksum":
            continue
        digest.update(_canonical(name, arrays[name]))
    return digest.hexdigest()


def _validate(case: CapturedCase, index: int, path: Path) -> None:
    """The layout contract both reference faces enforce, checked at load."""
    where = f"{path}: case_{index}"
    for name in ("q", "k", "v"):
        arr = np.asarray(getattr(case, name))
        if arr.ndim != 4:
            raise CaptureFormatError(f"{where}_{name}: expected a 4-d array, got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise CaptureFormatError(f"{where}_{name}: holds non-finite values")
    if not case.q.shape[0] == case.k.shape[0] == case.v.shape[0]:
        raise CaptureFormatError(
            f"{where}: batch mismatch — q {case.q.shape}, k {case.k.shape}, v {case.v.shape}"
        )
    if case.k.shape[1] != case.v.shape[1]:
        raise CaptureFormatError(
            f"{where}: k/v head-count mismatch: k {case.k.shape}, v {case.v.shape}"
        )
    if case.q.shape[2] != case.k.shape[2] or case.q.shape[2] != case.v.shape[2]:
        raise CaptureFormatError(
            f"{where}: k/v sequence length must match q ({case.q.shape[2]}), "
            f"got k {case.k.shape[2]}, v {case.v.shape[2]}"
        )
    if case.q.shape[3] != case.k.shape[3]:
        raise CaptureFormatError(f"{where}: head-dim mismatch between q and k")
    n_kv = int(case.k.shape[1])
    if n_kv <= 0 or int(case.q.shape[1]) % n_kv != 0:
        raise CaptureFormatError(
            f"{where}: n_kv_heads must divide n_heads: n_heads={case.q.shape[1]}, n_kv={n_kv}"
        )
    if int(case.block_size) <= 0:
        raise CaptureFormatError(f"{where}: block_size must be positive, got {case.block_size!r}")


def write_capture(
    path: Path | str,
    cases: list[CapturedCase] | tuple[CapturedCase, ...],
    *,
    capture_model: str,
    kind: str = REAL_ACTIVATION_KIND,
    metadata: dict | None = None,
) -> CaptureFile:
    """Write a versioned, checksummed capture and return its parsed form.

    ``kind`` is the evidence class and the stored ``activations_kind`` is
    ``f"{kind}:{capture_model}"`` — the class and its source, so a record
    carrying it names both rather than a bare label. The stored file is
    deterministic for identical inputs (callers that want a timestamp put it in
    ``metadata``).
    """
    if not cases:
        raise CaptureFormatError("refusing to write a capture with no cases")
    if not capture_model:
        raise CaptureFormatError("a capture must name the model it came from")
    if not kind or kind == FIXTURE_BATTERY_KIND:
        raise CaptureFormatError(
            f"kind {kind!r} is not a capture class; {FIXTURE_BATTERY_KIND!r} is the JSON path's"
        )

    arrays: dict[str, object] = {
        "format_version": np.int64(FORMAT_VERSION),
        "activations_kind": np.array(f"{kind}:{capture_model}"),
        "capture_model": np.array(capture_model),
        "case_count": np.int64(len(cases)),
        "metadata": np.array(json.dumps(metadata or {}, sort_keys=True)),
    }
    for index, case in enumerate(cases):
        _validate(case, index, Path(path))
        arrays[f"case_{index}_q"] = np.ascontiguousarray(case.q, dtype=np.float32)
        arrays[f"case_{index}_k"] = np.ascontiguousarray(case.k, dtype=np.float32)
        arrays[f"case_{index}_v"] = np.ascontiguousarray(case.v, dtype=np.float32)
        arrays[f"case_{index}_causal"] = np.bool_(case.causal)
        arrays[f"case_{index}_block_size"] = np.int64(case.block_size)
        arrays[f"case_{index}_layer"] = np.int64(case.layer)
        arrays[f"case_{index}_window"] = np.int64(case.window)
    arrays["checksum"] = np.array(checksum(arrays))

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    return read_capture(out)


def read_capture(path: Path | str) -> CaptureFile:
    """Parse and validate a capture file; a damaged or foreign file is refused."""
    resolved = Path(path)
    if not resolved.is_file():
        raise CaptureFormatError(f"{resolved}: no such capture file")
    try:
        with np.load(resolved, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
    except (
        OSError,
        EOFError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:  # not an npz, or a damaged zip
        raise CaptureFormatError(f"{resolved}: not a readable npz capture ({exc})") from exc

    missing = [name for name in _SCALARS if name not in arrays]
    if missing:
        raise CaptureFormatError(
            f"{resolved}: missing {sorted(missing)}; this is not a versioned capture "
            f"(format_version {FORMAT_VERSION} is required)"
        )
    version = int(arrays["format_version"])
    if version != FORMAT_VERSION:
        raise CaptureFormatError(
            f"{resolved}: format_version {version}, this reader speaks {FORMAT_VERSION} — "
            "re-capture with the committed tool"
        )
    expected = checksum(arrays)
    stored = str(arrays["checksum"].item())
    if stored != expected:
        raise CaptureFormatError(
            f"{resolved}: checksum mismatch — the file stores {stored} but its contents hash "
            f"to {expected}; the capture is damaged or edited"
        )
    kind = str(arrays["activations_kind"].item())
    if not kind:
        raise CaptureFormatError(f"{resolved}: activations_kind is empty")
    capture_model = str(arrays["capture_model"].item())

    count = int(arrays["case_count"])
    cases: list[CapturedCase] = []
    for index in range(count):
        names = {field: f"case_{index}_{field}" for field in _CASE_FIELDS}
        absent = [names[field] for field in ("q", "k", "v") if names[field] not in arrays]
        if absent:
            raise CaptureFormatError(f"{resolved}: case_count={count} but {absent} are missing")
        case = CapturedCase(
            q=arrays[names["q"]],
            k=arrays[names["k"]],
            v=arrays[names["v"]],
            causal=bool(arrays[names["causal"]].item()) if names["causal"] in arrays else False,
            block_size=int(arrays[names["block_size"]].item())
            if names["block_size"] in arrays
            else 64,
            layer=int(arrays[names["layer"]].item()) if names["layer"] in arrays else -1,
            window=int(arrays[names["window"]].item()) if names["window"] in arrays else -1,
        )
        _validate(case, index, resolved)
        cases.append(case)
    return CaptureFile(
        path=resolved,
        format_version=version,
        activations_kind=kind,
        capture_model=capture_model,
        checksum=stored,
        metadata=json.loads(str(arrays["metadata"].item())),
        cases=tuple(cases),
    )
