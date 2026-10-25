#!/usr/bin/env python
"""Regenerate a model directory's `attention_gate.json` (change qk-sage-attention).

The `gpu:sage` server face is opt-in only for models whose SHIPPED gate record
passes ([SA] §4.3 metrics vs the exact face). This tool is the record's
producer, so the face's enabling evidence is auditable rather than asserted:

    uv run python tools/gen_attention_gate_record.py --model-dir <model dir> \
        --activations crates/quantibbler-kernels/tests/fixtures/attention_sage.json

It re-derives, from the model directory and its activation battery, exactly the
numbers the server's startup gate reads:

- per captured attention call (case), the [SA] §4.3 accuracy metrics —
  Cosine Sim, Relative L1, RMSE, verbatim from §4.3's formulas — of the
  quantized reference (`quantibbler.kernels.attention_sage`) against the exact
  tiled face (`quantibbler.kernels.attention_tiled`), and their worst over the
  battery;
- the clause set the record was generated under, so the record can never be
  read as having passed a LOOSER gate than the server enforces.

Two battery formats, two evidence classes
-----------------------------------------
`--activations` accepts either, and the record's `activations_kind` names the
one it read:

- **`.json`** — the committed device-parity *fixture battery*
  (`crates/quantibbler-kernels/tests/fixtures/attention_sage.json`): f32 q/k/v
  drawn i.i.d., each record carrying the reference's own metrics. The record
  says `"activations_kind": "fixture-battery"`, the development class, and every
  case's recomputed metrics are cross-checked against the metrics
  `tools/gen_attention_sage_fixtures.py` stored (two independent computations of
  the same quantity — a mismatch aborts).
- **`.npz`** — a captured *real-activation* record written by
  `tools/capture_attention_activations.py` (versioned, checksummed f32; format
  owned by `tools/attention_activations.py`). The record carries that file's
  `activations_kind`, which names the evidence class AND the model captured from
  (`"real-activation:Qwen/Qwen3-0.6B"`). There is no second stored computation
  to cross-check against here — a capture stores the operands, not a reference's
  verdict — which is why the file is checksummed: the operands cannot be edited
  without the reader refusing the file.

**Large-file gate.** The real-activation capture this tool consumes is
deliberately outside the repo's 500 KB `check-added-large-files` gate — one
seq-256 case at 16 heads / 128 dim is ~4 MB of f32 operands, so a real battery
is orders past it, and the exemption is a decision rather than an accident. The
exemption is named in the commit that lands this capture path; the capture
itself is minted by a real-model run (`tools/capture_attention_activations.py`)
and is not committed with the tool.

Honest inputs, named rather than papered over. The activation battery this
change ships is the committed device-parity battery (fixture inputs, not a
captured forward pass of a real model): task 4.2's capture is a real-model run
held for an explicit @user GO, so the shipped record carries
`"activations_kind": "fixture-battery"` and `"ppl": null` until it exists. The
server gate names the missing PPL clause in its startup log and admits the face
on the metrics alone; it never sees a PPL number this tree cannot measure.

`--check` recomputes the metrics and compares them with the record already on
disk (exit 1 on drift), which is how "the record regenerates" is verified
without writing: run it after any change to the activation battery or to the
reference, and regenerate the record with the same command.

Drift guard: every JSON case's recomputed metrics are also compared against the
metrics STORED in the fixture by `tools/gen_attention_sage_fixtures.py` — two
independent computations of the same quantity — and a mismatch aborts.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from attention_activations import (  # noqa: E402
    FIXTURE_BATTERY_KIND,
    CaptureFormatError,
    read_capture,
)

from quantibbler.kernels.attention_sage import attention_sage  # noqa: E402
from quantibbler.kernels.attention_tiled import attention_tiled  # noqa: E402

RECORD_NAME = "attention_gate.json"
SCHEMA = "quantibbler.attention_gate/1"
DEFAULT_ACTIVATIONS = "crates/quantibbler-kernels/tests/fixtures/attention_sage.json"

# The gate's clauses ([SA] §4.3 class). These are the
# SERVER's constants (`crates/quantibbler-server/src/model/mod.rs`) recorded
# here so the record states the gate it was generated under; the server refuses
# a record that declares anything else.
COSINE_MIN = 0.999
RELATIVE_L1_MAX = 0.02

# Recomputation vs the fixture's own stored metrics: same quantities, two
# independent code paths, so they must agree to f64 round-off.
_FIXTURE_AGREEMENT_TOL = 1e-9


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """[SA] §4.3 accuracy metrics, verbatim (flatten, then the paper's formulas)."""
    a = np.asarray(reference, dtype=np.float64).reshape(-1)
    b = np.asarray(candidate, dtype=np.float64).reshape(-1)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    rel_l1 = float(np.sum(np.abs(a - b)) / np.sum(np.abs(a)))
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    return {"cosine": cos, "relative_l1": rel_l1, "rmse": rmse}


def _close(a: float, b: float, tol: float = _FIXTURE_AGREEMENT_TOL) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _verdict(value: float, lo: float, hi: float) -> str:
    return "PASS" if lo <= value <= hi else "FAIL"


def _model_id(model_dir: Path) -> str:
    manifest = model_dir / "manifest.toml"
    with manifest.open("rb") as fh:
        parsed = tomllib.load(fh)
    model_id = parsed.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise SystemExit(f"{manifest}: no model_id")
    return model_id


def _code_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def _case_metrics(case: dict) -> dict[str, float]:
    """One case: quantized reference vs the exact tiled face."""
    q = np.asarray(case["q"], dtype=np.float64)
    k = np.asarray(case["k"], dtype=np.float64)
    v = np.asarray(case["v"], dtype=np.float64)
    causal = bool(case["causal"])
    exact = attention_tiled(q, k, v, causal=causal)
    quantized = attention_sage(q, k, v, causal=causal, block_size=int(case["block_size"]))
    return metrics(exact, quantized)


def _json_cases(activations: Path) -> list[dict]:
    """The committed JSON fixture battery: flat inputs plus a shape record."""
    records = json.loads(activations.read_text())
    if not isinstance(records, list) or not records:
        raise SystemExit(f"{activations}: expected a non-empty battery")
    cases: list[dict] = []
    for index, record in enumerate(records):
        shape = record["shape"]
        inputs = record["inputs"]
        cases.append(
            {
                "index": index,
                "q": np.asarray(inputs["query"], dtype=np.float64).reshape(
                    shape["batch"], shape["n_heads"], shape["seq"], shape["head_dim"]
                ),
                "k": np.asarray(inputs["key"], dtype=np.float64).reshape(
                    shape["batch"], shape["n_kv_heads"], shape["seq"], shape["head_dim"]
                ),
                "v": np.asarray(inputs["value"], dtype=np.float64).reshape(
                    shape["batch"], shape["n_kv_heads"], shape["seq"], shape["val_dim"]
                ),
                "causal": bool(record["causal"]),
                "block_size": int(record["block_size"]),
                "stored": record.get("reference_metrics", {}),
            }
        )
    return cases


def _npz_cases(activations: Path) -> tuple[list[dict], str]:
    """A captured real-activation record: the versioned, checksummed f32 npz.

    The reader refuses a wrong format version or a checksum mismatch before it
    hands back any case, so a damaged or edited capture cannot be measured.
    """
    capture = read_capture(activations)
    cases = [
        {
            "index": index,
            "q": np.asarray(case.q, dtype=np.float64),
            "k": np.asarray(case.k, dtype=np.float64),
            "v": np.asarray(case.v, dtype=np.float64),
            "causal": bool(case.causal),
            "block_size": int(case.block_size),
            "stored": {},
        }
        for index, case in enumerate(capture.cases)
    ]
    return cases, capture.activations_kind


def load_cases(activations: Path) -> tuple[list[dict], str]:
    """Load a battery and the evidence class its record must name.

    Returns ``(cases, activations_kind)``: the JSON path's reserved
    ``fixture-battery`` class, or the capture file's own, which names both the
    evidence class and the model it was captured from.
    """
    if activations.suffix == ".npz":
        try:
            return _npz_cases(activations)
        except CaptureFormatError as exc:
            raise SystemExit(str(exc)) from exc
    return _json_cases(activations), FIXTURE_BATTERY_KIND


def build_record(
    model_dir: Path, activations: Path, activations_source: str, ppl: dict | None
) -> dict:
    cases, activations_kind = load_cases(activations)

    calls: list[dict] = []
    for case in cases:
        measured = _case_metrics(case)
        # The JSON fixture battery stores the reference's own metrics; an npz
        # capture stores operands only (its checksum, not a second computation,
        # is what makes its contents trustworthy). Cross-check where they exist.
        stored = case["stored"]
        for name in ("cosine", "relative_l1", "rmse"):
            if name in stored and not _close(measured[name], float(stored[name])):
                raise SystemExit(
                    f"{activations}: case {case['index']} {name} recomputed as "
                    f"{measured[name]!r} but the fixture stores {stored[name]!r} — the fixture "
                    "and this tool disagree; regenerate the fixture or fix the tool before "
                    "minting a record"
                )
        calls.append(
            {
                "case": case["index"],
                "block_size": int(case["block_size"]),
                "cosine": measured["cosine"],
                "relative_l1": measured["relative_l1"],
                "rmse": measured["rmse"],
            }
        )

    worst = {
        "cosine": min(call["cosine"] for call in calls),
        "relative_l1": max(call["relative_l1"] for call in calls),
        "rmse": max(call["rmse"] for call in calls),
    }
    return {
        "schema": SCHEMA,
        "model_id": _model_id(model_dir),
        "activations_source": activations_source,
        # Honest provenance: the evidence class this record rests on, named
        # rather than implied — `"fixture-battery"` for the committed synthetic
        # battery (task 4.2's real-model capture is held for an explicit @user
        # GO), or the capture file's own `real-activation:<source>`. Audit
        # field; the gate reads the metrics, and the evidence-class distinction
        # the planner ratified keys on this value.
        "activations_kind": activations_kind,
        "generated_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_commit": _code_commit(),
        "clauses": {"cosine_min": COSINE_MIN, "relative_l1_max": RELATIVE_L1_MAX},
        "calls": calls,
        "worst": worst,
        "ppl": ppl,
    }


def _metric_view(record: dict) -> dict:
    """The parts `--check` compares: everything the metrics determine."""
    return {
        "model_id": record["model_id"],
        "activations_source": record["activations_source"],
        "activations_kind": record.get("activations_kind"),
        "clauses": record["clauses"],
        "calls": record["calls"],
        "worst": record["worst"],
        "ppl": record["ppl"],
    }


def _compare(shipped: dict, recomputed: dict, path: Path) -> list[str]:
    drift: list[str] = []
    if shipped.get("schema") != SCHEMA:
        drift.append(f"schema is {shipped.get('schema')!r}, expected {SCHEMA!r}")
    shipped_view, fresh_view = _metric_view(shipped), _metric_view(recomputed)
    for key in ("model_id", "activations_source", "activations_kind", "clauses", "ppl"):
        if shipped_view[key] != fresh_view[key]:
            drift.append(f"{key}: shipped {shipped_view[key]!r} != recomputed {fresh_view[key]!r}")
    if len(shipped_view["calls"]) != len(fresh_view["calls"]):
        drift.append(
            f"calls: shipped {len(shipped_view['calls'])} != recomputed {len(fresh_view['calls'])}"
        )
    else:
        for old, new in zip(shipped_view["calls"], fresh_view["calls"], strict=False):
            case = old["case"]
            for key in ("case", "block_size"):
                if old[key] != new[key]:
                    drift.append(f"calls[{case}].{key}: {old[key]!r} != {new[key]!r}")
            for key in ("cosine", "relative_l1", "rmse"):
                if not _close(float(old[key]), float(new[key])):
                    drift.append(
                        f"calls[{case}].{key}: shipped {old[key]!r} != recomputed {new[key]!r}"
                    )
    for key in ("cosine", "relative_l1", "rmse"):
        if not _close(float(shipped_view["worst"][key]), float(fresh_view["worst"][key])):
            drift.append(
                f"worst.{key}: {shipped_view['worst'][key]!r} != {fresh_view['worst'][key]!r}"
            )
    if drift:
        drift.insert(0, f"{path} does not match the tree:")
    return drift


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", required=True, type=Path, help="qpk1 model directory")
    parser.add_argument(
        "--activations",
        default=DEFAULT_ACTIVATIONS,
        help=(
            "activation battery: the committed JSON fixture battery, or a captured "
            f"real-activation .npz (repo-relative by default: {DEFAULT_ACTIVATIONS})"
        ),
    )
    parser.add_argument(
        "--ppl-eval",
        type=Path,
        default=None,
        help='measured eval-set PPL report: {"eval_set": str, "delta": f, "bound": f}',
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute and compare against the record on disk; write nothing",
    )
    args = parser.parse_args(argv)

    model_dir: Path = args.model_dir
    activations_source = args.activations
    activations = Path(activations_source)
    if not activations.is_absolute():
        activations = REPO_ROOT / activations

    ppl = json.loads(args.ppl_eval.read_text()) if args.ppl_eval else None
    if ppl is not None:
        missing = {"eval_set", "delta", "bound"} - set(ppl)
        if missing:
            raise SystemExit(f"{args.ppl_eval}: missing keys {sorted(missing)}")

    record = build_record(model_dir, activations, activations_source, ppl)
    path = model_dir / RECORD_NAME

    if args.check:
        if not path.is_file():
            print(f"MISSING {path}", file=sys.stderr)
            return 1
        drift = _compare(json.loads(path.read_text()), record, path)
        if drift:
            print("\n".join(drift), file=sys.stderr)
            print(
                f"re-run: uv run python tools/gen_attention_gate_record.py --model-dir {model_dir}",
                file=sys.stderr,
            )
            return 1
        print(f"OK {path}: {len(record['calls'])} calls reproduce the shipped metrics")
        return 0

    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {path} ({path.stat().st_size} bytes, {len(record['calls'])} calls)")
    print(f"  activations_kind {record['activations_kind']}")
    print(
        f"  worst cosine {record['worst']['cosine']:.9f} "
        f"[{_verdict(record['worst']['cosine'], COSINE_MIN, 1.0)} >= {COSINE_MIN}]"
    )
    print(
        f"  worst relative L1 {record['worst']['relative_l1']:.9f} "
        f"[{_verdict(record['worst']['relative_l1'], 0.0, RELATIVE_L1_MAX)} <= {RELATIVE_L1_MAX}]"
    )
    print(
        "  eval-set PPL delta: "
        + (
            f"{record['ppl']['delta']!r} (bound {record['ppl']['bound']!r})"
            if record["ppl"]
            else "NOT MEASURED (task 4.2's real-model eval run is its only producer)"
        )
    )
    if record["ppl"] is not None and record["ppl"]["delta"] > record["ppl"]["bound"]:
        print("  the recorded PPL delta exceeds its bound: the server will refuse this face")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
