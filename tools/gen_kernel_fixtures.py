#!/usr/bin/env python
"""Generate kernel parity fixtures for the Rust-side cargo tests.

Emits one JSON file per scenario family into
``crates/quantibbler-kernels/tests/fixtures/``. Every record carries the
scenario ID, the inputs, the NumPy-reference outputs (the executable
specification), and the dtype/width metadata the Rust tests need.

Records carry a ``faces`` object; the ``numpy-reference`` face is always
present. When ``quantibbler_kernels`` is importable (i.e. after
``uv run maturin develop``), the generator additionally records the PyO3
face's outputs under ``faces["pyo3"]`` so cargo-side face-parity tests can
compare the rlib face against both records.

Regenerate with ``uv run python tools/gen_kernel_fixtures.py``; fixture
drift is a reviewable diff, never a manual copy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import quantibbler.kernels as k

FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "crates" / "quantibbler-kernels" / "tests" / "fixtures"
)


def _factor(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=shape)


def _pyo3_available() -> bool:
    return k.HAVE_KERNELS and all(
        hasattr(k._kernels, n)
        for n in (
            "pack_bits",
            "unpack_bits",
            "sign_estimate",
            "packed_matvec",
            "packed_matmul",
            "joint_itq",
        )
    )


def _record(
    scenario: str,
    inputs: dict,
    numpy_outputs: dict,
    dtypes: dict,
    pyo3_outputs: dict | None = None,
) -> dict:
    rec = {
        "scenario": scenario,
        "inputs": inputs,
        "faces": {"numpy-reference": numpy_outputs},
        "dtypes": dtypes,
    }
    if pyo3_outputs is not None:
        rec["faces"]["pyo3"] = pyo3_outputs
    return rec


def _pack_family(pyo3: bool) -> list[dict]:
    out = []
    for scenario, shape, seed in [
        ("pack_roundtrip_small", (5, 11), 101),
        ("pack_roundtrip_exact8", (4, 8), 102),
        ("pack_roundtrip_padded", (3, 9), 103),
    ]:
        factor = _factor(shape, seed)
        packed = k.pack_bits(factor, backend="numpy-reference")
        unpacked = k.unpack_bits(packed, shape[1], "int8", backend="numpy-reference")
        pyo3_out = None
        if pyo3:
            pyo3_out = {
                "packed": np.asarray(k.pack_bits(factor, backend="rust")).tolist(),
                "unpacked": np.asarray(
                    k.unpack_bits(packed, shape[1], "int8", backend="rust")
                ).tolist(),
            }
        out.append(
            _record(
                scenario,
                {"factor": factor.tolist()},
                {"packed": packed.tolist(), "unpacked": unpacked.tolist()},
                {"factor": "int8", "packed": "uint8", "unpacked": "int8", "n_bits": shape[1]},
                pyo3_out,
            )
        )
    # pack_layout_bitorder: single +1 at each of 16 positions
    rows = []
    expected = []
    for pos in range(16):
        row = np.full((1, 16), -1, dtype=np.int8)
        row[0, pos] = 1
        rows.append(row)
        exp = np.zeros((1, 2), dtype=np.uint8)
        exp[0, pos // 8] = 1 << (pos % 8)
        expected.append(exp)
    stacked = np.concatenate(rows, axis=0)
    packed = k.pack_bits(stacked, backend="numpy-reference")
    pyo3_out = None
    if pyo3:
        pyo3_out = {"packed": np.asarray(k.pack_bits(stacked, backend="rust")).tolist()}
    out.append(
        _record(
            "pack_layout_bitorder",
            {"factor": stacked.tolist()},
            {
                "packed": packed.tolist(),
                "expected_bytes": np.concatenate(expected, axis=0).tolist(),
            },
            {"factor": "int8", "packed": "uint8", "n_bits": 16},
            pyo3_out,
        )
    )
    # pack_roundtrip_empty
    for shape in [(2, 0), (0, 7)]:
        factor = np.zeros(shape, dtype=np.int8)
        packed = k.pack_bits(factor, backend="numpy-reference")
        pyo3_out = None
        if pyo3:
            pyo3_out = {"packed": np.asarray(k.pack_bits(factor, backend="rust")).tolist()}
        out.append(
            _record(
                "pack_roundtrip_empty",
                {"factor": factor.tolist(), "shape": list(shape)},
                {"packed": packed.tolist()},
                {"factor": "int8", "packed": "uint8", "n_bits": shape[1]},
                pyo3_out,
            )
        )
    return out


def _sign_family(pyo3: bool) -> list[dict]:
    x = np.array([0.0, -0.0, 1.5, -2.5, 0.0, -1e-9, 1e-300, -1e300, 2.5], dtype=np.float64)
    out_np = k.sign_estimate(x, backend="numpy-reference")
    pyo3_out = None
    if pyo3:
        pyo3_out = {"signs": np.asarray(k.sign_estimate(x, backend="rust")).tolist()}
    zeros = k.sign_estimate(np.zeros((2, 3)), backend="numpy-reference")
    return [
        _record(
            "sign_zeros",
            {"x": x.tolist(), "all_zeros": np.zeros((2, 3)).tolist()},
            {"signs": out_np.tolist(), "all_zeros_signs": zeros.tolist()},
            {"x": "float64", "signs": "int8"},
            pyo3_out,
        ),
        _record(
            "sign_extremes",
            {"x": x.tolist()},
            {"signs": out_np.tolist()},
            {"x": "float64", "signs": "int8"},
            None if not pyo3 else pyo3_out,
        ),
    ]


def _matvec_family(pyo3: bool) -> list[dict]:
    factor = _factor((6, 13), 201)
    vec = np.random.default_rng(202).standard_normal(13)
    packed = k.pack_bits(factor, backend="numpy-reference")
    out = k.packed_matvec(packed, 13, vec, backend="numpy-reference")
    dirty = packed.copy()
    dirty[:, -1] |= 0b11111110
    out_dirty = k.packed_matvec(dirty, 13, vec, backend="numpy-reference")
    pyo3_out = None
    if pyo3:
        pyo3_out = {
            "result": np.asarray(k.packed_matvec(packed, 13, vec, backend="rust")).tolist(),
            "result_dirty": np.asarray(k.packed_matvec(dirty, 13, vec, backend="rust")).tolist(),
        }
    return [
        _record(
            "matvec_padded",
            {"packed": packed.tolist(), "packed_dirty": dirty.tolist(), "vec": vec.tolist()},
            {"result": out.tolist(), "result_dirty": out_dirty.tolist()},
            {"packed": "uint8", "vec": "float64", "result": "float64", "n_bits": 13},
            pyo3_out,
        ),
        _record(
            "matvec_matches_dense_reference",
            {"packed": packed.tolist(), "vec": vec.tolist()},
            {"result": out.tolist()},
            {"packed": "uint8", "vec": "float64", "result": "float64", "n_bits": 13},
            None,
        ),
    ]


def _matmul_family(pyo3: bool) -> list[dict]:
    factor = _factor((5, 12), 301)
    mat = np.random.default_rng(302).standard_normal((12, 3))
    packed = k.pack_bits(factor, backend="numpy-reference")
    out = k.packed_matmul(packed, 12, mat, backend="numpy-reference")
    pyo3_out = None
    if pyo3:
        pyo3_out = {"result": np.asarray(k.packed_matmul(packed, 12, mat, backend="rust")).tolist()}
    return [
        _record(
            "matmul_wide",
            {"packed": packed.tolist(), "mat": mat.tolist()},
            {"result": out.tolist()},
            {"packed": "uint8", "mat": "float64", "result": "float64", "n_bits": 12},
            pyo3_out,
        )
    ]


def _itq_family(pyo3: bool) -> list[dict]:
    rng = np.random.default_rng(8)
    u_hat = rng.standard_normal((7, 3))
    v_hat = rng.standard_normal((5, 3))
    out = []
    for scenario, kwargs in [
        ("itq_converged", {"max_iter": 50, "tol": 1e-12, "seed": 42}),
        ("itq_maxiter", {"max_iter": 50, "tol": 0.0, "seed": 42}),
        ("itq_deterministic_seed", {"max_iter": 50, "tol": 1e-12, "seed": 42}),
        ("itq_preserves_product", {"max_iter": 50, "tol": 1e-12, "seed": 42}),
    ]:
        u_t, v_t, n_iter, converged = k.joint_itq(u_hat, v_hat, backend="numpy-reference", **kwargs)
        pyo3_out = None
        if pyo3:
            u_r, v_r, n_r, c_r = k.joint_itq(u_hat, v_hat, backend="rust", **kwargs)
            pyo3_out = {
                "u_t": np.asarray(u_r).tolist(),
                "v_t": np.asarray(v_r).tolist(),
                "n_iter": int(n_r),
                "converged": bool(c_r),
            }
        rec = _record(
            scenario,
            {"u_hat": u_hat.tolist(), "v_hat": v_hat.tolist(), "kwargs": kwargs},
            {
                "u_t": np.asarray(u_t).tolist(),
                "v_t": np.asarray(v_t).tolist(),
                "n_iter": int(n_iter),
                "converged": bool(converged),
            },
            {"u_hat": "float64", "v_hat": "float64", "u_t": "int8", "v_t": "int8"},
            pyo3_out,
        )
        if scenario == "itq_deterministic_seed":
            u2, v2, n2, c2 = k.joint_itq(u_hat, v_hat, backend="numpy-reference", **kwargs)
            rec["faces"]["numpy-reference"]["repeat_u_t"] = np.asarray(u2).tolist()
            rec["faces"]["numpy-reference"]["repeat_v_t"] = np.asarray(v2).tolist()
            rec["faces"]["numpy-reference"]["repeat_n_iter"] = int(n2)
            if pyo3:
                ur2, vr2, nr2, cr2 = k.joint_itq(u_hat, v_hat, backend="rust", **kwargs)
                pyo3_out["repeat_u_t"] = np.asarray(ur2).tolist()
                pyo3_out["repeat_v_t"] = np.asarray(vr2).tolist()
                pyo3_out["repeat_n_iter"] = int(nr2)
        out.append(rec)
    return out


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    pyo3 = _pyo3_available()
    families = {
        "pack_bits": _pack_family(pyo3),
        "sign_estimate": _sign_family(pyo3),
        "packed_matvec": _matvec_family(pyo3),
        "packed_matmul": _matmul_family(pyo3),
        "joint_itq": _itq_family(pyo3),
    }
    for fname, records in families.items():
        path = FIXTURE_DIR / f"{fname}.json"
        path.write_text(json.dumps(records, indent=1) + "\n")
        print(f"wrote {path} ({len(records)} scenarios, pyo3 face: {pyo3})")


if __name__ == "__main__":
    main()
