"""The solver probe is a measurement, so it is pinned like one.

It must be inert when unset (a default run is unchanged), must record BOTH
cosines (they answer different questions; one number hid the pack bottleneck
on the SVD arm), and must fail loudly if it is enabled but cannot write --
a probe that cannot write its numbers has measured nothing.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from quantibbler.ptq import block_recon
from quantibbler.ptq.config import PTQConfig


def _planted():
    gen = np.random.default_rng(0)
    u = np.where(gen.random((8, 4)) < 0.5, -1.0, 1.0)
    v = np.where(gen.random((6, 4)) < 0.5, -1.0, 1.0)
    bal = SimpleNamespace(U=u, V=v, s1=np.ones(8), s2=np.ones(6))
    return u @ v.T, bal, SimpleNamespace(n_iter=7, converged=False)


def test_probe_is_inert_when_unset(tmp_path, monkeypatch):
    w, bal, res = _planted()
    path = tmp_path / "probe.jsonl"
    monkeypatch.delenv("QK_SOLVER_PROBE", raising=False)
    block_recon._record_solver_probe("planted", w, bal, res, PTQConfig(rank=4))
    assert not path.exists()


def test_probe_records_both_cosines_and_the_arm(tmp_path, monkeypatch):
    w, bal, res = _planted()
    path = tmp_path / "probe.jsonl"
    monkeypatch.setenv("QK_SOLVER_PROBE", "1")
    monkeypatch.setenv("QK_SOLVER_PROBE_PATH", str(path))
    monkeypatch.delenv("QK_RHO_END", raising=False)
    monkeypatch.delenv("_fp_w", raising=False)

    block_recon._record_solver_probe("planted", w, bal, res, PTQConfig(rank=4))
    row = json.loads(path.read_text().strip())

    # an exact rank-1 +-1 target: continuous and packed reconstructions both hit 1
    assert row["cos_raw"] == pytest.approx(1.0, abs=1e-12)
    assert row["cos_pack"] == pytest.approx(1.0, abs=1e-12)
    assert row["rel_raw"] == pytest.approx(0.0, abs=1e-12)
    assert row["wq_raw_fro"] == pytest.approx(row["w_fro"], rel=1e-12)
    assert row["n_iter"] == 7
    assert row["converged"] is False
    assert row["solver"] == PTQConfig(rank=4).solver
    # the arm must be self-describing: value AND source, so a probe row cannot
    # be mislabelled about which pass produced it
    assert row["rho_end"] == PTQConfig(rank=4).rho_end
    assert row["rho_end_source"] == "config"

    monkeypatch.setenv("QK_RHO_END", "1e2")
    block_recon._record_solver_probe("planted", w, bal, res, PTQConfig(rank=4))
    rows = [json.loads(x) for x in path.read_text().strip().splitlines()]
    assert len(rows) == 2, "one JSONL row per solve, appended"
    assert rows[-1]["rho_end"] == 1e2 and rows[-1]["rho_end_source"] == "env"


def test_probe_row_carries_the_compute_dtype(tmp_path, monkeypatch):
    """Two arms differing only in precision must not be indistinguishable.

    Same provenance-gap class as an unlabelled rho_end, and sharper here: an
    fp32 row's cos is not comparable to an fp64 row's, so a row that omits the
    dtype actively invites exactly that comparison. Stated rather than implied
    -- the fp64 default rides along so a reader never has to infer it.
    """
    w, bal, res = _planted()
    path = tmp_path / "probe.jsonl"
    monkeypatch.setenv("QK_SOLVER_PROBE", "1")
    monkeypatch.setenv("QK_SOLVER_PROBE_PATH", str(path))
    monkeypatch.delenv("QK_RHO_END", raising=False)
    monkeypatch.delenv("_fp_w", raising=False)

    block_recon._record_solver_probe("planted", w, bal, res, PTQConfig(rank=4))
    block_recon._record_solver_probe(
        "planted", w, bal, res, PTQConfig(rank=4, compute_dtype="float32")
    )
    rows = [json.loads(x) for x in path.read_text().strip().splitlines()]
    assert [r["compute_dtype"] for r in rows] == ["float64", "float32"]


def test_probe_raises_rather_than_measuring_nothing(tmp_path, monkeypatch):
    w, bal, res = _planted()
    monkeypatch.setenv("QK_SOLVER_PROBE", "1")
    monkeypatch.setenv("QK_SOLVER_PROBE_PATH", str(tmp_path / "nope" / "probe.jsonl"))
    with pytest.raises(RuntimeError, match="cannot write"):
        block_recon._record_solver_probe("planted", w, bal, res, PTQConfig(rank=4))
