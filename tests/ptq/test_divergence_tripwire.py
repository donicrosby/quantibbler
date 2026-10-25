"""Tripwire proof: a diverged layer aborts the FULL pipeline before export
(qk-divergence-guards, task 3.3 — end-to-end scenario evidence).

Monkeypatches magnitude_balance to inflate one block-0-like layer's scales
past the band (the incident signature) and asserts the public entry point
aborts naming that layer — with no artifact bytes produced.
"""

import pytest

import quantibbler.ptq.lb_admm as lb_mod
from quantibbler.ptq.api import quantize_model
from quantibbler.ptq.block_recon import QuantizationDivergenceError
from quantibbler.ptq.config import PTQConfig
from quantibbler.ptq.e2e import TinyGateModel, build_gate_batches


def test_scn_e2e_gate_detects_divergence_before_export(monkeypatch):
    real_balance = lb_mod.magnitude_balance

    def inflated_balance(p_u, p_v, d_out_tilde, d_in_tilde, **kw):
        bal = real_balance(p_u, p_v, d_out_tilde, d_in_tilde, **kw)
        # Inflate the FIRST balanced layer the pipeline solves (a
        # block-0 stand-in): 1e6x the scales — the divergence signature.
        if not inflated_balance.done:
            inflated_balance.done = True
            bal.s1 = bal.s1 * 1e6
        return bal

    inflated_balance.done = False
    monkeypatch.setattr(lb_mod, "magnitude_balance", inflated_balance)
    # ...but block_recon imported the symbol directly; patch through the
    # caller's namespace too (both faces must see the inflated balance).
    import quantibbler.ptq.block_recon as br_mod

    monkeypatch.setattr(br_mod, "magnitude_balance", inflated_balance)

    model = TinyGateModel(seed=0)
    cfg = PTQConfig(solver_backend="numpy", seed=0, rank=4)
    with pytest.raises(QuantizationDivergenceError) as ei:
        quantize_model(model, build_gate_batches(seed=0), cfg)
    msg = str(ei.value)
    assert "outside band" in msg
    # The inflated layer is named — no silent pass-through, no partial export.
    assert "layer" in msg


def test_scn_e2e_gate_detects_divergence_artifact_bytes_absent(monkeypatch):
    """Same tripwire from the serialize side: qm.serialize() never happens
    because quantize_model raised — assert serialize-on-success is the only
    path (i.e. the exception fires inside quantize_model, not later)."""
    real_balance = lb_mod.magnitude_balance

    def inflated_balance(p_u, p_v, d_out_tilde, d_in_tilde, **kw):
        bal = real_balance(p_u, p_v, d_out_tilde, d_in_tilde, **kw)
        if not inflated_balance.done:
            inflated_balance.done = True
            bal.s2 = bal.s2 * 1e9
        return bal

    inflated_balance.done = False
    import quantibbler.ptq.block_recon as br_mod

    monkeypatch.setattr(br_mod, "magnitude_balance", inflated_balance)

    model = TinyGateModel(seed=0)
    cfg = PTQConfig(solver_backend="numpy", seed=0, rank=4)
    with pytest.raises(QuantizationDivergenceError):
        quantize_model(model, build_gate_batches(seed=0), cfg)
    # Reaching this point means the abort happened inside quantize_model,
    # before any QuantizedModel (let alone serialized bytes) existed.
