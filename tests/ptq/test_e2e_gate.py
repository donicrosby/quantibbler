"""Tasks 7.1: end-to-end tiny-model gate (scenarios e2e_tiny_gate_passes,
e2e_bpw_and_parity, e2e_gate_detects_corruption).

Seeded tiny transformer (d=32, wide MLP 128, 2 blocks), 8 calibration
samples x 64 tokens (OQ-4 proposal), quantize -> serialize qpk1 -> reload
-> forward on the probe batch. Assertions:

(a) max |logit_fp - logit_q| <= 1.0 absolute (A-8; recorded threshold,
    revisit clause belongs to the orchestrator, not this test);
(b) effective BPW < 1.0 computed from serialized bytes via the
    qk-representation accounting (qpk1.bpw_from_bytes over the aggregate
    artifact; never parameter counts);
(c) pack/unpack parity — bit identity through the qk-kernels dispatch
    (NumPy reference and, when present, the Rust extension);
(d) a deliberately corrupted artifact FAILS the gate (corruption
    self-test, A-8's wrong-choice signature).

Test-speed stage overrides live in e2e.gate_config (A-10, documented);
the paper defaults remain the PTQConfig defaults.
"""

import numpy as np
import pytest
import torch

from quantibbler import qpk1
from quantibbler.ptq.api import QuantizedModel, quantize_model
from quantibbler.ptq.e2e import (
    TinyGateModel,
    build_gate_batches,
    gate_config,
    run_e2e_gate,
)


def _quantize_gate_fixture():
    """One deterministic quantize -> serialize -> reload round for the
    fixture assertions (identical to run_e2e_gate's happy path)."""
    cfg = gate_config(seed=0)
    model = TinyGateModel(seed=cfg.seed)
    batches = build_gate_batches(seed=cfg.seed)
    probe = batches[0]
    with torch.no_grad():
        fp_logits = model(probe)
    qm = quantize_model(model, batches, cfg)
    blobs = qm.serialize()
    reloaded = QuantizedModel.deserialize(blobs, config=cfg)
    host = TinyGateModel(seed=cfg.seed)
    reloaded.attach_to(host)
    with torch.no_grad():
        q_logits = reloaded.forward(probe)
    return cfg, qm, blobs, reloaded, fp_logits, q_logits


def test_scn_e2e_tiny_gate():
    """e2e_tiny_gate_passes: the full gate on the honest artifact — max
    absolute logit delta vs the fp teacher within the recorded A-8 bound."""
    result = run_e2e_gate()
    assert result["bits_identical"] is True
    assert result["logit_delta"] <= 1.0, (
        f"max logit delta {result['logit_delta']} exceeds the recorded A-8 "
        "bound 1.0 — if this reproduces on a correct implementation, A-8's "
        "revisit clause is an orchestrator decision, not a test edit"
    )


def test_scn_e2e_bpw_and_parity():
    """e2e_bpw_and_parity: byte-derived BPW < 1.0 through the qk-
    representation accounting, and pack/unpack bit identity through the
    qk-kernels dispatch (both backends when present)."""
    cfg, qm, blobs, reloaded, fp_logits, q_logits = _quantize_gate_fixture()

    # Byte accounting from the serialized container only (D5): aggregate
    # BPW via qpk1.bpw_from_bytes over the whole artifact — never from
    # parameter counts.
    total_bytes = sum(len(b) for b in blobs.values())
    total_entries = sum(st.Ub.shape[0] * st.Vb.shape[0] for st in reloaded.layers.values())
    bpw = qpk1.bpw_from_bytes(total_bytes, 1, total_entries)
    assert bpw == reloaded.effective_bpw()
    assert bpw < 1.0, f"effective BPW {bpw} from {total_bytes} serialized bytes is not < 1.0"

    # Serialize(reloaded) == original streams: byte-exact round trip.
    assert reloaded.serialize() == blobs

    # Pack/unpack parity through the qk-kernels dispatch: the artifact's
    # +/-1 factors are packed and re-unpacked bit-identically on every
    # backend present, and NumPy-reference bits equal Rust bits.
    from quantibbler import kernels

    for _name, st in reloaded.layers.items():
        for backend in ("numpy-reference", "rust"):
            if backend == "rust" and not kernels.HAVE_KERNELS:
                continue
            packed_u = kernels.pack_bits(st.Ub, backend=backend)
            packed_v = kernels.pack_bits(st.Vb, backend=backend)
            assert np.array_equal(
                kernels.unpack_bits(packed_u, st.Ub.shape[1], "int8", backend=backend), st.Ub
            )
            assert np.array_equal(
                kernels.unpack_bits(packed_v, st.Vb.shape[1], "int8", backend=backend), st.Vb
            )
    if kernels.HAVE_KERNELS:
        for st in reloaded.layers.values():
            assert np.array_equal(
                kernels.pack_bits(st.Ub, backend="numpy-reference"),
                kernels.pack_bits(st.Ub, backend="rust"),
            )
            assert np.array_equal(
                kernels.pack_bits(st.Vb, backend="numpy-reference"),
                kernels.pack_bits(st.Vb, backend="rust"),
            )


def test_scn_e2e_gate_detects_corruption():
    """e2e_gate_detects_corruption: the deliberately corrupted artifact
    FAILS the gate (A-8's wrong-choice signature — the gate is not vacuous)."""
    with pytest.raises(AssertionError):
        run_e2e_gate(flip_one_bit=True)
