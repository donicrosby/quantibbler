"""Device-mismatch auditor (qk-ptq: "The pipeline SHALL place every stage on the
declared device without host round-trips", scenario device_auditor_catches_mismatch).

A grep cannot see this class: the boundary is crossed at call time and the
offending tensor is often produced elsewhere (a NumPy round-trip, a tokenizer
batch, an accumulator), so the auditor wraps ``nn.Module.__call__`` instead. The
driver's ``--device-audit`` flag is what turns it on for a debugging run.
"""

import importlib

import pytest
import torch
from torch import nn


@pytest.fixture
def fresh_auditor(monkeypatch):
    """A clean module per test: the install latch is process-global."""
    monkeypatch.delenv("QUANTIBBLER_DEVICE_AUDIT", raising=False)
    mod = importlib.reload(importlib.import_module("quantibbler.ptq.device_audit"))
    original_call = nn.Module.__call__
    yield mod
    nn.Module.__call__ = original_call
    mod._INSTALLED = False


def test_scn_device_auditor_catches_mismatch(fresh_auditor, monkeypatch):
    """A mismatched tensor argument raises, naming module + both devices."""
    monkeypatch.setenv("QUANTIBBLER_DEVICE_AUDIT", "1")
    assert fresh_auditor.install_device_auditor() is True

    module = nn.Linear(4, 4, dtype=torch.float64)  # CPU module
    # A 'cuda'-labelled tensor is not constructible on this box, so the check is
    # exercised with the mismatch the auditor actually inspects: devices differ.
    with pytest.raises(RuntimeError, match="device auditor"):
        module(torch.zeros(1, 4, dtype=torch.float64, device="meta"))


def test_auditor_is_off_by_default(fresh_auditor, monkeypatch):
    """Unset env: no wrapper is installed and calls behave normally."""
    monkeypatch.delenv("QUANTIBBLER_DEVICE_AUDIT", raising=False)
    assert fresh_auditor.install_device_auditor() is False
    module = nn.Linear(4, 4, dtype=torch.float64)
    assert module(torch.zeros(1, 4, dtype=torch.float64)).shape == (1, 4)
