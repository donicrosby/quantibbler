"""Opt-in device auditor: the fence for "CPU tensor meets device-resident module".

A grep cannot see this class. The boundary is crossed at call time, and the
offending tensor is often produced elsewhere -- a NumPy round-trip, a tokenizer
batch, an accumulator -- so a source-shape check misses exactly the sites that
bite (multi-line calls, aliased names). This wraps ``nn.Module.__call__`` and
raises on any tensor argument whose device disagrees with the module's own
parameters, naming the call site. One smoke run then enumerates every remaining
boundary in minutes instead of one fire per site.

Enable with ``QUANTIBBLER_DEVICE_AUDIT=1``. The campaign fire runs with it off:
the smoke's clean pass over the same code is the proof, and a long run must not
die to its own instrument.
"""

from __future__ import annotations

import os
import traceback

import torch

_INSTALLED = False


def install_device_auditor() -> bool:
    """Wrap nn.Module.__call__ to fail closed on device disagreement. Idempotent."""
    global _INSTALLED
    if _INSTALLED or os.environ.get("QUANTIBBLER_DEVICE_AUDIT") != "1":
        return False
    original = torch.nn.Module.__call__

    def audited(self, *args, **kwargs):  # noqa: ANN001
        # Explicit None checks, never truthiness: a Parameter's truth value is
        # ambiguous, and `next(...)` without a default raises StopIteration on a
        # parameterless module -- both have already cost a run today.
        ref = next(iter(self.parameters()), None)
        if ref is None:
            ref = next(iter(self.buffers()), None)
        if ref is not None:
            want = ref.device

            def check(value, where):  # noqa: ANN001
                if torch.is_tensor(value) and value.device != want:
                    frame = traceback.extract_stack()[-3]
                    raise RuntimeError(
                        f"device auditor: {where} is on {value.device} but "
                        f"{type(self).__name__} parameters are on {want} -- "
                        f"{frame.filename}:{frame.lineno}"
                    )

            for i, value in enumerate(args):
                check(value, f"arg[{i}]")
            for key, value in kwargs.items():
                check(value, f"kwarg {key}")
        return original(self, *args, **kwargs)

    torch.nn.Module.__call__ = audited
    _INSTALLED = True
    return True
