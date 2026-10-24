"""The capture path must fit the container at paper geometry.

A run at paper calibration (128 sequences x 8 = 1024 rows) OOM-killed the
container in a single 30-second window: the block's MLP expands the capture to
(1024, 2048, 3072), so ``act(gate(x)) * up(x)`` needs two 12 GiB intermediates
live at bf16. dmesg recorded anon-rss 33,480,244kB -- 99.8% of the 32 GiB cap.
Chunking bounds that peak by chunk size instead of by capture geometry, which is
what makes it survive rank/seq/calib growth rather than tuned to today's numbers.
fp32 doubles the intermediates, so this is required on every device, not a CPU
workaround.

Two properties are gated, and only two:

1. chunked output equals unchunked output -- within floating-point rounding, NOT
   bit-for-bit. BLAS picks its reduction order from the batch shape, exactly as
   it does from the thread count the suite pins; asserting bit-identity would
   encode a false requirement and fail for the wrong reason.
2. the batch a module is actually handed never exceeds ``chunk_rows``, which is
   the property that bounds the peak. Call counts are asserted alongside so a
   "chunking" that quietly forwards the whole tensor cannot pass.
"""

from __future__ import annotations

import torch
from torch import nn

from quantibbler.ptq.block_recon import CAPTURE_CHUNK_ROWS, _forward_chunked
from quantibbler.ptq.calibration import collect_stats


class _Recorder(nn.Module):
    """Row-wise module that records the batch size it was handed."""

    def __init__(self) -> None:
        super().__init__()
        self.batches: list[int] = []

    def forward(self, x):
        self.batches.append(int(x.shape[0]))
        return x * 2.0


def _mlp(dtype):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16)).to(dtype)


def test_chunked_matches_unchunked_within_rounding():
    for dtype in (torch.float64, torch.float32):
        m = _mlp(dtype)
        x = torch.randn(40, 16, dtype=dtype)
        torch.testing.assert_close(_forward_chunked(m, x, 8), m(x), rtol=1e-5, atol=1e-6)


def test_module_never_sees_more_than_chunk_rows():
    """The peak-bounding property, at the real capture row count."""
    m = _Recorder()
    x = torch.randn(1024, 8, dtype=torch.float64)
    y = _forward_chunked(m, x, 64)

    assert max(m.batches) <= 64, f"a chunk handed the module {max(m.batches)} rows"
    assert m.batches == [64] * 16, m.batches[:5]
    torch.testing.assert_close(y, x * 2.0, rtol=0, atol=0)


def test_no_op_at_or_below_the_chunk_and_on_bad_chunk():
    m = _Recorder()
    x = torch.randn(CAPTURE_CHUNK_ROWS, 8, dtype=torch.float64)
    _forward_chunked(m, x)
    assert m.batches == [CAPTURE_CHUNK_ROWS], "a small capture must make one call"

    m2 = _Recorder()
    small = torch.randn(32, 8, dtype=torch.float64)
    for bad in (0, -1, None):
        _forward_chunked(m2, small, bad)
    assert m2.batches == [32, 32, 32], "a nonsense chunk must fall back, not split"


def test_capture_forward_and_stats_backward_chunking_compose():
    """The composed path: chunked capture forward over the same data whose
    backward the stats pass chunks separately. Both chunking sites run at once
    here; a chunking interaction cannot hide between them."""
    m = _mlp(torch.float64)
    x = torch.randn(32, 16, dtype=torch.float64)
    batches = [x[i : i + 4] for i in range(0, 32, 4)]

    torch.testing.assert_close(_forward_chunked(m, x, 8), m(x), rtol=1e-5, atol=1e-6)

    # Fixed caller-supplied cotangent, identical object both calls, shaped to
    # ONE batch (collect_stats reshapes it against every batch's output):
    # parity is about chunk re-association, measured on a loss-bearing gradient.
    with torch.no_grad():
        probe = m(batches[0])
    cot = torch.randn(probe.shape, generator=torch.Generator().manual_seed(9), dtype=probe.dtype)
    ref = collect_stats(m, batches, cotangent=cot)
    chunked = collect_stats(m, batches, chunk_rows=4, cotangent=cot)
    assert ref.layer_names == chunked.layer_names
    for a, b in zip(ref, chunked, strict=True):
        torch.testing.assert_close(
            torch.from_numpy(a.d_in_raw), torch.from_numpy(b.d_in_raw), rtol=1e-6, atol=1e-9
        )
        torch.testing.assert_close(
            torch.from_numpy(a.d_out_raw), torch.from_numpy(b.d_out_raw), rtol=1e-6, atol=1e-9
        )
