"""Lane 2: dual activation streams with an fp-stream teacher capture.

The contract under test (`qk-nanoquant-alignment-lane2`): block reconstruction
advances TWO streams -- an fp stream through pristine blocks producing the
teacher target ``Y_b = B^FP_b(X_fp)``, and a compressed stream through quantized
blocks producing the student input ``X_b``. ``Y_b`` must not be a function of the
compressed stream.

This deviates from [NQ] :735's literal ``Y_b <- B^FP_b(X_b)`` deliberately; the
spec records the deviation and design.md carries the rationale. These fixtures
assert the deviation's consequence, not the paper's literal line.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from quantibbler.ptq.block_recon import (
    _forward_chunked,
    _model_input_stem,
    reconstruct_model,
)
from quantibbler.ptq.calibration import collect_stats
from quantibbler.ptq.config import PTQConfig
from quantibbler.ptq.e2e import HiddenTrunk


def _block(dim: int, seed: int) -> nn.Module:
    """One direct-child block holding a single quantizable Linear."""
    block = nn.Sequential(nn.Linear(dim, dim, bias=True, dtype=torch.float64))
    with torch.no_grad():
        g = torch.Generator().manual_seed(seed)
        block[0].weight.copy_(torch.randn(dim, dim, generator=g, dtype=torch.float64) * 0.3)
    return block


class _TinyLM(nn.Module):
    """Three quantizable blocks plus the e2e.py trunk/head anatomy.

    The anatomy is not decoration: without a reachable trunk and head this model
    cannot run whole-model next-token CE, and the experiment then cannot build the
    Phase-1 statistics book -- which is the path a real solve uses. Constructing
    the book from the pristine model (before anything is quantized) is what keeps
    the compressed stream the tuning input rather than the statistics source.
    """

    def __init__(self, dim: int = 8, vocab: int = 16, seed: int = 0) -> None:
        super().__init__()
        # nn.Linear's default init draws from the GLOBAL rng, and the explicit
        # generators below only pin the tensors they copy. Without this the model
        # differs per process and the fixture's baseline drifts, so a witness run
        # can't be reproduced even on identical bytes.
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab, dim, dtype=torch.float64)
        self.blk0, self.blk1, self.blk2 = (_block(dim, seed * 10 + i) for i in range(3))
        self.head = nn.Linear(dim, vocab, bias=False, dtype=torch.float64)
        with torch.no_grad():
            g = torch.Generator().manual_seed(seed + 99)
            self.embed.weight.copy_(torch.randn(vocab, dim, generator=g, dtype=torch.float64) * 0.5)
            self.head.weight.copy_(torch.randn(vocab, dim, generator=g, dtype=torch.float64) * 0.1)

    def _hidden(self, ids: torch.Tensor) -> torch.Tensor:
        # Same construction the driver advances: the stem discovered by
        # _model_input_stem, then each block's own output with no residual added
        # between steps (a block that wants one carries it internally).
        x = _forward_chunked(_model_input_stem(self, [ids]), ids)
        for blk in (self.blk0, self.blk1, self.blk2):
            x = _forward_chunked(blk, x)
        return x

    @property
    def trunk(self) -> nn.Module:
        return HiddenTrunk(self._hidden)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self._hidden(ids))


def _cfg() -> PTQConfig:
    return PTQConfig(rank=2, admm_max_iter=20, paper_faithful=False)


def _wide_block(dim: int, seed: int) -> nn.Sequential:
    """A block holding TWO quantizable linears -- the K=2 case TuneFP per-target needs."""
    torch.manual_seed(seed)
    with torch.no_grad():
        g = torch.Generator().manual_seed(seed + 7)
        fc1 = nn.Linear(dim, dim, bias=True, dtype=torch.float64)
        fc2 = nn.Linear(dim, dim, bias=True, dtype=torch.float64)
        fc1.weight.copy_(torch.randn(dim, dim, generator=g, dtype=torch.float64) * 0.4)
        fc1.bias.copy_(torch.randn(dim, generator=g, dtype=torch.float64) * 0.1)
        fc2.weight.copy_(torch.randn(dim, dim, generator=g, dtype=torch.float64) * 0.4)
        fc2.bias.copy_(torch.randn(dim, generator=g, dtype=torch.float64) * 0.1)
    return nn.Sequential(fc1, nn.Tanh(), fc2)


class _WideLM(nn.Module):
    """_TinyLM's anatomy with two quantizable linears per block (K=2).

    Task 4's claim is about TuneFP's per-target machinery, which only iterates when a
    block holds more than one quantizable linear. _TinyLM's one-linear blocks cannot
    reach that path. This model exists rather than thickening _TinyLM because that
    model's weights pin the witnessed reds for tasks 1 and 3 -- changing it would move
    their numbers and retire two witnesses for nothing.

    The trunk/head anatomy is copied from _TinyLM deliberately: without a reachable
    trunk and head, whole-model CE cannot run and no statistics book can be built.
    """

    def __init__(self, dim: int = 8, vocab: int = 16, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab, dim, dtype=torch.float64)
        self.blk0, self.blk1, self.blk2 = (_wide_block(dim, seed * 10 + i) for i in range(3))
        self.head = nn.Linear(dim, vocab, bias=False, dtype=torch.float64)
        with torch.no_grad():
            g = torch.Generator().manual_seed(seed + 99)
            self.embed.weight.copy_(torch.randn(vocab, dim, generator=g, dtype=torch.float64) * 0.5)
            self.head.weight.copy_(torch.randn(vocab, dim, generator=g, dtype=torch.float64) * 0.1)

    def _hidden(self, ids: torch.Tensor) -> torch.Tensor:
        x = _forward_chunked(_model_input_stem(self, [ids]), ids)
        for blk in (self.blk0, self.blk1, self.blk2):
            x = _forward_chunked(blk, x)
        return x

    @property
    def trunk(self) -> nn.Module:
        return HiddenTrunk(self._hidden)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self._hidden(ids))


def _events(model: nn.Module, teacher: nn.Module, batches, book) -> list[dict]:
    """Run reconstruction and collect every recorder event."""
    events: list[dict] = []
    reconstruct_model(
        model,
        batches,
        _cfg(),
        teacher_model=teacher,
        stats_book=book,
        recorder=lambda event, **fields: events.append({"event": event, **fields}),
    )
    return events


def _captures(events: list[dict]) -> list[dict]:
    return [e for e in events if e["event"] == "capture"]


def test_scn_stream_separation_fixture() -> None:
    """Corrupting the compressed stream must not move the teacher target Y_b.

    The corruption is applied to the STUDENT lineage only: two identical models
    are built from the same seed and one has its first block perturbed before
    reconstruction, so the quantized prefix -- and therefore ``X_b`` at block 2 --
    differs between the runs.

    The teacher is pinned explicitly and BEFORE the perturbation. ``reconstruct_
    model`` deep-copies the live model at call time when ``teacher_model=None``,
    so perturbing first and letting the harness copy would give the two runs
    different teachers; ``Y_b`` would then differ on correct bytes too, and the
    test would fail for a reason that has nothing to do with stream separation.
    """
    dim, seed = 8, 0
    batches = [
        torch.randint(0, 16, (4, 6), generator=torch.Generator().manual_seed(7)) for _ in range(2)
    ]
    clean = _TinyLM(dim, seed=seed)
    perturbed = copy.deepcopy(clean)

    # Pinned before any perturbation: one pristine teacher for both runs.
    pristine = copy.deepcopy(clean)

    # Perturb the BIAS, not the weights. A diagonal rescale of the weights is
    # absorbed by magnitude_balance -- it refits the per-row s1 AND per-column s2,
    # which are exactly a single-row scale's degrees of freedom, so a weight
    # perturbation can leave the quantized block's output unchanged and the
    # fixture would assert nothing. The bias enters additively after the
    # +-1-bitplane product, no diagonal rescale can absorb it, and block_recon's
    # write-back carries the host's own bias (the artifact has no bias field), so
    # it reaches the quantized prefix deterministically.
    with torch.no_grad():
        perturbed.blk0[0].bias.add_(0.1)

    book = collect_stats(pristine, batches)
    clean_ev = _events(clean, pristine, batches, book)
    pert_ev = _events(perturbed, pristine, batches, book)
    clean_cap, pert_cap = _captures(clean_ev), _captures(pert_ev)

    assert clean_cap, f"no capture events; events seen: {sorted({e['event'] for e in clean_ev})}"
    assert len(clean_cap) == len(pert_cap)

    for clean_c, pert_c in zip(clean_cap, pert_cap, strict=True):
        if clean_c["capture_index"] == 1:
            # The first block's input IS the calibration batch: there is nothing
            # upstream of it to corrupt, so X_b cannot move. The spec scopes this
            # requirement to "every block after the first", and so does the guard.
            continue
        # First prove the corruption reached the compressed stream. Without this
        # the separation assertion below is vacuous: two identical runs would
        # agree on Y_b for reasons that have nothing to do with stream separation.
        assert not torch.equal(clean_c["x_b"], pert_c["x_b"]), (
            f"block {clean_c['capture_index']}: X_b did not move, so the compressed "
            "stream was never actually corrupted and this fixture proves nothing"
        )
        assert torch.equal(clean_c["y_b"], pert_c["y_b"]), (
            f"block {clean_c['capture_index']}: teacher target Y_b moved when ONLY the "
            f"compressed stream was corrupted "
            f"(max|d|={float((clean_c['y_b'] - pert_c['y_b']).abs().max())}); "
            "Y_b is reading the compressed stream, not the fp stream"
        )


def _fp_activation(pristine: nn.Module, batches: list[torch.Tensor], upto: int) -> torch.Tensor:
    """The fp stream's activation at block ``upto``: the pristine prefix on a batch.

    Built from production's OWN primitives, not a reimplementation: the stem comes
    from ``_model_input_stem`` (the same call ``:1799`` makes, so integer batches
    get the Embedding composition and float batches get identity), each step is
    ``_forward_chunked`` exactly as ``:1867`` advances the running stream, and the
    driver adds no residual between steps. Re-deriving any of those here would
    make the fixture measure my reconstruction instead of the code's behaviour --
    which is precisely the class of defect this lane exists to remove.
    """
    # Concatenate the way production does it, from the same source: :1795 builds
    # ``_batches = torch.cat(batches, dim=0)`` and :1799 feeds that tensor to the
    # stem. The order and dim are copied rather than chosen, because the stream
    # production advances IS this concatenation -- and pre-concatenating with a
    # different dim would give matching shapes and diverging rows.
    x = _forward_chunked(_model_input_stem(pristine, batches), torch.cat(batches, dim=0))
    for idx in range(upto - 1):
        x = _forward_chunked(getattr(pristine, f"blk{idx}"), x)
    return x


def test_scn_teacher_target_reads_fp_stream() -> None:
    """Y_b is B^FP_b(X_fp): the pristine block applied to the FP-stream activation.

    Scope is blocks 2+. At block 1 the fp and compressed streams hold the same
    tensor, so "equals the fp reading AND differs from the compressed reading"
    could not both hold; the spec scopes the requirement the same way.

    The negative half asserts the INPUTS differ before the outputs, so
    "not equal to B^FP_b(X_b)" cannot pass vacuously on near-identical streams.
    """
    dim, seed = 8, 0
    batches = [
        torch.randint(0, 16, (4, 6), generator=torch.Generator().manual_seed(7)) for _ in range(2)
    ]
    model = _TinyLM(dim, seed=seed)
    pristine = copy.deepcopy(model)
    book = collect_stats(pristine, batches)
    caps = _captures(_events(model, pristine, batches, book))
    assert caps, "no capture events"

    for cap in caps:
        idx = cap["capture_index"]
        if idx == 1:
            continue
        pristine_block = getattr(pristine, f"blk{idx - 1}")
        x_fp = _fp_activation(pristine, batches, idx)
        x_b = cap["x_b"]

        # Inputs must differ, or the output comparison below proves nothing.
        assert not torch.equal(x_fp, x_b), (
            f"block {idx}: the fp and compressed activations are identical, so this "
            "fixture cannot distinguish which stream Y_b came from"
        )
        y_from_fp = _forward_chunked(pristine_block, x_fp)
        assert torch.equal(cap["y_b"], y_from_fp), (
            f"block {idx}: Y_b does not equal B^FP_b(X_fp) "
            f"(max|d|={float((cap['y_b'] - y_from_fp).abs().max())}); "
            "the teacher target is not the fp-stream reading"
        )


def test_scn_target1_gradient_is_float_noise_at_first_target() -> None:
    """The ruling's "vacuous" justification, MEASURED instead of asserted.

    At the first target (idx==0) the block is entirely still-FP and the teacher is
    its deepcopy -- the same function on the same input -- so the TuneFP gradient is
    float noise. That, not a flag, is why the per-target loop skips idx==0. This
    pins the *because*: if the gate flips to ``idx >= 0`` the absence check above
    fails, and if the vacuity ever stops holding this measurement fails.

    Under Lane 2's dual-stream fix the target-1 gradient is no longer ~0 (Y_b comes
    from a different input than the block's forward), so the skip's justification
    shifts from "vacuous" to "nothing to compensate yet" -- recorded for design.md.
    """
    dim, seed = 8, 3
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 11)

    # idx == 0 state: nothing factorized yet, teacher is a deepcopy of the student.
    student = _wide_block(dim, seed)
    teacher = copy.deepcopy(student)
    x_same = torch.randn(4, dim, generator=g, dtype=torch.float64)

    same_loss = ((student(x_same) - teacher(x_same)) ** 2).mean()
    same_loss.backward()
    same_gn = sum(float(p.grad.norm()) for p in student.parameters() if p.grad is not None)

    # idx == 1 state: target 1 factorized, so the block is degraded and the teacher
    # reads a DIFFERENT input -- genuine reconstruction work exists.
    student2 = _wide_block(dim, seed)
    teacher2 = copy.deepcopy(student2)
    with torch.no_grad():
        student2[0].weight.mul_(0.5)  # degrade fc1: the factorized stand-in
    x_a = torch.randn(4, dim, generator=g, dtype=torch.float64)
    x_b = torch.randn(4, dim, generator=g, dtype=torch.float64) * 1.3
    diverged_loss = ((student2(x_a) - teacher2(x_b)) ** 2).mean()
    diverged_loss.backward()
    diverged_gn = sum(float(p.grad.norm()) for p in student2.parameters() if p.grad is not None)

    print(f"DIAG_GRAD same_state={same_gn:.3e} diverged_state={diverged_gn:.3e}")
    assert same_gn < 1e-12, (
        f"first-target gradient is {same_gn:.3e}, expected float noise (<1e-12): the "
        "teacher is the student's deepcopy at idx==0, so there is nothing to compensate"
    )
    assert diverged_gn > 1e-6, (
        f"post-factorization gradient is {diverged_gn:.3e}, expected clearly nonzero: "
        "once target 1 is factorized the block is degraded and TuneFP has real work"
    )


def test_scn_target1_tunefp_vacuity_gone_by_construction() -> None:
    """Target-1 TuneFP is nonzero at block >= 2, and the block-1 skip stays.

    With a single stream the teacher target and the student input are the same
    tensor at every block, so the target-1 objective is identically zero and
    TuneFP reports a vacuous event. Under dual streams the streams have diverged
    by block 2, so a zero objective there is a construction defect, not
    nothing-to-compensate.

    Block 1 keeps its skip: the prefix is the identity, the streams coincide, and
    that is the single precise identity-prefix case the design records.
    """
    dim, seed = 8, 0
    batches = [
        torch.randint(0, 16, (4, 6), generator=torch.Generator().manual_seed(7)) for _ in range(2)
    ]
    model = _WideLM(dim, seed=seed)
    pristine = copy.deepcopy(model)
    book = collect_stats(pristine, batches)
    tune_events = [e for e in _events(model, pristine, batches, book) if e["event"] == "tune_fp"]
    print(
        "DIAG_WIDE",
        [
            (e["capture_index"], e["factorized_in_block"], e["vacuous"], e["steps"])
            for e in tune_events
        ],
    )
    assert tune_events, "no tune_fp events; the fixture is not exercising the path"

    # Block 1's skip is by OMISSION, not by a flag: the per-target loop gates both
    # the TuneFP call and its recorder on ``idx >= 1``, so no event is emitted for
    # the first target and no ``skipped`` field exists to assert on. Pinning the
    # absence is the faithful check; asserting a field the code never sets would
    # pass or fail for reasons unrelated to the target-1 behaviour.
    zero_target = [ev for ev in tune_events if ev["factorized_in_block"] == 0]
    assert not zero_target, (
        f"{len(zero_target)} tune_fp event(s) carry factorized_in_block == 0; the first "
        "target must skip TuneFP by omission, so no such event can exist"
    )

    fired_blocks = []
    for ev in tune_events:
        idx = ev["capture_index"]
        assert ev["factorized_in_block"] >= 1, (
            f"block {idx}: tune_fp emitted for a target before the first factorization"
        )
        fired_blocks.append(idx)
        # The lane's core claim concerns blocks where the streams have DIVERGED.
        # Block 1's prefix is the identity, so its streams coincide by construction
        # and its behaviour is pinned by the absence check above rather than by a
        # vacuity expectation. ``steps`` is deliberately not asserted: its semantics
        # (per-target steps? total?) are unverified, and guessing a field's meaning
        # is what produced the two setup reds already spent on this fixture.
        if idx >= 2:
            assert ev["vacuous"] is False, (
                f"block {idx}: TuneFP is vacuous. Under dual streams the streams have "
                "diverged by block 2, so a zero objective is a construction defect "
                "(stream cross-talk, or a capture reading the compressed state), not a "
                "nothing-to-compensate condition."
            )
    assert fired_blocks, "no tune_fp events; the fixture proved nothing"


def test_scn_dual_stream_advances_in_lockstep() -> None:
    """Both streams advance together, and the capture records both readings.

    `x_b` is the compressed-stream activation already recorded today. `x_fp` is the
    fp-stream reading at the same block, which Phase 2 adds. Without it the artifact
    cannot distinguish "advanced in lockstep" from "advanced at all" -- a single
    recorded stream is no evidence about the other one.

    The perturbation diverges the streams before capture (task 1's construct), so the
    lockstep claim is measured on divergent inputs rather than a model where both
    streams coincide and the assertion would prove nothing.
    """
    dim, seed = 8, 0
    batches = [
        torch.randint(0, 16, (4, 6), generator=torch.Generator().manual_seed(7)) for _ in range(2)
    ]
    clean = _TinyLM(dim, seed=seed)
    # Pinned before any perturbation: one pristine reference for both streams.
    pristine = copy.deepcopy(clean)
    perturbed = copy.deepcopy(clean)
    with torch.no_grad():
        perturbed.blk0[0].bias.add_(0.1)
    book = collect_stats(pristine, batches)

    caps = _captures(_events(perturbed, pristine, batches, book))
    assert caps, "no capture events"

    for cap in caps:
        idx = cap["capture_index"]
        assert "x_b" in cap, f"block {idx}: compressed-stream activation missing"
        assert "x_fp" in cap, (
            f"block {idx}: capture records no fp-stream activation. The lockstep "
            "advance is unverifiable from the artifact without it -- the driver must "
            "record what the pristine reference stream saw at this block, not only "
            "the compressed reading it feeds the student."
        )
