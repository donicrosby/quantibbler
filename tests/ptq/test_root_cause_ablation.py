"""qk-root-cause-ablation: the three-knob measurement harness.

RED-first against the frozen pin list (D-1..D-6 after the D-1 amendment):

* the branch contract of the ONE helper -- binary/grad -> passthrough
  (gradient must flow: STE needs d/dU = 1), binary/no_grad -> literal
  ``torch.sign`` bitwise (``torch.equal`` on the f64 tier), fp/either -> the
  latent itself, regime chosen by an explicit ``torch.is_grad_enabled()`` test;
* one definition, three expression sites, no ``torch.sign`` on latents outside
  it;
* REACHABILITY observed twice -- the live STE path consults the helper, and
  ``_set_linear``'s written weight is bit-equal to the helper's no_grad output
  (a pin satisfied by a dead call site proves nothing);
* the negative row: a helper that returns literal sign in the GRAD-ENABLED
  branch must red, so the dead-STE catastrophe is a named, tested failure;
* the scope filter, the zero-refinement ablation, and the fp export boundary.
"""

import ast
import pathlib
import re

import numpy as np
import pytest
import torch
from torch import nn

from quantibbler.ptq import _factorized, block_recon, calibration
from quantibbler.ptq.block_recon import FactorizedLinear, _set_linear, reconstruct_block
from quantibbler.ptq.checkpoints import config_fingerprint
from quantibbler.ptq.config import PTQConfig

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
#: The D-1 cluster -- ``_factor_terms`` and the three expression sites that must
#: consult it -- lives in ``_factorized``. The pin parses the module that OWNS
#: the helper, and the three sites must stay in
#: it: separating them is what the pin exists to forbid.
MODULE_PATH = REPO_ROOT / "src" / "quantibbler" / "ptq" / "_factorized.py"
MODULE_SRC = MODULE_PATH.read_text()

_HELPER = "_factor_terms"
_SITES = ("forward", "frozen_forward", "_set_linear")


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _fac(d: int = 6, r: int = 3, seed: int = 0, **kwargs) -> FactorizedLinear:
    gen = torch.Generator().manual_seed(seed)
    u = torch.randn(d, r, generator=gen, dtype=torch.float64)
    v = torch.randn(d, r, generator=gen, dtype=torch.float64)
    fac = FactorizedLinear(u, v, torch.zeros(d, dtype=torch.float64), **kwargs)
    with torch.no_grad():
        fac.s1.copy_(torch.linspace(-1.0, 1.0, d, dtype=torch.float64))
        fac.s2.copy_(torch.linspace(1.0, -1.0, d, dtype=torch.float64))
    return fac


def _helper():
    fn = getattr(_factorized, _HELPER, None)
    assert fn is not None, (
        f"_factorized has no {_HELPER!r} -- D-1 requires ONE definition of a "
        "factorized layer's effective terms, consulted by all three sites"
    )
    return fn


class _ScopedBlock(nn.Module):
    """A block whose linears sit under `self_attn.` and `mlp.` module paths."""

    def __init__(self, d: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.self_attn = nn.ModuleDict({"q_proj": nn.Linear(d, d, dtype=torch.float64)})
        self.mlp = nn.ModuleDict({"gate_proj": nn.Linear(d, d, dtype=torch.float64)})
        with torch.no_grad():
            for m in (self.self_attn["q_proj"], self.mlp["gate_proj"]):
                m.weight.copy_(torch.randn(d, d, generator=gen, dtype=torch.float64) * 0.4)
                m.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp["gate_proj"](torch.nn.functional.gelu(self.self_attn["q_proj"](x)))


def _calib(d: int, n: int = 4, seq: int = 8, seed: int = 1) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(n, seq, d, generator=gen, dtype=torch.float64) for _ in range(2)]


def _fixed_cot(module, batches, *, seed: int = 9):
    """Caller-supplied output cotangent (spec: caller_supplied_head_for_bare_block).

    A bare block carries no scalar loss, so the local statistics pass takes
    the gradient at the module output from the caller. The probe runs the
    CONCATENATED batch stream — the local pass feeds ``[x_b]`` with
    ``x_b = cat(batches)``, so the cotangent must carry those rows. Fixed
    seed so runs asserting on recorded events stay reproducible.
    """
    x = torch.cat(batches, dim=0) if len(batches) > 1 else batches[0]
    with torch.no_grad():
        probe = module(x)
    return torch.randn(
        probe.shape, generator=torch.Generator().manual_seed(seed), dtype=probe.dtype
    )


def _events(cfg: PTQConfig, block: nn.Module, d: int = 6) -> list[tuple[str, dict]]:
    seen: list[tuple[str, dict]] = []

    def recorder(event: str, **info) -> None:
        seen.append((event, info))

    calib = _calib(d)
    reconstruct_block(block, calib, cfg, cotangent=_fixed_cot(block, calib), recorder=recorder)
    return seen


# --------------------------------------------------------------------------
# knobs: validation + fingerprint (D-6)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "good", "bad"),
    [
        ("factor_mode", ("binary", "fp"), "half"),
        ("ablation", ("none", "untuned-signs"), "untuned"),
        ("layer_scope", ("all", "self_attn", "mlp"), "attn"),
    ],
)
def test_config_validates_knob_choices(field, good, bad):  # noqa: ANN001
    """Unknown value = parse/validation error, never a silent default (D-6)."""
    for value in good:
        assert getattr(PTQConfig(solver_backend="numpy", **{field: value}), field) == value
    with pytest.raises(ValueError):
        PTQConfig(solver_backend="numpy", **{field: bad})


@pytest.mark.parametrize(
    ("field", "value"),
    [("factor_mode", "fp"), ("ablation", "untuned-signs"), ("layer_scope", "mlp")],
)
def test_knobs_are_solver_state_in_the_fingerprint(field, value):  # noqa: ANN001
    """A resumed fp run must never resume as a binary run (D-6)."""
    assert config_fingerprint(
        PTQConfig(solver_backend="numpy", **{field: value})
    ) != config_fingerprint(
        PTQConfig(
            solver_backend="numpy",
        )
    )


def test_default_knobs_are_the_pre_knob_pipeline():
    """The three defaults ARE today's behavior (regression contract)."""
    cfg = PTQConfig(
        solver_backend="numpy",
    )
    assert (cfg.factor_mode, cfg.ablation, cfg.layer_scope) == ("binary", "none", "all")


# --------------------------------------------------------------------------
# D-1: one definition, three sites, no stray sign
# --------------------------------------------------------------------------


def _helper_pin_findings(src: str) -> list[str]:
    """Findings for the D-1 shape, as a function so reviewers can attack it."""
    findings: list[str] = []
    tree = ast.parse(src)
    defs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == _HELPER]
    if len(defs) != 1:
        return [f"expected exactly one {_HELPER!r} definition, found {len(defs)}"]
    helper = defs[0]
    lo, hi = (
        helper.lineno,
        max(getattr(n, "lineno", helper.lineno) for n in ast.walk(helper) if hasattr(n, "lineno")),
    )
    # Every sign applied to a LATENT lives inside the helper.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "sign":
            continue
        if not any(
            isinstance(arg, ast.Attribute) and arg.attr in ("u_latent", "v_latent")
            for arg in node.args
        ):
            continue
        if not lo <= node.lineno <= hi:
            findings.append(
                f"line {node.lineno}: sign() on a latent outside {_HELPER!r} -- D-1 has "
                "ONE definition of the effective terms, and a second hand-rolled sign site "
                "is the divergence class this pin exists to forbid"
            )
    # The grad-enabled branch must keep the straight-through passthrough:
    # literal sign there gives exactly zero gradient and TuneLatentSTE dies
    # silently — the failure mode this whole change exists to prevent, so it is
    # a named, tested failure rather than a lucky escape.
    grad_branches = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.If) and "is_grad_enabled" in ast.unparse(node.test)
    ]
    if len(grad_branches) != 1:
        findings.append(
            f"expected exactly one `is_grad_enabled()` branch in {_HELPER!r}, "
            f"found {len(grad_branches)}"
        )
    elif not any(
        isinstance(inner, ast.Attribute) and inner.attr == "detach"
        for inner in ast.walk(grad_branches[0])
    ):
        findings.append(
            f"the grad-enabled branch of {_HELPER!r} carries no `.detach()` — that branch "
            "must keep the straight-through passthrough (d/dU = 1); a literal sign there "
            "zeroes the gradient and TuneLatentSTE dies silently"
        )
    # All three expression sites consult the helper.
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in _SITES:
            continue
        called = {
            inner.func.id
            for inner in ast.walk(node)
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
        }
        if _HELPER not in called:
            findings.append(
                f"{node.name!r} does not consult {_HELPER!r} -- three expression sites "
                "(module forward, live STE forward, write-back) must share it"
            )
    return findings


def test_pin_one_helper_three_sites_no_stray_sign():
    findings = _helper_pin_findings(MODULE_SRC)
    assert findings == [], findings


def _dead_ste_helper(src: str) -> str:
    """The catastrophic mutation: literal sign on the GRAD-ENABLED branch.

    This is the shape that passes every other gate while STE is silently dead.
    """
    marker = "if torch.is_grad_enabled():"
    assert marker in src, "the branch marker drifted; the mutation is a no-op"
    head, tail = src.split(marker, 1)
    body_end = tail.index("\n    elif") if "\n    elif" in tail else tail.index("\n    else:")
    return f"{head}{marker}\n        ub = torch.sign(fac.u_latent){tail[body_end:]}"


def test_negative_row_literal_sign_in_the_grad_branch_reds():
    """The dead-STE catastrophe must be a named, tested failure."""
    mutated = _dead_ste_helper(MODULE_SRC)
    assert mutated != MODULE_SRC
    assert _helper_pin_findings(mutated), "literal sign in the grad branch was NOT caught"


# --------------------------------------------------------------------------
# D-1: the branch contract
# --------------------------------------------------------------------------


def test_binary_no_grad_is_bitwise_sign():
    """Under no_grad the helper returns literal sign -- bitwise, not approximately."""
    fac = _fac(seed=3)
    with torch.no_grad():
        left, right = _helper()(fac, torch.float64, "binary")
    assert torch.equal(left, fac.s1.unsqueeze(1) * torch.sign(fac.u_latent))
    assert torch.equal(right, fac.s2.unsqueeze(1) * torch.sign(fac.v_latent))


def test_binary_grad_keeps_the_passthrough():
    """With gradients on, d(terms)/dU must be the scales -- never zero."""
    fac = _fac(seed=4)
    fac.u_latent.requires_grad_(True)
    left, _ = _helper()(fac, torch.float64, "binary")
    left.sum().backward()
    assert fac.u_latent.grad is not None
    expected = fac.s1.unsqueeze(1).expand_as(fac.u_latent)
    assert torch.allclose(fac.u_latent.grad, expected), (
        "the grad-enabled branch must keep the straight-through passthrough; "
        "literal sign there gives a zero gradient and STE dies silently"
    )


@pytest.mark.parametrize("grad_enabled", [True, False])
def test_fp_terms_have_no_sign_in_either_regime(grad_enabled):  # noqa: ANN001
    fac = _fac(seed=5, factor_mode="fp")
    with torch.set_grad_enabled(grad_enabled):
        left, right = _helper()(fac, torch.float64, "fp")
    assert torch.equal(left, fac.s1.unsqueeze(1) * fac.u_latent)
    assert torch.equal(right, fac.s2.unsqueeze(1) * fac.v_latent)


# --------------------------------------------------------------------------
# D-1 reachability, observed twice
# --------------------------------------------------------------------------


def test_set_linear_writeback_is_bitwise_the_helper_no_grad_output():
    """The weights the probe and the gate actually measure."""
    fac = _fac(seed=6)
    linear = nn.Linear(fac.din, fac.dout, dtype=torch.float64)
    _set_linear(linear, fac, factor_mode="binary")
    with torch.no_grad():
        left, right = _helper()(fac, torch.float64, "binary")
    assert torch.equal(linear.weight, left @ right.T)


def test_live_ste_path_consults_the_helper(monkeypatch):
    """Reachability #1: the pin cannot be satisfied by a dead call site."""
    calls: list[str] = []
    real = _helper()

    def spy(fac, activation_dtype, factor_mode):
        calls.append(factor_mode)
        return real(fac, activation_dtype, factor_mode)

    monkeypatch.setattr(_factorized, _HELPER, spy, raising=False)
    block = _ScopedBlock(6, seed=7)
    _events(PTQConfig(solver_backend="numpy", rank=3), block)
    assert calls, "the live STE path never consulted the helper"


def test_set_linear_consults_the_helper_by_spy_not_just_by_equality(monkeypatch):
    """Reachability #2 by SPY.

    Bit-equality alone is a weaker guard than the STE leg's: a future drifted
    copy with identical arithmetic satisfies equality while bypassing the
    helper, so the write-back path is observed directly too.
    """
    calls: list[str] = []
    real = _helper()

    def spy(fac, activation_dtype, factor_mode):
        calls.append(factor_mode)
        return real(fac, activation_dtype, factor_mode)

    monkeypatch.setattr(_factorized, _HELPER, spy, raising=False)
    fac = _fac(seed=14)
    linear = nn.Linear(fac.din, fac.dout, dtype=torch.float64)
    _set_linear(linear, fac, factor_mode="binary")
    assert calls == ["binary"], "the write-back must consult the shared helper"


def test_detached_alias_is_invisible_to_the_shape_pin_by_design():
    """The AST pin is a SHAPE guard; the runtime contract is the real proof.

    A detached alias keeps the pin's `.detach()` clause satisfied while the
    gradient through it is exactly zero — so the shape pin must never be cited
    alone as the guard on STE living (D-1's named residual), and the analytic
    gradient contract is what actually catches this class.
    """
    src = MODULE_SRC.replace(
        "        u = torch.sign(fac.u_latent)\n"
        "        u = u + fac.u_latent - fac.u_latent.detach()  # STE passthrough",
        "        ua = fac.u_latent.detach()\n"
        "        u = torch.sign(ua)\n"
        "        u = u + ua - ua.detach()",
        1,
    )
    assert src != MODULE_SRC, "the mutation is a no-op: the helper drifted"
    assert _helper_pin_findings(src) == [], (
        "by design the shape pin cannot see the alias form — if it now does, "
        "this residual is closed and the docs should say so"
    )
    # The attack is REAL: an alias form severs the gradient path entirely.
    u = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    ua = u.detach()
    aliased = torch.sign(ua) + ua - ua.detach()
    assert not aliased.requires_grad, (
        "the detached alias must leave no gradient path to the latent at all — "
        "if it does, this attack does not reproduce and the contract test is not "
        "the backstop it is claimed to be"
    )
    # The SHIPPED helper's contract, by contrast, hands the latent its gradient.
    fac = _fac(seed=15)
    fac.u_latent.requires_grad_(True)
    left, _ = _helper()(fac, torch.float64, "binary")
    left.sum().backward()
    assert torch.any(fac.u_latent.grad != 0)


# --------------------------------------------------------------------------
# ablations: zero refinement, and the scope filter
# --------------------------------------------------------------------------


def test_untuned_signs_runs_zero_polish_and_zero_ste():
    block = _ScopedBlock(6, seed=8)
    seen = _events(PTQConfig(solver_backend="numpy", rank=3, ablation="untuned-signs"), block)
    names = [event for event, _ in seen]
    assert "ste" not in names, "untuned-signs must run zero TuneLatentSTE steps"
    lb = [info for event, info in seen if event == "lb_admm"]
    assert lb and all(info.get("polish_rounds") == 0 for info in lb), (
        "untuned-signs must run zero polish rounds; the lb_admm event carries the count"
    )


def test_default_run_records_the_polish_rounds():
    block = _ScopedBlock(6, seed=9)
    seen = _events(PTQConfig(solver_backend="numpy", rank=3), block)
    lb = [info for event, info in seen if event == "lb_admm"]
    assert lb and all(
        info.get("polish_rounds")
        == PTQConfig(
            solver_backend="numpy",
        ).polish_rounds
        for info in lb
    )


@pytest.mark.parametrize("scope", ["self_attn", "mlp"])
def test_scope_leaves_unscoped_weights_bit_identical(scope):  # noqa: ANN001
    block = _ScopedBlock(6, seed=10)
    untouched = "mlp.gate_proj" if scope == "self_attn" else "self_attn.q_proj"
    linears = {name: m for name, m in block.named_modules() if isinstance(m, nn.Linear)}
    before = {name: m.weight.detach().clone() for name, m in linears.items()}
    _events(PTQConfig(solver_backend="numpy", rank=3, layer_scope=scope), block)
    after = {name: m for name, m in block.named_modules() if isinstance(m, nn.Linear)}
    assert torch.equal(after[untouched].weight, before[untouched])
    in_scope = "self_attn.q_proj" if scope == "self_attn" else "mlp.gate_proj"
    assert not torch.equal(after[in_scope].weight, before[in_scope])


# --------------------------------------------------------------------------
# D-5: the measurement / artifact boundary
# --------------------------------------------------------------------------


def test_extract_and_pack_refuses_fp_latents_at_the_conversion_site():
    """The primary wall: the ONLY bit-making function refuses fp latents.

    Asserted at the conversion site itself, for any caller — a caller that
    politely skips it can be re-plumbed around; the site that owns
    ``np.where(latents >= 0, +1, -1)`` cannot.
    """
    u = np.ones((4, 2))
    v = np.ones((4, 2))
    with pytest.raises(ValueError, match="factor_mode='fp'"):
        block_recon.extract_and_pack(u, v, name="probe", factor_mode="fp")
    # ...and the wall is mode-specific, not a blanket refusal.
    st = block_recon.extract_and_pack(u, v, name="probe")
    assert st.Ub.shape == (4, 2)
    assert set(np.unique(st.Ub).tolist()) == {1}


def test_fp_run_makes_no_layers_and_writes_a_different_weight():
    """An fp run fabricates no binary states, and its write-back really drops
    the sign constraint: same seed, same geometry, DIFFERENT host weight."""
    fp = _ScopedBlock(6, seed=12)
    binary = _ScopedBlock(6, seed=12)
    calib = _calib(6, seed=13)
    result = reconstruct_block(
        fp,
        calib,
        PTQConfig(solver_backend="numpy", rank=3, factor_mode="fp"),
        cotangent=_fixed_cot(fp, calib),
    )
    assert result.layers == [], "an fp measurement must fabricate no binary layer states"
    reconstruct_block(
        binary,
        calib,
        PTQConfig(solver_backend="numpy", rank=3),
        cotangent=_fixed_cot(binary, calib),
    )
    w_fp = fp.self_attn["q_proj"].weight
    w_binary = binary.self_attn["q_proj"].weight
    assert not torch.equal(w_fp, w_binary), (
        "the fp write-back equals the binary one — the sign constraint was not removed"
    )


# --------------------------------------------------------------------------
# D-7: streamed statistics are part of rung 1
# --------------------------------------------------------------------------

_CALIBRATION_SRC = (
    pathlib.Path(__file__).resolve().parents[2] / "src/quantibbler/ptq/calibration.py"
)


class _TwoLinear(nn.Module):
    """Two stacked linears: enough to exercise two hook pairs."""

    def __init__(self, din: int = 8, dmid: int = 5, dout: int = 4, seed: int = 5) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.a = nn.Linear(din, dmid, dtype=torch.float64)
        self.b = nn.Linear(dmid, dout, dtype=torch.float64)
        for lin in (self.a, self.b):
            with torch.no_grad():
                lin.weight.copy_(
                    torch.randn(lin.weight.shape, generator=gen, dtype=torch.float64) * 0.5
                )
                lin.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.tanh(self.a(x)))


def _d7_batches(rows: int = 6, seq: int = 4, din: int = 8, seed: int = 11) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(rows, seq, din, generator=gen, dtype=torch.float64)]


def test_d7_streamed_statistics_match_the_reference_at_1e_12():
    """The streamed path is a re-association of the same sums, nothing more."""
    model = _TwoLinear()
    batches = _d7_batches()
    # Fixed caller-supplied cotangent (spec: caller_supplied_head_for_bare_block)
    # — identical object both calls, so the chunking parity below is
    # apples-to-apples on a loss-bearing gradient, not a vacuous uniform one.
    with torch.no_grad():
        probe = model(batches[0])
    cot = torch.randn(probe.shape, generator=torch.Generator().manual_seed(9), dtype=probe.dtype)
    reference = calibration.collect_stats(model, batches, cotangent=cot)
    streamed = calibration.collect_stats(model, batches, chunk_rows=4, cotangent=cot)

    assert streamed.layer_names == reference.layer_names
    for got, want in zip(streamed, reference, strict=True):
        np.testing.assert_allclose(got.d_in_raw, want.d_in_raw, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(got.d_out_raw, want.d_out_raw, rtol=1e-12, atol=1e-12)


def test_d7_streamed_path_never_materializes_full_activations(monkeypatch):
    """The chunked path must not touch the materializing reducer at all."""
    seen: list[tuple[int, ...]] = []

    def _spy(x):  # noqa: ANN001
        seen.append(tuple(x.shape))
        raise AssertionError("the chunked path materialized a full activation tensor")

    # Same fixed caller-supplied cotangent for both paths (see the parity
    # test above): the spy criterion is chunking behavior, not the gradient.
    with torch.no_grad():
        probe = _TwoLinear()(_d7_batches()[0])
    cot = torch.randn(probe.shape, generator=torch.Generator().manual_seed(9), dtype=probe.dtype)
    monkeypatch.setattr(calibration, "_reduce_tokens", _spy)
    streamed = calibration.collect_stats(_TwoLinear(), _d7_batches(), chunk_rows=4, cotangent=cot)
    assert seen == [], f"chunked collect_stats called the materializing reducer: {seen}"
    assert len(streamed) == 2

    # Non-vacuity: the spy really can detect a materializing path.
    with pytest.raises(AssertionError, match="materialized"):
        calibration.collect_stats(_TwoLinear(), _d7_batches(), cotangent=cot)
    assert seen, "the spy never fired on the reference path — it proves nothing"


def _collect_stats_streaming_findings(src: str) -> list[str]:
    """Structural guard for D-7 (see calibration.py).

    Token greps cannot carry this: `_reduce_tokens` must exist in the file for
    the reference path, so presence proves nothing. The shape that matters is
    WHICH branch calls it.
    """
    tree = ast.parse(src)
    fn = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "collect_stats"),
        None,
    )
    if fn is None:
        return ["collect_stats is missing from calibration.py"]
    branch = next(
        (n for n in ast.walk(fn) if isinstance(n, ast.If) and "streaming" in ast.unparse(n.test)),
        None,
    )
    if branch is None:
        return [
            "collect_stats has no `streaming` branch: the chunked path is "
            "indistinguishable from the reference path"
        ]
    streaming_src = "\n".join(ast.unparse(n) for n in branch.body)
    reference_src = "\n".join(ast.unparse(n) for n in branch.orelse)
    findings = []
    if "_reduce_tokens" in streaming_src:
        findings.append(
            "the streaming branch calls _reduce_tokens: full f64 activations are retained again"
        )
    if re.search(r"feats_(?:in|out)\[[^\]]+\]\.append\(", streaming_src):
        findings.append(
            "the streaming branch appends per-chunk payloads: state grows with the chunk count"
        )
    # LOAD-BEARING TWICE — do not "simplify" this clause away.
    # 1. Contract: both the input and the output statistic must be streamed. A
    #    leg that streams one side and retains the other is a half-fix that
    #    still OOMs at real widths.
    # 2. Accidentally, an alias tripwire: any realistic form that replaces a
    #    streaming leg with a materializer drops the count below 2 and reds
    #    here. That includes materializing through a module-level alias
    #    (`_rt = _reduce_tokens`), which carries neither the `_reduce_tokens`
    #    token nor an append — it evades the two checks above AND the runtime
    #    spy, which is blind to import-time binding.
    # Named residual (poison-review, accepted, NOT fixed): a contrived composite
    # that keeps two dead `_accumulate` calls purely to satisfy this count while
    # materializing through an alias passes the shape pin and the spy alike.
    # Only a bounded-memory measurement can see that one; the fixture's cost and
    # flake were judged not worth it. If it ever must be closed: measure, don't
    # grep harder.
    if streaming_src.count("_accumulate") < 2:
        findings.append("the streaming branch does not stream both input and output statistics")
    if "_reduce_tokens" not in reference_src:
        findings.append("chunk_rows=None no longer materializes: the reference path was disturbed")
    return findings


def test_d7_pin_streaming_branch_cannot_materialize_activations():
    findings = _collect_stats_streaming_findings(_CALIBRATION_SRC.read_text())
    assert findings == [], findings


def test_d7_negative_row_reintroduced_append_reds_the_pin():
    src = _CALIBRATION_SRC.read_text()
    lines = src.splitlines()
    anchor = next(i for i, ln in enumerate(lines) if ln.strip() == "if streaming:")
    mutated = "\n".join(
        [
            *lines[: anchor + 1],
            "            feats_out[key].append(_reduce_tokens(out))",
            *lines[anchor + 1 :],
        ]
    )
    assert mutated != src, "the negative row is a no-op"
    ast.parse(mutated)  # the poison is a real code shape, not a syntax error
    findings = _collect_stats_streaming_findings(mutated)
    assert findings, "the pin admitted a reintroduced append of full activations"
    assert any("append" in f or "_reduce_tokens" in f for f in findings), findings
