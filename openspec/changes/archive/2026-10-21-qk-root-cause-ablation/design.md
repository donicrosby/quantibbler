# Design: qk-root-cause-ablation

## Context

The r512-d1 abort (depth 2, ratio 21,161x) and the all-or-nothing archive
(943x -> 1.16e6x, no intermediates) say the collapse is a defect, not capacity.
This change is the measurement harness that locates the defect. It ships no
fix by itself: it ships the three cheapest ways to *see* which stage is lying.

## Goals / Non-Goals

- Goals: three knobs (fp factors, untuned signs, layer scope) that run through
  the *existing* pipeline stages unchanged except for the named constraint;
  per-depth collapse curves on every ablation run; provenance that names the
  knobs in the report; a hard export boundary so a measurement can never
  masquerade as an artifact.
- Non-goals: no tuning of the pipeline against ablation results (the verdicts
  are expected `failed` — that is the point); no new solver; no container
  changes; no seq-2048 bar claims from these runs (they are diagnostics at the
  current geometry until the ladder narrows the field).

## Decisions

### D-1: one helper, three sites, bitwise-contract (the rung-1 bug class)

The sign expression exists at three hand-rolled sites: `FactorizedLinear.forward`
(unreachable on a real trunk — never installed into a block — but kept as the
module's own forward), `_quantized_forward_block.frozen_forward` (the live STE
path, sole caller `_tune_latent_ste`), and `_set_linear` (the write-back whose
weights the probe and the end-of-run gate actually measure). The last two
already diverge in kind: `sign` versus the `sign + u - u.detach()` passthrough —
and a helper that drifts one ulp rotates every downstream capture, moving the
measurement baseline under the 21,161x depth-2 anchor and every archived ratio.

One shared helper `(fac, activation_dtype, factor_mode)` produces the latent
terms for ALL three sites, with the frozen branch contract:

```
binary, grad enabled : sign(u) + u - u.detach()   # passthrough — STE needs d/dU = 1
binary, no_grad      : sign(u)                    # bitwise: torch.equal against torch.sign
fp,     either regime: u                          # no sign in either regime
```

The regime is selected by an explicit `torch.is_grad_enabled()` test, never by
leaning on autograd. The grad-enabled branch MUST return the passthrough: a
literal `sign` there silently zeroes the gradient to the latents and turns
TuneLatentSTE into a no-op — indistinguishable in probe history from "the
sub-1-bit problem is genuinely hard", which is the failure mode this change
exists to rule out. The `no_grad` branch MUST return literal `torch.sign`
bitwise (the passthrough is bit-unequal under `no_grad` too — measured:
111/1000 elements, maxdiff 4.44e-16; it is float evaluation order, not
autograd), so the write-back and every downstream capture stay bit-identical
to the pre-knob pipeline.

The pins: (a) one definition of the helper; (b) no `torch.sign` on latents
anywhere in the module outside it; (c) reachability observed at BOTH live
paths — spies confirm the STE path and `_set_linear` each consult the helper,
then `_set_linear`'s written weight is additionally `torch.equal` to the
helper's `no_grad` output on the f64 fixture tier; (d) `fp` mode's terms contain
no sign in EITHER regime; (e) a negative battery row: a helper returning
literal sign in the grad-enabled branch must red (the dead-STE catastrophe,
named and tested).

The AST pin is a shape guard, not the proof that STE is alive. Bug-hunter's
measured detached-alias attack (`ua = u_latent.detach()` then passthrough on
`ua`) is shape-legal and can evade a syntactic `.detach()` check while its
latent gradient is exactly zero. The runtime gradient contract — real
`backward()`, analytic `d/dU = s1` equality — is therefore the load-bearing
dead-STE guard. Neither the AST finding nor an output-value equality may be
cited alone as proof of gradient flow.

### D-2: fp mode = binary mode minus the binary

`fp` mode skips exactly one stage (greedy bit-flip polish — it is discrete-only)
and changes exactly one operation (the sign/STE pair becomes the identity).
Everything else — capture, TuneFP, solver (svd/admm), balance, Eq. 8 scale
init, TuneLatentSTE refinement on continuous latents, probe, report — runs
unchanged. This is what makes rung 1 interpretable: any difference from the
binary run is attributable to the discretization alone.

### D-3: untuned signs = sign of the balance latents, zero refinement

`untuned-signs` runs the full binary pipeline up to and including balance, then
exports `sign(U_bal)` / `sign(V_bal)` with the Eq. 8 scale fold — no polish
rounds and no TuneLatentSTE steps. The recorder must prove it: zero `ste`
events and zero polish invocations in the run log.

### D-4: layer scope filters what gets factorized, nothing else

`self_attn` / `mlp` scope restricts the set of linears that are factorized
(via the existing `_quantizable_names`/`facs` scope mechanism). Unscoped
linears remain fp in the prefix capture, in refinement, and in export. A
scoped run's unscoped weights must be byte-identical to their pre-run values.

### D-5: measurements never become artifacts

- `fp`-mode: `extract_and_pack` itself — the function that owns the sole
  `np.where(latents >= 0, +1, -1)` conversion — refuses fp latents at that
  boundary, for every caller. `reconstruct_block` does not route around it and
  does not fabricate a `BlockReconResult.layers` binary payload: it completes
  by writing the continuous reconstruction into the host through the helper's
  fp branch. The progress probe and end-of-run held-out gate therefore measure
  that live host; the report is written normally with its verdict,
  `artifact_exported: false`, and `effective_bpw: null`.
- Scale calibration is deliberately skipped in fp mode: it serves a packed
  artifact, while rung 1 measures host-weight reconstruction fidelity. This is
  the ablation boundary, not an omission.
- The driver-level `export_model_directory` refusal remains as defense in depth:
  if future plumbing ever presents an fp run to export, it raises with the flag
  named rather than producing a container.
- `untuned-signs` and scoped runs are binary and MAY export; provenance records
  the knobs either way.
- The verdict band is unchanged in every mode: status is always derived from
  the held-out PPL ratio.

### D-6: knobs are solver state

`factor_mode`, `ablation`, and `layer_scope` join the checkpoint fingerprint
(a resumed fp run must never resume as a binary run), are `choices`-validated
at the CLI (unknown value = parse error, never a silent default), and appear in
report provenance.

### D-7: memory discipline — streamed statistics are part of rung 1

The first rung-1 launch (`rank=512`, `seq_len=2048`, `tune_seq_len=512`,
`factor_mode=fp`) was kernel-provably OOM-killed in `layers.0` immediately
after TuneFP: Docker emitted `oom`, kernel memcg killed Python at 33,480,576 KiB,
and breadcrumbs rose 22.18 -> 31.07 GiB before exit 137. No depth-2 probe or
numerics verdict was produced.

The failure is localized to `collect_stats`, not the 512-window split. Its
`chunk_rows=4` path bounds each backward graph, but both forward hooks still
append `_reduce_tokens(...)` — a full float64 activation tensor — for EVERY
chunk and concatenate them only after all chunks finish. At Qwen3-0.6B's real
linear widths, the retained hook payload is 22,528 input+output features per
token: 2,048 tokens/chunk x 8 bytes = **352 MiB/chunk**; 128 chunks retain
**44.0 GiB** before the final concatenation, exceeding the 32 GiB memcg by
construction. Making chunks smaller does not change that total; reducing
calibration rows changes the NQ method; raising the cap hides an unbounded
implementation.

For `chunk_rows is not None`, hooks SHALL stream float64 per-feature sum-of-
squares plus a token count and discard the full chunk tensor immediately.
State is O(sum of feature widths), independent of row/chunk count. Gradient
accumulation remains exactly as today. The `chunk_rows=None` float64 reference
path keeps its landed list/concatenate behavior byte-identical. A tight fixture
compares streamed and reference statistics at `rtol=1e-12, atol=1e-12`; a
structural poison proves the streamed hooks cannot append full activation
tensors. Rung 1 does not retry until that RED->GREEN and bug-hunter review land.

Ablation runs continue to use the existing tune/eval geometry split
(`--tune-seq-len`); the measured evaluation protocol is unchanged. Container
runs follow the launch protocol (announce the launch line first) — the
unattributable-corpse rule applies even to expected failures.

## Measurement adjudication (2026-10-21)

All four registered rungs completed. Continuous fp factors failed at depth 2
(108,550.4041x), so signs are not the necessary kill site. Untuned signs failed
in the same class (84,692.9561x), so refinement is neither necessary nor
sufficient for the explosion. Attention-only failed at depth 2 (257.5404x),
while MLP-only was inside the line at depth 2 (1.8699x) and failed at depth 3
(67.7508x). Both projection families can independently collapse; the selected
branch is machinery shared by their continuous factorization, scale handling,
or host write-back. Because fp mode deliberately skipped global scale
calibration and still failed, that later stage is not a necessary cause of the
early observed collapse; this does not exonerate it from separate later-stage
quality risk.

`ratio ** (1/depth)` is only a geometric mean over the observed prefix, not a
claim of uniform per-block damage. The MLP history demonstrates the distinction:
the depth-2 ratio is 1.8699355654649084x and the depth-3 ratio is
67.75079573777128x, an actual adjacent-depth multiplier of
36.23162048416727x. Existing geometric-mean labels remain aggregate summaries;
they must not be narrated as each block's measured contribution.

The MLP run's exact announced geometry was already live under planner session
`proc_e450f3f44e64` and wrote
`/opt/data/tmp/rca-rung3-mlp/quantize_report.json`; no duplicate run was fired
for the later announced `/opt/data/tmp/rca-rung4-mlp` path. The next diagnostic
should measure per-linear norm ratio, directional cosine, and scale magnitude at
each stage on a failing production-scale block. A scoped-fp rerun tests an
interaction already downstream of the selected branch but does not identify the
stage, so it is not added to this completed ladder.

## Out of scope

- Per-layer rank schedules, solver changes, KD/STBLLM-adjacent objectives.
- Any recalibration of the verdict band on the strength of ablation results.
