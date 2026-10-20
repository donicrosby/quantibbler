# qk-flash-attention — Design

## User directive (2026-10-20): consume CubeK for the attention kernels

User: "we'd want to use CubeK for the attention kernels." Recorded as a
binding design steer for this change (and qk-sage-attention): consume
tracel-ai/cubek's `cubek-attention` crate (variants `mma` / `unit` /
`multi-rows`) rather than hand-rolling the CubeCL tiled attention face.
The spec's fail-closed construction and double-gated hardware-test
requirements stay — they constrain the wrapper regardless of kernel
provenance. Verified 2026-10-20 (crates.io + repo): `cubek-attention`
v0.2.0 exists (~217K downloads, MIT OR Apache-2.0, active), and its
`cpu-reference` feature is exactly the third parity face this change
already names. Status 2026-10-20 (coder registry probe + sec-eng
provenance audit): version alignment is RESOLVED — `cubek-attention`
0.3.0 is published (2026-10-06, not yanked) with normal dependencies
`cubecl ^0.11.0` / `cubecl-common ^0.11.0`, exactly this repo's pin; a
scratch lockfile probe shows exactly one cubecl 0.11.0 in the graph
(the 0.2.0 dual-major hazard does not exist on this route); 0.3.0
declares no rust-version, so no MSRV change. Provenance: all four cubek
crates ship from tracel-ai/cubek — the same publisher as cubecl itself —
zero new publishers in the graph. Adapter invariants the re-spec MUST
carry: (1) the forward face has a HIDDEN softmax scale 1/sqrt(head_dim)
(partition_attention.rs:75) — the adapter must NOT pre-scale Q; the
proposal's pre-scale item is superseded and the no-pre-scale invariant
goes in the delta, not task-time discovery; (2) the GQA gap is confirmed
(single num_heads derived from query.shape[1]) — the adapter expands K/V
to Q's head count at the 2x KV traffic cost the perf probe must beat;
(3) `cpu-reference` ships inside cubek-test-utils — the shipped server
never enables it. Executable instrument, per the binary-purity lesson
(a link-based test cannot see a feature flag): a SIBLING test file next
to binary_purity.rs reads the RESOLVED graph (`cargo tree -p
quantibbler-server -e normal,build --locked`), package-scoped — never
workspace-wide, never `--all-features` (the kernels crate's own test
build legitimately carries cubek-attention/cpu-reference as the third
parity face, and resolver-2 keeps dev-dependency features out of the
server graph) — and asserts a POSITIVE+NEGATIVE pair: cubek-attention
IS reachable from quantibbler-kernels under `-F gpu`, and
cubek-test-utils is NOT reachable from quantibbler-server. The pair is
self-guarding: dropping the dep or moving it out from under the `gpu`
feature reds the positive half instead of the negative rotting into
decoration. The tripwire lands in the SAME commit as the cubek
adoption — a negative-only assertion today is vacuously green; (4) the commit introducing cubek-attention carries the
regenerated Cargo.lock in the SAME commit — the --locked referee gates
make a split landing red.

## User directive (2026-10-27): reference benchmark vs pip flash-attn

User: "getting the attention face working so that we're not exploding
memory. There already exists a flash-attn pip package we can benchmark
against; if ours meets it or greater then we should be set." Recorded
as binding: the perf section gains a REFERENCE benchmark (task 4.2) —
pip `flash-attn` (ROCm build, gfx1151, same rocm container as the GPU
probes) run at identical shapes, dtype, masking, warmup and batch; the
face is "set" only when it meets or beats the reference on BOTH wall
time and peak memory. The reference is a mature CUDA/ROCm kernel
suite; a first-version shortfall is recorded honestly, never papered
over (same discipline as the 1.5× CPU probe).

Scope fact recorded with the directive: the measured memory explosion
(2026-10-27: seq-2048 tune killed at the 32g cap; N² autograd
retention ~7×, 100-200 GiB bracket) lives in the PTQ TUNE path — a
torch training loop on CPU — which this server-side face does not
reach. Fixing the server's decode memory is in-scope here; fixing tune
memory is a separate candidate workstream (fused attention inside the
torch loop — SDPA mem-efficient/flash backend, or the pip package as
backend; no new kernel required), named in the bar change's constraint
section. The two share the "never materialize N²" requirement, not the
code path.

## Method provenance (rule: every claim cited)

- Tiling + online softmax + rescaling identity: [FA] §3.1, Algorithm 1 —
  running statistics m (block max) and ℓ (block sum of exp), each new block
  rescales prior accumulators by exp(m_old − m_new). Implemented verbatim;
  no algorithmic novelty is claimed.
- IO-complexity motivation: [FA] §3.2, Theorem 2.
- Numerical-stability form of softmax (subtract max before exp): [FA] §3.1.
- f32 compute substrate: engine activations are f32 (pinned qk-server
  convention); RDNA3 fp32 measures 4.80 TFLOP/s (2026-10-18 rocBLAS probe)
  vs 0.44 TF fp64 — the rate argument that does NOT exist for the solver.

## Discarded alternatives

- **Port FlashAttention CUDA/ROCm reference kernels** — banned: config
  forbids C/C++/CUDA-C++ in the build graph; evidence rule says papers
  only. Discarded, not deferred.
- **rocBLAS-based split (two sgemms + fused softmax kernel)** — simpler
  kernels but materializes S per block and doubles global-memory traffic;
  [FA] §3.1's fusion is precisely the win being specified. Revisit only if
  the tiled kernel misses its perf goal.
- **f64 attention for parity bit-identity** — would throw away 11× of
  measured hardware rate to satisfy the wrong gate; per the 3.5 lesson the
  gate follows the arithmetic class (reassociating ⇒ bounded rel error),
  not the other way round.
- **kv-split / flash-decoding variants** — speculative scheduling beyond
  [FA] v2's pinned text; out of scope by the evidence rule.

## Informed assumptions

- **Assumption:** a naive-tiled first kernel (one workgroup per Q block,
  K-loop inside) reaches ≥ 1.5× the scalar CPU face on decode-shaped
  attention (seq ≤ 2k, d=128, GQA 8:1) on gfx1151. Rationale: 4.8 TF fp32
  vs measured ~0.46 TF effective CPU, with attention's arithmetic intensity
  amortizing launch overhead. **Wrong-if:** it surfaces as the perf probe
  (task 5) failing to clear 1.5× — then the face stays opt-in and the
  change records the miss instead of shipping a regression.
- **Assumption:** CubeCL f32 `Tensor` indexing/`ABSOLUTE_POS` forms from
  the landed gpu.rs (tasks 3.1–3.4) extend to 2D tiling without new API
  surface (shared-memory arrays via `cubecl::std` comptime-sized arrays).
  **Wrong-if:** compile failures force new forms — captured as
  crystallization entries, cost bounded by the proven 0.11 pin.

## Substrate decision (2026-10-19): cubek-attention 0.3.0

Registry evidence (crates.io, verified 2026-10-19): `cubek` 0.3.0 stable
(2026-10-06), 447k recent downloads, Apache-2.0, MSRV unset, by the CubeCL
maintainers; `cargo add cubek@0.3.0` in the throwaway bench resolves with
`cargo tree` unifying on OUR pinned `cubecl 0.11.0` — no fork. Source
audit of the published `cubek-attention-0.3.0` tarball: `AttentionProblem`
exposes `causal`, `masked` (mask tensor [b, h, sq, skv]), per-tensor
`AttentionGlobalTypes`, and an accumulator-precision option; tests cover
masking, ragged tails, and LSE; a `cpu-reference` feature ships an
independent CPU implementation and benchmark catalog (a third parity face
against our NumPy reference). Backward is scaffolded `todo!()` — irrelevant
(inference only). Two contract gaps, adapter-shaped: single `num_heads`
(no first-class GQA → materialize repeated K/V heads; traffic cost counted
against the perf probe) and no softmax-scale field → pre-scale Q by 1/√d
in f32 (exact to round-off). Fallback if the contract misses at
integration: the hand-rolled tiled kernel per the discarded-alternatives
note — the specs' parity gates do not care which substrate computes.

## Open Questions

- OQ-1 (owner: orchestrator, probe = task 5): does the tiled face clear
  1.5× CPU decode attention on gfx1151 at production shapes? Status: OPEN
  — resolved only by the timed probe, never by argument.
- OQ-2 (owner: orchestrator): workgroup/tiling constants (B_c, B_r, d
  specialization) — opened during implementation, closed by measurement
  over the fixture battery; defaults follow [FA] Algorithm 1 line 1's
  SRAM-size formula with gfx1151 LDS limits substituted.

## Implementation crystallization (2026-10-20, tasks 1.1–1.2)

- **`quantibbler.kernels` is now a package.** The spec's own Requirement
  says "the *kernels package* SHALL provide a NumPy tiled attention
  reference" and the proposal pins `src/quantibbler/kernels/attention_tiled.py`,
  but the repo shipped `src/quantibbler/kernels.py` — a module, which cannot
  coexist with a same-named directory (the RED collection error said so in
  as many words: `'quantibbler.kernels' is not a package`). Resolution: the
  dispatch module moved verbatim to `kernels/__init__.py` (sha256
  `129d3f74c9e9…` identical before and after — a rename, not an edit) and
  the reference landed beside it. Import compatibility is unchanged for every
  existing caller and is verified three ways: the full suite is green, the
  built wheel ships `quantibbler/kernels/{__init__,attention_tiled}.py`, and
  `import quantibbler.kernels` still resolves to the same dispatch surface.
  The rejected alternative — a flat module path such as
  `quantibbler/attention_tiled.py` — contradicts the Requirement's wording,
  so if the flat layout is preferred the spec delta must change first and the
  test's import moves with it.
- **The 1e-12 parity gate needed a formulation that attention can actually
  meet, and the difference is measured, not argued.** Literal per-element
  relative error is unattainable here: softmax outputs cancel, so an entry can
  be `4e-4` while its row norm is `3.4`, and the absolute error of a correct
  tiled implementation (1.1e-15, one to two ulp) then *reads* as `3.6e-11`
  relative. The gate implemented in `tests/test_attention_tiled.py` is
  two-part: (1) infinity-norm relative `max|a−b| ≤ 1e-12·max|a|,|b|`, and
  (2) per-element relative `≤ 1e-12` restricted to entries within `1e-2` of
  the row-max magnitude. Measured on the shipped battery: part (1)
  **3.3e-16**, part (2) **6.2e-15** — three orders inside the gate — while a
  masking, GQA or rescaling defect lands at O(1) (nine mutations, all
  caught). **Proposed spec wording (owner: orchestrator)** — replace
  "every output agrees with relative error ≤ 1e-12" with "every output agrees
  to round-off: infinity-norm relative error ≤ 1e-12, and per-element
  relative error ≤ 1e-12 for entries within two orders of the row-max
  magnitude"; the GPU scenario's "rel ≤ 1e-5 vs the NumPy reference" is
  unaffected (it is a 1e5×-looser envelope).
  **Decision (2026-10-27, planner): ACCEPTED** — the spec delta now
  carries the two-part gate with the rationale inline (per-element
  relative error is undefined near softmax cancellation; the
  inf-norm clause covers the whole row, the restricted per-element
  clause keeps stringency where it is meaningful). Rationale beyond
  the measurements: every real defect the battery models (dropped
  rescale, off-by-one causal, GQA mixup, masked-then-softmax, tail
  truncation) lands at O(1) — orders beyond either clause — so the
  reformulation trades no detection power. Bug-hunter's attack on
  this formulation is invited before landing; it is the only
  tolerance in the change shaped outside the inherited spec text.
- **Design-silent API choices, now pinned by the tests:** signature
  `attention_tiled(q, k, v, *, causal=False, mask=None, block_size=64,
  return_lse=False)`; GQA broadcast is `repeat_interleave` (head `h` reads kv
  head `h // group`), matching the engine convention; the `1/√d` scale is
  applied to `Q·Kᵀ` (the substrate carries no softmax-scale field, and the
  GPU face pre-scales Q for the same reason); causal masking removes
  strictly-upper scores *before* the row maximum, so masked keys contribute
  exactly zero — proven by bit-identity, not by tolerance: moving a key's
  value by 1e3 leaves every earlier row's output unchanged; a row whose every
  score is masked yields NaN exactly as the dense formula's `0/0` does, with
  no invented fallback the GPU face would not reproduce; `block_size` defaults
  to 64 while the GPU face's tiling constants remain open (OQ-2, above).
  **Correction (task 1.4):** "yields NaN" was true, but the *statistic* that
  produced it was not the one the scenario states — an all-masked row had
  `ℓ = NaN` by propagation through `exp(−inf − (−inf))`, not `ℓ = 0`. The
  row is now genuinely empty (`m = −inf, ℓ = 0`, `lse = −inf`) so the output's
  NaN is the same `0/0` the dense formula computes; until that fix the
  scenario's own premise was unmet and the demonstrated fallback mutation was
  dead code rather than a detectable escape.
- **The wide-dynamic-range fixture family exists because the mutation battery
  demanded it.** Replacing the running `m` with each block's own max is
  mathematically exact but numerically unstable; with the original fixtures
  (scores O(1)) it was bit-identical and escaped all 206 cases. The added
  family (q,k × 2000 → scores ±3e3, block maxima ~1e3 apart) makes the
  running-max line load-bearing: that mutation produces **824 non-finite
  outputs** there while the shipped implementation matches the dense oracle
  with inf-norm relative error **0.0**. A battery is what makes "the gate is
  real" a measurement instead of a claim.

## Implementation crystallization (2026-10-20, tasks 2.1–2.2)

- **The substrate applies the softmax scale itself; the audit was half
  wrong and the parity battery is what said so.** The substrate decision's
  source read recorded "no softmax-scale field → pre-scale Q by 1/√d" — the
  field is indeed absent, but the standard `1/√head_dim` factor is applied
  *inside* the routine. The first adapter therefore squared the scale, and
  the fixture battery caught it on its first device run: inf-norm relative
  error **3.52e-1**, with the device matching the dense reference at
  `1/(2√d)`. Removing the host-side pre-scale gives **9.0e-08** at `1/√d`
  across all nine cases. The adapter now passes Q unscaled and the module
  docstring says why; the correction is as much the battery's product as the
  code is. Nothing about the substrate decision changes — the wrapper simply
  implements it correctly.
- **`head_dim ≠ val_dim` is now part of the contract on both faces.** The
  substrate exposes `head_dim` and `val_dim` separately, and a square value
  projection cannot distinguish `1/√head_dim` from `1/√val_dim`: every
  square fixture would pass under either convention. The NumPy reference
  gained `val_dim` independence (scale keyed to the *query/key* dimension,
  output carrying `val_dim`), a Python scenario pins it, and the GPU battery
  carries two non-square cases — `d=4,val=8` and `d=128,val=32`.
- **GQA remains a host-side materialization**, as designed: `n_heads` single
  in `AttentionDims`, so K/V heads are repeated `repeat_interleave`-wise
  before upload. This is a real traffic cost at seq 2048 and belongs to the
  perf probe (section 4) — recorded here so it is argued with numbers there
  rather than assumed away.
- **Named residual, deliberately not asserted in 2.2:** repeated-run
  bit-identity. The blueprint is inferred per call (`BlueprintStrategy::
  Inferred`) and the substrate may select or autotune a routine, so
  determinism is not yet a contractual property of this face. The parity
  budget is unaffected (1e-5 absorbs reassociation), but if any engine
  consumer later depends on run-to-run identity it needs its own scenario
  and a pinned strategy — mechanism before reliance, the repo's usual order.
  **Resolution shape (bug-hunter, 2026-10-27):** when that day comes, the
  resolution is a pinned blueprint strategy (or equivalent determinism
  mechanism), never a tolerance change — nobody "fixes" run-to-run drift by
  loosening the 1e-5 budget; the budget is the parity contract and stands
  independent of determinism.
