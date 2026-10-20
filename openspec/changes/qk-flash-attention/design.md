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
