# qk-fix-admm-solver — design

## T² exit rescale (landed at `df1e6cd`)

Entry: `w = w / target_scale` (`lb_admm.py:290`). Exit: was `× target_scale` on each of u, v, P_u, P_v. For a two-factor product `UVᵀ`, scaling both factors by `t` scales the product by `t²`. Fix: `factor_scale = sqrt(target_scale)` on all four returns (`:356-361`). Entry comment corrected to state the real invariant. Red-first test (`test_lb_admm_scale_equivariance`) verifies: scale target by c, returned product scales by c.

## SVID proxy — two-sided optimum

### Paper definition

[NQ] :592: "We apply Sign-Value Independent Decomposition (SVID) (Pouransari et al., 2020; Xu et al., 2024) to derive the **optimal rank-1 approximation that preserves the sign structure**."

The SVID of matrix P with sign structure S = sign(P) is: Z = S ⊙ (g hᵀ) where (g, h) minimize ‖P − S ⊙ (g hᵀ)‖_F. Since S² = 1 entrywise, this is equivalent to finding (g, h) as the top rank-1 SVD of (S∘P). Then Z = S ⊙ (g hᵀ). For the consensus variable P = U + Λ ∈ R^{d_out×r}, the two vectors g ∈ R^{d_out}, h ∈ R^r are computed from the top singlet of the d_out×r matrix S∘P via power iteration (O(d_out·r) per call, strictly cheaper than the existing O(r³) Cholesky).

### Current implementation

Column-only: `out[:, j] = s[:, j] * (s[:, j]ᵀ z[:, j] / ‖s[:, j]‖²)`. One scalar per column, row vector h fixed at all-ones. The docstring records three column-only variants tried; the two-sided optimum was never in the set.

### Fix

Replace the column-wise projection with the two-sided optimum: compute `M = sign(P) ∘ P` (entrywise), compute top rank-1 SVD of M → σ, u₁, v₁, set `g = u₁ * √σ`, `h = v₁ * √σ` (or equivalent split), return `Z = sign(P) ⊙ (g hᵀ)`. The SVD is O(r³) — same complexity class as the existing Cholesky solves (r ≤ 512). Dead columns (all-zero input): Z gets the positive `DEFAULT_SCALE_FLOOR`.

## Rho schedule — scale-aware endpoints

### Paper

[NQ] App. C :3008: "We employed a linear ADMM penalty scheduler for 400 factorization steps, for each weight matrix across all models." The line does NOT specify the rho range [1e-2, 1e2] — that range was an informed assumption calibrated on r137-scale targets.

### Defect

The iteration runs in the normalized space (`w/T`, O(1) norm). At rho=1e2, the consensus RHS `rho*(zu−lu)` dominates the data RHS `vᵀwᵀ` (O(1)) by two orders of magnitude, crushing U onto the ±1-shaped proxy before the continuous fit converges. Measured: capping rho at 1e-1 lifted raw cos on 6/7 linears (mean +0.11). The solver reports `converged=False` after every measured run — the tolerance is unreachable within 400 iterations at the shipped endpoints.

### Fix

Keep the linear schedule shape; cap `rho_end` at `1e-1` in the normalized regime. `rho_start` stays at `1e-2`. The data term is O(1) by construction (target divided by its own Frobenius norm at entry), so `rho_end = 1e-1` keeps the consensus term at ≤10% of the data term magnitude at the final iteration — sufficient to nudge latents toward the binary proxy while preserving the continuous fit.

The paper's schedule is a shape prescription, not an endpoint prescription. The endpoints are an implementation choice; a `rho_end` that prevents convergence is a defect, not a deviation.

### Discarded alternatives

- **Scale-relative rho (rho × ‖W̃‖_F):** "converges" at iteration 1 with product unmoved — trivial dominance, not adjustment.
- **Scaling epsilon:** bit-identical to capping rho in measured probe (the tolerance is not the binding constant).
- **Adaptive rho:** adds a tunable heuristic the paper doesn't describe; the fixed linear schedule with constrained endpoints is simpler and stays closer to the paper's algorithm shape.

### Open Questions

None — endpoints are an implementation choice the paper leaves unspecified, and the measured defect is the rho_end that prevents convergence.

## Interaction with pack-stage bottleneck

The decomposition showed post-pack cos clamped to 0.34–0.45 regardless of raw fit quality (0.32–0.84). Fixing rho and SVID improves the raw continuous fit; the pack stage remains the binding constraint, and its fix is a separate change. This change improves the solver's input to the pack stage; it does not claim to move the shipped number alone.

## Implementation crystallization — 2026-10-25 (Phase 2 convergence-test revision)

`tests/ptq/test_admm_convergence.py` was revised to the Phase-2 ruling. One clause of that
ruling is refuted by measurement and is deliberately NOT asserted.

Environment for every number below: dev venv, python 3.12.14, torch 2.14.1+cpu
(`torch.version.hip=None`, `torch.cuda.is_available()=False`), numpy 2.5.4,
`OPENBLAS_NUM_THREADS=1`, numpy solver backend, K=400. Note the tree has moved past this
change's `rho_end = 1e-1`: lane-3's normalized regime landed at `97a4725` and the schedule is
now `rho in [0.0, 1.0]` with `admm_reg = 3e-2`.

Gated, as measured:

- Fixture is a planted rank-32 target, rank-64 factorization, gamma = 0.2, and the ruling's
  premise is now **measured in the fixture** instead of assumed: the rank-64 truncated-SVD
  warm start represents that target at cos **0.999715** (>= 0.99).
- `cos(U V^T, W) = 0.933478 > 0.5`. The consensus-proxy product that `block_recon` actually
  hands to `magnitude_balance`, `P_u P_v^T`, measures 0.941705; `U V^T` is the operand the
  ruling names, so that is what is gated and the proxy value is recorded beside it.
- `cos(U V^T, W) = 0.933478 > cos(sign(P_u) sign(P_v)^T, W) = 0.883145`. `sign(P_u)` is the
  ruling's `sign(zu)`: `svid_proxy` preserves its input's sign structurally.
- `converged` is recorded, not gated: 400/400 iterations, `converged=False`.

**Refuted: "decreases monotonically across the first 200 iterations"** (tasks row 29 item 1,
and the delta scenario `rho_end_bounded_for_convergence`). The residual is not monotone on any
measured target: 67 of the 199 first-half steps rise on the shipped fixture (peak 0.877 at
iteration 5; first half 0.4243 -> 0.3443), and all 20 recorded real `layers.0` rows in
`artifacts/lane0-fire-r3-20261024/rows.jsonl` are non-monotone (2-66 rises per row; one row —
arm B, `self_attn.o_proj` — does not even net-decrease over the first 200). The shape is
structural, not noise: the consensus term pulls the warm start onto the +-1 proxy before the
data term reasserts itself, so the residual rises first and then descends to its floor
(terminal 0.01738 == trace min). The requirement of record folded in
`openspec/specs/qk-ptq/spec.md` carries no adverb — "the consensus residual ... decreases
across iterations" — and that is the form the test gates, in kill-only shape (net decrease
over the first half plus a terminal value at the schedule's own floor). Asserting the adverb
would assert a convex descent this non-convex iteration does not have.

**Finding for the planner (not actioned here).** The delta scenario
`rho_end_bounded_for_convergence` over-claims relative to the folded requirement of record,
and its `WHEN` is "a rank-512 target from a real transformer linear layer" while the unit
fixture is a synthetic rank-64 target (a real layer would put a model in the test lane; the
probe rows above are the supporting evidence). Amending a SHALL is the planner's call, so the
delta spec is left untouched and reported instead — otherwise archiving folds an adverb into
the requirements of record that 20/20 measured runs contradict.

## Implementation crystallization — 2026-10-25 (planner: delta-spec amendment)

The delta scenario `rho_end_bounded_for_convergence` was amended by the planner (t_a60cc8b7):

- **Adverb dropped.** "decreases monotonically" → "makes net progress across the first 200
  iterations (net decrease from start of the second half)". The requirement of record in
  `openspec/specs/qk-ptq/spec.md:76` carries no adverb — "decreases across iterations" — and
  the measured evidence (67/199 first-half rises on the shipped fixture; 20/20 real
  `layers.0` probe rows non-monotone, one row not net-decreasing) refutes monotonicity. The
  shape is structural: the consensus term pulls the warm start onto the ±1 proxy before the
  data term reasserts itself. Asserting the adverb would fold a knowingly-false gate into the
  requirements on archive.

- **WHEN updated to name the fixture it is gated by.** The old text said "a rank-512 target
  from a real transformer linear layer" — the unit convergence gate (the fixture the test
  actually exercises) is a planted rank-32 target at rank-64 factorization (`gamma=0.2`,
  warm-start cos ≥0.99). A real layer would put a model in the test lane. The real-layer
  probe rows (`artifacts/lane0-fire-r3-20261024/rows.jsonl`) are cited in the WHEN as
  supporting evidence that the non-monotone shape is structural, not a fixture artifact.

- `openspec validate qk-fix-admm-solver --strict` re-run after the edit (green below).

## Implementation crystallization — 2026-10-25 (rho_end reconciliation, ruling R4)

**The delta's `rho_end ≤ 1e-1` bound was STALE against the regime the solver
accepts, and is amended in place to the closed interval `[0.0, 1.0]`.**

- **The shipped regime moved and the delta did not.** Lane-3's normalized
  regime landed at `97a4725`; `PTQConfig.rho_end` defaults to **`1.0`**
  (`src/quantibbler/ptq/config.py:181`) with `rho_start = 0.0` and
  `admm_reg = 3e-2`, and the schema refuses `rho_end < rho_start`
  (`config.py:303-304`). Folding `≤ 1e-1` into `openspec/specs/qk-ptq/spec.md`
  would have written a requirement of record that the shipped default
  violates on every run — the delta out of step with its own code.
- **The tighter endpoint was refuted operationally.** The faithful-ADMM probe
  card `t_83ef0ef7` was killed by @user on 2026-10-24 on the forensics that the
  **rho endpoint cap dominates the SVID-fix effect** in faithful runs — the
  rho cap, not the proxy, is the dominant lever on the measured residual and
  cos trajectories. An endpoint retained only because it was once written down,
  against both the shipped default and the measurement, is precisely the defect
  this requirement's own last sentence names ("an endpoint that prevents
  convergence is a defect"). It therefore cannot ride into the requirements of
  record.
- **The bounded-shape intent is preserved, only the interval moved.** The
  requirement still constrains `rho_end` to a stated interval in the normalized
  regime, and now also names the ordering constraint the schema enforces. The
  paper prescribes the linear *shape* ([NQ] App. C :3008), never the endpoint —
  the endpoints are explicitly an implementation choice, which is what makes
  amending them a reconciliation rather than a paper deviation.
- `npx --yes @fission-ai/openspec validate qk-fix-admm-solver --strict` re-run
  after this edit → valid, RC=0.

### Superseded wording

The prior delta sentence read: *"The rho endpoint SHALL be bounded so the
consensus term does not overwhelm the unit-scale data term: in the normalized
regime …, `rho_end` SHALL NOT exceed `1e-1`."* That text is retracted and must
not be quoted as the requirement: it contradicted `config.py:181`. The amended
sentence above replaces it in the same file.
