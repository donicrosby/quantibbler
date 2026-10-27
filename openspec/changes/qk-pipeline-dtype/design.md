# Design: qk-pipeline-dtype

## Architecture: dtype-agnostic pipeline

The pipeline becomes a function of `compute_dtype: torch.dtype` — one dtype
selects the precision for every tensor the pipeline creates or transforms.
The declared dtype is resolved once at entry from `PTQConfig.compute_dtype`
(validated: `"bf16"`, `"float32"`, `"float64"`) and flows through every stage
as an explicit parameter.

### Storage vs accumulation vs compute

Three distinct precision roles:

| Role | Default under bf16 | Scope | Justification |
|------|-------------------|-------|---------------|
| **Storage/compute** | `bf16` | Latents (U, V), scales (s1, s2), bias, activation captures, STE forward, TuneFP/STE/KD gradients, packed sign extraction | User mandate: one dtype end-to-end |
| **Accumulation** | `float32` | Dot-product accumulators (ADMM `VᵀV`, `UᵀU` normal equations), Cholesky factorization, robust-diagonal running sums, CE-gradient squared-statistic accumulation, KL-divergence sum | bf16's 7-bit mantissa: a dot product over d=4096 terms accumulates ~1.4 ULPs per term → ~12 bits lost. float32's 23-bit mantissa recovers ~16 guard bits relative to bf16, safe for ranks up to ~4096 before needing float64. |
| **Parity reference** | `float64` (unchanged) | NumPy solver backend (unit-test-only), kernel parity probes | The executable specification stays at the highest precision — the oracle does not move |

The accumulation dtype is a per-site named constant, not an adaptive heuristic.
Each site that widens declares its dtype and the arithmetic bound that justifies
it. The pipeline SHALL NOT silently upcast any tensor — every widening is an
explicit `tensor.to(accum_dtype)` call with a comment naming the justification.

### Dtype flow diagram

```
HF weights (bf16)
    │
    ├─ Calibration ──► bf16 captures, float32 accumulators
    │
    ├─ Preconditioner ──► bf16 D~_in / D~_out (diagonals, no product)
    │
    ├─ ADMM solve
    │   ├─ Normal equations ──► float32 accumulators
    │   ├─ Cholesky ──► float32 factorization
    │   ├─ Latents ──► bf16 storage
    │   └─ Scales ──► bf16 storage
    │
    ├─ Block reconstruction
    │   ├─ TuneFP ──► bf16 forward/backward
    │   ├─ STE ──► bf16 (sign extraction, gradient passthrough)
    │   └─ TuneScalesKD ──► bf16
    │
    ├─ Sign extraction ──► bf16 → packed bits (±1 → 0/1)
    │
    └─ Export ──► artifact remainder dtype (default bf16)
```

### `FactorizedLinear` dtype contract

- Constructor: latents stored as `nn.Parameter` at `compute_dtype` (was
  always `double()`/`torch.float64`). The `from_weight` SVD init: on CPU,
  uses the NumPy float64 path then casts to `compute_dtype` before storing;
  on GPU, uses `torch.linalg.svd` at **float32** accumulation dtype
  (explicit, not the device dtype) — `torch.linalg.svd` at bf16 is
  platform-dependent and may not converge; float32 SVD is a justified
  accumulation site under class (b) in the evidence gate. The parity
  oracle compares against the NumPy float64 SVD path.
- `_factor_terms`: operates at `compute_dtype` — the `activation_dtype`
  parameter becomes the pipeline's `compute_dtype`. The STE passthrough
  `sign(u) + u - u.detach()` works identically at any float dtype.
- `forward`: the matmul `(s2 ⊙ f(V)) @ ((f(U)ᵀ @ (s1 ⊙ U)ᵀ) @ x) + b`
  executes in `compute_dtype`. No intermediate widening.

### PTQConfig changes

```python
# OLD:
_COMPUTE_DTYPES = ("float64", "float32")
compute_dtype: str = "float64"

# NEW:
_COMPUTE_DTYPES = ("bf16", "float32", "float64")
compute_dtype: str = "bf16"
```

`np_compute_dtype` becomes a property mapping `"bf16"` → **raises a named error** (`NumPyBackendUnsupportedDtypeError`: NumPy backend is float64-only; bf16 and float32 compute require the torch backend). The torch backend maps `"bf16"` to `torch.bfloat16`.

The `remainder_dtype` default changes from `"fp32"` to `"bf16"` in the driver
(`tools/quantize_hf_model.py`) — separately settable, but default-tracks-compute
per the user ruling.

**Orchestrator ruling (2026-10-27):** This change OWNS the `--remainder-dtype` default (bf16, tracking compute). The active `qk-artifact-dtype` change (same qk-ptq capability, same flag) carries a contradictory fp32-default with byte-identity scenario RDW-1.1. Resolution: qk-artifact-dtype MUST either fold this default flip into its delta/tasks (absorbs the ruling) or sequence after pipeline-dtype with an explicit `## Supersedes` note — it cannot land with the fp32 default intact. The export writer, manifest `[sidecar] dtype`, and report `remainder_dtype` field are co-owned: pipeline-dtype sets the default; qk-artifact-dtype defines the format contract.

### Driver changes

```python
parser.add_argument("--compute-dtype", default="bf16", choices=("bf16", "float32", "float64"))
# --remainder-dtype default: follows compute_dtype when unset
parser.add_argument("--remainder-dtype", default=None, choices=("bf16", "fp32"))
# Post-parse: if remainder_dtype is None, set from compute_dtype mapping.
```

## Architecture: device-generic execution

### Current state (v6-class fire)

| Stage | Device | Wall fraction (est.) | GPU idle reason |
|-------|--------|---------------------|-----------------|
| Calibration capture | Accelerator (torch) | ~10% | — |
| TuneFP | Accelerator | ~5% | — |
| **Solver (ADMM/SVD)** | **CPU (NumPy)** | **~60%** | `.cpu().numpy()` round-trip per layer, NumPy SVD/Cholesky on one socket |
| STE | Accelerator | ~5% | — |
| TuneScalesKD | Accelerator | ~15% | — |
| Sign/pack/export | CPU | ~5% | — |

The solver is the dominant wall consumer and runs on CPU, leaving the GPU idle
for ~60% of wall time. Each layer crosses the host↔device boundary twice:
`weight.cpu().numpy()` → solve → `torch.from_numpy().to(device)`.

**Wall-fraction note:** The ~60% solver estimate derives from the per-layer
breakdown above (ADMM + SVD on CPU/NumPy, one-socket). The driver's own
`--device` help text records a MEASUREMENT: torch stages ≈ 69% of wall on
the CPU path and solver crossings \"cost seconds rather than hours\". Both
numbers are consistent in context — the 69% is torch-stage fraction on the
CPU-only path (solver rounds are the smallest component there), and the ~60%
is the GPU-idle fraction on an accelerated path (solver is the largest
component when autograd stages accelerate). The solver port's benefit is the
elimination of per-block host round-trips (hundreds of `.cpu().numpy()` calls)
and the GPU-idle window — not the crossing time itself.

### Target state

| Stage | Device | Notes |
|-------|--------|-------|
| Calibration capture | Accelerator | Unchanged |
| TuneFP | Accelerator | Unchanged |
| **Solver (ADMM/SVD)** | **Accelerator (torch)** | `torch.linalg.svd` / custom Cholesky on device tensors; NumPy path stays unit-test-only |
| STE | Accelerator | Unchanged |
|| Sign/extract/pack | CPU (host) | Boundary to packed-bit representation; avoids `torch.sign(0)=0` vs the `x>=0→+1` convention required for pack-bit parity |
| TuneScalesKD | Accelerator | Unchanged |
| Eval | Accelerator | Unchanged |
| Export | CPU (host) | Write to disk — inherent host path |

The solver moves to `torch` tensors on the accelerator device. The two
host↔device round-trips per layer are eliminated. The NumPy backend remains
the float64 parity oracle, reached only by `--solver-backend numpy` in unit
tests — the torch backend on the accelerator is the production default.

### Device selection

- `--device auto` (default): takes the first available CUDA/ROCm device, or
  REFUSES with a named error (no silent fallback to CPU — the v6 autograd
  path already follows this rule). When no accelerator is present, `auto`
  refuses; the user must explicitly choose `--device cpu`.
- `--device cpu`: explicit CPU path with good-guest thread sizing.
- `--device cuda:N`: specific device index.

The solver is NOT a separate device decision — it runs on the same device as
every other stage. The torch tensor on the device is the solver's input and
output; there is no NumPy intermediary.

### Solver torch backend: dtype-aware paths

The torch solver backend (`solver_backend.py`, `lb_admm.py`) parameterizes on
`compute_dtype`. The key operations and their accumulation dtype:

| Operation | Storage | Accumulation | Justification |
|-----------|---------|-------------|---------------|
| `VᵀV` normal equation | bf16 input | float32 output | Sums over `din` terms per `(r, r)` entry |
| Cholesky `L @ Lᵀ = VᵀV + (ρ+λ)I` | float32 | float32 | Factorization needs ~23-bit mantissa for r≤4096 |
| Continuous update `solve(L, rhs)` | float32 | float32 | Back-substitution preserves input precision |
| SVID proxy `sign(P) ⊙ (g hᵀ)` | bf16 | float32 SVD | Top-1 SVD needs stable convergence |
| Dual update `Λ ← Λ + U − Z` | bf16 | bf16 | Element-wise add, no summation widening |
| Balance `η = sqrt(‖V̂‖_F / ‖Û‖_F)` | bf16 | float32 norms | Frobenius norm sums over all entries |
| Scale extraction `mean(|û_i|)` | bf16 | float32 sums | Sum over r terms per row |

The torch backend SHALL produce results within the tolerance declared per
compute_dtype (see "Tolerances" below) relative to the NumPy float64 reference.

### Per-dtype tolerances

**Superseded by C1 below.** The table's tolerances were declared on the converged-output cosine metric; measurement proved that metric measures basin selection, not dtype fidelity (C1). The correct metric is the first-update residual relative difference, and the tolerance values (1e-2 / 1e-4 / 1e-12) transfer unchanged — the bounds were correctly derived from format ULP analysis; the defect was the metric, not the values.

| Compute dtype | ADMM recon cos tolerance vs float64 | Parity gate class |
|---------------|-------------------------------------|-------------------|
| float64 | 1e-12 (existing) | float64-parity |
| float32 | 1e-4 (backward error on ADMM round-off) | float32-parity |
| bf16 | 1e-2 (2^-7 per element, accumulated over r·d terms) | bf16-parity |

These tolerances are declared, not calibrated — they derive from the format
ULP bounds. The parity suite extends across the dtype×device matrix:
`{bf16, float32, float64} × {GPU, CPU}` with per-dtype bounds as above.
The pipeline owns these tolerance definitions independently — they are
pipeline-level end-to-end reconstruction cosine bounds measured at the
block output (unpacked reconstruction vs float64-NumPy reference output),
distinct from the per-operation backward-error bounds in
`qk-kernels-dtype-generic` (which govern kernel-level precision for
individual matmuls and pack operations). The first production run SHALL
confirm these declared bounds per dtype.

**dtype×device matrix provenance:** The six-cell matrix `{bf16, float32, float64} × {GPU, CPU}` cannot be fully proven on CI (GPU cells require an accelerator). The CPU cells SHALL run always (they are the torch CPU backend path). The GPU cells SHALL run on the designated GPU hardware lane: the repo's rocm-family image `quantibbler-ptq:rocm-official` (digest-pinned) on the Strix Halo box (gfx1151), with exact skip-count discipline — every skipped GPU cell MUST produce a documented skip event naming the reason (no GPU present). The first fire on the GPU lane proves the three accelerator cells.

### Hardcoded-dtype site enumeration (task 6.1, populated)

The grep against the full pattern set
(`grep -rnE '\.double\(\)|torch\.float32|torch\.float64|np\.float32|np\.float64|astype\(.*float' src/quantibbler/ptq/ tools/quantize_hf_model.py`)
is the evidence gate. Resolution classes:

- **(a) Parametric:** the site uses the declared `compute_dtype` / accumulation
  dtype, or is the declaration/mapping itself.
- **(b) Accumulation-site:** a named accumulation/host grade with the arithmetic
  or contract justification in the comment beside it.
- **(c) Parity-oracle:** a deliberate float64 literal in the NumPy parity path,
  a unit test, or a reference fixture.

`tools/quantize_hf_model.py` returns **zero** hits: every dtype decision on the
driver path is the declared one.

| Site | Class | Note |
|------|-------|------|
| `calibration.py` `_TINY` | (b) | named dead-channel floor constant |
| `calibration.py` `_accumulator_dtype` (`torch.float32`) | (a) | the bf16→float32 accumulation mapping itself |
| `calibration.py` `_to_numpy_stat` | (b) | statistic STORAGE schema boundary (documented) |
| `solver_backend.py` `host_np_dtype` (`np.float32`) | (a) | the declared host-grade map itself |
| `_factorized.py` accelerator SVD (`torch.float32`) | (b) | named float32 accumulation for `torch.linalg.svd` |
| `_factorized.py` NumPy SVD init (`torch.float64`) | (c) | `# parity-oracle: float64 by design` |
| `_factorized.py` `check_layer_magnitude` | (b) | catastrophe gate: Frobenius ratio over the whole block |
| `lb_admm.py` `DEFAULT_SCALE_FLOOR` | (b) | named dead-channel floor constant |
| `api.py` `scale_dtype` dispatch | (a) | the declared `config.scale_dtype` storage knob |
| `api.py` `_phase1_chunk_rows`, `_block_stage` branch | (a) | dtype comparisons, not casts |
| `api.py` finiteness reads, `deserialize` loads | (b) | validation/load at the widest host grade |
| `scale_calib.py` KD logit cache dtype | (a) | derived from the measured logit dtype |
| `_stats.py` weight fingerprint digest | (b) | identity canonicalization: the digest must not depend on the run's dtype |
| `_stats.py` loss-weight host vectors | (b) | host statistic, cast to the compute dtype at the loss site |
| `checkpoints.py` state archive + torch reload | (b) | persisted-payload storage schema (int8 bits, float64 scales) |
| `_probe.py:160` | — | comment text only, no call site |
| `e2e.py` tiny gate fixture | (c) | reference fixture |

**Flagged for the planner (one delta).** Three of the (b) rows above are the
SAME shape and the requirement's three-class vocabulary does not name it: a
**persisted-payload / identity-canonicalization grade** (`checkpoints.py`'s
archive schema, `_stats.py`'s weight-fingerprint digest, `api.py`'s load path).
These are not compute paths — a checkpoint's float64 scales and a fingerprint's
float64 canonical form are BYTE CONTRACTS, and re-grading them to the compute
dtype would move the resume admission and the identity digest without any
numerics changing. They are enumerated as (b) here because (b) is the closest of
the three; the class list needs a fourth entry ("storage/identity grade, not a
compute path") or an explicit exclusion, and that is a spec change with an owner.

## Architecture: good-guest resource discipline

### Anti-pattern (loadavg-47–70 incident)

The v6 fire's `OMP_NUM_THREADS=16` (hardcoded in the container wrapper's
OMP environment block) left 0 free CPU cores on this host's
48-thread box during BLAS-heavy ADMM solves. Combined with the solve's
single-socket restriction, the host's loadavg spiked to 47–70 —
starving co-tenant agents and services. This is the anti-pattern the
good-guest rule exists to prevent.

### CPU thread sizing

At pipeline startup, the thread count is computed:

```python
import os

try:
    n_cores = len(os.sched_getaffinity(0))  # cpuset-respecting
except AttributeError:  # sched_getaffinity unavailable (non-Linux)
    n_cores = os.cpu_count()
loadavg_1m = os.getloadavg()[0]
if loadavg_1m > n_cores:
    n_threads = max(1, n_cores // 4)  # heavy load: quarter the cores
elif loadavg_1m >= n_cores * 0.5:  # moderate: at or above half (spec: thread_sizing_moderate_load)
    n_threads = max(2, n_cores - 4)  # moderate load: leave 4 cores
else:
    n_threads = max(2, n_cores - 2)  # light load: leave 2 cores
```

**Mechanism:** The computed value sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` in `os.environ` **before** any BLAS or PyTorch import
— the pipeline driver SHALL execute this sizing block as the very first
statements before `import torch` or any linear-algebra import. Additionally,
`torch.set_num_threads(n_threads)` is called after torch import for the
PyTorch-internal thread pool. A `--max-threads N` override exists for explicit
opt-out. A startup assertion SHALL confirm that `torch.get_num_threads()` and
effective BLAS threads reflect the computed value (not a stale default).

### GPU memory discipline

- Default: `torch.cuda.set_per_process_memory_fraction(0.85)`
- `--gpu-memory-fraction F` override for opt-out.
- **Unified-memory lane (APUs, Strix Halo/gfx1151):** On iGPUs where device
  memory IS host RAM (kfd-granted, outside the container fence), the
  per-process fraction cap alone can still drain the host. For APU devices,
  the pipeline SHALL additionally measure `max_memory_reserved()` at startup
  and enforce an absolute-GB cap (`--gpu-memory-gb G`, defaulting to 0.85 ×
  host-visible RAM on APUs). The peak reserved memory is recorded in the
  run report.
- No unbounded allocations: gradient checkpointing stays on (bounded retained
  graph). Batch sizes stay at the pipeline's declared small values — the
  default batch sizes are `bs=4` for TuneFP, `bs=1` for STE (verified
  against the stage-parameter defaults in the current codebase) — not
  paper-claimed defaults.

### The good-guest contract

The pipeline's resource consumption SHALL be bounded at startup based on
observed host state, not hardcoded constants. A `--no-good-guest` flag
disables the bounds (explicit opt-out — for benchmarking). The default is
considerate.

## Configuration migration

Users with existing `PTQConfig(compute_dtype="float64")` calls will see a
behavior change: the default is now bf16. Explicit `compute_dtype="float64"`
calls are unchanged. The config validation accepts all three values.

Artifacts produced under the old default carry `"compute_dtype": "float64"` in
their provenance — they are distinguishable from bf16 artifacts.

## Open Questions

### OQ-1: bias-zero root cause (resolved 2026-10-27 by t_59815b89)

**Status:** RESOLVED — not dtype-related.

The parallel bug-hunt (t_59815b89) completed: the v6b artifact's 196/196 bias
vectors are exactly 0.0 because Qwen3-0.6B is bias-free by config
(`attention_bias=false`, no MLP bias). Zeros are legit-zero math — born at
`_block_stage.py:438-441`, carried faithfully through journal, checkpoints,
and artifact. Bias-carry is exonerated; it is NOT a silent no-op. The bias
path is nonetheless rewritten to dtype-agnostic form in this change as
defense-in-depth — the `bias.double()` hardcodes at `_factorized.py:91,92,95`
are replaced with `compute_dtype`. Two peripheral findings from the hunt
(None-semantics overload, bias-free-model coverage gap) are routed to a
separate card (t_373bdeff) and do NOT expand this spec's scope.

### OQ-4: v6b 14.14x PPL damage — expected fix validation

**Status:** OPEN — owned by planner, resolved by first production fire.

With bias-carry exonerated (t_59815b89), the dtype-mixing suspect class is the
SOLE standing explanation for the v6b NOT-SANE verdict (PPL 285.79 vs 20.21).
This change is the damage fix: its dtype-consistency enforcement (one
declared dtype through every stage, no silent upcast, accumulation-site
discipline, grep-evidenced hardcoded-dtype elimination) is load-bearing, not
hygiene. The first production fire under bf16-default MUST measure PPL sanity
against the 285.79 baseline — full wikitext-2-raw-v1 evaluation with greedy decoding
on the same split (the driver's actual eval corpus, the one the 285.79 baseline
was measured on), recorded in the provenance report. (No fire is in scope of
this specification phase; same resolve-by-production pattern as OQ-2/3.)

### OQ-2: float32 accumulation sufficiency at rank 512+

**Status:** OPEN — owned by planner, resolved by first production measurement.

The float32 accumulation for ADMM normal equations is justified arithmetically
for ranks up to ~4096. The first production run at rank 512 under bf16 compute
+ float32 accumulation MUST measure the ADMM convergence trajectory (residual
norm per iteration) and compare it against a float64 run on the same layer.
Divergence (more than 10% additional iterations to converge, or failure to
converge) would indicate float32 accumulation is insufficient at this rank.

### OQ-3: bf16 STE gradient fidelity

**Status:** OPEN — owned by planner, resolved by first production measurement.

The STE passthrough `sign(u) + u - u.detach()` at bf16 involves three
operations with ~7-bit mantissa each. The gradient scale (`d/dU = s1`) may
lose precision relative to the float64 path. The first production run MUST
compare the per-linear STE gradient norm against the float64 reference path on
at least one layer and record the relative difference. A difference exceeding
1e-3 relative would indicate bf16 STE gradients need a float32 intermediate
(the accumulation dtype can widen the STE path without widening the latents).

## Discarded alternatives

### D1: float32 default instead of bf16

**Discarded.** The user ruling explicitly demands bf16 end-to-end. float32
would be a defensible middle ground (23-bit mantissa, no accumulation needed
for moderate ranks) but contradicts the ruling. float32 remains selectable.

### D2: Keep solver on CPU, only move autograd to GPU

**Discarded.** The solver is ~60% of wall time. Moving only the autograd
stages while keeping the solver on CPU would leave the GPU idle for most of
the run and retain the per-block host↔device round-trips. The ruling demands
GPU for every compute-heavy stage.

### D3: Accumulation dtype as a separate config knob

**Discarded.** A user-tunable `accumulation_dtype` invites misconfiguration
(bf16 accumulation for a rank-4096 solve would silently produce garbage). The
accumulation dtype is a per-site named constant derived from the format
properties, not a user decision. The one user-reachable override is
`compute_dtype` itself — choosing float32 or float64 eliminates the
accumulation question entirely.

### D4: NumPy solver as CPU fallback when no GPU present

**Discarded.** The existing spec rule (torch backend without accelerator
refuses) is correct: a CPU torch backend doing ~400 Cholesky solves per layer
on one socket is the same performance as NumPy, and NumPy is the parity oracle
— never a fallback. The explicit `--device cpu` path uses torch CPU tensors.

### D5: Remainder dtype always tracks compute dtype (no separate knob)

**Discarded.** `compute bf16 + store fp32` is a legal declared combination for
users who want bf16-quality pipeline numerics with fp32-precision artifact
sidecars (e.g. for bit-exact comparison against pre-bf16 artifacts). The
default tracking is the user ruling's intent; the separate knob preserves the
declared-separation principle from `qk-artifact-dtype`.

## Implementation crystallization (2026-10-27)

Dated record of what the implementation found, per the repo's
"behavior change lands with its crystallization entry" rule. Measured numbers
replace authoring-time estimates; superseded statements are kept beside them so
they cannot be quoted as current.

### C1: the `admm_parity_per_dtype` tolerance cells do NOT reproduce as written

**Superseded claim (authored):** `admm_parity_per_dtype` — the reconstruction
cosine against the float64 NumPy reference satisfies **≥0.99 (bf16)**,
**≥0.9999 (float32)**, **≥1−1e-12 (float64)**; and the per-dtype tolerances table
above (1e-2 / 1e-4 / 1e-12) is "declared, not calibrated".

**Measured (torch 2.14.1+cpu, explicit `QUANTIBBLER_DEVICE=cpu`, 3 seeds,
`admm_max_iter=1200`, converged arms):** the converged-output cosine does not
separate dtypes at all — including the float64 arm against the float64 NumPy
reference:

| Arm | 1 − cos vs converged float64 NumPy reference (3 trials) |
|-----|--------------------------------------------------------|
| bf16 | 0.32 – 0.68 |
| float32 | 0.23 – 0.64 |
| **float64 (torch)** | **0.25 – 0.68** |

The float64-torch vs float64-NumPy pair is the control: it differs by 0.25–0.68
at the CONVERGED point while agreeing to 2.5e-15 at the first update. The
SVID proxy is a sign-quantized rank-1 projection, so the iterate is a discrete
sign pattern: round-off differences between two backends select among
near-equal discrete optima, and hundreds of iterations amplify the selection.
A converged-output cosine therefore measures **basin selection**, not dtype
fidelity, and cannot carry any per-dtype bound in either direction — the cells
are not merely unsatisfied, they are not a measurement of the thing they name.
Resolution: the reachable metric is the EARLY trajectory against the reference,
which does separate the dtypes at their declared classes:

| compute dtype | measured | declared class |
|---------------|----------|----------------|
| bf16 | 1.0e-3 | 1e-2 (holds) |
| float32 | 3.8e-7 | 1e-4 (holds) |
| float64 | 2.5e-15 | 1e-12 (holds) |

(metric: relative difference of the FIRST ADMM update residual vs the float64
NumPy reference; `tests/ptq/test_pipeline_dtype_generic.py::test_scn_admm_parity_cpu_cell`)

Also measured, same fixture: all three dtypes converge within the budget
(bf16 344 / float32 347 / float64 360 iterations at `admm_max_iter=900`), so the
budget question is not what separates the cells.

**Owner: planner.** This is a measurement row, so the numbers above replace the
authored ones here and in the tolerance table; the `admm_parity_per_dtype`
scenario's metric in `specs/qk-ptq/spec.md` (a converged-output cosine bound)
needs a ruling — either the scenario moves to the early-trajectory metric, or it
declares the basin-selection caveat explicitly. The same applies to
`accumulation_identity_at_higher_dtypes`' float64-torch clause.

**Orchestrator ruling (2026-10-27):** The scenario moves to the early-trajectory
metric. The first-update residual relative difference is the correct parity
definition for all three compute dtypes, and the tolerance values (bf16 ≤1e-2,
float32 ≤1e-4, float64 ≤1e-12) are confirmed.

*Justification.* The first ADMM update computes the continuous latent update
from identical initial conditions, target, and regularization — only the
arithmetic precision varies. It is measured BEFORE the first sign projection,
so no discrete basin selection has occurred. The float64-torch vs float64-NumPy
control arm proves the metric's validity: the two backends agree to 2.5e-15 on
the first update (matching the format ULP bound) while diverging to 0.25–0.68
cosine distance at convergence. The converged-iterate cosine is explicitly
NOT a parity metric — it measures which discrete basin the iterate selected,
not the fidelity of the floating-point arithmetic. Two dtype runs that land
in different basins are both "correct" by their own arithmetic; no per-dtype
bound on converged cosine is satisfiable by construction.

*Tolerance derivation from first principles (not vibes).* bf16 has 7 mantissa
bits + implicit 1 ≈ 8 bits ≈ 2.4 decimal digits. The normal equations form at
float32 accumulation, but the input terms carry bf16's ~2e-3 relative error
per element; for din ≈ 4096, the accumulated error ≈ 2e-3/√4096 ≈ 3e-5 per
entry, with Cholesky amplifying by the condition number. Conservative bound:
**≤1e-2** (matches measured 1.0e-3 with ~10× headroom). float32 has 23
mantissa bits ≈ 7.2 decimal digits; accumulated error ≈ 6e-8/√din ≈ 9e-10
per entry. Conservative bound: **≤1e-4** (matches measured 3.8e-7 with
~260× headroom). float64 has 53 mantissa bits; both backends use IEEE 754
double precision. Expected agreement ~1e-15, conservative bound: **≤1e-12**
(matches measured 2.5e-15). All three bounds derive from the format's
mantissa width and the operation's term-count, not from calibration — the
calibration confirms they are reachable.

*Spec amendment (this commit).* `admm_parity_per_dtype` scenario and
`accumulation_identity_at_higher_dtypes`' float64-torch clause both amended
to the first-update residual metric. Design.md tolerance table annotated as
superseded, pointing to this ruling. No tolerance value changed — the bounds
were correctly derived; the defect was the metric.

### C2: the polish entry was an int8 Gram (fixed here)

`greedy_polish` receives the binary factors as int8 (their storage type) and
formed `ub @ vb.T` at int8. NumPy's integer `@` accumulates in the INPUT dtype,
so the Gram overflows past rank 127 (measured: 300 ones → `44`). Every
production rank (>127) silently polished against a corrupted objective. The
dtype-generic port floats the bits at the declared work grade, which fixes it;
the tiny-rank fixtures (r ≤ 8) never overflowed, which is why it shipped.

### C3: GPU cells deferred, not dropped

The accelerator cells of the dtype×device matrix (`{bf16, float32, float64} ×
{GPU}`) cannot run in a CPU-only worker sandbox. `test_scn_admm_parity_gpu_cells_skip_with_reason`
emits a documented skip event naming the reason and the deferral target; the
cells run on the GPU lane card `t_f8e3b1d8` (host-side container discipline).

### C4: `ste_dtype_generic` deferred — the bf16-relative-to-f64 STE gradient record is not implemented

**Superseded claim (authored):** `ste_dtype_generic` (`specs/qk-ptq/spec.md`,
STE-refinement requirement) asserts that at bf16 compute the per-linear STE
gradient norm is recorded RELATIVE TO the float64 reference, and that a
deviation exceeding 1e-3 relative triggers a `ste_bf16_gradient_deviation`
event.

**Measured (this implementation):** `_ste.py` records `gradient_norm` per
linear at the declared compute dtype, and nothing more. There is NO float64
reference run inside the stage and no relative-deviation record:
`ste_bf16_gradient_deviation` has zero producers and zero readers tree-wide
(grep over `src/` and `tests/`), and both STE test files' diffs are solely
`compute_dtype` pins. The scenario is wholly unexecuted.

**Disposition: DEFERRED, declared.** Satisfying the scenario means running each
layer's STE a SECOND time at float64 purely to compute the comparison, inside
the production stage, plus a report event and its test — an instrumentation
change on the numerics path, not a pin. It also intersects **OQ-3**, which
already frames the same measurement as a first-production-run obligation and
whose outcome (whether bf16 STE gradients need a float32 intermediate) would
change what the event should assert. It is therefore declared here and routed
to the planner rather than improvised under this card. The shipped record —
`gradient_norm` at the declared compute dtype — is honest as far as it goes;
the relative-to-f64 record and the `ste_bf16_gradient_deviation` event are
named debt, tracked by the open task row and by OQ-3, and routed to the planner
card `t_76cceaa5`.

### C5: `bias_zero_assertion_gate`'s zero-bias-fails arm re-ruled — the gate fires on dropped bias, never on bias-free architecture

**Superseded claim (authored):** `bias_zero_assertion_gate`
(`specs/qk-ptq/spec.md`) asserted "a zero-bias artifact fails the gate even if
PPL passes" — a clause whose WHEN carries no architecture discriminator.

**Measured/exonerated (t_59815b89, recorded as OQ-1 above):** Qwen3-0.6B is
architecturally bias-free (`attention_bias=false`, no MLP bias); its artifact's
196/196 exactly-zero bias fields are legit-zero math — the faithful
serialization of absence, born at `_block_stage.py:438-441` and carried through
journal, checkpoints, and wire. The clause as written would REFUSE exactly this
artifact class, so it is not an implementable assertion — nearly every fixture,
including the e2e gate's own `TinyGateModel` (declared biases whose source
values are zeroed at construction, `e2e.py:120-129`), would fail it.

**Ruling (2026-10-27, reviewer R2, comment 477):** the gate's bias clause is a
dropped-bias detector, not a zero-bias detector. It fires iff a quantized
linear DECLARES a bias in the live model (`linear.bias is not None`) whose
SOURCE value is non-zero, and that linear's exported bias field is exactly zero
(or absent) — "bias SHOULD exist but didn't arrive". It never fires on
zero-source declared biases (a zeroed-at-source bias carries zero faithfully;
there is nothing to drop, and no observable difference) and performs no bias
assertion at all for bias-free architectures, whose all-zero fields are the
expected outcome.

*Spec amendment (this commit):* the scenario's WHEN is re-scoped to the
architecture-declared fire condition with PPL-independence preserved, and a
second scenario pins the legit-zero pass arm naming the Qwen3-0.6B trait
(`attention_bias=false`, no MLP bias). The requirement body's "no stage SHALL
zero or drop the bias" is re-scoped to "a bias the module declares".

**Implementing test acceptance shape (corrected):** both arms test through
`run_e2e_gate` (the gate holds the live model at quantize time, so the declared
bias is readable there). Fire arm — a corruption-injection pass in the
mechanism of the existing `flip_one_bit` self-test: a bias-bearing fixture (a
`TinyGateModel` variant whose linear biases are seeded non-zero, mirroring the
bias-bearing fixtures in `tests/ptq/test_bias_carry.py`) quantizes green with
every declared-bias linear's exported bias non-zero; a second run zeroing the
exported bias field while logits stay intact MUST fail the gate
(`pytest.raises`, the shape of `test_scn_e2e_gate_detects_corruption`). Pass
arm — the stock `TinyGateModel` (declared, zero-source biases) and a
bias-free variant (`bias=False` linears) both pass with all-zero bias fields;
the pack-boundary contract for the absent-bias case is already pinned by
`test_biasless_block_packs_zero_bias_v2` (PRESENT, exactly-zero v2 field). No
unit gate loads a real Qwen3 checkpoint; `attention_bias=false` is the config
trait the discriminator reads off the live model's modules.
