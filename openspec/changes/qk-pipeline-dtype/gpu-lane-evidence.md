# GPU-lane evidence: dtype x device matrix, per-stage timings, good-guest

Card: **t_f8e3b1d8** (GPU lane for `qk-pipeline-dtype`, gated on the CPU-side port
`t_4625e9ca`). Measured 2026-10-27 on the Strix Halo box (AMD Radeon 8060S,
`gfx1151`), via the host-side container discipline (worker sandboxes are CPU-only;
the GPU cells run in a container on the host daemon).

Base revision: branch `coder/t_4625e9ca-pipeline-dtype` @ `425cbc3` (checked out
into the lane worktree; trunk untouched). This document is the deliverable's
evidence; the patch carries the test legs that assert it.

## 0. Lane identity and launch discipline

| Field | Value |
|-------|-------|
| Image | `quantibbler-ptq:rocm-official-v6`, digest `sha256:7e397b0d8866ef6be22ad9b2da708df4703b3a06593cde0932e5a1b7139833fd` |
| Interpreter | python 3.12.3, `torch 2.14.0+rocm10.1.0`, `hip 7.16.26385`, `cuda.is_available()=True`, 1 device |
| Device | `AMD Radeon 8060S Graphics` (`gcnArchName=gfx1151`), `total_memory` 133 143 986 176 B (~124 GiB, unified/APU) |
| Container flags | `--rm --memory 16g --memory-swap 16g --device /dev/kfd --device /dev/dri --group-add <kfd gid> --network none -v <worktree>:/repo:ro -v <out>:/out -e PYTHONPATH=/repo/src -w /repo` |
| Labels | `qkb_launch_id=qk-pdtype-gpu-<UTC>Z`, `qkb_launch_owner=coder`, `qkb_launch_card=t_f8e3b1d8` |
| Breadcrumbs | `/opt/data/tmp/quantibbler-breadcrumbs.log`, written BEFORE every launch |

**Image deviation (stated, not silent):** the design's provenance note names
`quantibbler-ptq:rocm-official`. That image carries `pytest` (9.1.1) but not
`pydantic`, so the suite cannot run green there: on the lane's own files the
generic file collects cleanly (10 collected) and the bf16-default end-to-end leg
fails at runtime — `ModuleNotFoundError: No module named 'pydantic'` at
`src/quantibbler/ptq/checkpoints.py:55`, reached by the e2e leg's serialize path —
leaving 9 of 10 legs green. `quantibbler-ptq:rocm-official-v6` is the same
ROCm/torch build with the deps installed (the variant the `quantibbler` skill
documents for exactly this reason); it is what was run, digest-pinned above.
*Owner: planner* — the provenance sentence should name the variant the lane can
actually run.

## 1. Prerequisite reproduced: the CPU-side claim

Claim from `t_4625e9ca`: bf16 default runs end to end on CPU, pinned in
`tests/ptq/test_pipeline_dtype_generic.py`, with the three CPU cells of the matrix in.

At the **pristine parent branch tip** (`425cbc3`) the file collects **11** legs. On a
CPU-only host that is **10 passed, 1 skipped** — the single skip is the parent's own
`test_scn_admm_parity_gpu_cells_skip_with_reason`; on an accelerator host the same
pristine tree is **1 failed, 10 passed** instead (that failed assertion is F1 below).
The prereq's substance holds at the base: the bf16 e2e leg and the three CPU cells
pass there.

The run below is the **post-patch** CPU tier — this deliverable's patch, which moves
the accelerator cells out to the `integration`-marked GPU module and so is
skip-free (it is not a pristine-base measurement):

```
docker run <lane flags> --entrypoint python quantibbler-ptq:rocm-official-v6 \
  -m pytest tests/ptq/test_pipeline_dtype_generic.py -v -p no:cacheprovider
=> 10 passed in 6.17s   (0 skipped, 0 failed)
```

The 10 legs include `test_scn_admm_parity_cpu_cell[bf16|float32|float64]` and
`test_bf16_default_end_to_end_on_cpu`.

## 2. dtype x device parity matrix (six cells)

Fixture and metric are the CPU cells' own (design.md C1's reachable metric):
`w = default_rng(11).standard_normal((32, 24))`, rank 4, `admm_max_iter=900`, arm
vs the float64 NumPy reference. **Gated** metric = relative difference of the FIRST
ADMM update residual vs the reference. The per-dtype class is the declared
format-ULP bound from design.md "Per-dtype tolerances" — identical for both
devices, never widened.

| Device | compute_dtype | Declared class | Measured `rel_first` | Verdict | iters (converged) | returned U device / dtype |
|--------|---------------|----------------|----------------------|---------|-------------------|---------------------------|
| CPU  | bf16    | 1e-2  | 1.017e-3 | PASS | 378 | `cpu` / `torch.bfloat16` |
| CPU  | float32 | 1e-4  | 3.132e-7 | PASS | 416 | `cpu` / `torch.float32`  |
| CPU  | float64 | 1e-12 | 7.578e-16| PASS | 371 | `cpu` / `torch.float64`  |
| GPU  | bf16    | 1e-2  | 1.022e-3 | PASS | 370 | `cuda:0` / `torch.bfloat16` |
| GPU  | float32 | 1e-4  | 3.264e-6 | PASS | 436 | `cuda:0` / `torch.float32`  |
| GPU  | float64 | 1e-12 | 5.873e-15| PASS | 387 | `cuda:0` / `torch.float64`  |

All six cells: the arm converged inside the budget, executed on the declared
device, and landed inside its declared class. **bf16 is honest at ~1e-3 — two
orders inside the 1e-2 class, and never silently widened per device.**

Test-backed by:
`tests/ptq/test_pipeline_dtype_gpu.py::test_scn_admm_parity_gpu_cell[bf16|float32|float64]`
(GPU) and `tests/ptq/test_pipeline_dtype_generic.py::test_scn_admm_parity_cpu_cell[...]`
(CPU).

### 2a. The CONVERGED-output cosine, recorded (per design.md C1)

The metric the spec delta's `admm_parity_per_dtype` originally named — converged
`1 - cos(recon_pack, recon_pack_ref)` — is **recorded here, not gated**, because C1
measured it as basin selection rather than dtype fidelity. The accelerator cells
corroborate that finding:

| Device | compute_dtype | `rel_first` (gated) | `1 - cos_pack` vs converged f64 NumPy ref | `cos_pack` vs `W` |
|--------|---------------|---------------------|-------------------------------------------|-------------------|
| CPU | bf16    | 1.017e-3  | 0.932 | 0.324 |
| CPU | float32 | 3.132e-7  | 0.918 | 0.369 |
| CPU | float64 | 7.578e-16 | 0.864 | 0.343 |
| GPU | bf16    | 1.022e-3  | 0.964 | 0.244 |
| GPU | float32 | 3.264e-6  | 0.780 | 0.416 |
| GPU | float64 | 5.873e-15 | 0.735 | 0.388 |

The **float64-torch vs float64-NumPy control** differs by 0.74–0.86 at the
converged point while agreeing to ~1e-15 at the first update, on BOTH devices. The
converged cosine therefore does not separate dtypes on the accelerator any more
than on the CPU, and would fail every cell of both rows outright — it measures
which discrete optimum the sign-quantized iteration selected. This is evidence for
the pending ruling `t_5a964ad9` (planner), not a new claim.

## 3. GPU end-to-end bf16-default run (tiny/toy model path)

The spec's own verification scale — `TinyGateModel` + `gate_config`, bf16 default,
torch backend on the accelerator. **Not a fire.**

```
tests/ptq/test_pipeline_dtype_gpu.py::test_scn_default_numerics_bf16_gpu_end_to_end PASSED
```

- Model and calibration batches on `cuda`; `QUANTIBBLER_DEVICE=cuda`.
- Artifact: 4 qpk1 blobs for 4 layer states; every layer state carries a bias vector;
  reloaded effective BPW 0.682 (< 1.0).
- Accelerator stage records name `cuda:0` (`ste`, `tune_scales_kd`; the capture /
  solve / pack rows carry no device field by design — the device is witnessed
  separately by the `ste.per_linear[].device` assertion and by the returned tensors
  in §2).

## 4. Per-stage GPU timing rows (measured, not estimated)

Rows are the recorder's own `elapsed_s` per stage, summed over the two tiny blocks,
plus `export` (`QuantizedModel.serialize`) and `eval` (probe forward on the
reloaded model), for the same tiny path. Two independent runs are given because the
box carried co-tenant load (see §5): the timing is **not** presented as a clean
benchmark, it is presented as measured under a stated load.

| Stage | Run A (GPU, load1 ≈ 7–15) | Run B (GPU, load1 ≈ 41.5) | Run A (CPU, load1 ≈ 47–57) |
|-------|--------------------------|--------------------------|----------------------------|
| capture | 0.0005 | 0.0045 | 0.248 |
| tune_fp | 0.1391 | 1.0120 | 1.553 |
| solver (`lb_admm`) | 1.3552 | 2.5362 | 83.375 |
| ste | 0.1466 | 0.1168 | 2.347 |
| sign/extract/pack | 0.0299 | 0.0320 | 0.026 |
| tune_scales_kd | 0.0634 | 0.0691 | 2.221 |
| export | 0.0001 | 0.0017 | 0.002 |
| eval | 0.0003 | 0.0005 | 0.095 |
| **pipeline total (wall)** | **1.85 s** | **3.39 s** | **90.00 s** |

Context: the v6-class full fire is the ~8 h baseline these rows exist to be
compared against — the tiny fixture is verification scale, not that comparison.
The GPU rows' spread (1.85 s vs 3.39 s for identical work) is host-load variance:
the CUDA arms are ~2.5x slower to launch and synchronize under a loaded box. The
CPU column is included only to show the solver's share: on the CPU path the solver
is 93 % of the tiny-path wall and the accelerator removes essentially all of it.

`tests/ptq/test_pipeline_dtype_gpu.py::test_scn_per_stage_timing_and_device_on_gpu`
asserts the shape (every stage emits a measured `elapsed_s`, the accelerator stages
report `cuda:0`, `qk_report.stage_map` carries `row_time_s`); the table above is the
measured content.

## 5. Good-guest evidence under GPU load

### 5.1 GPU memory discipline (tasks.md 5.4, deferred here)

`tests/ptq/test_pipeline_dtype_gpu.py::test_scn_gpu_memory_cap_recorded` PASSED:

- `is_unified_memory_device()` → **True** (gfx1151 is an APU: device memory IS host
  RAM), so the absolute-GB bound is the one that protects co-tenants.
- `apply_gpu_memory_cap(plan)` → **0.85** applied on device 0.
- `resolve_memory_fraction(0.85, gb=8.0, total=133 143 986 176)` → **0.0645** —
  i.e. `--gpu-memory-gb 8` narrows the cap ~13x below the bare fraction. That
  narrowing is the APU clause's whole purpose.
- `max_memory_reserved()` recorded under a live `1 << 22`-element (16 MiB) f32
  allocation: 16 777 216 B, well inside the cap.

### 5.2 CPU thread sizing — measured, and the anti-pattern measured beside it

The shipped contract: the sizing block runs **before** the BLAS/torch imports and
sets `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS`, then
`torch.set_num_threads(n_threads)` after torch import.

Measured at load1 = 37.4 (n_cores = 32, so the **heavy** branch): `n_threads = 8`;
`OMP/MKL/OPENBLAS = 8`; `torch.get_num_threads() = 8` — the startup assertion the
spec asks for holds.

Same fixture, same host, same instant — torch's own default pool vs the plan's:

| Arm | Threads | iters/s (120-iter ADMM arm) | wall |
|-----|---------|-----------------------------|------|
| torch default pool (anti-pattern) | 32 | **24.5 /s** | 4.89 s |
| good-guest plan (heavy branch) | 8 | **2043.6 /s** | 0.059 s |

83x throughput from *reducing* the pool, and the container's own CPU consumption
fell with it (2.07 CPU-seconds over the 0.138 s measured arm; ~15 cores average
even at 8 threads, i.e. still oversubscribed — the tiny-fixture regime). This is
the thread-thrash artifact the driver's own comment records (unpinned 235.5 s vs
pinned 3.98 s), reproduced on the GPU lane.

### 5.3 Host liveness — the loadavg-47–70 anti-pattern, measured both ways

The lane ran with a 2-second host sampler (`/proc/loadavg`, `MemAvailable`).

- **Uncapped lane work (no plan applied, default 32-thread pools), 2026-10-27
  17:42–17:46Z:** the host's 1-minute load average rose from **3.6 → 57.6** and
  stayed above 40 for four minutes; the sampled peak was 57.58 (mean 36.8 over
  303 samples). The container's own CPU arm averaged **29.3 cores** of a 32-core
  box. This is the incident shape the spec names, reproduced by this lane's own
  CPU-side comparison arms (the GPU cells themselves are 1.8–3.4 s of GPU work).
- **Plan applied (good-guest sizing, same work), 17:47Z:** host load1 went
  **37.39 → 37.23** across the run — the lane added nothing measurable, and the
  arm finished 83x faster (§5.2).

Two findings fell out of this, both routed rather than fixed here:

- **G1 (finding, routed).** The "sizing must precede every BLAS import" precondition
  is **not** satisfied by the shipped import ordering: `from quantibbler.good_guest
  import ...` executes `quantibbler/__init__.py`, which imports `.kernels`, which
  does `import numpy as np` — so NumPy's BLAS is loaded *before* `apply_thread_env`
  runs. Measured, not inferred: immediately after the `good_guest` import,
  `"numpy" in sys.modules` is **True** (`"torch" in sys.modules` is False). The
  driver's comment claims the import is stdlib-only and drags nothing heavy in.
  Consequence: the env trio may be a no-op for NumPy's own pool on this box (the
  measured 83x gain came through `torch.set_num_threads`, which is applied after
  import by design). Fixing it means making the package `__init__` (or `kernels`)
  lazy — a shared-package change, not this lane's. *Owner: planner/coder on the
  parent change.*
- **G2 (lane guidance, measured).** The plan is evaluated **once at startup**. On a
  box already at load1 = 20 (< n_cores = 32) it takes the *moderate* branch
  (`n_cores - 4` = 28 threads ≈ the whole box), so a lane starting on a
  moderately-loaded host can still add ~28 cores of pressure. The heavy branch is
  correct once load1 exceeds n_cores, but a lane that must run CPU-heavy comparison
  arms should either pass `--max-threads N` or bound the container (`--cpus`), and
  should re-check the host before launching. Recorded here as operating guidance;
  the formula is the spec's, and this lane does not change it.

## 6. What changed in the repo (patch)

| File | Change |
|------|--------|
| `tests/ptq/test_pipeline_dtype_generic.py` | Removed `test_scn_admm_parity_gpu_cells_skip_with_reason` (see §7); the accelerator cells now live in the GPU module and are `integration`-marked. |
| `tests/ptq/test_pipeline_dtype_gpu.py` | **New.** Six accelerator legs: three parity cells, the bf16 GPU end-to-end run, the GPU per-stage timing/device rows, and the GPU memory cap. |
| `openspec/changes/qk-pipeline-dtype/gpu-lane-evidence.md` | **New.** This document. |
| `openspec/changes/qk-pipeline-dtype/tasks.md` | Rows 3.4 and 5.4 ticked with dated evidence; 5.1/3.3 annotated with the accelerator measurement. |

No production code changed. No numerics changed.

Reproduce (GPU lane):

```bash
docker run --rm --memory 16g --memory-swap 16g \
  --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" \
  --network none --label qkb_launch_id=qk-pdtype-gpu-$(date -u +%Y%m%dT%H%M%SZ) \
  --label qkb_launch_owner=coder --label qkb_launch_card=t_f8e3b1d8 \
  -v <worktree>:/repo:ro -e PYTHONPATH=/repo/src -w /repo \
  --entrypoint python quantibbler-ptq:rocm-official-v6 \
  -m pytest -m integration tests/ptq/test_pipeline_dtype_gpu.py -v
# => 6 passed
```

Reproduce (CPU tier — must stay skip-free for CI's zero-skip referee):

```bash
docker run ... --entrypoint python quantibbler-ptq:rocm-official-v6 \
  -m pytest tests/ptq/test_pipeline_dtype_generic.py -v
# => 10 passed
```

## 7. Findings the reviewer must read

- **F1 — the parent's GPU cell could never pass, and would have reddened CI.**
  `test_scn_admm_parity_gpu_cells_skip_with_reason` asserted
  `res.U.device.type == "cuda"` while the module's autouse fixture pinned
  `QUANTIBBLER_DEVICE=cpu`; the torch backend's device comes from that env, so the
  arm executed on **CPU** and the assertion failed on an accelerator host —
  measured: `assert 'cpu' == 'cuda'` (1 failed, 10 passed). It was also unmarked,
  so on a GPU-less runner it *skipped in the default tier*, which is exactly what
  CI's `full-python` job refuses (`tools/ci/pytest_summary.py --expect-skipped 0`):
  the CPU-side port's own suite would have failed CI on the skip, not on the code.
  Fixed by moving the accelerator cells to the `integration`-marked GPU module and
  having each leg set the device explicitly.
- **F2 — the spec's declared metric does not hold on either device** (§2a). Not a
  new claim; corroborated on the accelerator for the pending ruling `t_5a964ad9`.
- **F3 (G1 above)** — the good-guest "before any BLAS import" precondition is
  violated by the package import chain (numpy in `sys.modules` before the env is
  set).

## 8. Honesty / caveats

- The image is `rocm-official-v6`, not the `rocm-official` the design's provenance
  note names (§0) — the deviation and its real reason (the named image carries
  `pytest` but lacks `pydantic`, so the bf16-default e2e leg fails at runtime) are
  stated, not silent.
- Timing rows carry their load context; the box was shared and its load1 ranged
  3.6–57.6 during the lane. No timing number here is presented as a clean baseline.
- The planner's parity-metric ruling (`t_5a964ad9`) was still `ready` when this run
  was taken. The GPU cells mirror the CPU cells' metric **by construction** (same
  fixture, same reachable metric, same classes); if the ruling changes the CPU
  metric, these cells must be re-run against the amended tip before this evidence
  is read as current.
