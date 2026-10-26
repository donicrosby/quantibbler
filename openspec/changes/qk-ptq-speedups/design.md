# qk-ptq-speedups — design

## Sequencing mandate (user directive, 2026-10-23)

GPU acceleration lands PRIOR to the next production run. The user has ruled the numerics question settled by the measured A/B campaign (fp32 GPU ≡ fp64 CPU within tolerance on tested layers). This is a SEQUENCING constraint, not a tolerance waiver: the fp32-GPU-vs-fp64-CPU parity scenarios remain the acceptance gate, and the torch-autograd stages become the production path only after per-stage parity passes.

## min_eigs gating rationale

- Benchmark (measured, not comment-inferred): 2× `eigvalsh(512×512)` ≈ 142 ms/iteration at 16 OpenBLAS threads vs 62 ms for the 2× matmul + 2× Cholesky it sits next to; the code's "cheap r × r" comment is wrong at production thread settings.
- Redundancy: `_cholesky_solve` immediately below IS the SPD proof, and `trace.n_jitter` already records every non-SPD event — so the gated instrument guards nothing the always-on path misses.
- Scale note: fixed per-linear tax that does not shrink while GEMMs do — wrong cost to carry into a 4B run.

## STE/TuneFP GPU port rationale

- Pure torch autograd, zero device references in `block_recon.py` — GPU-movable with no algorithm change (sanctioned by config.yaml GPU policy: compressor/trainer on PyTorch).
- fp32 on gfx1151 is the earning configuration; fp64 throughput is ~1/16 of fp32, making fp64 GPU a wash against the 32-core CPU.

## collect_stats disposition (row: *Disposition of `collect_stats` …*; recorded 2026-10-25, t_97576d39)

**GPU-movable pieces: ported — device-true by construction, nothing left to move.**
Every piece follows its own tensor's device; no global device state is consulted and
nothing is pinned to the host:

| piece | site (`calibration.py`) | device authority |
|---|---|---|
| calibration batches | `collect_stats` head: `_audit_ref = next(iter(module.parameters()), None)`, then `[b.to(_audit_ref.device) for b in calibration_batches]` | the model's own parameter device (explicit `None` guard — never truthiness, never a bare `next`) |
| streamed accumulators (`sum_sq_in/_out`, `rows_*`, `grad_sq_out`) | hook setup | born on `m.weight.device` / `m.weight.dtype`; `_accumulate` re-coerces its accumulator to the activation's device at the first real input |
| per-token clip, row reduce, cotangent square | `_clip_token_norms`, `_reduce_tokens`, `cotangent_grad_sq_sum` | torch ops on the incoming tensor's device |
| forward / backward | `detected_trunk(...)` → `loss.backward()` | wherever the parameters are |

Audit basis: NumPy is reachable inside `collect_stats` only at its exit (next table) —
the pass contains no `.cpu()`, no NumPy array, and no `torch.set_default_device`. **The
accelerator leg of this claim is a CODE AUDIT, not an executed reading**: the box that
produced this record has no accelerator (`torch.cuda.is_available()=False`; the `torch`
extra is scoped to the `pytorch-cpu` index in `pyproject.toml`), so the claim's
verification is the container lane that produced the STE/TuneFP device witness above.

**Left CPU-only, with the measured reason.**

| piece | measured |
|---|---|
| `_to_numpy_stat` + `np.maximum(_TINY)` + `robust_diag(tau=None, gamma=0.2)`, two statistic vectors per linear | **0.299 s** for the whole 197-linear production book = **1.5 ms/linear** |

It runs once per fire, after the pass; against the one recorded Phase-1 batch (**42.3 s**
for 8×2048 rows over 197 linears, the cgroup reading in `api.py`) the host remainder is
**0.04 %** of the batch it follows, and the fire runs 16 of those batches. Disposition:
**leave**.

## Remaining CPU-bound pieces — per-piece disposition (row: *Enumerate every remaining CPU-bound piece …*; recorded 2026-10-25, t_97576d39)

Every remaining host-side piece on the per-block path. All numbers are rank 512 on real
per-layer shapes; the measurement regime (interpreter, device, load) is stated in the
crystallization entry below, and these are single-run upper bounds on a loaded box.

| piece | site | measured | disposition |
|---|---|---|---|
| greedy bit-flip polish | `bit_polish.greedy_polish`, called per layer from `block_recon` with `rounds=cfg.polish_rounds` (4) | 26.4 s per block (7 linears; 3.2–4.7 s/linear) with the bits already at their own optimum (0 flips); **27.3–33.4 s/linear** with 25 % of the stored bits perturbed (470k–576k flips, cos 0.24 → 0.81–0.84) | **measure-first** |
| post-balance divergence gate | `check_layer_magnitude`, per layer | 4.12 s per block (0.59 s/linear) | leave |
| sign-extract + pack | `extract_and_pack`, per layer, incl. the unpack parity round-trip | 1.31 s per block (0.19 s/linear) | leave |
| Phase-3 KD scale calibration | `calibrate_scales` | not executable on the accelerator as shipped (below); **no production reading exists** — of the 15 recorded reports, the 4 that carry a `stage_map` all read `tune_scales_kd: executed: false` (the other 11 predate the stage_map), i.e. no recorded fire has reached Phase 3 at all: every r512 fire died at the depth-2 probe | **port** (blocker) |
| eval path | driver call sites `perplexity_from_logits(lambda b: model(b.to(_device)), …)` | already device-true: the batch is moved to the logits' device inside the helper, and the model is on `_device` | leave (already ported) |
| checkpoint write | `checkpoints.py`, one `.npz` per block | 11.7 MB and 0.072 s per block | leave |
| Phase-1 host assembly | `calibration.py` exit (the `collect_stats` disposition section above) | 0.299 s per fire | leave |
| solver host-hop and SVD init | `block_recon`: `w = linear.weight.detach().to(float64).cpu().numpy()`, then `np.linalg.svd` when `solver="svd"` | the hop IS the solver contract (the solve's device is chosen by `solver_backend`, resolved at the solve site); `solver="svd"` is the diagnostic arm, production default is `admm` | leave (measure-first if ever promoted) |
| tokenization / wikitext load | driver `[2/6]` | caller-side, off the PTQ path | leave |

Fire-level total for the small pieces: (26.4 + 4.12 + 1.31) s × 28 blocks + 0.3 s + I/O
≈ **15 min**, ~1.1 % of the proposal's 21.8 h cost model. The polish is the only piece
whose *worst measured* state is material (~28 × 7 × 30 s ≈ 1.6 h, ~7 %) — and the one
whose start state varies most between fires, which is why it gets neither of the two
outright verdicts:

**`bit_polish` — measure-first, and the instrument it needs is missing.** Both ends are
measured here (0.9 % and ~7 % of the fire); the fire's own start state decides between
them, and the shipped per-stage timing cannot report it: `qk_report.stage_row_times`
reads a row's `elapsed_s`, the STE and TuneFP rows carry one, and the `lb_admm` record —
the only row the polish can land in, since its round count already travels there —
carries `polish_rounds` alone, so its `row_time_s` reads null. One field is the
prerequisite for the decision. What that decision weighs against is a port: the stage is
NumPy-only coordinate descent with exact-gain flip algebra, off the activation path, and
a torch rewrite buys a new parity surface for a cost bounded at ~1.6 h in its worst
measured state.

**`scale_calib` — port (blocker, not a speed item).** `calibrate_scales` builds every
operand on the host while the model it forwards is device-resident: `torch.from_numpy`
for the scale tensors (s1/s2 at entry) and for `ub`/`vb` inside `scale_forward`; the
teacher cache written through `.cpu().numpy()` into an `np.memmap`; the paged logits read
back through `torch.from_numpy`. The driver hands the stage the host model (`host = model`,
after `causal.to(_device)`), so on an accelerator the first `model(x)` crosses devices —
the failure class the STE leg already hit (`Expected all tensors to be on the same
device`). Two consequences:

1. **Named delta — owner: planner.** The requirement *"Torch-autograd stages SHALL run on
   the accelerator device"* enumerates STE and TuneFP in its body; Phase 3's KD is a
   torch-autograd stage of the same class and is named neither in the spec delta nor in
   this change's `tasks.md`. The requirement's own header covers the class, its
   enumeration does not. The port is not launched from this record: lane status is the
   user's call.
2. **Its sizing is not only wall time.** At the fire's geometry (128 rows × 2048 positions
   — 16 packed batches, as the r512 launch records) the teacher cache is **148.4 GiB** at
   float32 (296.8 GiB for a float64 model), written once, and paged back at **1.16 GiB per
   step over 1024 steps ≈ 1187 GiB** of reads under the paper-faithful budget
   (`stage_global` bs 1, 8 epochs; the gate-shaped budget is 128 steps and 594 GiB). The
   disk backing is load-bearing and must survive any port — only the model
   forward/backward is the accelerator's to take.

**`eval path` — leave (already ported).** The heavy half (18 batches × 8 × 2048 tokens
for the r512 window) already runs where the model is; what is left on the host is the
tokenizer and the per-row `float(nats)` hand-off, both outside the PTQ path.

## Deviation log

- None of the changes in this set alter solver arithmetic or optimizer configuration; the gating is behavior-neutral by scenario (bit-identical fingerprint, flag off vs on).
- The measured per-stage wall-time deltas are recorded in the run report and the commit body at landing time, superseding the proposal's cost-model estimates.

## Implementation crystallization (2026-10-25)

Landed with the min_eigs gating; the choices below were design-silent, so they
are recorded here rather than left to be re-litigated by the next reader.

1. **The gate is resolved from the ENVIRONMENT at the solve site, never from
   `PTQConfig`.** `lb_admm` reads `SOLVER_PROBE_ENV` once (beside `_rho_end` /
   `_admm_reg`) and guards both eigenvalue appends with it. A config field was
   discarded: `config_fingerprint` hashes the config, so a `probe` field would
   move the digest between a probe run and a production run of the same target
   — contradicting the requirement that the gating be fingerprint-neutral.
   Consequence: the flag is an operator knob with no effect on any persisted
   identity; it is NOT provenance and is deliberately absent from the solver
   probe row's `(value, source)` pairs, which attest the solve's *regime*.
2. **One name, owned by the solver.** `SOLVER_PROBE_ENV` is declared in
   `lb_admm` (the module that gates on it) and imported by `block_recon`'s
   per-linear JSONL writer. Two literals would let the gate and the writer drift
   apart; `block_recon` cannot own it because it imports `lb_admm` at module
   level (the import direction would invert into a cycle).
3. **The Lane-3 sweep turns the probe on for its own solves.**
   `tools/lane3_reg_sweep.py` is the live consumer of the `min_eigs` floor
   (lane-3's acceptance reads it out of the sweep rows), so it enables the flag
   around each solve instead of emitting a NaN floor that depends on whether the
   operator remembered the env var; it restores the flag afterwards, exactly as
   it already did for `QK_ADMM_REG`.
4. **Two pre-existing scenarios are diagnostic configurations by
   construction.** `test_scn_admm_spd_solves` and
   `test_scn_rho_endpoints_recalibrated_normalized_regime` assert ON the
   `min_eigs` column, so they enable the probe through a local
   `solver_probe_on` fixture. No other test in that file changed, and the
   suite's default configuration stays production (flag off).
5. **The gated column is not a lost signal.** The SPD proof a production run
   relies on is the Cholesky factorization itself, whose every non-SPD event is
   recorded always-on in `trace.n_jitter`; the requirement's
   `spd_signal_preserved` scenario pins that. An empty `min_eigs` list therefore
   reads as "no diagnostic requested", never as "not SPD" — the docstring on the
   field says so at the point of use.
6. **Reproduction regime for the benchmark claim above.** The "≈142 ms /
   iteration at 16 threads" line is a QUIET-BOX number. Measured on this shared
   host under co-tenant load (loadavg 47–70 on 32 cores) the same op pair reads
   102.9 s versus 0.585 s at one thread — i.e. the 16-thread reading there is a
   contention artifact, not a property of the code. Any per-linear delta quoted
   for this change states the loadavg it was taken at.
   **AMENDED 2026-10-25, same day (quiet window — superseded by the row-9
   measurement section at the END of this file): the two contended readings in
   this item (`102.9 s`, `0.585 s`) are SUPERSEDED as characterizations of the
   box.** On a quiet box the same op pair reads **20.10 ms** at one thread and
   **130.6 ms** at 16 threads: the `102.9 s` figure is ~790× the quiet 16-thread
   value and `0.585 s` is ~29× the quiet 1-thread value. The contention caveat in
   this item stands and is re-measured there; these two numbers do not.

## Implementation crystallization (coder, 2026-10-25) — tasks 17, 18

Ticket `t_c3fb5ce8` covers this file's rows 17 and 18 (section 4's first two
rows; the ticket's "tasks 17, 18" numbering is this file's row order). Row 19
belongs to ticket `t_97576d39` and is untouched here.

**Row 17 — landed and evidenced.** The device plumbing for the two stages
predates this ticket (`ptq: device-true packed weight in frozen_forward`
`e46bc0f` and its L1 follow-ups); this ticket adds the parity suite the row
demands and the device witness the record lacked.

- RED, literally executed at `4393b85` (= `e46bc0f^`) in
  `quantibbler-ptq:rocm-official` + `/dev/kfd`: the accelerator leg of the STE
  died `RuntimeError: Expected all tensors to be on the same device, but found
  at least two devices, cuda:0 and cpu!`, while the TuneFP accelerator leg and
  both CPU legs completed. At HEAD the same four legs all complete.
- GREEN: `tests/ptq/test_stage_device_parity.py` — 4 passed (container lane).
  The parity legs are isolated (same fp32-exact weights and same fp64 latents on
  both legs, host-computed target), so the comparison holds only the difference
  the scenario names: `device` and activation dtype.

**Decision — the ±0.002 class is asserted on the STAGE run, not through the
whole block pipeline.** Measured on the fixture at `cuda/fp64` vs `cpu/fp64`
(same precision, only torch's reduction order differs): the TuneFP record agrees
to 1.6e-10 and fc1's packed cos to 5.1e-12, yet fc2's packed cos lands **7.6e-3**
apart with STE gradient norms 8% apart. The solver + greedy-polish path is a
DISCRETE search: a sub-ULP input difference flips an accepted move, and at a
small fixture's geometry (d=48, r=12) that flip dominates the artifact metric.
The spec's ±0.002 is a production-layer property of the whole-path comparison
(qk-solver-backend's spec restates it as the campaign's measured record), so the
unit fixture asserts the class where the claim is stage-scoped and the
whole-pipeline pair is asserted for configuration identity and device placement
only. This is recorded rather than papered over: a fixture-sized cos gate on the
packed artifact would be measuring the discrete search, not the port.

**Decision — row-time rides in the existing stage inventory, not a new
top-level report key.** `qk_report.stage_map` already carries each stage's
``invocations`` and Algorithm 1 lines; it now carries ``rows``, ``elapsed_s``
and ``row_time_s`` beside them, and the stage records carry ``elapsed_s`` and
``device``. A new top-level report key would have to be mirrored into the
driver's abort-payload fixture (`tests/test_qk_report.py` derives both key sets
from the driver's literals by AST), for no reader benefit.

**Decision — an STE row time is attributed to a linear only when the call
refined exactly that linear** (the shipped per-target call pattern). A joint call
over several `facs` cannot split its wall time by linear, so those rows report
null rather than a fabricated share. `tune_fp` skips are counted as zero rows —
a skip did no work and spent no time.

**Measured per-stage delta (production layer shapes; row 18's measurement half,
recorded here because the run report does not carry a benchmark).** The
instrument is `tools/stage_row_time_bench.py` (landed with this row; same
shape as `tools/lane3_reg_sweep.py`), default geometry: block
`proj_in 1024→3072 → GELU → down_proj 3072→1024`, `x_b` (4, 2048, 1024), rank
512 — the shapes of the campaign model (Qwen3-0.6B `config.json`: hidden 1024,
intermediate 3072, 28 layers), weights synthetic, rows reduced. Identical stage
configuration on both legs; only (device, dtype) differs.

| run | stage | unit | cpu/fp64 | cuda/fp32 | speedup |
|---|---|---|---|---|---|
| loadavg 43→61 | TuneFP | per fire (4 steps, bs 4) | 17.04 s | 4.69 s | 3.63× |
| loadavg 43→61 | STE | per linear (16 steps, bs 1) | 58.49 s | 17.25 s | 3.39× |
| loadavg not recorded | TuneFP | per fire | 15.22 s | 8.60 s | 1.77× |
| loadavg not recorded | STE | per linear | 59.36 s | 14.57 s | 4.07× |

Extrapolated to the proposal's row counts (168 fires, 196 linears) from the
load-stamped run: TuneFP 47.7 → 13.1 min, STE 191.1 → 56.3 min. **The proposal's
10-30× estimate is NOT reproduced at this geometry**, and the two runs above
differ by 2× on the speedup itself — the peer section's reproduction rule
applies: both were taken on this shared host under co-tenant load
(`load average: 43.29, 49.72, 51.08` before / `60.97, 54.55, 52.69` after; the
unrecorded pair ran at ~52 as well), so the per-stage factor is a ~3× order of
magnitude with a large contention band, not a precise figure. A quiet-window
re-run is part of the same GO'd production measurement that closes row 18; the
named candidates for the gap against the estimate are the proposal's own batch-1
STE latency bound and fp64 latents/Adam (`FactorizedLinear` holds its parameters
in float64 by design) on gfx1151's 1/16 fp64 rate. Recorded as a finding, not a
defect: this measurement supersedes the estimate, which is what the row asks for.

The tool also prints the stage record's OWN `elapsed_s` beside the wall clock
(14.88 s vs 17.04 s cpu, 2.39 s vs 4.69 s cuda for TuneFP): the gap is the
fixture's construction, and the two agreeing to the stage's share is the check
that the report column measures the stage rather than the harness around it.

**Row 18 stays unticked.** Its report half shipped (see above). Its measurement
half says *production layers*: the numbers above are production SHAPES on
synthetic weights at a reduced row count, and a real-model measurement is a fire
— GO-gated by the lane rule, not launchable from a ticket. The row-time column
now makes the next fire's own stage numbers the measurement, so the half lands
with that fire.

**Ticket/repo deltas (recorded, not amended).** The ticket's gate line names
`tests/ci/` and `tests/qk_report/`, which do not exist — the files are
`tests/test_ci_referee.py` and `tests/test_qk_report.py`; running the literal
directories exits 4 (usage error) rather than green. Both files are run as files
in the gate below.

## Implementation crystallization (coder, 2026-10-25) — rows 19, 23

Ticket `t_97576d39` carries "tasks 19, 23" — `tasks.md` LINE numbers, this change's
convention (the sibling entry above records it for 17/18). Rows are cited by their text
here, because the numbers moved within the hour: at ticket-creation time the
`collect_stats` disposition row sat at line 19 and the remaining-CPU-bound enumeration at
line 23, and a concurrent landing (rows 17-18) had pushed them to 21 and 25 by the time
these ticks were written. The two rows this ticket covers are the `collect_stats`
disposition and the remaining-CPU-bound enumeration; the sibling entry covers the two
above them.

1. **The measurement regime, stated once for both new sections.** `uv run --no-sync
   python` → python 3.12.14, torch 2.14.1+cpu (CPU wheel, `torch.get_num_threads()` =
   16), numpy 2.5.4, **no accelerator**, 32 CPUs, host load average 66 (co-tenant lanes
   on a shared box). Single runs: **upper bounds**, and any number quoted for this change
   states the loadavg it was taken at. Shapes and rank are a real fire's
   (`artifacts/qwen3-0.6B-r512-admm-paperfaithful/checkpoints/block-1.npz`, rank 512);
   the fp **weights** are not on this box (no HF cache, and a model download is the
   caller's job), so the polish target is each layer's own artifact reconstruction — and
   that is exactly why the perturbed-start variant is measured beside it: the
   reconstruction alone is a fixed point of the polish and would have reported the
   degenerate floor as the cost.
2. **"Ported" is claimed only where the code makes the device authority explicit**, and
   the accelerator leg of that claim is recorded as an audit, not as a reading (no
   accelerator here). A stage that CANNOT follow the device is named with its mechanism
   rather than with a speed number — that distinction is why `scale_calib` is the only
   **port** in the table, and why the record does not rest on an unmeasured speedup.
3. **A disposition of `measure-first` must name the missing instrument.** For
   `bit_polish` that is the `elapsed_s` field on the `lb_admm` stage row: without it the
   fire's own polish cost lands nowhere a reader can find it, and the disposition is
   unfalsifiable. A `measure-first` that leaves the measurement unspecified is the same
   rot as a ticked box whose text is false.
4. **Nothing was ported and no lane was started under this ticket**, by design: both rows
   are disposition records. The STE/TuneFP port they depend on landed under the sibling
   ticket; the Phase-3 port this record names is unrouted — planner owns the spec text
   and the task row, the user owns the lane (config.yaml: fire-bearing tasks launch only
   on explicit GO, and lane status is the user's call alone).
5. **Row references by LINE NUMBER are this change's ticket convention and they rot**:
   `t_97576d39`'s "tasks 19, 23" are the row line numbers, and every insertion above a
   cited row invalidates them (the sibling entry records the same convention for rows
   17/18). Cite the row's text; the line numbers here are as of this record only.

## Implementation crystallization (coder, 2026-10-25) — row 9, the measured min_eigs probe tax

The row asks for solver wall time on **real production layers, flag on vs off**, with the
per-linear delta recorded. The blocked attempt could not take it (box at loadavg 47–70 /
32 cores); the row-9 ruling then authorized a labeled 1-thread CPU micro-measure and named
the quiet-window reading as an optional, non-blocking follow-up. **The quiet window arrived
(same day), so both were taken**, and this section supersedes item 6 above.

**Regime — stated once, because a number without it is not evidence.** `/opt/venv` python
3.12, **torch 2.14.1+cpu** (`torch.version.hip None`, `torch.cuda.is_available() False`,
32 CPUs) — the dev CPU wheel, so this is NOT the production (interpreter × device) pair.
Backend/device are declared, never implicit: default `solver_backend="torch"` with
`QUANTIBBLER_DEVICE=cpu` (the documented diagnostic reading; implicit torch refuses on this
host by design). Real weights from the local `Qwen3-0.6B` cache, **rank 512** (production),
float64. Thread counts set explicitly per arm
(`OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/`MKL_NUM_THREADS`); `/proc/loadavg` is stamped on
every pair by the instrument. Instrument: `/opt/data/scratch/speedups_measure.py` (paired
**interleaved** `off,on` within a rep, one process, `n_iter == K` and "the gate tracks the
flag" asserted after every solve), attached to ticket `t_2522c684`. Production layers:
`model.layers.0.self_attn.q_proj` (2048×1024) and `model.layers.0.mlp.down_proj` (1024×3072).

All Δ/ratio cells are the instrument's own medians (`ratio := median(on) / median(off)`,
`Δ := median(on − off)` over the reps of that row); a row with `reps ≥ 2` therefore also
carries interleaved replication, and the per-rep spread is named where it matters.

| thread env | layer | K | reps | off ms/iter | on ms/iter | Δ ms/iter | ratio | Δ s/linear (Δ × K/1e3) | loadavg |
|---|---|---|---|---|---|---|---|---|---|
| 1 | q_proj | 20 | 2 | 301.8 | 322.6 | **+20.75** | 1.069 | +8.3 (extrap.) | 2.8–3.1 |
| 1 | down_proj | 20 | 2 | 438.4 | 457.5 | **+19.08** | 1.044 | +7.6 (extrap.) | 2.8–3.1 |
| 1 | q_proj | **400** | 1 | 324.6 | 334.2 | **+9.61** | 1.030 | **+3.85 (measured)** | 2.9–3.9 |
| 16 | q_proj | 20 | 2 | 582.3 | 722.5 | **+140.24** | 1.241 | +56.1 (extrap.) | 2.8–3.4 |
| 16 | down_proj | 20 | 2 | 794.5 | 969.2 | **+174.67** | 1.220 | +69.9 (extrap.) | 2.8–8.8 |
| 16 | q_proj | 400 | 1 | 558.8 | 1202.2 | +643.5 | 2.152 | +257.4 (measured) | 3.6 — **flagged, see below** |
| 16 | q_proj | 60 | 3 | 2216.7 | 2481.7 | +328.7 | 1.120 | +131.5 (extrap.) | 2.7–7.2 — **contended**; per-rep ratios 1.18 / 0.69 / 1.18 |

**Per-linear delta (K=400, rank 512, real production linear):** **+3.9 s at 1 thread** —
directly measured, 129.84 s guarded vs 133.69 s probed (1.030×); that layer's own K=20 pair
extrapolates to +8.3 s (down_proj +7.6 s), so read the 1-thread cost as **~4–8 s/linear**. At 16 threads the
load-protected interleaved pair gives Δ 140.2 (q_proj) and 174.7 (down_proj) ms/iter, i.e.
**≈ +56–70 s/linear**, against a guarded solve of 582–795 ms/iter there.

**The benchmark claim is REPRODUCED (−8 % on the op pair).** Claim: "2× `eigvalsh(512×512)`
≈ 142 ms/iteration at 16 threads vs 62 ms for the 2× matmul + 2× Cholesky it sits next to."
Quiet-box op-level micro-bench (same interpreter, torch, `r=512`):

| ops | 1 thread | 16 threads |
|---|---|---|
| `eigvalsh(512)` | 9.44 ms | 60.42 ms (numpy sibling: 59.96 ms) |
| 2× `eigvalsh(512)` | 20.10 ms | **130.58 ms** (claim 142: −8 %) |
| 2× (GEMM + Cholesky) | 17.29 ms | **49.63 ms** (claim 62: −20 %) |

and the in-solve interleaved marginal at 16 threads is 140–175 ms/iter (median 157.5) —
the same quantity the row names, measured where it is actually paid.

**Findings (each one is a reason the gate exists):**

1. **The tax is a 16-thread pathology, not a fixed cost.** One thread is **6.4× faster**
   than 16 for `eigvalsh(512)` (9.44 vs 60.42 ms) and **1.72× faster** for the whole guarded
   solve on the same layer (324.6 vs 558.8 ms/iter, quiet box): OpenMP dispatch overhead
   dominates a 512³ problem, so the operator's own 16-thread setting is the worst case —
   which is where the probe taxed production solves (ratio 1.22–1.24) versus ~1.03–1.07 at
   one thread.
2. **The single K=400 16-thread pair is flagged, not quoted.** Its off arm
   (558.8 ms/iter) matches the interleaved K=20 off arm (582.3) while its on arm
   (1202.2 ms/iter) ran **1.66×** its K=20 sibling (722.5) — the signature of co-tenant
   load arriving inside the ~290 s on-arm window of an unreplicated sequential pair, not of
   a K-dependence (an `eigvalsh` on a fixed 512² matrix costs the same at iteration 400 as
   at iteration 1). The interleaved pair is the load-protected reading; this one is an upper
   bound.
3. **The contention band is measured, not asserted.** A third 16-thread run (K=60, reps=3,
   interleaved) came back at median Δ +328.7 ms/iter with **rep 1 reading ratio 0.69** — the
   probe arm FASTER than the guarded arm (215.1 s off vs 148.9 s on) — while off-arm cost
   inflated from 559 to 1808–3585 ms/iter, at per-pair loadavg 7.18 / 4.53 / 2.73 and 15.10
   observed mid-run (32 cores). Only the quiet-window interleaved pair and the 1-thread pairs
   are load-protected; no single 16-thread pair is a property of the code.
4. **The gated solve is ~22 % cheaper at the production thread setting and ~4–7 % cheaper at
   one thread** (interleaved medians: ratio 1.241 / 1.220 at 16 threads, 1.069 / 1.044 at 1),
   i.e. the +56–70 s/linear versus +4–8 s/linear above. It is a fixed per-linear tax that does
   not shrink while the GEMMs around it do (the `min_eigs` field's own comment), on all 196
   linears of a fire — which is what the row exists to establish.
5. **Optional follow-up (never blocking, per the ruling):** the production
   (interpreter × device) pair — `quantibbler-ptq:rocm-official` + `/dev/kfd` — is the only
   configuration in which this row's number is a production number; the dev CPU wheel cannot
   reach the accelerated path at all.

## Implementation crystallization (planner, 2026-10-26) — Phase 3 spec delta (task t_c78611dc)

**Decision: amend the existing requirement, do not author a new one.** The
requirement header *"Torch-autograd stages SHALL run on the accelerator device"*
covers the class. Phase 3's `TuneScalesKD` is a torch-autograd stage (forward and
backward both flow through PyTorch autograd), and the coder's audit plus the v6
fire (`qk-pertarget-v6-20261026T0030Z`) confirm it is the only remaining stage in
the class that is device-incompatible. One ADDED requirement covering all three
stages keeps the enumeration tight and prevents further drift.

The body now enumerates STE, TuneFP, and TuneScalesKD; Phase 3's distinguishing
feature (disk-backed teacher cache stays on the host — it is an I/O path, not a
compute path) is stated in both the requirement body and its dedicated scenario
(`tunescaleskd_gpu_zero_algorithm_change`).

**Task row added:** Task 6 (`tasks.md`) carries the acceptance contract the coder
specified: RED-first in the container lane (device-mismatch error at
`scale_calib.py:194`, confirmed by the v6 fire), GREEN with
activation-as-authority device plumbing (`linear.weight.device` authority,
`ub`/`vb` hoisted to stage scope, bias included), container-lane parity
acceptance (KL trace monotonic on both legs, scales finite and in trust
region), stage record with `device` and `elapsed_s`. Frozen-bits invariant
already tested by `test_scn_kl_scale_calibration`; packed-bit identity at
whole-pipeline scope is explicitly NOT asserted (the repo's own measurements:
packed cos 7.6e-3 apart due to discrete-search sensitivity).

**Why not a separate requirement:** TuneScalesKD's port shares every invariant
with the existing requirement — activation-as-authority device resolution, no
algorithm change, no globals. The differentiators (host-resident teacher cache,
f64-parameter/native-dtype-activation regime) are narrow enough for
scenario-level carve-outs. The fp32 tolerance class that covers STE and TuneFP
does not cover TuneScalesKD; its device equivalence is asserted through KL trace
agreement and scale finiteness (scenario `kd_gpu_device_parity`).
