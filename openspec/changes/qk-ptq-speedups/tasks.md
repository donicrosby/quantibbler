# qk-ptq-speedups — tasks

## 1. Gate the min_eigs computation

- [x] Wrap the two `min_eigs` appends (the U-side and V-side continuous-update blocks in `lb_admm.lb_admm` — the `lb_admm.py:411, :415` citation in this row was stale, those lines are 40 commits of edits away; cited by symbol per the repo rule) in the probe-flag condition; `trace.n_jitter` untouched. Red-first: a production-config test asserts no eigenvalue calls occur with the flag off (monkeypatch/recorder), and the diagnostic config still emits the column.
      — **landed 2026-10-25 (t_2522c684): VERIFIED.** Gate = `SOLVER_PROBE_ENV` resolved once at the solve site in `lb_admm.lb_admm`; RED observed twice (collection ImportError RC=2 at `6c09948`, then the assertion itself: recorder on `_LinalgProxy.eigvalsh` counted `eigvalsh_calls 800` = 2 × 400 iterations with the flag unset, `assert eig_calls == []` False). GREEN after the change: the same probe prints `eigvalsh_calls 0`, and `tests/ptq/test_min_eigs_gating.py` = 4 passed (one test per scenario id in the delta: `production_run_pays_no_diagnostic_tax`, `diagnostic_run_keeps_the_column`, `gating_is_behavior_neutral`, `spd_signal_preserved`).

## 2. Flag-on/off timing measurement

- [ ] Measure solver wall time on real production layers with the flag on vs off; record the per-linear delta in the commit body (benchmark claim: 2× eigvalsh ≈ 142 ms/iteration at 16 threads).
      — **OPEN 2026-10-25 (t_2522c684): NOT TAKEN — blocked on the box being unmeasurable at 16 threads, and on the @user GO this ticket's own body requires for a run-bearing measurement.** Evidence collected instead (paired, load recorded): at 1 BLAS thread the guarded op pair reads 2×`eigvalsh(512)` = 585 ms vs 2×(matmul+Cholesky) = 212 ms per iteration (ratio 2.8, consistent with the claim's 142/62 ms at 16 threads); at `OPENBLAS_NUM_THREADS=16` on this shared host under co-tenant load (loadavg 51–66 on 32 cores) the same pair reads 102.9 s vs 13.5 s — a 176× inflation over the 1-thread reading, i.e. a contention artifact, not a code property. The per-linear production-number needs a quiet window or the ROCm image; script and exact launch line are on the ticket.

## 3. Gates and landing

- [ ] Suite green in the torch/hf lane (quote env + count), `openspec validate qk-ptq-speedups --strict`, scoped commit.

## 4. STE/TuneFP accelerator port

- [x] Move the STE and TuneFP stages to the accelerator via the existing device authority path — zero algorithm change; red-first parity tests: same loss values, same grad norms within the tolerance class on a fixture block, GPU fp32 vs CPU fp64.
      — **landed 2026-10-25 (t_c3fb5ce8): VERIFIED.** The stage plumbing itself predates this ticket (`e46bc0f`, "device-true packed weight in frozen_forward", plus its L1 follow-ups); RED was literally executed at `4393b85` (= `e46bc0f^`) in `quantibbler-ptq:rocm-official` + `/dev/kfd`: `STE cuda/f32: RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!` while TuneFP's accelerator leg and both CPU legs completed — at HEAD all four legs complete. GREEN: `tests/ptq/test_stage_device_parity.py` = 4 passed (container lane); the parity legs are isolated (fp32-exact weights and identical fp64 latents per leg, host-computed target) and every metric sits inside the spec's ±0.002 class. The stage records now also NAME their device (`device`, from the activation), which is what makes the placement verifiable from the artifact.
- [ ] Report per-stage row-time in the run report; measure the on/off wall-time delta per stage on production layers.
      — **report half landed 2026-10-25 (t_c3fb5ce8); measurement half OPEN.** The stage records carry `elapsed_s` + `device`, and `qk_report.stage_map` carries `rows` / `elapsed_s` / `row_time_s` per stage (3 lean tests in `tests/test_qk_report.py`, plus the record assertions in the parity file). The on/off delta is MEASURED at production SHAPES, not production layers (`tools/stage_row_time_bench.py`, default geometry = Qwen3-0.6B's 1024/3072, loadavg 43.29→60.97 on 32 cores): TuneFP 17.04 s → 4.69 s (3.63×/fire), STE 58.49 s → 17.25 s (3.39×/linear). A second run under the same load read 1.77× / 4.07×, so read the factor as a ~3× order of magnitude with a contention band, not a precise figure. NOT ticked: a real-model production-layer measurement is a fire and needs @user GO (lane authority) — the row-time column now makes the next fire's own stage numbers that measurement.
- [x] Disposition of `collect_stats` (calibration.py, mixed torch/numpy) recorded in design.md: GPU-movable pieces ported, anything left CPU-only named with a measured reason.

## 5. Remaining accelerable sweep

- [x] Enumerate every remaining CPU-bound piece (bit_polish, scale_calib, eval path) with a per-piece disposition — port, leave, or measure-first — recorded in design.md before the next production run.
