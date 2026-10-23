# qk-ptq-speedups — tasks

## 1. Gate the min_eigs computation

- [ ] Wrap the two `min_eigs` appends (`lb_admm.py:411, :415`) in the probe-flag condition; `trace.n_jitter` untouched. Red-first: a production-config test asserts no eigenvalue calls occur with the flag off (monkeypatch/recorder), and the diagnostic config still emits the column.

## 2. Flag-on/off timing measurement

- [ ] Measure solver wall time on real production layers with the flag on vs off; record the per-linear delta in the commit body (benchmark claim: 2× eigvalsh ≈ 142 ms/iteration at 16 threads).

## 3. Gates and landing

- [ ] Suite green in the torch/hf lane (quote env + count), `openspec validate qk-ptq-speedups --strict`, scoped commit.

## 4. STE/TuneFP accelerator port

- [ ] Move the STE and TuneFP stages to the accelerator via the existing device authority path — zero algorithm change; red-first parity tests: same loss values, same grad norms within the tolerance class on a fixture block, GPU fp32 vs CPU fp64.
- [ ] Report per-stage row-time in the run report; measure the on/off wall-time delta per stage on production layers.
- [ ] Disposition of `collect_stats` (calibration.py, mixed torch/numpy) recorded in design.md: GPU-movable pieces ported, anything left CPU-only named with a measured reason.

## 5. Remaining accelerable sweep

- [ ] Enumerate every remaining CPU-bound piece (bit_polish, scale_calib, eval path) with a per-piece disposition — port, leave, or measure-first — recorded in design.md before the next production run.
