# qk-launch-attribution — Tasks

Rules honored: tests precede implementation (RED observed before
code); every task independently verifiable; final task is the full
gate suite.

## 1. Reference first (tests)

- [ ] 1.1 RED: author `tests/ptq/test_launch_attribution.py` covering
      the spec scenarios: `launch identity line inside a checkout`
      (fake argv, temp scratch, rev == repo HEAD), `launch identity
      line outside a checkout` (no .git reachable → literal
      `unknown`), `default path disambiguates concurrent launches`
      (two concurrent driver processes → two `breadcrumbs-<pid>.log`
      files, no interleave), `configured path honored exactly`
      (QUANTIBBLER_BREADCRUMBS set → exact path, identity line first).
      Run `uv run pytest tests/ptq/test_launch_attribution.py` and
      OBSERVE the failures (missing rev/out fields, unsuffixed shared
      path). Record the RED output in the task note.
- [ ] 1.2 GREEN: implement in `tools/quantize_hf_model.py` — extend
      the launch line with `rev` (runtime-resolved, `unknown` outside
      a checkout) and resolved `out` dir; default trail path becomes
      `breadcrumbs-<pid>.log` in the KD scratch; configured path
      honored exactly. Same command exits 0 on all scenarios.
- [ ] 1.3 Pin rides the same commit: the launch-line construction
      joins the driver-wiring tripwire set
      (`tests/test_qk_report.py::_driver_wiring_findings`) — a later
      edit dropping a field reds the tripwire test. Tripwire battery
      still green after the addition.

## 2. Wrapper interplay

- [ ] 2.1 Confirm the container contract unchanged: wrapper's
      `DRY_RUN=1` output still names the same mounted breadcrumb path;
      `tests/test_run_ptq_wrapper.py` green; the forensics read path
      (tail of the configured file) unchanged.

## 3. Landing

- [ ] 3.1 Full gate suite: `uv run pytest` (lean: expected skip count,
      full: 0 skipped), scoped pre-commit, `openspec validate
      qk-launch-attribution --strict`; scoped commit (spec + code +
      tests + crystallization together).
