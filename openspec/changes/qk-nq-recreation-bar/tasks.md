# Tasks: qk-nq-recreation-bar

- [x] (coder) Tune/eval window split, RED first: tests for `split_mode_preserves_eval_geometry` and `default_single_geometry_unchanged`; observe RED (flag absent). The measured interim geometry — option (a) of the design's constraint section; this is what converts the seq-2048 protocol from OOM to a number.
      — **RED observed** on the new `tests/ptq/test_driver_tune_seq_len.py`, all
      three tests failing for their intended reasons (not one shared accident):
      `--tune-seq-len` absent from `--help`; a real driver run completing and
      then `KeyError: 'tune_seq_len'` on its report (so the fixture path is
      proven live, not assumed); and the split invocation exit 2 with
      `unrecognized arguments: --tune-seq-len 16`. `3 failed in 12.35s`.
- [x] (coder) GREEN: `--tune-seq-len` in the driver with pipeline slicing; report records both lengths; wrapper `TUNE_SEQ_LEN` env knob passthrough; lean/full suites green.
      — **Landed.** Driver: `--tune-seq-len` (`_positive_int`, default `None`)
      resolving to `--seq-len` when unset; `calib` SLICED into windows of that
      length rather than re-sampled, so every calibration token survives (the
      sampler's 2048-token rows are the paper's protocol — a 512-token sample
      would shrink the calibration budget to a quarter); `calibration_rows`
      counted BEFORE the slice because it counts sequences and slicing adds
      none. The report records the RESOLVED `tune_seq_len` on both paths
      (`report_single_shape_both_paths` holds). Wrapper: `TUNE_SEQ_LEN`
      passthrough, empty default, flag appended only when set — the wrapper does
      not author a tuning geometry. Suites: **full 438 passed**; **lean 353
      passed / 16 skipped**, `skip discipline holds`; pre-commit **RC=0**;
      `validate --strict` valid. Design-silent choices, the RED transcript and a
      guard finding are recorded in this change's design.md
      ("Implementation crystallization 2026-10-20").
- [x] (bug-hunter) Review: the seq-2048 eval clause cannot be satisfied by eval-at-512; default (flag unset) geometry is behavior-identical to pre-flag; any memory claim comes only from the 32g-container microbench with crumb-format peaks.
      — **Reviewed on landed bytes @ HEAD 075e7ec.** All three clauses PASS.
      (1) Eval geometry is `--seq-len` only: the report records `seq_len` and
      `tune_seq_len` as separate fields, and `qk_report.geometry_deviations`
      emits a `tune_geometry_split` deviation that `fidelity_conditions` feeds
      into the computed `paper_faithful` verdict, so a split run cannot carry
      the faithful label; the split-run transcript shows `calib 2 batches,
      eval 2 batches` + `calib sliced to 4 windows of 16` (slicing ran, the
      sequence count did not move). (2) Default path: diff 4617038 shows unset
      → `tune_seq_len = args.tune_seq_len or args.seq_len` with the slice
      branch gated on `tune_seq_len != args.seq_len`, so the pre-flag path is
      skipped entirely (no slicing, no new print); `calibration_rows` keeps
      the identical `len(calib) * BATCH` expression, hoisted above the slice;
      `_positive_int` closes the `--tune-seq-len 0` back door. (3) Every
      memory number in the change cites the death record (container
      `2d2d4b6df1e0`, 33,480,724 kB vs the 32g memcg, crumb 30.14G at
      15:57:20Z) or the measured seq-512 transient, and the test tier asserts
      no memory fact (docstring assigns the location claim to the microbench).
      RESIDUAL (foreign, not this change's bytes): the two subprocess tests
      are red on the current tree — `lb_admm.py:586` resolves
      `get_backend(backend)` with no device and no `QUANTIBBLER_DEVICE` read,
      so the torch-default solve refuses on CPU mid-driver-run; introduced by
      8d1bcc0 (2026-10-23, torch-default routing; the split landing predates
      `solver_backend.py`'s existence), 97a4725's env-threading repaired
      `lb_admm.py:422` but missed this site. Witness: 1 passed / 2 failed in
      41.10s + two standalone driver runs, byte-identical RuntimeError at
      [4/6]. Fix carded to coder; blocks the landing gate's full pytest.
- [x] (coder) Suite test first per `bar_constants_asserted`/`bar_not_purchased_with_bits`: a torch-free module or test module pinning the four row constants + tolerance arithmetic with the derivation audit (constants asserted, not recomputed). Observe RED.
      — **RED observed** on `tests/test_qk_bar.py` before any bar module existed:
      `ImportError: cannot import name 'qk_bar' from 'quantibbler'`, `1 error
      during collection`, RC=2. Then: the four constants pinned as literals
      (`FP16_BASELINE_PPL` + the three `BAR_ROWS` entries), the tolerance
      computed as `TOLERANCE * row` and asserted UNEQUAL to the design table's
      2dp renderings (31.69/38.86/60.88 are renderings of 31.694/38.8585/60.881),
      and the derivation audit (`_derivation_findings`, parsing the module's own
      source) rejecting a row recomputed at import time — four poisoned-source
      cases (`6.33 * 2.0`, `115 / 100`, `float("27.56")`, a bare name) each
      produce a named finding against the module's real text, so the tripwire is
      non-vacuous. Provenance re-read from the pinned copy: line 881 → 12.66,
      1035 → 27.56, 1079 → 33.79, 1123 → 52.94, each value unique in the file.
      `9 passed in 0.43s`, RC=0.
- [x] (coder) Wire the test location per repo layering (torch-free constants live where the band constants live — `qk_report` or a sibling bar module if the change spirit demands separation); GREEN in lean env.
      — **Landed as a SIBLING module, `src/quantibbler/qk_bar.py`** (torch-free,
      package root, the `qk_report.py` layering) — task row 27's second option,
      taken because this change's own `bar_distinct_from_landing_gate` /
      `bar_never_recalibrates_the_band` spirit wants the bar structurally
      separate: `qk_report` does not import it and nothing on the run path reads
      it, so the bar cannot become a per-run verdict. `uv sync --all-groups
      --extra kernels` RC=0. Lean gate `uv run --no-sync pytest -q` →
      **437 passed, 0 skipped, RC=0** (428 before this change's 9 tests).
      Interpreter: a private `UV_PROJECT_ENVIRONMENT` (`uv sync --all-groups
      --extra kernels`, `torch` absent); the shared `/opt/venv` was re-synced
      with `--extra torch` by a co-tenant lane mid-run, which collects the ptq
      tier and its 56 known dev-CPU-wheel `resolve_device` refusals — none of
      them this file — so the hermetic env is what makes the lean number
      attributable. `ruff check` + `ruff format --check` RC=0 on both new files.
      Design-silent choices recorded in this change's design.md
      ("Implementation crystallization 2026-10-25 — bar constants, tasks 26-27").
- [x] (orchestrator) Author the acceptance-run ledger format (fields per `bar_recorded_not_gated` scenario) as a doc/format note in this change, not code.
      — **Authored 2026-10-25: `ledger-format.md` in this change dir.** One
      append-only JSONL ledger at `docs/acceptance-ledger.jsonl`, one entry per
      acceptance evaluation, human-authored at gate time from
      `quantize_report.json`. REQUIRED fields carry the scenario's eight
      verbatim — `model_id`, `commit` (40-hex), `targeted_row` (closed
      `qk_bar.BAR_ROWS` label enum; nothing below 0.55 bits is writable),
      `eval_protocol` (corpus/partition/seq_len/tune_seq_len/calibration),
      `measured` (`ppl`, `ratio` = ppl/row, `ratio_vs_baseline`,
      `effective_bpw`, plus copied `row_ppl`/`row_bits`/`bar_ceiling`),
      `tolerance_provenance` (provisional 1.15 spec pin), `bar_met`
      (`row_meets_bar` semantics AND paper protocol — surrogate-only entries
      stay `false`), `operational_gate_verdict` (a recorded COPY of the band +
      report path, never computed) — plus the human gate (`acceptance`:
      decision `met`/`not_met`/`equivalence`, `decided_by`, `decided_at`) and
      integrity fields (`recorded_at`, `run_command`, `report_path`,
      `artifact_path`, optional `protocol_deviation.equivalence_argument`,
      `supersedes`). Doc-only by design (proposal Impact: tooling when the
      first acceptance run is scheduled). Design-silent choices recorded in
      this change's design.md ("Implementation crystallization 2026-10-25 —
      ledger format, task 28").
- [x] (bug-hunter) Review: the constants match the pinned paper copy by grep (docs/papers/nanoquant-2602.06694v3.md rows 27.56/33.79/52.94/12.66); tolerance is 1.15x verbatim, not rounded in code; the seq-2048 clause cannot be satisfied by seq-512 data; the bar never feeds a per-run verdict.
      — Paper-match half DELIVERED early (bug-hunter, 2026-10-25, @ HEAD
      075e7ec): a programmatic parse of the pinned copy's Table 2 puts
      Q3-0.6 at cell index 15 of 21, and the four values sit exactly there —
      line 866 FP16 16.00 bits → 12.66; line 1020 NanoQuant Total Bits 1.00 →
      27.56; line 1064 @0.80 → 33.79; line 1108 @0.55 → 52.94 (grep lines
      881/1035/1079/1123 agree). The change's pin is verbatim `1.15 * row`
      (design table: 31.69/38.86/60.88); no bar code exists yet, so nothing
      rounds the tolerance and nothing gates a per-run verdict on it (the
      `qk_report` band is LB2/STBLLM-derived: OK_LINE=4, COLLAPSE_LINE=40).
      BOX STAYS OPEN: the code-side half — constants asserted not recomputed,
      1.15 never rounded IN CODE — is reviewable only when the constants test
      lands (tasks 26-27); re-review then.
      — **Code-side half REVIEWED 2026-10-25 on landed bytes @ 60c075f; all
      four clauses PASS.** Byte chain: worktree sha256 of `qk_bar.py`
      (32500b47…), `test_qk_bar.py` (0ec3b1db…), `ledger-format.md`
      (ca413560…) equals the commit's blobs, so the suite ran the LANDED
      bytes: `uv run pytest tests/test_qk_bar.py -v` → **9 passed, RC=0**
      (Python 3.12.14). (1) Constants are COPIES, and the audit BITES:
      `git show 60c075f:docs/papers/nanoquant-2602.06694v3.md` lines
      881/1035/1079/1123 read `| 12.66 / | 27.56 / | 33.79 / | 52.94` — the
      exact `FP16_BASELINE_PPL` / `BAR_ROWS` literals; `_derivation_findings`
      demands module-level `ast.Constant` literals for the scalars and the
      ppl/bits columns (labels + paper-line columns are pinned by full-tuple
      equality with the test oracle, whose line column the suite re-reads
      from the paper with a uniqueness count — no hole), and the 4
      poisoned-source cases (`6.33 * 2.0`, `115 / 100`, `float("27.56")`,
      bare `_ROW_PPL`) each yield their NAMED finding against the real
      module text (mutation-applied asserted in-test). (2) `1.15` verbatim:
      `bar_tolerance` returns `TOLERANCE * row_ppl` — no rounding anywhere;
      the suite asserts `bar_tolerance(27.56) != 31.69` (+ the other two 2dp
      renderings) and the torch-free subprocess asserts
      `== 1.15 * 27.56` exactly; `row_meets_bar` compares the UNROUNDED
      product. (3) seq-512 cannot buy the bar: ledger-format.md defines
      `bar_met` = `row_meets_bar` semantics AND paper protocol
      (`seq_len == 2048`, WT2 test) — surrogate-only stays `false`; format
      rule 1 makes `decision: "met"` require `bar_met: true`, and
      `equivalence` requires `bar_met: false` + a non-empty
      `equivalence_argument` (a recorded human acceptance of a deviation,
      never a flip) — exactly scenario `bar_requires_paper_protocol`.
      (4) The bar never feeds a verdict: `git grep -c qk_bar 60c075f --
      src/quantibbler/qk_report.py` → 0 matches (re-checked at main tip
      8c226fa: still 0); `row_meets_bar` has no caller outside `qk_bar.py` +
      its test at either commit; `docs/acceptance-ledger` has ZERO
      references in src/tools/tests @ 60c075f — the ledger is doc-only by
      construction; design.md crystallization records the unwired state as
      declared debt ("wired to the ledger, never to `verdict()`"). NIT for
      the future ledger tooling (non-blocking): `_measured` coerces bools
      (`float(True) == 1.0`), so the tooling should reject bools explicitly.
- [x] (orchestrator) `npx --yes @fission-ai/openspec validate qk-nq-recreation-bar --strict`; scoped stage (spec + code ONE commit); gate-chained commit; fenced push; CI watch.
      — **DONE 2026-10-25.** Validate (`npx --yes @fission-ai/openspec@1.14.3`,
      exactly ONE id): `Change 'qk-nq-recreation-bar' is valid`, RC=0, re-run
      after the final tasks.md edit. Scoped stage: ONE landing commit
      `60c075f` = 5 files (design.md, tasks.md, ledger-format.md,
      `src/quantibbler/qk_bar.py`, `tests/test_qk_bar.py`), +714/−5, verified
      `git show --stat`; companion fix commit `aff8fe6` (+6/−1 — exactly the
      balance hunk; first attempt d2d4eab discarded unpushed after a sibling
      lane's in-flight `lb_admm.py` edits rode the `git add`, blob rebuilt
      surgically, worktree untouched — see the card comment). Gate-chained:
      validate && scoped add && commit, RC=0. Fenced push: unwrap → push →
      re-fence inside one command, `6c09948..60c075f HEAD -> main`,
      **PUSH_RC=0**, pre-push secret fence **0 hits** (face1 tree + face2
      outgoing range), fence restored. HTTPS had no credentials (the e2e
      precedent); pushed with the sandbox's gh GITHUB_TOKEN via a throwaway
      credential-helper script, removed after the push.
      **CI watch — run `40169509339` (event push, head `60c075f`), per-job
      verdicts read back from the GitHub API:** rust kernels ✅ · openspec ✅
      (every active change, strict) · seed-weights ✅ · **lean python ❌**
      `5 failed, 414 passed, 18 skipped` (expected 16) — the SAME five
      `tests/test_run_ptq_wrapper.py` env-guard failures and the same
      skip-count delta as the base run `40168399393` @ `6c09948` (`405
      passed` there; +9 = this landing's bar tests): **inherited main red,
      foreign to this range** · **full python ❌** `57 failed, 608 passed,
      2 skipped` — base was `62 failed, 594 passed`: **this landing's
      companion fix `aff8fe6` repaired exactly 5** (the 2 canaries + 3 driver
      subprocess tests); the remaining 57 + 2 zero-skip violations are the
      documented 8d1bcc0 default-torch-on-accelerator-less-runner class ·
      **driver smoke ❌ / abort leg ❌** with the same failure text as base
      (driver dies before writing quantize_report.json — same class). Zero
      new reds introduced; the base run was already 4-of-7 red.
- [x] (orchestrator + team) Full gate suite before landing — `uv run pytest`, `cargo test` in `crates/quantibbler-kernels`, `cargo clippy --all-targets -- -D warnings`, `cargo fmt --all -- --check`, `uv run maturin build --manifest-path crates/quantibbler-kernels/Cargo.toml` — and record results.
      — **Observed 2026-10-25 on the landing tree (shared `/opt/venv`, torch
      2.14.1+cpu, no accelerator):** `tests/test_qk_bar.py` → **9 passed**,
      RC=0; `tests/ptq/test_solver_backend.py` → **12 passed**, RC=0 (the
      t_7190f18b canary suite — refusal pins intact with the balance-site
      fix). Full tier `uv run --no-sync pytest -q -rs` → **53 failed /
      621 passed / 3 integration deselected** (documented default marker
      filter), RC=1, 177.32s, 677 collected — the documented dev-CPU-wheel
      class (implicit torch resolve refuses without an accelerator; driver
      subprocesses exit 1 and their assertions fail downstream): 56 measured
      on HEAD+fix by t_7190f18b; the −3 failed / +6 collected delta rides
      sibling lanes' IN-FLIGHT test edits in the shared tree
      (`test_admm_convergence.py`, `test_geometry_conformance.py` — not this
      landing's paths); zero failures in any path this landing stages. Rust,
      CI's stronger `-p` form: `cargo test --locked -p quantibbler-kernels -p
      quantibbler-server --all-targets` + `cargo clippy --locked -p
      quantibbler-kernels -p quantibbler-server --all-targets -- -D warnings`
      + `cargo fmt --manifest-path crates/quantibbler-kernels/Cargo.toml
      --all -- --check` + `uv run --no-sync maturin build --manifest-path
      crates/quantibbler-kernels/Cargo.toml` — **chain exit 0**.
      `npx --yes @fission-ai/openspec@1.14.3 validate qk-nq-recreation-bar
      --strict` → `Change 'qk-nq-recreation-bar' is valid`, **RC=0** (exactly
      one id per invocation). pre-commit on all six changed paths (new files
      named explicitly) → **RC=0**.
