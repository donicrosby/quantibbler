# qk-flash-attention — Tasks

Rules honored: tests precede implementation per capability (RED observed
before code); every task independently verifiable; final task is the full
gate suite.

## 1. Reference first (Python face)

- [x] 1.1 RED: author `tests/test_attention_tiled.py` covering scenario IDs
      `ref-matches-formula` and `ref-causal-mask` (fixtures per
      specs/qk-kernels delta: seq 8/10/64, d 4/128, heads (2,1)/(8,2));
      run `uv run pytest tests/test_attention_tiled.py` and OBSERVE the
      failure (module absent). Record the RED output in the task note.
      — **RED observed:** `uv run pytest tests/test_attention_tiled.py` →
      collection error, rc=2:
      `ModuleNotFoundError: No module named 'quantibbler.kernels.attention_tiled';
      'quantibbler.kernels' is not a package`. The RED message itself surfaced a
      structural conflict (the spec's path is a *package* path while the repo had
      `src/quantibbler/kernels.py`) — resolved in the design.md crystallization
      entry below, and the file now covers the two named scenarios plus the
      wide-dynamic-range family and the contract edges (206 cases).
- [x] 1.2 GREEN: implement `src/quantibbler/kernels/attention_tiled.py`
      ([FA] §3.1 Algorithm 1, f64, ragged-tail safe); same command exits
      0 with all scenarios passing.
      — **GREEN:** `uv run pytest tests/test_attention_tiled.py` → **206 passed,
      rc=0**; full suite → **425 passed, 0 skipped, rc=0**; lean env (`/opt/data/tmp/ci-lean-venv`)
      → **343 passed, 16 skipped** and `tools/ci/pytest_summary.py` reports
      `skip discipline holds`; `pre-commit run --files` → **RC=0** twice (no
      rewrite round); `openspec validate qk-flash-attention --strict` → RC=0;
      built wheel ships `quantibbler/kernels/{__init__,attention_tiled}.py`.
      **Gate proven non-vacuous by a 9-mutation battery against the real module
      source** (each mutation asserted to change the text): dropped online
      rescale, ℓ not rescaled, causal off-by-one, causal ignored, GQA mapping
      wrong, scale dropped, mask applied post-softmax, ragged tail truncated,
      block-max instead of running max — **all 9 caught**; the last one is
      caught only by the wide-dynamic-range fixture family added after the
      battery showed it escaping the first battery (824 non-finite outputs).
- [x] 1.4 Fixture gap 1 (bug-hunter, demonstrated green-when-broken):
      the fully-masked-row NaN contract is unpinned — `_assert_roundoff`
      asserts all-finite by design, so it can never carry it. Add one
      dedicated test: a mask killing every score in one row, asserting
      the NaN pattern matches dense position-for-position (`isnan`
      both sides), explicitly not routed through the roundoff gate.
      Prove the fixture catches the demonstrated mutation (zeroing
      fully-masked rows via `np.where(ell == 0, 0.0, …)` must fail it).
      — **Landed, and the mutation had to be made live first.** New
      `test_ref_fully_masked_row_yields_the_dense_nan_pattern`: a mask with
      `-inf` across row 3, `isnan` compared position-for-position against the
      dense `0/0`, plus non-vacuity asserts (the dense oracle really carries
      NaNs, in exactly that row). **The demonstrated mutation was dead code
      as written**: the scenario's own premise says "row max −inf, **ℓ = 0**",
      but the implementation propagated NaN through `exp(−inf − (−inf))`, so
      ℓ was NaN and `ell == 0` could never fire. Fixed the statistic to match
      the spec: an empty row now yields `m = −inf, ℓ = 0` exactly (guarded
      α and weights), so the output's NaN comes from the same `0/0` the dense
      formula produces and `return_lse` is `−inf` rather than NaN. With that,
      the mutation is live and **fails the test** (rc=1), as the acceptance
      requires. Mutation evidence, real source mutated in place and restored
      byte-identically (sha256 `dfd109fd4534…` before and after): zero-the-row
      `np.where(ell == 0, 0.0, acc/ell)` **caught**; `np.nan_to_num(acc/ell)`
      **caught**; empty-row guard removed (`empty = zeros`) **caught** — that
      last one only by the `lse == −inf` clause, which is why the clause is
      asserted and not decorative.
- [x] 1.5 Fixture gap 2 (bug-hunter, demonstrated green-when-broken):
      no causal×mask combined fixture exists — `mask is not None and
      not causal` silently disables the mask on the causal path. Add
      one combined case: causal=True plus a strong negative bias on
      allowed (lower-triangle) positions; the dense reference is the
      oracle and the demonstrated mutation must fail it.
      — **Landed.** New `test_ref_combined_causal_and_additive_mask`:
      `causal=True` with a strong negative bias (−50·U[0,1]) on every
      *allowed* position, oracle `_dense(causal=True, mask=bias)`, plus a
      non-vacuity assert that dropping the bias changes the result (so a
      fixture that silently disabled the mask could not pass) and a
      finiteness assert. Mutation `if bias is not None and not causal:`
      **caught** (rc=1) by the same in-place mutation harness, source restored
      byte-identically. Benefit: the earlier 9-mutation battery re-run after
      the ℓ=0 fix still catches **9/9** with identical case counts — the
      statistic fix weakened nothing (the battery flagged its own stale anchor
      on the rescaled line rather than passing vacuously, which is the
      behaviour that makes it worth keeping).

## 2. GPU attention face (kernels crate, `gpu` feature, cubek substrate)

- [x] 2.1 RED: author `crates/quantibbler-kernels/tests/gpu_attention_parity.rs`
      (hardware-gated per house pattern: unset env = clean skip notice, set
      = device run) with scenario `gpu-parity-budget` (rel ≤ 1e-5 vs the
      NumPy reference) and `fail-closed-construction`; OBSERVE compile/run
      failure before wiring exists; record it.
      — **RED observed:** `cargo test -p quantibbler-kernels --features gpu
      --test gpu_attention_parity --no-run` → **rc=101**, 4 errors:
      `E0432 no AttentionShape in gpu` and `E0599 no method named 'attention'`
      / `no associated function ... 'init_at'` on `GpuContext`. The battery
      rides `tests/fixtures/attention_tiled.json` (9 cases: seq 8/10/64, d 4/128,
      heads (2,1)/(8,2), causal on/off, batch 2, and **two cases with
      head_dim ≠ val_dim** that separate `1/√head_dim` from `1/√val_dim`),
      generated by the new `tools/gen_attention_fixtures.py` from the NumPy
      reference — a device face is never compared against itself.
- [x] 2.2 GREEN: wire the cubek-attention forward pass behind the `gpu`
      feature — `cubek-attention = "0.3.0"` dep (verified: resolves on our
      cubecl 0.11 pin), GQA adapter (repeat K/V heads), Q pre-scale 1/√d,
      causal flag from the engine's mask, fail-closed typed init; hardware
      gate green on gfx1151 in BOTH env states; clippy/fmt green.
      — **GREEN, with one measured correction to the task's own wording.**
      *Resolution:* `cubek-attention 0.3.0` added offline (local registry
      cache) with `cubek-matmul/std/tile`; `Cargo.lock` keeps **cubecl 0.11.0
      unified** — no fork — and `cargo check --locked` passes.
      *Correction — Q must NOT be pre-scaled:* the substrate applies the
      standard `1/√head_dim` internally, so the audited "no softmax-scale
      field" reading was half wrong. The first adapter pre-scaled Q as
      specified and the battery caught the duplicate factor immediately
      (inf-norm rel **3.52e-1**); the device matched the dense reference at
      scale `1/(2√d)` and, with the host pre-scale removed, at `1/√d` with
      error **9.0e-08**. The measurement is the correction; the adapter now
      passes Q unscaled and says why.
      *Both gate states, real runs:* unset env → clean skip notice, 2 passed
      (parity skipped, fail-closed exercised); `QUANTIBBLER_GPU_TESTS=1` +
      staged SDK on gfx1151 → **9/9 fixture cases ran on the device**, all
      within the 1e-5 budget, 2 passed, 13.3 s.
      *Gates:* `cargo fmt --all -- --check` RC=0; `cargo clippy --all-targets
      -- -D warnings` RC=0 (default feature set) and RC=0 with `--features gpu`;
      default suite `parity.rs` 24 passed; `--features gpu` full suite RC=0;
      Python full suite 426 passed; lean 344 passed/16 skipped (`skip
      discipline holds`); pre-commit RC=0.
      *Named residual:* parity is asserted per fixture, single-shot — no
      repeated-run bit-identity assertion (the blueprint is inferred per call
      and the substrate may autotune, so determinism is not yet a contractual
      property of this face; if the engine later depends on it, it needs its
      own scenario and a pinned strategy).
- [x] 2.2a Binary-purity pair rides THIS commit (design invariant (3),
      sec-eng sequencing catch 2026-10-27): a sibling test file next to
      `binary_purity.rs` runs `cargo tree -p <pkg> -e normal,build
      --locked`, package-scoped, never workspace-wide, never
      `--all-features`, and asserts a POSITIVE+NEGATIVE pair —
      `cubek-attention` IS reachable from `quantibbler-kernels` under
      `-F gpu`, and `cubek-test-utils` is NOT reachable from
      `quantibbler-server`. The pair is self-guarding (dropping the dep
      reds the positive half; leaking the test-utils dep into the
      server graph reds the negative half). Acceptance: both halves
      assertable green on the current tree BEFORE 2.3 enables
      `cpu-reference`, so the hazard window never opens; clippy/fmt
      green; file hash pinned with the section-2 evidence.
      — **Landed.** New `crates/quantibbler-server/tests/cubek_purity.rs`
      (sha256 `f071847fab04`), 2 tests green. Commands are exactly the
      invariant's shape: `cargo tree -e normal,build --locked -p <pkg>`
      (`± -F gpu`), package-scoped, no workspace filter, no
      `--all-features`. **One refinement the literal acceptance cannot
      carry:** the crate-absence half is vacuous as a name grep, because
      `cubek-test-utils` does not appear in `Cargo.lock` at all (count 0)
      until someone enables `cpu-reference` — so "cubek-test-utils
      unreachable" can only ever pass. The negative half is therefore
      asserted at the FEATURE level, on the same invocation, with
      `--format "{p} {f}"`: the server graph must render
      `cubek-attention v0.3.0 std` and must not carry `cpu-reference`
      anywhere; non-vacuity is asserted in that same graph (the attention
      crate must be present *with* a rendered feature list), and the
      name-absence grep is kept as a documented backstop rather than the
      guard. Both halves were then shown load-bearing by mutating the real
      manifests in place and restoring every touched file byte-identically
      (kernels `Cargo.toml` `ef4afa092f9e`, `Cargo.lock` `0d3a994d3309`):
      dropping `dep:cubek-attention` from the `gpu` feature **reds the
      positive half** (rc=101); enabling `cpu-reference` on the attention
      dep **reds the negative half** (rc=101).
      — **BLOCKING FINDING, and it is my own defect in the referee:** the
      `rust-kernels` job is manifest-scoped
      (`--manifest-path crates/quantibbler-kernels/Cargo.toml`), so **no CI
      job runs `quantibbler-server`'s test targets at all** — meaning this
      new pair, *and the pre-existing `binary_purity.rs` no-python guard*,
      are local-only today. "Green on the current tree" is satisfied
      locally but unenforced where it matters. Exact replacement verified
      locally before proposing it: `cargo test --locked -p
      quantibbler-kernels -p quantibbler-server --all-targets` →
      **150 passed / 0 failed**, with `binary_purity.rs` (12) and
      `cubek_purity.rs` (2) both executing; clippy `-D warnings` clean on
      both crates; `cargo fmt --all --check` clean. This is a job-scope
      change to the CI referee (needs its own scenario delta in the active
      `qk-ci-referee` change) and it moves the already-transferred
      `ci.yml` sign-off hash `b637a0601e3b` — @planner ruled it rides **this**
      commit (a commit that ships a guard nothing executes fails its own
      acceptance on arrival), and it is landed: the rust job's test/clippy gates
      now select `-p quantibbler-kernels -p quantibbler-server`, the referee pins
      that selection against a five-row bypass battery (bug-hunter's addition),
      and `ci.yml` moved `b637a0601e3b` → `8cfb8511f5e3` with the fresh transfer
      hashes recorded in the active `qk-ci-referee` change's tasks. The pair is
      CI-enforced, not merely green locally.
- [ ] 2.3 Third-face cross-check: run the cubek `cpu-reference` feature
      against our NumPy reference on the fixture battery (both CPU, exact
      f64-class comparison where the reference allows) — independent
      implementations agreeing is the strongest available guard against a
      shared misreading of the conventions; record deltas.
- [ ] 2.4 Re-sync the PyO3 face (`uv sync --reinstall-package
      quantibbler-kernels`) and confirm the CHANGED behavior (new symbol
      computes, not just imports).

## 3. Server attention face (dispatch + parity)

- [ ] 3.1 RED: author `crates/quantibbler-server/tests/attention_dispatch.rs`
      (scenario `face-selection-health`, `soft-fallback-init-failure` —
      the fallback test must run WITHOUT hardware); observe RED.
- [ ] 3.2 GREEN: engine wiring mirroring MatvecFace: AttentionFace enum,
      select-once at load, env opt-in, soft fallback, /health
      `attention_face` label; default-feature suite green (scalar path
      untouched).
- [ ] 3.3 Hardware parity: scenario `cross-face-token-identity` on gfx1151
      (fixture pipeline greedy generation, both faces, tokens identical,
      logits rel ≤ 1e-5).

## 4. Perf probe (OQ-1 — resolved only by measurement)

- [ ] 4.1 Timed decode-attention benchmark, CPU scalar face vs GPU tiled
      face at (seq 512, seq 2048) × d=128 × GQA 8:1 on gfx1151; record
      the table in design.md crystallization; OQ-1 answered WITH numbers.
      Outcome gate: ≥ 1.5× at seq 2048 keeps the face opt-in-recommended;
      a miss keeps it opt-in-experimental and records why.
- [ ] 4.2 Reference benchmark (user directive 2026-10-27): install pip
      `flash-attn` (ROCm build for gfx1151) in the sanctioned rocm
      container and run the SAME decode-shaped timed benchmark
      (seq 512/2048 × d=128 × GQA 8:1) through its interface; record
      wall time AND peak memory for both faces side by side in the
      design.md crystallization table. Outcome gate per the directive:
      ours meets-or-beats on both axes = the face is set; a shortfall
      records the gap and stays opt-in-experimental until it closes.
      The harness pins equal shapes, dtype, causal masking, warmup, and
      batch for both — no benchmark-adjacent advantage may differ
      without being recorded in the table's own header.

## 5. Landing

- [ ] 5.1 Full gate suite: `uv run pytest`, `cargo test` (kernels + server,
      both feature states), `cargo clippy --all-targets -D warnings`,
      `cargo fmt --all -- --check`, `uv run maturin build`, and
      `openspec validate qk-flash-attention --strict`; scoped commit
      (spec + code + crystallization together).
