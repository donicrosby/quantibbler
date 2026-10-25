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
      sec-eng sequencing catch 2026-10-20): a sibling test file next to
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
- [x] 2.3 Third-face cross-check: run the cubek `cpu-reference` feature
      against our NumPy reference on the fixture battery (both CPU, exact
      f64-class comparison where the reference allows) — independent
      implementations agreeing is the strongest available guard against a
      shared misreading of the conventions; record deltas.
      — **LANDED, with the budget reformulated exactly as pre-registered.**
      `cargo test -p quantibbler-kernels --features cubek-cpu-reference
      --test cubek_cpu_reference_parity --offline` → **3 passed, 0 failed**,
      with **9/9 fixture cases executing on CPU** (the old single-clause budget
      reddened at case #0 and stopped the battery there, so cases #1-#8 had
      never run). The budget shape is now the two-part rule this change already
      accepted for the Python and device faces, LOCAL to this file: clause 1
      `max|a-e| <= 1e-5 * max(|a|,|e|)`, clause 2 per-element `<= 1e-5` within
      `BAND = 1e-1` of the row max. **`RTOL` is unchanged — the shape changed,
      not the number** — and the unbanded per-element statistic is still printed
      as a RECORDED DELTA instead of being dropped. Measured margins: worst
      clause 1 **2.004e-07** (50x under), worst clause 2 **3.941e-06** (2.5x
      under). Branch selection, the per-case table for BOTH faces, the
      element-level localization, and the citation defect on the
      pre-registration's `atol + rtol·|ref|` citation (no such rule was ever
      written: the rule text occurs once in the openspec corpus, in the sentence
      citing it, and `atol` is a tolerance formalism nowhere in the repo — the
      other `atol` tokens are numpy `allclose(atol=…)` call sites in `tests/ptq`)
      are recorded in design.md, "2.3 witness budget — control result and
      landing"; @planner ruled to keep the landed gate and fix the citation.
      *Proof-of-power row landed as a mandatory component* (sec-eng): one
      injected O(1) defect of the device battery's family — GQA mapping
      replaced by modulo, pushed through the REAL reference — misses BOTH
      clauses by **7.6e+05x** the budget, with the healthy leg asserted first so
      the row cannot pass against a broken baseline. The shared GPU budget
      (`gpu_attention_parity.rs`, sha256 `315b7cdd25b2…`, byte-identical) and
      the clamp-divergence tripwire are both untouched; one defect class per
      budget.
      *Two further defects fixed in the same file:* `expand_heads` was
      batch-blind (case #5, the first batch=2 case ever reached, failed the
      length assert `160 != 80`), and the power row's first draft targeted a
      fixture with one kv head, where the injected mapping is a no-op — its own
      assert caught that before it could pass vacuously.
- [x] 2.4 Re-sync the PyO3 face (`uv sync --reinstall-package
      quantibbler-kernels`) and confirm the fresh build COMPUTES — artifact
      identity plus behavior, never import (`a stale .so imports cleanly and
      runs old code`).
      — **Re-synced and confirmed.** `uv sync --all-groups --extra torch
      --extra kernels --extra hf --locked --reinstall-package
      quantibbler-kernels` → **RC=0** (`Building`/`Built quantibbler-kernels`,
      2.54 s). *Artifact identity:* the wheel uv built in that sync and the
      installed `/opt/venv/lib/python3.12/site-packages/quantibbler_kernels/
      quantibbler_kernels.abi3.so` are byte-identical (sha256
      `4e7ba876a413…` both), so the face under test IS the fresh build, not a
      cached predecessor. *Behavior, measured on that face:* all six v0.1
      kernels re-computed against the NumPy reference — `pack_bits`
      bytes-identical (int8 AND float64 boundaries), `unpack_bits` exact in
      both dtypes, `packed_matvec` / `packed_matmul` / `joint_itq` **max rel
      0.000e+00** against the 1e-12 gate, rejection contract intact
      (`ValueError … got 0.5`); `uv run pytest tests/test_kernels_parity.py
      tests/test_kernels_dispatch.py -q` → **27 passed, 0 skipped** (Rust leg
      active).
      — **Row text corrected by measurement, and this is the correction.**
      The row was authored 2026-10-19, before the CubeK directive, when §2
      still meant a hand-written kernel exposed *through the extension* (row
      2.1's own text reads "vs the NumPy reference **via the PyO3 face** or
      fixture vectors"). The 2026-10-20 substrate decision moved the attention
      face to the **rlib** side behind the non-default `gpu` feature
      (`gpu_attention` is `#[cfg(feature = "gpu")]` in `lib.rs` and is never
      exported through `#[pymodule]`) and moved its parity proof to
      `cargo test --features gpu`. Measured on the built extension: the
      exported surface is exactly the v0.1 six plus `backend_version`, and a
      default-feature dependency graph contains no `cubek-attention` at all —
      so §2 contributes **zero bytes** to the extension (which is why the
      re-sync reproduced the predecessor byte-for-byte). No requirement of
      record asks for a PyO3 attention binding: `openspec/specs/qk-kernels`
      enumerates the dual-face surface and its PyO3-parity requirement over
      those six kernels only. The row is therefore ticked for what it can
      truthfully mean (re-sync + identity + behavior), NOT expanded into an
      unspec'd binding; recorded in design.md, "Implementation crystallization
      (2026-10-25, task 2.4)". If a Python-callable attention face is wanted
      after all, the spec delta must change first (same rule as the 1.1–1.2
      package-path resolution) — the row text alone cannot order it.
      — **Attention face behavior re-confirmed in BOTH gate states on the
      current tree** (announced before firing, this ticket's board comment;
      `LD_LIBRARY_PATH=/opt/data/tmp/rocm-sdk/lib cargo test --locked -p
      quantibbler-kernels --features gpu --test gpu_attention_parity`):
      unset → **RC=0, 2 passed** with the clean skip notice; with
      `QUANTIBBLER_GPU_TESTS=1 HSA_OVERRIDE_GFX_VERSION=11.5.1` → **RC=0,
      2 passed**, printing `attention parity: 9/9 fixture cases ran on the
      device` (378 s under co-tenant load). The Python form of the same face:
      `uv run pytest tests/test_attention_tiled.py -q` → **209 passed**.

## 3. Server attention face (dispatch + parity)

- [x] 3.1 RED: author `crates/quantibbler-server/tests/attention_dispatch.rs`
      (scenario `face-selection-health`, `soft-fallback-init-failure` —
      the fallback test must run WITHOUT hardware); observe RED.
      — **RED observed:** `cargo test -p quantibbler-server --test
      attention_dispatch --no-run` → **rc=101**, `error[E0599]: no method
      named 'attention_face' found for struct
      Arc<engine::ModelWeights>` at the test's own call site. **GREEN after
      3.2:** default build → **14 passed, 0 failed** (the three scenario
      tests + the eleven embedded module unit tests); `--features gpu` →
      **14 passed, 0 failed**. Both scenario halves are real, not nominal:
      unset gate prints the clean skip notice, and with
      `QUANTIBBLER_GPU_TESTS=1` + staged SDK on gfx1151 the DEVICE arm selects
      `gpu:tiled` on the real device (rc=0). The fallback arm needs no
      hardware in either build — the probe index is `usize::MAX / 2`, the
      same deterministic construction the kernels crate's own
      `fail_closed_construction_without_a_usable_device` uses, and it fired
      for real (`device probe failed at index 4611686018427387903:
      NoSuchDevice { runtime: Hip, available: 0 }` bare, `available: 1` with
      the staged SDK — so it is not a lucky-absence pass). **No scenario label
      in the file is a phantom:** the delta's scenarios carry HEADINGS only
      (`grep` over `specs/` finds neither kebab id), the ids are `tasks.md`'s
      own, and the docstring says so. Gate proven non-vacuous: mutating the
      shipped label string (`"cpu:scalar"` → `"cpu:scalar-MUTATED"`) reds all
      three label assertions with their own messages (11 passed / 3 failed,
      rc=101), source restored byte-identical (sha256 compared before and
      after — the value is not quoted here, since the file has gained
      comment-only edits since and a hash in prose is rot by construction).
- [x] 3.2 GREEN: engine wiring mirroring MatvecFace: AttentionFace enum,
      select-once at load, env opt-in, soft fallback, /health
      `attention_face` label; default-feature suite green (scalar path
      untouched).
      — **Landed.** `AttentionFace { CpuScalar, GpuTiled(GpuContext) }` +
      `AttentionFaceDecision { cpu_scalar_selected, select_from_env,
      describe }` in `engine/mod.rs` — the `FaceDecision` shape verbatim —
      selected ONCE in `ModelWeights::from_layers` beside the matvec decision,
      read back by `ModelWeights::attention_face()` and served by `/health` as
      `attention_face` next to `matvec_face`. The opt-in NAME is the one thing
      the delta leaves open (it says only "an env opt-in"): `QUANTIBBLER_ATTN=1`,
      its own variable so the two frozen decisions stay independently
      selectable — recorded as a design-silent decision in design.md, since a
      task row cannot mint spec vocabulary. Soft fallback is the matvec policy
      exactly: a failed `GpuContext::init_at` logs the typed staging hint and
      returns `cpu:scalar`; startup never aborts. The forward dispatch sits in
      `gqa_attention` (a one-line face match); the shipped scalar body moved
      VERBATIM to `gqa_attention_scalar` — no expression changed, so the
      default face is the code that shipped (that is what "scalar path
      untouched" means) — and `gqa_attention_tiled` expresses one decode step
      as a single query position (`seq_q = 1`) against the block's whole cache
      (`seq_kv = pos + 1`), `causal = true`, `val_dim = head_dim`; no gather
      was needed because `KvCache` already stores a block as one contiguous
      `[seq_kv, kv_dim]` stack.
      *Commands and counts:* `cargo test -p quantibbler-server` → **140 passed,
      0 failed, 10 ignored** over 12 targets (incl. `binary_purity` 12,
      `cubek_purity` 2, `gpu_dispatch` 13, `load` 14, `attention_dispatch` 14),
      rc=0; `cargo clippy -p quantibbler-server --all-targets -- -D warnings` →
      rc=0; `cargo check -p quantibbler-server --features gpu --all-targets` →
      rc=0 with no warnings (and `clippy` clean there too); `cargo fmt -p
      quantibbler-server -- --check` → rc=0; `cargo fmt --all -- --check` →
      rc=0; kernels gates unchanged (`cd crates/quantibbler-kernels && cargo
      test` → 24 passed rc=0, clippy rc=0, fmt rc=0).
      *Named residual:* the device arm of the dispatch is
      `#[cfg(feature = "gpu")]`, so NOTHING in the default build executes it —
      this row proves selection, labelling, soft fallback and that the gpu arm
      COMPILES; the device face's number-identity (`cross-face-token-identity`)
      is row 3.3, left unchecked below.
- [x] 3.3 Hardware parity: scenario `cross-face-token-identity` on gfx1151
      (fixture pipeline greedy generation, both faces, tokens identical,
      logits rel ≤ 1e-5).
      — **Landed, and it was RED first on the device.**
      New `crates/quantibbler-server/tests/cross_face_token_identity.rs`
      drives the FIXTURE pipeline (`server-model` + its tokenizer, greedy
      argmax) through both faces — two constructions under the two env
      states, the `gpu_dispatch.rs` shape — and asserts the face LABELS
      (`cpu:scalar` / `gpu:tiled`) before comparing, so a device leg that
      soft-fell-back to scalar cannot pass by comparing scalar with itself.
      *RED, real hardware:* the first device run selected `gpu:tiled` on
      gfx1151 and produced token streams identical but final logits at
      **inf-norm rel 1.634708e-1** — O(1), four orders past the budget, i.e.
      the dispatch was handing the device the wrong bytes.
      *Cause:* `KvCache::keys/vals` expose the cache's ROW-MAJOR
      `(seq_kv, n_kv_heads, head_dim)` stack, while `GpuContext::attention`
      documents `(B, n_kv_heads, seq_kv, head_dim)` (head-major); the two
      orders coincide only when `n_kv_heads == 1`, and the fixture has 2.
      *Fix:* `KvCache::keys_head_major` / `vals_head_major` — a host-side
      gather — used by `gqa_attention_tiled`; 3.2's "no gather was needed"
      claim is retracted and corrected in design.md.
      *GREEN, same device and command:* **inf-norm rel 0.000000e0, max|delta|
      0.000000e0** over a 14-key context, 8 identical tokens, RC=0.
      *Both gate states:* unset → clean skip notice, 12 passed, RC=0; set
      (`QUANTIBBLER_GPU_TESTS=1 LD_LIBRARY_PATH=/opt/data/tmp/rocm-sdk/lib
      HSA_OVERRIDE_GFX_VERSION=11.5.1 cargo test --locked -p
      quantibbler-server --features gpu --test cross_face_token_identity
      -- --nocapture --test-threads=1`) → the device run above, RC=0.
      *Honest weakness of the token clause on THIS fixture:* the fixture's
      greedy stream is constant (token `1` for all 8 steps), so "tokens
      identical" carries little detection power here; the LOGITS clause is
      what caught the O(1) dispatch defect and it is the clause with measured
      power (1.63e-1 → 0.0). Recorded rather than dressed up.
      *Named finding, sibling change, NOT this card's scope:*
      `gqa_attention_sage` hands `attention_sage` the same row-major cache
      stack although that kernel documents the same `(B, n_kv_heads, S, d)`
      contract — the INT8 face therefore carries the identical latent O(1)
      layout defect for `n_kv_heads > 1`; its tests call the kernel directly
      and never reach the engine call site. Routed as its own card.

## 4. Perf probe (OQ-1 — resolved only by measurement)

- [x] 4.1 Timed decode-attention benchmark, CPU scalar face vs GPU tiled
      face at (seq 512, seq 2048) × d=128 × GQA 8:1 on gfx1151; record
      the table in design.md crystallization; OQ-1 answered WITH numbers.
      Outcome gate: ≥ 1.5× at seq 2048 keeps the face opt-in-recommended;
      a miss keeps it opt-in-experimental and records why.
      — **MISS, WITH NUMBERS; face stays opt-in-experimental (as
      pre-registered).** Probe: throwaway `/opt/data/tmp/attn-decode-bench`
      (gemm-bench pattern, NOT repo code), `cargo build --release`, built and
      run INSIDE the sanctioned container (`quantibbler-ptq:rocm-official`;
      the sandbox here carries a 5 GB cgroup cap that SIGKILLs rustc
      mid-release-build). Command: `docker exec qk-attn-refbench bash -lc
      'LD_LIBRARY_PATH=…/_rocm_sdk_devel/lib ./target/release/attn-decode-bench'`
      → **RC=0**; device gfx1151 (Radeon 8060S). Pins (in the table header):
      batch=1, n_heads=8, n_kv_heads=1 (GQA 8:1), d=val_dim=128, f32,
      causal=true, warmup=3. cpu:scalar vs gpu:tiled, median (min) ms:
      `decode kv=1` **0.001 (0.001) vs 1.065 (0.581)** [diagnostic fixed-cost
      row, not a gate row]; `decode kv=512` **0.297 (0.278) vs 21.918 (9.284)**;
      `decode kv=2048` **0.997 (0.788) vs 123.744 (103.378)**;
      `causal seq=512` **67.008 (66.364) vs 32.815 (26.866)** = 2.04×;
      `causal seq=2048` **1096.130 (1096.130) vs 152.108 (150.797)** = 7.21×.
      Gate: seq-2048 decode is **124× SLOWER** (0.01×), seq-512 **74× slower**
      → MISS. Every row asserts both faces compute the same thing BEFORE
      printing (inf-norm rel 2.62e-06 at decode 2048, 2.09e-07 at causal 2048)
      — the probe's own first draft was caught by that assert (rel 1.48) for
      comparing two plane orders. `HSA_OVERRIDE_GFX_VERSION=11.5.1` re-run
      reproduces the verdict (110.492 ms vs 0.997 ms; 7.83×). Why, as far as
      the probe can say: ~1 ms fixed per-call floor plus a term linear in the
      cached context (42.8–60.4 µs/key vs 0.5–0.6 µs/key for the scalar face);
      the two named candidates (per-call GQA materialisation/upload/readback;
      the design's own `BlueprintStrategy::Inferred`-per-call residual) are NOT
      separated by this probe and the decomposition probe that would separate
      them is named as the next experiment. Tables + provenance in design.md,
      "Implementation crystallization (2026-10-25, tasks 4.1–4.2 — measured)".
- [x] 4.2 Reference benchmark (user directive 2026-10-20): install pip
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
      — **SHORTFALL ON BOTH AXES; gap recorded; the face is NOT "set" →
      opt-in-experimental until it closes.** Installed via pip
      (`python -m pip install --no-build-isolation`, `BUILD_TARGET=rocm`)
      TWO ways, both in the sanctioned container, same shape/dtype/mask/
      warmup/batch pins as 4.1: **(a)** the literal PyPI sdist
      `flash-attn 2.8.3.post2` with `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`
      (its in-tree Triton AMD path) — the primary reference; **(b)**
      `ROCm/flash-attention @2.8.4` with the same flags, Triton backend via
      `third_party/aiter` + `AITER_TRITON_ONLY=1` — a cross-check. Both
      produced IDENTICAL output checksums. Recorded deviation from the row's
      literal wording: the CK/`hipcc` backend was not used — it is
      fp16/bf16-only (our face is f32) and its documented build is hours of
      hipcc template instantiation; the Triton backend supports fp32 and RDNA,
      which is why it is the one that can compare like-for-like. Command:
      `docker exec qk-attn-refbench bash -lc 'FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
      AITER_TRITON_ONLY=1 python ref_bench.py'` → **RC=0** (torch
      2.14.0+rocm10.1.0, device AMD Radeon 8060S). Wall time, median (min) ms,
      (a)/(b): `decode kv=1` 0.665 (0.428)/0.237 (0.156); `decode kv=512`
      1.758 (0.887)/1.807 (0.872); `decode kv=2048` 4.950 (2.968)/4.943
      (1.942); `causal seq=512` 3.266 (2.869)/2.273 (1.663); `causal seq=2048`
      33.488 (32.730)/15.274 (13.717). Peak memory: reference device peak
      0.0/0.8/3.0/6.5/26.1 MiB (`torch.cuda.max_memory_allocated`, reset per
      row) with host-RSS deltas ≤12.2 MiB; ours 598 MiB whole-probe host peak
      (137 MiB worst single-row delta). Against the BETTER reference build
      ours is **12.5×/25.0× slower at the decode shapes and 14.4×/10.0×
      slower at the causal shapes**, and worse on the only comparable memory
      axis → shortfall on wall time AND memory → **not set**. The two memory
      instruments differ (torch device allocator vs `/proc/self/status`), which
      the table states instead of papering over. Ledger note that matters for
      the next reader: the CPU scalar face — what the server runs today — BEATS
      the reference at both decode shapes (5.9×/5.0× faster) and loses at both
      causal shapes (29.5×/71.8×). Both tables in design.md.

## 5. Landing

- [ ] 5.1 Full gate suite: `uv run pytest`, `cargo test` (kernels + server,
      both feature states), `cargo clippy --all-targets -D warnings`,
      `cargo fmt --all -- --check`, `uv run maturin build`, and
      `openspec validate qk-flash-attention --strict`; scoped commit
      (spec + code + crystallization together).
      — **Suite RED on the live shared checkout, for two sibling-owned reasons;
      box deliberately left UNTICKED** (kanban `t_5d79c4f8`, measured
      2026-10-25 11:03–11:22Z, HEAD `8c226fa`, tree dirty with 90 paths).
      *Interpreter + device, declared once for every Python number below:*
      `/opt/venv`, `torch 2.14.1+cpu`, `hip=None`, `cuda.is_available()=False`,
      pytest 9.1.1. (A bare `uv run pytest` re-syncs this shared venv and drops
      the extras — every invocation below is `--no-sync` with the full extra
      set.)
      *Python — the literal single invocation* `uv run --no-sync --extra torch
      --extra kernels --extra hf pytest -q -p no:cacheprovider --tb=no -rf` →
      **8 failed, 1007 passed, 3 skipped, 4 deselected in 143.05s, RC=1**
      (`/opt/data/tmp/t5d7-gates/uvrun_single.log`). Independently reproduced
      by the chunked form (5 disjoint processes, because this sandbox's cgroup
      is `memory.max=5368709120` and a whole-tree run is SIGKILLable): 754/4 +
      77/4 + 57/0 + 59/0 + 60/0 = **1007 passed, 8 failed, 3 skipped** over
      **1018 collected** (`--collect-only` agrees: `1018/1022 tests collected
      (4 deselected)`; the 4 are the `integration` marker).
      *Rust — every leg RC=0:* `cd crates/quantibbler-kernels && cargo test
      --locked` → **24 passed, 0 failed**; `--features gpu` → **30 passed, 0
      failed** (the 2 `gpu_attention_parity` cases skip cleanly with the gate
      unset); `cargo clippy --all-targets -- -D warnings` → RC=0 default and
      RC=0 `--features gpu`; `cargo test --locked -p quantibbler-server` →
      **179 passed, 0 failed, 10 ignored, 15 targets** (incl.
      `attention_dispatch` 14, `cross_face_token_identity` 12,
      `binary_purity` 12, `cubek_purity` 2); `--features gpu` → **180 passed,
      0 failed, 10 ignored**; server `clippy --all-targets -- -D warnings` →
      RC=0 in both feature states; `cargo fmt --all -- --check` → RC=0.
      `uv run --no-sync maturin build --manifest-path
      crates/quantibbler-kernels/Cargo.toml` → RC=0, wheel
      `target/wheels/quantibbler_kernels-0.1.0-cp311-abi3-manylinux_2_35_x86_64.whl`.
      `npx --yes @fission-ai/openspec validate qk-flash-attention --strict` →
      `Change 'qk-flash-attention' is valid`, RC=0 (one change id per
      invocation).
      *The 8 reds are two classes, both outside this change's blast radius and
      both already owned by blocked sibling cards:* (1) **4 reds in
      `tests/ptq/test_progress.py`** from the uncommitted
      `src/quantibbler/ptq/progress.py` ETA unit-model edit — card
      **t_993f1df5** already records this same `4 failed, 21 passed` in its own
      blocker comment (its four reds assert the superseded `(total-done)*rate`
      formula). (2) **4 reds in `tests/test_qk_report.py`** from the
      uncommitted `tools/quantize_hf_model.py` refactor of the KD scratch
      default into a named `_scratch_root()` — card **t_14977580** owns that
      helper and the per-run checkpoint dir under it. Neither failing
      assertion names anything this change added, and neither file is in its
      path set. *Causality, measured rather than inferred from co-location:*
      the checker `_scratch_dir_findings` — committed at HEAD, unmodified by
      anyone — returns `[]` for HEAD's driver source and exactly those 3
      findings (`no platform-scratch fallback in the default: _scratch_root()`,
      `the sandbox path is not existence-probed: _scratch_root()`, `the sandbox
      scratch preference is gone: _scratch_root()`) for the working tree's.
      *Baseline A/B:* a detached worktree at `8c226fa` reads **116 passed**
      (91 + 25) on those two files, so neither red exists at HEAD.
      *The change's own surface is green in both feature states*:
      `tests/test_attention_tiled.py` → **209 passed**;
      `tests/test_kernels_parity.py` + `tests/test_kernels_dispatch.py` → **27
      passed**; server `attention_dispatch` 14 and `cross_face_token_identity`
      12 green in the default build *and* under `--features gpu`.
      *Named residual (why no tree here yields the row's "all gates green"):*
      HEAD itself is red — `64 failed, 603 passed, 3 deselected` (670
      collected) on the same interpreter — that is main's post-wave-push red,
      owned by `t_07c6b629` / `t_fa083bb3`; the shared tree is *greener* (8 vs
      64) because sibling fixes for that class are sitting uncommitted in it.
