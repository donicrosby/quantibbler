# qk-sage-attention — Tasks

Dependency: qk-flash-attention lands first (tiled scaffold + parity
harness are the foundation). Tests precede implementation per capability.

## 1. Quantized reference (Python face)

- [x] 1.1 RED: `tests/test_attention_sage.py` for scenario IDs
      `gamma-shift-invariance` and `quantized-budget` (fixtures + a
      captured-activation fixture hook); observe RED, record it.
      — **RED observed** (module absent), `uv run --no-sync pytest
      tests/test_attention_sage.py` → collection error, **rc=2**, 18.18 s:
      `ModuleNotFoundError: No module named 'quantibbler.kernels.attention_sage'`.
      The battery that then came green: *lossless-grid* fixtures where γ(K) is
      bit-exactly representable (per K-block: zero column sums plus a
      per-block shared bias, every row peaking at exactly ±127, so δ_K =
      127/127 = 1.0) across seq 8/10/64 × d 4/128 × block 1/3/4/8/64/128 ×
      causal × additive-mask; general fixtures over **34 (shape, block) pairs**
      × causal — every block size on every shape except `block_size=1` at
      seq 64, where a case is 2080 tiles of per-block assertions under
      co-tenant load for a branch the short shapes and
      `test_attention_sage_contract_edges` pin symbol-for-symbol; the budget
      metrics; the strictly-positive-scale clause; the [SA] §4.3 granularity
      axes; and the captured-activation hook, which **names** an absence
      (`absent: no captured activations at …`) instead of `pytest.skip` — the
      CI legs pin their skip counts, so a silent skip would have shortened the
      battery invisibly.
- [x] 1.2 GREEN: `src/quantibbler/kernels/attention_sage.py` ([SA] §4
      scheme; γ per K-block; exposed quantizers); suite green.
      — **GREEN:** `uv run --no-sync pytest tests/test_attention_sage.py` →
      **291 passed, rc=0** (30.6 s / 37.4 s across runs). *Measured margins* on
      the 68-case budget battery vs `attention_tiled`: worst **Cosine Sim
      0.99988958** (gate ≥ 0.999) and worst **Relative L1 0.01432540**
      (gate ≤ 0.02), both at (seq 64, d 128, 8/2 heads, block 64/128,
      non-causal), RMSE 4.76e-3; min per-token Q scale 8.7e-4, min per-token K
      scale 1.7e-3 (strictly positive, no zero-range token). *γ exactness*: on
      the lossless-grid fixtures **|ψ⁻¹(ψ(γ(K))) − γ(K)| = 0.0e+00** and
      **|dequantized-then-unsmoothed K − K| = 0.0e+00** (bit-exact); on general
      fixtures the unsmoothed path's deviation from the exact scores equals the
      dequantized operands' deviation with residual **3.55e-15** (1.8e-15
      relative, gate 1e-12) — γ and the shift contribute nothing. *Gate
      non-vacuity*: a 6-mutation battery against the real module source
      (shift-back dropped; γ from the whole-cache mean; γ = identity; δ_K
      broadcast along the wrong axis; V per-token instead of per-channel; P̃
      static scale off 1/127) is **caught 6/6**, source restored
      byte-identically (sha256 `9954de43…` before and after). *Gates*:
      `pre-commit run --files <both files>` **rc=0**;
      `uv build --wheel` ships `quantibbler/kernels/attention_sage.py`; **lean
      tier** (`/opt/data/tmp/ci-lean-venv`: numpy + pytest + pyyaml, no torch,
      the CI lean leg's interpreter) → **718 passed, 16 skipped, rc=0** in
      23.3 s with `tools/ci/pytest_summary.py --expect-skipped 16` reporting
      `skip discipline holds` (the pinned skip count is unchanged by the new
      file). *Suite gates* (interpreter this box actually has: python 3.12.14,
      numpy 2.5.4, **torch 2.14.1+cpu**, quantibbler-kernels 0.1.0 — the CPU
      wheel, so no accelerated path is reachable): full suite `uv run --no-sync
      pytest -q -rs` → **14 failed, 968 passed, 3 skipped, 3 deselected** in
      344.2 s; the ticket's trio (`tests/ptq tests/test_ci_referee.py
      tests/test_qk_report.py` — the ticket's `tests/ci/` and `tests/qk_report/`
      paths do not exist, these are the two files it names) → **14 failed, 400
      passed, 3 skipped**. **All 14 failures carry ONE signature** —
      `RuntimeError: solver backend 'torch' requires an accelerator: implicit
      resolution refuses to run torch on CPU (…)` — the documented fail-closed
      refusal of the CPU-wheel interpreter, in files this change neither edits
      nor imports (`test_lb_admm`, `test_relative_lr`, `test_dual_stream`,
      `test_e2e_gate`; `grep -rn attention_sage src/` finds no import edge, so
      the ptq tier is bit-identical with and without this slice). The 3 skips
      are `no accelerator: (…) needs one` in another
      lane's untracked `tests/ptq` files. The two runs this change CAN move are
      the lean tier and its own file — 718/16 and **291 passed, rc=0** — and
      that file's healthy time is **3.5 s**, not the 30-130 s it took while
      co-tenant lanes saturated the 32-core box (load 45-60, 14-18 concurrent
      pytest processes): timings here are a property of the window, the counts
      are a property of the tree. Design-silent decisions (the reference
      unsmooths the scores; the zero-range δ = 1.0 convention a ragged
      one-token tail needs; P̃ quantized after the running-max rescale with ℓ
      full precision; V per channel *within the tile*) are recorded in
      design.md, "Implementation crystallization (2026-10-25, tasks 1.1–1.2)",
      together with the captured-activation fixture gap.

## 2. INT8 kernel path (kernels crate, `gpu` feature)

- [x] 2.1 Probe (cheap, pre-code): compile-run a CubeCL INT8 dot form on
      gfx1151 (i8 tensor vs u32-packed fallback); record the working idiom
      as crystallization. Command: throwaway bench in /opt/data/tmp
      (gemm-bench pattern), NOT repo code.
      — **Run green 2026-10-25; the working idiom is recorded in design.md,
      "Implementation crystallization (2026-10-25, task 2.1 — feasibility
      probe)".** Throwaway crate `/opt/data/tmp/sage-int8-probe-11` (NOT repo
      code; gemm-bench pattern; source + all six run logs archived at
      `/opt/data/tmp/sage-int8-probe-11/evidence-2.1/`, `probe-main.rs` sha256 `f84d722b…`, binary `2c9b1fab…`),
      `cubecl` pinned `= 0.11.0` — the repo's own lock version (a caret
      requirement resolved 0.11.3 first, which is NOT the version under probe).
      `cargo build --release --offline` **BUILD_RC=0**; run
      `LD_LIBRARY_PATH=/opt/data/tmp/rocm-sdk/lib HSA_OVERRIDE_GFX_VERSION=11.5.1
      /opt/data/tmp/gemm-bench/target/release/sage-int8-probe-11` →
      **RUN_RC=0**, `device ok` on the HIP device (gfx1151 via the staged ROCm
      userspace). **All four dot forms compile, run and are exact against a host
      i64 reference** at (m=8,k=128,n=8), i32 device accumulator: `i8 max|dev| =
      0`, `i32-packed manual = 0`, `u32-packed manual = 0`, `dp4a = 0` — i.e.
      the row's literal pair (i8 tensor vs u32-packed fallback) plus the
      `dp4a()` op, each verified rather than assumed. Recorded in the same
      entry: the three idiom traps that cost probe iterations (a `#[cube]`
      helper is not callable as `helper::expand(a,b)` from a launch body; the
      packed operand needs one word per *thread*, not per four lanes, or the
      device reads out of bounds; the host packing buffer must be sized by the
      packed shape), the source-read finding that `Dp4aOp` lowers to
      `expand_dp4a_polyfill` on **both** HIP backends (so no byte-dot
      instruction exists in this stack — corroborating task 4.1's verdict and
      settling the substrate assumption's fallback branch), and the
      register-resident ALU isolation (`dp4a` only **1.2×** the hand-written
      4-lane polyfill, 1.64× `f32` per lane-MAC). Shape numbers from this probe
      are labelled indicative-only (contended box); task 4.1's table stays the
      rate of record. **Go/no-go: GO on feasibility for 2.2/2.3, NO-GO on the
      "byte dot is the win" premise — no ≥2× speed claim from this stack.**
- [x] 2.2 RED: `tests/gpu_attention_sage.rs` hardware-gated; observe RED.
      — **RED observed 2026-10-25, literally.** `cd crates/quantibbler-kernels &&
      cargo test --offline --features gpu --test gpu_attention_sage` →
      **RED_RC=101** in 2m05s: `error[E0599]: no method named `attention_sage`
      found for struct `GpuContext` in the current scope` (with the compiler's
      own hint `there is a method `attention` with a similar name`), then
      `error: could not compile `quantibbler-kernels` (test "gpu_attention_sage")
      due to 9 previous errors`. The other 8 errors of that same pass are
      `error: unknown format trait `f`` — rustc 1.98.1 on this box rejects
      `{x:.6f}` outright (verified standalone), so the file formats with `e`.
      The test file lands with this row (hardware-gated `#![cfg(feature =
      "gpu")]` + `QUANTIBBLER_GPU_TESTS=1`, mirroring
      `gpu_attention_parity.rs`), and so does its fixture battery:
      `tools/gen_attention_sage_fixtures.py` →
      `tests/fixtures/attention_sage.json` (10 cases, 516 143 B, sha256
      `7d2755019315af36…`), each record carrying BOTH the quantized reference's
      output and the exact face's output plus the reference's own [SA] §4.3
      metrics, so the envelope clause compares the device to measured numbers
      rather than to a hand-written constant.
- [x] 2.3 GREEN: INT8 tile path in gpu_attention.rs; both gate states
      green on gfx1151; clippy/fmt green.
      — **GREEN on the real device 2026-10-25** (gfx1151; `/dev/kfd` +
      `/dev/dri/renderD128` present in this sandbox; staged ROCm SDK). *Gate
      SET*: `QUANTIBBLER_GPU_TESTS=1 LD_LIBRARY_PATH=/opt/data/tmp/rocm-sdk/lib
      HSA_OVERRIDE_GFX_VERSION=11.5.1 cargo test --offline --features gpu
      --test gpu_attention_sage -- --nocapture` → **RUN_RC=0**:
      `sage INT8 attention: 10/10 fixture cases ran on the device; worst parity
      band rel 1.201e-6 (budget 1e-4); worst Cosine Sim 9.99968778e-1; worst
      Relative L1 7.21831548e-3`, `1 passed; 0 failed`, 1.21 s. *Gate UNSET*:
      the same binary prints `skip: set QUANTIBBLER_GPU_TESTS=1 (plus the staged
      ROCm SDK env) to run the INT8 attention parity battery against real
      hardware` → `1 passed; 0 failed`, 0.00 s, RC=0 — a clean skip carrying a
      notice, not a silent zero. *Envelope*: on the very same 10 cases the
      quantized reference's own worst Cosine Sim is 0.9999687782 and its worst
      Relative L1 7.2183043e-3 (case index 7 for both), and the device lands at
      0.999968778 / 7.21831548e-3 — the device's extra f32 error does not move
      the quality metrics at all, so it is inside the reference's measured
      envelope rather than merely inside the spec gate; parity is ~80× inside
      the 1e-4 budget. *Gates*: `cargo test` (default features) **24 passed, 0
      failed, RC=0**; `cargo clippy --all-targets -- -D warnings` **RC=0**;
      `cargo clippy --all-targets --features gpu -- -D warnings` **RC=0** (the
      default-feature run cannot see the new kernel at all — it is behind
      `gpu`); `cargo fmt --all -- --check` **RC=0**.
      `uv sync --all-groups --extra torch --extra kernels --extra hf
      --reinstall-package quantibbler-kernels` **RC=0** (6.5 s): the extension
      re-imports and computes (`pack_bits([[1,-1,1]]) → [[5]]`), and the
      rebuilt default-feature `.so` contains **0** occurrences of
      `sage_int8_tile_kernel` (the gpu-feature rlib carries 1687) — the new path
      is behind the feature and the Python face is untouched, which is also why
      `uv`'s rebuild exercises no changed behaviour: the changed behaviour IS
      the hardware-gated device path, re-run above after the last source edit.
      Design-silent decisions (the finite masking sentinel; no additive-bias
      `mask`; `seq_q == seq_kv` required with a typed refusal; plain f32
      statistics kept because the requirement names them) and the cubecl 0.11
      idioms the build cost are in design.md, "Implementation crystallization
      (2026-10-25, tasks 2.2–2.3)".

## 3. Server face + quality gate

- [x] 3.1 RED: `tests/sage_dispatch.rs` — `gate-pass-enables`,
      `gate-fail-refuses` (no-hardware runnable),
      `soft-fallback-device-only`; observe RED.
      — **RED observed 2026-10-25, literally, before any server code changed.**
      `cd crates/quantibbler-server && cargo test --offline --test sage_dispatch`
      → **RED_RC=101**: `error[E0425]: cannot find function
      `attention_gate_check` in module `model`` (2 sites — the pass and
      regenerates arms), `error: could not compile `quantibbler-server` (test
      "sage_dispatch") due to 2 previous errors`. The file lands with this row:
      the three scenario arms the spec delta writes plus the
      record-regeneration pin. (One earlier pass of the same file also carried
      4 × `E0277: `LoadedModel` doesn't implement `Debug`` from the test's own
      `expect_err` use; `LoadedModel` carries no `Debug` by design, so the file
      was switched to an explicit `load_err` helper and RED re-taken — the errors
      above are the missing API alone.)
- [x] 3.2 GREEN: `gpu:sage` face, startup gate on the shipped record,
      refusal typing, fallback chain; default-feature suite green.
      — **GREEN 2026-10-25, both feature states, real output.** *Face*:
      `AttentionFace::GpuSage(GpuContext)` + `gqa_attention_sage` +
      `AttentionFaceDecision::sage_selected()` + `SAGE_BLOCK_SIZE` in
      `crates/quantibbler-server/src/engine/mod.rs`; opt-in
      `QUANTIBBLER_ATTN=sage` (`=1` stays the exact tiled face, unset the scalar
      root). *Gate*: `crates/quantibbler-server/src/model/mod.rs` —
      `AttentionGateError` (5 typed variants), `attention_gate_check`,
      `sage_face_from_env`, called from `model::load` **before the weights are
      built and before any device is touched**. *Refusal classes observed*
      (`scenario_gate_fail_refuses_the_face`, one arm each): missing record;
      non-JSON; a per-call metric below `cosine ≥ 0.999`; a `worst` that
      contradicts the record's own calls (`worst relative_l1 is 0.001, but the
      calls aggregate to …`); a record declaring a looser clause set than the
      gate enforces (`declares cosine_min=0.5, the gate requires 0.999`); a
      declared `ppl_delta` above its bound; another model's `model_id`; and at the
      PROCESS boundary the binary exits non-zero with `attention gate` in stderr
      (`--model-dir` = a copy of the fixture dir minus the record) — all
      no-hardware, in both builds, because the gate runs before device init.
      *Soft fallback* (`scenario_soft_fallback_device_only_keeps_the_exact_face`,
      run WITH the staged SDK present): the impossible test device index makes
      init fail after the gate passes →
      `the gate admitted the gpu:sage face but the INT8 device face is
      unavailable; staying on the scalar CPU face. (… device probe failed at
      index 4611686018427387903: NoSuchDevice { runtime: Hip, available: 1 })`
      and `/health` reports `cpu:scalar`, frozen across two reads — quality is
      never the fallback trigger (an out-of-bounds record ABORTS; it does not
      fall back). *Gate-pass arm, hardware*
      (`QUANTIBBLER_GPU_TESTS=1 LD_LIBRARY_PATH=/opt/data/tmp/rocm-sdk/lib
      HSA_OVERRIDE_GFX_VERSION=11.5.1 cargo test --offline --features gpu --test
      sage_dispatch -- --nocapture --test-threads=1`) → **16 passed; 0 failed**,
      `face=gpu:sage` on both `/health` reads (gfx1151 via the staged ROCm
      userspace). The same gpu build with the env UNSET → **16 passed; 0 failed**
      with the clean skip notice `skip: set QUANTIBBLER_GPU_TESTS=1 (plus the
      staged ROCm SDK env) to run the gpu:sage face-selection arm`. *Suites*:
      `cargo test --offline` (server, default features) **RC=0 — 155 passed, 0
      failed, 10 ignored** across 13 targets (the ignored are the pre-existing
      hardware/load arms); `cargo clippy --all-targets -- -D warnings` **RC=0**
      and `cargo clippy --all-targets --features gpu -- -D warnings` **RC=0**
      (the default-feature build cannot see `gqa_attention_sage` at all — it is
      behind `gpu`); `cargo fmt --all -- --check` **RC=0**. One flake, attributed
      and not the diff: the full gpu-feature sweep stopped once at
      `load.rs::load_failure_never_ready` (`variant eos_out_of_range: diagnostic
      must name eos_token_id; got: ` — the empty-diagnostic shape the runbook
      names, 200 × 25 ms under load 46); it re-runs **1 passed; 0 failed** in
      isolation, and the default-feature suite was green twice. Design-silent
      decisions (the record's location/schema, the PPL clause's presence rule,
      the padded-query mapping forced by the kernel's `seq_q == seq_kv` refusal,
      `SAGE_BLOCK_SIZE`) are recorded in design.md, "Implementation
      crystallization (2026-10-25, tasks 3.1–3.3)" — the PPL rule is offered
      there as a ruling request, not presented as settled.
- [x] 3.3 Gate tool: committed script regenerating the record from a model
      dir + activation fixture ([SA] §4.3 metrics verbatim);
      `gate-record-regenerates` scenario green.
      — **Committed and exercised 2026-10-25.**
      `tools/gen_attention_gate_record.py` recomputes, from the model directory
      and the activation fixture, exactly what the server gate reads: the [SA]
      §4.3 metrics of `attention_sage` vs `attention_tiled` per case, their
      worst, and the clause set — cross-checking every case against the metrics
      `gen_attention_sage_fixtures.py` stored (two independent computations;
      disagreement aborts). Command:
      `uv run --no-sync python tools/gen_attention_gate_record.py --model-dir
      tests/fixtures/server-model` → `wrote
      tests/fixtures/server-model/attention_gate.json (2206 bytes, 10 calls)`,
      `worst cosine 0.999968778 [PASS >= 0.999]`, `worst relative L1 0.007218304
      [PASS <= 0.02]` (the same two numbers task 2.3's device run reports:
      0.9999687782 / 7.2183043e-3), `eval-set PPL delta: NOT MEASURED`. *It
      regenerates*: a second run into a private copy produced a metric view equal
      to the shipped record's on every compared key (`model_id`,
      `activations_source`, `clauses`, `calls`, `worst`, `ppl` — only the audit
      `generated_at`/`code_commit` move). *It detects drift*: `--check` against
      the shipped record → `OK …: 10 calls reproduce the shipped metrics` (RC=0);
      `--check` against a copy of the fixture with one query element moved → RC=1
      `case 2 cosine recomputed as 0.9999855861400737 but the fixture stores
      0.9999854257394872`; `--check` against a record whose declared PPL clause a
      fresh run does not reproduce → RC=1 naming the mismatched key. *The PPL
      clause's other half is exercised too*: `--ppl-eval` with a within-bounds
      delta lands in the record (`{'eval_set': 'unit-test eval', 'delta': 0.011,
      'bound': 0.05}`, RC=0) and with an out-of-bounds delta the tool exits 1
      (`the recorded PPL delta exceeds its bound: the server will refuse this
      face`). *Scenario*: `scenario_gate_record_regenerates_matches_the_shipped_fixture`
      green — present, bound to the served `model_id`, within the clauses it
      declares, naming a shipped fixture whose case count matches the record's.
      **Named gap and hazard:** the fixture the record is generated from is the
      committed device-parity battery, not a captured forward pass (the record
      says so: `activations_kind: "fixture-battery"`; task 4.2's capture is the
      real-model run held for an @user GO), the `ppl` clause is therefore `null`
      — a spec-reading decision recorded in design.md and flagged there for a
      ruling — and `tools/gen_server_fixtures.py` rebuilds
      `tests/fixtures/server-model` with `shutil.rmtree`, so a fixture
      regeneration drops the record and the regenerates test fails loudly until
      the one-line command above is re-run.
- [x] 3.4 Hardware parity for the INT8 face: scenario
      `quantized-face-token-identity` on gfx1151 (fixture pipeline greedy
      generation, exact scalar face vs `gpu:sage`, tokens identical, final
      logits inside the [SA] §4.3 gate — Cosine Sim ≥ 0.999, Relative L1 ≤
      0.02; NOT the exact face's 1e-5, which this lossy face cannot meet).
      — **RED first on the device, then GREEN.** New
      `crates/quantibbler-server/tests/cross_face_sage_identity.rs` drives the
      FIXTURE pipeline through the exact scalar face and the quantized face —
      two constructions under the two env states, the
      `cross_face_token_identity.rs` shape — and asserts the face LABELS
      (`cpu:scalar` / `gpu:sage`) before comparing, so a device leg that
      soft-fell-back to scalar cannot pass by comparing scalar with itself.
      *Why the scenario exists:* nothing else walks the engine call site
      `gqa_attention_sage` — the kernels battery calls the INT8 kernel
      DIRECTLY with fixtures already in its documented layout, and 3.1–3.3
      read labels, not bytes; the sibling tiled face's identical defect was
      found the same way (task 3.3) and this is its mirror.
      *RED, real hardware:* with the shipped call site's raw row-major stack
      injected (`&cache.k[block][..]` / `&cache.v[block][..]`) the run read
      **cosine 9.904153659e-1** (gate ≥ 0.999) and **Relative L1
      1.304462818e-1** (gate ≤ 0.02) — 8 identical tokens, context 14 keys,
      RC=101.
      *Cause:* `gqa_attention_sage` handed `attention_sage` the cache's
      ROW-MAJOR `(seq_kv, n_kv_heads, head_dim)` stack while the kernel
      contract is head-major `(B, n_kv_heads, S, d)` (the kernel reads
      `kv_tok = (b * n_kv + kvh) * seq`, and `attention_sage` documents the
      same layout as `GpuContext::attention`); the two orders coincide only at
      `n_kv_heads == 1`, and the fixture has 2.
      *Fix:* `KvCache::keys_head_major` / `vals_head_major` — the host-side
      gather the tiled face already uses — at the sage call site, exactly as
      task 3.3 fixed `gqa_attention_tiled`. The now-unread raw accessors
      `KvCache::keys` / `vals` were then cut (grep-confirmed zero readers) and
      their layout warning folded into `keys_head_major`'s doc; `KvCache`
      storage order is untouched.
      *GREEN, same device and command:* **cosine 9.999999884e-1, Relative L1
      1.698875021e-4** over a 14-key context, 8 identical tokens, 12 passed,
      RC=0. Both metrics move by orders and both cross the gate, so this
      scenario has measured detection power (unlike the sibling's token
      clause, whose fixture stream is constant).
      *Both gate states:* unset → clean skip notice, 12 passed, RC=0; set
      (`QUANTIBBLER_GPU_TESTS=1 LD_LIBRARY_PATH=/opt/data/tmp/rocm-sdk/lib
      HSA_OVERRIDE_GFX_VERSION=11.5.1 cargo test --offline --locked -p
      quantibbler-server --features gpu --test cross_face_sage_identity
      -- --nocapture`) → the device run above, RC=0. Default features (no
      `gpu`) → the opt-in inertness arm green, 12 passed, RC=0.
      *Spec-first:* the delta carried NO scenario for the engine call site, so
      this row ADDED one to `specs/qk-server` ("Serving parity across the
      quantized attention face" / "Quantized-face token identity on the
      fixture pipeline") before the code changed — the requirement is in the
      delta, not minted by this row. The kebab id is this row's, per the
      sibling's convention; the delta's scenarios carry headings only.

## 4. Probes resolving the Open Questions (measurement, never argument)

- [x] 4.1 INT8 vs fp32 dot microbench at attention shapes on gfx1151 —
      resolves OQ-1 WITH numbers into design.md crystallization.
      — **Measured 2026-10-25; the ≥ 2× assumption is FALSIFIED.** Throwaway
      instrument `/opt/data/tmp/sage-int8-microbench` (NOT repo code,
      gemm-bench pattern; `src/main.rs` sha256 `fe93fccc…`, `Cargo.toml`
      `c7c89293…`, `Cargo.lock` `7aa73978…`), built+run in
      `quantibbler-ptq:rocm-official` (`--device /dev/kfd --device /dev/dri`,
      `HSA_OVERRIDE_GFX_VERSION=11.5.1`,
      `LD_LIBRARY_PATH=/opt/rocm/lib:/opt/data/tmp/rocm-sdk/lib`, gfx1151):
      `cargo build --release --offline` **BUILD_RC=0** (55.4 s) and the
      binary **RUN_RC=0**, twice. Four forms as one-thread-per-C-element naive
      GEMM (dot form the only variable), best-of-3 after warmup, timed through
      the blocking read-back. All three INT8 forms **bit-exact** vs a host i64
      reference at (8,128,8) — `i8`, packed-manual and `dp4a` `max|dev| = 0`.
      Best INT8 vs the f32 dot, run 1 / run 2 GOP/s: QKᵀ (2048,128,2048)
      **1.64× / 1.48×** (235.4/449.3 vs 143.2/302.6); QKᵀ (4096,128,4096)
      **1.06× / 1.11×** (126.5/142.0 vs 119.9/127.9); P̃Vᵀ (2048,2048,128)
      1.25× / 2.08×. The two runs are reported because run 2's `dp4a` moved
      more than the effect under test; run 2's load average was 47.69 on 32
      cores (co-tenant window — counts are the tree's, timings are the
      window's). Table, pins, instrument provenance (three source fixes against
      the byte-identical task-2.1 copy) and the `dp4a`-is-not-the-fast-form
      observation: design.md, "Implementation crystallization (2026-10-25,
      task 4.1 — measured; task 4.2 measured in its own entry below)".
- [x] 4.2 Quality gate run on captured REAL activations + quantized-PPL
      delta on the eval set — resolves OQ-2; face ships as
      opt-in-recommended only if both gates pass, else recorded as
      opt-in-experimental with the failing numbers.
      *(Row TEXT amended 2026-10-25 to name what was actually measured, in the
      same landing as the tick: the "r137 activations" this row originally asked
      for cannot be produced — every `qwen3-0.6B-r137*` qpk1 artifact is
      `failed` (v0 PPL 1 156 577, v2 943.49, v3/v4 NaN, v7 5 517.8, v8
      463 742), so their activations come from a dead model and a quality gate
      measured on them measures nothing. The capture is therefore the healthy
      base Qwen3-0.6B those artifacts quantize — the model whose outlier
      statistics the gate exists to be robust to — and a capture from the served
      qpk1 artifact would need a capture surface in the Rust engine (a
      server-contract change, not a probe). The substitution and its reasoning
      are recorded as judgment call (1) of the design.md task-4.2 entry.)*
      — **MEASURED 2026-10-25 (real-model run, @user GO given); OQ-2 resolved;
      the face is recorded `opt-in-experimental`.** Sections 2/3 having landed,
      the earlier BLOCKED reading is retracted: the gate tool (3.3) exists, the
      INT8 path (2.2/2.3) is landed, and the GO was given. Throwaway instrument
      `/opt/data/tmp/sage-real-activations/capture_and_ppl.py` (NOT repo code —
      the section-4 probe pattern 4.1 set) on `/opt/venv/bin/python3`, **torch
      2.14.1+cpu** (`cuda.is_available()=False`; CPU wheel, no accelerated path
      reachable), numpy 2.5.4, transformers 5.20.1, bfloat16. The two faces run
      *inside* `Qwen3Attention.forward` (registered attention implementations),
      so they receive exactly the tensors they document — post-`q_norm`/`k_norm`,
      post-RoPE, GQA un-repeated, `(B, n_heads, S, d)` / `(B, n_kv_heads, S, d)`
      — i.e. REAL activations. Model: local `models--Qwen--Qwen3-0.6B` snapshot
      (bf16, 1 503 300 328 B; 28 layers, 16/8 heads, d 128); eval:
      `wikitext-2-raw-v1` **test**. Self-validation, measured: exact face vs HF
      eager on the same captured q/k/v worst **8.52e-7**, and HF `sdpa` vs exact
      PPL within **0.098%** (the harness noise floor). Two runs, both RC=0:
      *run 1* (4 layers × 2 windows, 4 096 eval tokens) worst Cosine Sim
      **0.994972772**, worst Relative L1 **0.122862964**, ppl_exact 36.970209 vs
      ppl_sage 36.511942 (delta **−0.458267**); *run 2* (28 layers × 3 windows =
      **84 cases**, 8 192 tokens) worst Cosine Sim **0.987266027** (L5 w2) vs the
      **≥ 0.999** gate and **0/84** cases meeting Relative L1 ≤ 0.02 (worst
      **0.148357076**, 7.4× the ≤ 0.02 gate; 14/84 meet cosine alone),
      ppl_exact 31.328512 vs ppl_sage 31.059789 (delta **−0.268724**, ratio
      **0.991422396**). **Both gates:** the [SA] §4.3 metric gate **FAILS**
      (every clause, both runs); the PPL gate **passes**, in the favouring
      direction. Per this row's own rule the face is therefore **recorded as
      opt-in-experimental with the failing numbers**, and the finding that
      outlives the status is that the shipped *fixture-battery* record
      (0.999968778 / 7.2183e-3) over-states the face by orders of magnitude
      against the 84 real-activation cases — [SA] Table 3's warning landing on
      us. Design-silent decisions (base-model capture rather than a dead r137
      artifact; the PPL clause value with bound 0.05 from the change's own
      shipped-test convention; and why no committed record is minted) are
      recorded in design.md, "Implementation crystallization (2026-10-25, task
      4.2 — the real-activation quality gate, measured; OQ-2 resolved AGAINST
      the face)", and OQ-2 is marked RESOLVED there.

- 4.3 Compact real-activation capture path for the gate tool: deferred to
     coder card `t_0ed63061` (planner ruling t_fccd1ced, Decision 4).
     The gate tool consumes JSON today; a real-shape case is ~82 KB/unit
     of seq, past the repo's large-file gate. Card delivers: npz/f32-binary
     path in the gate tool + committed capture script producing
     `tests/fixtures/attention_sage_activations.npz`.

- Planner rulings (2026-10-25, card t_fccd1ced): all four decisions landed
  in design.md "Planner rulings" and the spec delta `specs/qk-server`:
  (1) evidence-class distinction (fixture-battery → experimental only,
  real-activation → enabling class); (2) face status ratified as
  `opt-in-experimental`; (3) PPL bound 0.05 ratified as policy threshold;
  (4) compact capture path deferred to `t_0ed63061`.

## 5. Landing

- [x] 5.1 Full gate suite (pytest; cargo test kernels+server in both
      feature states; clippy -D warnings; fmt --check; maturin build;
      `openspec validate qk-sage-attention --strict`) and scoped commit
      (spec + code + crystallization together).
      — **GATE SUITE GREEN 2026-10-25, both feature states; box left `[ ]`
      for the commit half.** Card `t_711ae2a4`. Reproduction revision **HEAD
      `cfa3311`** (live shared checkout, dirty with this change's uncommitted
      capture-path slice and two peer paths); interpreter
      `/opt/venv/bin/python` = CPython 3.12.14, **torch 2.14.1+cpu**,
      `torch.cuda.is_available()=False`, numpy 2.5.4, pytest 9.1.1 — the only
      interpreter this box has, and it installs the CPU wheel, so no accelerated
      path is reachable; every number below is that (interpreter × device) pair.
      Logs under `/opt/data/tmp/t_711ae2a4/rerun/`, each gate's RC in
      `gates.log`. *Gates*: `cargo fmt --all -- --check` **RC=0**; kernels
      `cargo test --offline --locked` **RC=0 — 24 passed, 0 failed** (parity.rs;
      7 other targets 0-test); kernels `cargo test --offline --locked
      --features gpu` (`LD_LIBRARY_PATH=/opt/data/tmp/rocm-sdk/lib
      HSA_OVERRIDE_GFX_VERSION=11.5.1`) **RC=0 — 30 passed, 0 failed**
      (parity 24, gpu_parity 3, gpu_attention_parity 2, gpu_attention_sage 1);
      kernels clippy `--all-targets -- -D warnings` **RC=0** in BOTH states;
      server `cargo test --offline --locked` **RC=0 — 179 passed, 0 failed,
      10 ignored**, 15 targets (incl. `sage_dispatch` 15,
      `cross_face_sage_identity` 12); server `cargo test --offline --locked
      --features gpu` **RC=0 — 180 passed, 0 failed, 10 ignored**, 15 targets
      (`sage_dispatch` 16 — the hardware-gated arm); server clippy **RC=0** in
      BOTH states; `uv run --no-sync --locked maturin build` **RC=0** in BOTH
      states → `target/wheels/quantibbler_kernels-0.1.0-cp311-abi3-manylinux_2_35_x86_64.whl`
      (the gpu state runs ONLINE — maturin's `cargo metadata` must resolve a
      target-conditional dep absent from the offline cache); `uv run --no-sync
      pytest --color=no -q -rs` **RC=0 — 1036 passed, 0 failed, 0 skipped,
      7 deselected by the integration marker**, 239.55 s; `npx --yes
      @fission-ai/openspec validate qk-sage-attention --strict` (ONE change id)
      **RC=0 — `Change 'qk-sage-attention' is valid`**. The change's own Python
      surface: `tests/test_attention_sage.py` + the capture-path
      `tests/test_attention_gate_record.py` → **298 passed**.
      `pre-commit run --files` on the change's 9 paths **RC=0** (trailing
      whitespace, eof, large-files, merge-conflict, private-key, ruff-check,
      ruff-format; cargo-fmt/clippy skipped — no `.rs` in that set, and the
      workspace `cargo fmt`/`clippy` ran directly above). `--no-sync` is
      deliberate: a bare `uv run pytest` re-syncs this shared `/opt/venv` to the
      floor and UNINSTALLS the extras, yielding a silent lean suite.
      **Why the suite card left the box `[ ]`:** the row's text is a conjunction
      — full gate suite **and** scoped commit. The suite half was observed green
      above; the scoped-commit half had NOT happened then (this change still
      carried uncommitted spec/code/crystallization — `design.md`, `tasks.md`,
      `tests/test_attention_sage.py`, `tools/gen_attention_gate_record.py`, plus
      the untracked `tests/test_attention_gate_record.py`,
      `tools/attention_activations.py`, `tools/capture_attention_activations.py`),
      and per the ticket's landing discipline the leaf never commits (the
      orchestrator is sole committer). Ticking then would have asserted a commit
      that did not exist, so the row was left open with this evidence recorded
      inline.
      — **LANDED 2026-10-25 (card `t_894ccbd4`).** The scoped-commit half is
      done: commit `be76022` lands this change's seven sage-owned paths
      (`design.md`, `tasks.md`, `tests/test_attention_sage.py`,
      `tools/gen_attention_gate_record.py`, `tests/test_attention_gate_record.py`,
      `tools/attention_activations.py`, `tools/capture_attention_activations.py`)
      — 7 files, +1457 −75, zero foreign bytes (`git show --stat be76022`).
      Scoped re-quote on the landing tree: `pytest
      tests/test_attention_sage.py tests/test_attention_gate_record.py -q`
      **298 passed** (interp `/opt/venv/bin/python3` = torch 2.14.1+cpu, no
      accelerator; `PYTHONPATH` pinned to this tree), `openspec validate
      qk-sage-attention --strict` **valid**, `pre-commit run --files` (those
      paths) **RC=0**. Both conjuncts of this row are now true, so the box is
      ticked. The change heads to archive per its own fold row —
      **ready-for-archive**.
