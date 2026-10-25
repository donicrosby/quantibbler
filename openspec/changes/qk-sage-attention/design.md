# qk-sage-attention — Design

## User directive (2026-10-20): consume CubeK for the attention kernels

Same binding steer as qk-flash-attention (full record there). Version
alignment RESOLVED: cubek-attention 0.3.0 on the registry (cubecl
^0.11.0, our exact pin; no rust-version; same tracel-ai publisher) —
no git rev, no MSRV change. The hidden-scale, GQA-expansion,
cpu-reference purity-test, and same-commit Cargo.lock invariants from
the flash record apply here unchanged. Sage-specific: the score path
isolates its softmax construction to one seam
(partition_attention.rs:75), so the γ(K)-mean-shift has one place to
live — whether it is pluggable without a custom stage is deferred to
when this change opens, not guessed now.

## Method provenance (rule: every claim cited)

- γ(K) = K − mean(K) and the shift-invariance exactness argument: [SA]
  §4.2, Eq. 6. Derived property we rely on: the mean may be taken over the
  K-block currently in flight (any constant shift preserves softmax), so
  the kernel never needs a whole-cache pass — this is algebra, not
  assumption.
- Granularity and dtypes: [SA] §4.3 — Q, K per-token INT8 (per-channel is
  structurally impossible at QKᵀ's inner axis); P̃ per-block with STATIC
  scale 1/127 (row max of P̃ = exp(s − rowmax) is 1 by construction); V
  per-channel INT8; FP32 accumulator ([SA] §4.3, Table 4: fp32 acc is the
  most accurate of their variants).
- Quality metrics (Cosine Sim, Relative L1, RMSE vs fp attention):
  [SA] §4.3 "Accuracy metrics" — adopted verbatim as our gate metrics.
- Worst-case warning that motivates the real-activation gate: [SA] Table 3
  (INT8 rows degrading to 56–79% cos-sim on outlier layers).

## Discarded alternatives

- **FP8 Q/K or V** — [SA] §4.3 accuracy is worse than INT8 for Q/K, and
  RDNA3 exposes no fp8 dot instruction; the paper's fp8 rationale
  (2×-slower-than-int8 on NVIDIA) does not transfer. Discarded.
- **Whole-cache mean(K) (precomputed at load)** — breaks with KV-cache
  growth per token and adds a per-step recompute; per-block mean is exact
  by shift invariance and free. Discarded for the per-block form.
- **Post-hoc static calibration of Q/K scales** — [SA] uses dynamic
  per-token scales inside the kernel; static scales would deviate from the
  pinned text to save nothing measurable. Discarded.
- **INT8 PV accumulation (i32 accumulator into softmax)** — P̃·Vᵀ must
  dequant per-channel V in f32 and the online-softmax statistics are f32;
  integer accumulation there would corrupt m/ℓ. Discarded; i32 appears only
  as the intermediate of INT8 dot products before f32 scaling.

## Informed assumptions

- **Assumption:** RDNA3 (gfx1151) byte-domain INT8 dot (`v_dot4`) yields
  ≥ 2× the fp32 dot rate at attention shapes, making the quantized path
  faster than the exact tiled face, not just narrower. Rationale: RDNA3
  ISA exposes vector byte-dot instructions; [SA]'s own speed mechanism is
  INT8 dot-rate. **Wrong-if:** the task-4 microbench measures < 2× — then
  the change records the number and the face justifies itself (or not) on
  bandwidth alone; no speed claim ships unmeasured. **FALSIFIED
  2026-10-25 (task 4.1, measured): 1.48–1.64× at S=2048 and 1.00–1.11× at
  S=4096 — the wrong-if branch fired; see "Implementation crystallization
  (2026-10-25, task 4.1)". No ≥ 2× claim is live in this change.**
- **Assumption:** the quality gate on fixture + real activations
  (cos-sim ≥ 0.999, rel-L1 ≤ 0.02 class) predicts acceptable generation
  quality on the r137 artifact. **Wrong-if:** quantized-PPL delta vs the
  exact face exceeds the recorded bound — the face stays
  opt-in-experimental with the failure recorded, per the same discipline
  as the solver's fp32 discussion: accelerators are opt-in, never silent.
  **PARTLY FALSIFIED 2026-10-25 (task 4.2, measured on real activations).**
  The PPL half holds: delta **−0.268724** (ratio 0.991422396) over 8 192
  wikitext-2 test tokens, i.e. the INT8 face does **not** degrade eval-set PPL.
  The premise's *other* half — that the gate measured on the fixture battery
  predicts real-activation quality — is what died: on 84 captured
  real-activation cases the worst Cosine Sim is **0.987266** (gate ≥ 0.999) and
  **zero** cases meet Relative L1 ≤ 0.02 (worst **0.148357**, 7.4× the gate),
  while the fixture battery the shipped record is built from passes both at
  0.999968778 / 7.2183e-3. See the task-4.2 entry below.
- **Assumption:** cubek-quant 0.3.0's symmetric quantization primitives
  (per-block / per-tensor, q8) cover or compose into [SA]'s per-token-Q/K,
  static-P̃, per-channel-V axes on device. **Wrong-if:** the task-2.1
  compile/run probe shows a granularity mismatch — fall back to hand-rolled
  quantize epilogues on the landed gpu.rs forms (the i8/u32-packed ladder),
  recorded as crystallization; the parity gates are substrate-agnostic.

## Implementation crystallization (2026-10-25, tasks 1.1–1.2)

The reference is landed (`src/quantibbler/kernels/attention_sage.py`) and four
choices the [SA] text leaves open had to be fixed before it could be measured.
All four are recorded here because the device INT8 kernel (task 2.3) must
reproduce them within its parity budget; the fifth is a fixture gap.

1. **The reference *unsmooths*: the per-block shift is added back into the
   scores before the softmax.** [SA] §4.2's invariance argument is written for a
   single whole-K mean; with **per K-block** γ each key block's shift is a
   *different* constant, so the running `m`/`ℓ` would not be comparable across
   key blocks unless the shift is undone. Scores are therefore
   `δ_Q δ_K (Q̂K̂ᵀ) + q̃·meanᵀ` — [SA] Eq. 5's dequantization plus the shift, and
   exactly the "dequantized-then-unsmoothed score path" the spec's
   `gamma-shift-invariance` scenario is written against, with the mean taken
   from the *dequantized* Q so the residual is the operands' INT8 error alone.
   Measured: on a fixture whose γ(K) lies on the INT8 grid the path is
   bit-identical to the exact scores (0.0e+00), and on general fixtures the
   deviation is 3.6e-15 (1.8e-15 relative) — i.e. γ and the shift contribute
   nothing. This resolves the question the substrate note deferred ("whether it
   is pluggable without a custom stage") for the *reference*: a kernel may
   instead fold the per-block shift into its running max — the algebra is the
   same — but the exposed score path, which the parity harness compares, is the
   unsmoothed one.

2. **A zero-range row keeps a defined, positive scale: δ = 1.0, all-zero
   codes.** [SA] §3.2's `δ = max|A|/127` is 0/0 when a row's peak is zero, and
   that is not an exotic case: at any ragged tail of one token (`seq % block ==
   1`) a per-block γ gives `γ(K_block) ≡ 0` by construction, and `block_size=1`
   is that case for every token. The value of δ cannot move any output (zero
   codes dequantize to zero for *any* δ, and the shift then restores the exact
   keys), so the choice is free — but it must be *made*, or a quantizer that
   follows the paper literally emits NaN on exactly those tails. The reference
   fixes δ = 1.0 with zero codes and `block_size=1` is therefore a working
   configuration whose score path is exact, not a refusal. The kernel needs the
   same convention on its tail block.

3. **P̃ is quantized after the running-max rescaling** (`exp(S − m_new)`, the
   FlashAttention form the tiled reference already uses), not against the
   block's own rowmax; `|P̃| ≤ 1` either way, so [SA] §4.3 reason 2's static
   `1/127` scale still covers the range and the reference stays on the kernel's
   loop shape. **ℓ uses the full-precision P̃** while only the `P̂V̂` product uses
   the quantized tile — [SA] Eq. 5 literally (`(m′, P) = σ̃(m, S)` full
   precision, "online softmax is left in full-precision", §4.1).

4. **V is quantized per channel *within the tile*** — one scale per channel of
   the V-block the inner loop has loaded, not over the whole cache. The axis is
   the one [SA] §4.3 reason 3 names (the outer axis of P̃Vᵀ, which per-channel
   handling addresses the V outliers), and a tile-local scale is the only form
   the kernel can compute without a whole-cache pass.

5. **Fixture gap recorded, not papered over:** the `quantized-budget` scenario
   also runs "captured real activations", and none are shipped in the repo yet
   (task 4.2 produces them, into the model directory). The hook
   (`load_captured_activations`, `QUANTIBBLER_SAGE_ACTIVATIONS` or
   `tests/fixtures/attention_sage_activations.npz`) *names* the absence instead
   of skipping — a skip would silently shorten the battery and move the CI skip
   counts — and its present branch is exercised in-test on a synthesized
   capture. Skipping was rejected on purpose: a quiet `pytest.skip` is how a
   quality gate stops running without anyone noticing.

## Implementation crystallization (2026-10-25, task 4.1 — measured; task 4.2 measured in its own entry below)

**OQ-1 is measured, and the ≥ 2× assumption is falsified at the shapes the
mechanism is claimed for.** Instrument: throwaway crate
`/opt/data/tmp/sage-int8-microbench` (`src/main.rs` sha256 `fe93fccc…`,
`Cargo.toml` `c7c89293…`, `Cargo.lock` `7aa73978…`), **NOT repo code** — the
gemm-bench pattern the task names. Four forms, every one a
one-thread-per-C-element naive GEMM so the dot form is the only variable:
`i8`-tensor scalar dot, `u32`-packed manual byte decode, `u32`-packed `dp4a()`,
and an `f32` scalar dot as the baseline. Best-of-3 after one warmup per form;
the timed unit is the launch plus the blocking read-back (the synchronisation
the real call sites pay).

Provenance: the kernel bodies and host-side CubeCL calls are inherited verbatim
(sha256 `bc374749…`) from the sibling lane's task-2.1 probe
`/opt/data/tmp/sage-int8-probe/src/main.rs`, itself copied from the landed,
hardware-validated recipe in `crates/quantibbler-kernels/src/gpu.rs` and
`/opt/data/tmp/gemm-bench`. Three lines had to diverge before it would run, and
they are recorded because they are the difference between an instrument and a
file: the missing `use cubecl::std::tensor::TensorHandle;`; `dp4a`'s frontend
op takes `i32` (`cubecl-core-0.11.0/src/frontend/operation/dp4a.rs:8`), so
`i32::cast_from(wa)` (bit pattern unchanged); and `pack_u32`'s A-operand branch
sized its buffer `rows*cols` while writing `rows*(k/4)` — an
index-out-of-bounds panic on the first call. The source this was derived from
carried all three (byte-identical at derivation), so any task-2.1 number
reported from that copy needs the same three fixes before it is comparable.

Pins: `cubecl 0.11.0` (the repo's own lock version); container
`quantibbler-ptq:rocm-official` with `--device /dev/kfd --device /dev/dri`;
`HSA_OVERRIDE_GFX_VERSION=11.5.1`; `LD_LIBRARY_PATH=/opt/rocm/lib:/opt/data/tmp/rocm-sdk/lib`;
device gfx1151 (Radeon 8060S). The build ran in that container because this
sandbox's own 5 GiB memcg SIGKILLs the root-crate LTO link (dmesg
`Memory cgroup out of memory`, `task=rustc` — twice). Container wart worth
knowing before the next Rust GPU build here: the ROCm image ships
`/usr/local/bin/hipconfig` reporting `7.16.26385-0000000`, whose patch is **not**
among `cubecl-hip-sys 7.14.6085001`'s declared `hip_*` bindings features, so the
sys crate builds with **no bindings** and `cubecl-hip` dies with 87 `E0425`s;
parking `hipconfig` for the build (the crate's documented "no hipconfig on the
system" fallback, which warns and selects its latest declared feature) builds
clean in 55 s.

Correctness before rate, in both runs: all three INT8 forms are **bit-exact**
against a host i64 reference at (8,128,8) — `i8 max|dev| = 0`,
`packed-manual = 0`, `dp4a = 0`.

**Table — INT8 vs fp32 dot at attention shapes** (two runs back to back; run 2's
box load average `47.69 47.45 47.72` on 32 cores — the co-tenant saturation the
1.1–1.2 entry describes — run 1's load was not captured, which is why there are
two columns rather than one. Best-of-3 is the noise-resistant reading; `×f32` is
against that run's own `f32` baseline.)

| shape (m,k,n) | role | form | run 1 GOP/s | run 2 GOP/s | run 1 ×f32 | run 2 ×f32 |
|---|---|---|---|---|---|---|
| 2048,128,2048 | QKᵀ (S=2048, d=128) | i8-scalar | 212.8 | 428.2 | 1.49× | 1.41× |
| | | u32-packed manual | 235.4 | 449.3 | 1.64× | 1.48× |
| | | u32-packed `dp4a` | 138.3 | 173.1 | 0.97× | 0.57× |
| | | f32-scalar (baseline) | 143.2 | 302.6 | 1.00× | 1.00× |
| 4096,128,4096 | QKᵀ (S=4096, d=128) | i8-scalar | 125.5 | 134.2 | 1.05× | 1.05× |
| | | u32-packed manual | 119.5 | 133.4 | 1.00× | 1.04× |
| | | u32-packed `dp4a` | 126.5 | 142.0 | 1.06× | 1.11× |
| | | f32-scalar (baseline) | 119.9 | 127.9 | 1.00× | 1.00× |
| 2048,2048,128 | P̃Vᵀ (S=2048, d=128) | i8-scalar | 190.6 | 936.2 | 0.52× | 1.86× |
| | | u32-packed manual | 370.5 | 1049.6 | 1.01× | 2.08× |
| | | u32-packed `dp4a` | 458.7 | 982.4 | 1.25× | 1.95× |
| | | f32-scalar (baseline) | 368.2 | 504.5 | 1.00× | 1.00× |

**Verdict.** At the two QKᵀ shapes — the dot the γ mechanism and the per-token
Q/K quantizers exist for — the best INT8 form is **1.48–1.64×** the f32 dot at
S=2048 and **1.00–1.11×** at S=4096: under the 2× bar in every run, and at
S=4096 effectively at parity. The P̃Vᵀ shape straddles the bar (1.25×, then
2.08×), so it neither carries the claim nor refutes it. Two things register with
that number rather than being argued around it:

- **No ≥ 2× speed claim ships** — the assumption's own wrong-if clause. The
  face's case has to rest on measured bandwidth or on a tiled formulation, and
  this microbench establishes neither: these are naive per-C-element loops,
  latency/ILP-bound, so the table is evidence about the *dot form at these
  shapes*, not a throughput ceiling for the face or for a tiled kernel. The
  byte-dot op is also not automatically the fast form here: `dp4a` is the
  slowest of the three at both QKᵀ shapes in run 1 (138 vs 235 GOP/s at
  S=2048) and 2.6× slower than the manual packed form in run 2 (173 vs 449).
- **The paper's speed mechanism is not the one available to us.** [SA]'s win is
  tensor-core INT8 dot rate; RDNA3's byte-domain path does not deliver ≥ 2×
  over f32 on this instrument at attention shapes. What remains is the other
  lever the proposal names — halved K/V memory traffic — which this instrument
  does **not** measure (it times compute at fixed operand widths, not the
  cache-resident traffic a real KV cache sees). Task 2.3 is where that would
  have to be measured, if at all.

**Task 4.2 — SUPERSEDED 2026-10-25: the row was measured in its own entry
below, and OQ-2 is thereby RESOLVED (against the face).** The retracted reading
is kept here only so it cannot be quoted as current: it was true when written —
no captured-activation record existed, the gate tool (3.3) and the INT8 kernel
path (2.2/2.3) had not landed, and both halves of the row were real-model runs
held for an explicit @user GO. All three of those conditions then changed: 2.2/
2.3 and 3.1–3.3 landed, and the GO was given. Nothing above is a substitute for
the real-activation gate — which is exactly what the measurement below shows,
and it shows it the hard way.

## Implementation crystallization (2026-10-25, task 4.2 — the real-activation quality gate, measured; OQ-2 resolved AGAINST the face)

**The gate has been run on REAL activations, and the INT8 face FAILS it.** By
this change's own task-row rule — *"face ships as opt-in-recommended only if
both gates pass, else recorded as opt-in-experimental with the failing
numbers"* — the face is recorded **opt-in-experimental**.

Instrument: `/opt/data/tmp/sage-real-activations/capture_and_ppl.py` — **NOT
repo code**, the section-4 probe pattern task 4.1 set (measurement, never
argument) — driven by `uv run --no-sync python` on `/opt/venv/bin/python3` with
**torch 2.14.1+cpu** (`torch.cuda.is_available() = False`: the CPU wheel, so no
accelerated path is reachable), numpy 2.5.4, transformers 5.20.1, bfloat16.

**Where the measurement sits — the tensors are the faces' own documented
operands.** The two faces run *inside the model*, registered as attention
implementations (`ALL_ATTENTION_FUNCTIONS.register`), so `Qwen3Attention.forward`
hands them precisely what they document: the post-`q_norm`/`k_norm`, post-RoPE
`query_states`/`key_states`/`value_states`, GQA **un-repeated**, in the
`(B, n_heads, S, d)` / `(B, n_kv_heads, S, d)` layout both modules' docstrings
pin. Everything captured is therefore a REAL activation of the model, not a
fixture. Eval set is the driver's own held-out one: `wikitext-2-raw-v1` **test**,
seq 256, block-shuffled windows (the driver's selection rule). Model: the local
HF snapshot `models--Qwen--Qwen3-0.6B` (bf16, 1 503 300 328 B; 28 layers,
16 heads / 8 kv heads, head_dim 128) — the model the `qwen3-0.6B-r137*` qpk1
artifacts quantize.

**Instrument self-validation, measured rather than asserted.** Per captured
case, the exact face vs HF's own eager attention on the *same* captured q/k/v:
worst relative **8.52e-7** (run 2) / 6.78e-7 (run 1). Model-level, HF's native
`sdpa` and our exact face give the same eval PPL to **0.098%** (run 2) / 0.063%
(run 1). That is the harness's noise floor, and every number below is quoted
against it.

**Two runs (both RC=0):**

| | run 1 — 4 layers × 2 windows | run 2 — 28 layers × 3 windows | gate |
|---|---|---|---|
| cases | 8 | 84 | |
| worst Cosine Sim | 0.994972772 | **0.987266027** (L5 w2) | ≥ 0.999 → **FAIL** |
| worst Relative L1 | 0.122862964 | **0.148357076** (L5 w0) | ≤ 0.02 → **FAIL** |
| best Cosine Sim | 0.998514466 | 0.999674763 | |
| best Relative L1 | 0.050310011 | 0.024245284 | |
| cases passing cosine ≥ 0.999 | 0/8 | 14/84 | |
| cases passing Relative L1 ≤ 0.02 | 0/8 | **0/84** | |
| cases passing BOTH | 0/8 | **0/84** | |
| eval tokens | 4 096 | 8 192 | |
| ppl_exact (`attention_tiled`) | 36.970209 | 31.328512 | |
| ppl_sage (`attention_sage`) | 36.511942 | 31.059789 | |
| ppl_native (HF `sdpa`, swap check) | 36.993469 | 31.359278 | |
| **PPL delta (sage − exact)** | **−0.458267** | **−0.268724** | |
| ratio | 0.987604423 | 0.991422396 | |

Run 2's per-layer worst Cosine Sim / Relative L1 (28 layers, 3 windows each)
shows the error is spread across depth rather than confined to a tail:
L0 0.997620/0.051510, L1 0.999652/0.024897, L2 0.999329/0.028834, L3
0.994365/0.107359, L4 0.989546/0.126718, **L5 0.987266/0.148357**, L6
0.997591/0.066592, L7 0.994119/0.093224, L8 0.994353/0.099011, L9
0.997983/0.063201, L10 0.996687/0.070437, L11 0.999280/0.037095, L12
0.995327/0.088874, L13 0.998652/0.048257, L14 0.996996/0.077367, L15
0.997452/0.071684, L16 0.999318/0.039550, L17 0.998800/0.047422, L18
0.998108/0.058647, L19 0.998269/0.064781, L20 0.997810/0.077427, L21
0.995660/0.119932, L22 0.996317/0.104208, L23 0.996541/0.107791, L24
0.996669/0.113647, L25 0.993755/0.139814, L26 0.996749/0.099755, L27
0.994973/0.122863.

**Verdict.** The [SA] §4.3 metric clause **fails on real activations in both
runs, in both directions, and not marginally**: the best cosine in 84 cases
(0.999675) still misses 0.999, and **zero of 84 cases** meet Relative L1 ≤ 0.02
(best 0.024245, worst 0.148357 — 7.4× the gate). The PPL half passes, and passes
in the *favouring* direction: `delta = −0.268724` at an eval size of 8 192
tokens, i.e. the INT8 face measures **lower** PPL than the exact face, by 2.9×
the harness noise floor (so it is not noise, and it is not a degradation).
Measured, not argued: the face's quality case cannot be made on the metrics and
does not need to be made on PPL.

**The consequence that matters more than the verdict.** The shipped record
passes these *same two clauses* at **0.999968778 / 7.2183e-3** (tasks 2.3/3.3)
— because it is measured on the committed *fixture battery*, whose inputs are
i.i.d. Gaussians, while 84 real-activation cases fail Relative L1 by 7.4×. This
is [SA] Table 3's warning landing on us in its own terms: **a fixture battery
does not see outlier layers, so the shipped enabling evidence over-states the
face by orders of magnitude.** The face as shipped would be *enabled* on the
fixture record while failing the gate on the activations it will actually see.
Routed to the orchestrator (OQ-2's owner) and to the planner.

**Judgment calls, named rather than buried — and each is a place a reader may
reasonably disagree.**

1. **The capture is the healthy base Qwen3-0.6B, not a dead `r137` qpk1
   artifact.** Every `qwen3-0.6B-r137*` artifact is `failed` (v0 PPL 1 156 577,
   v2 943.49, v3/v4 NaN, v7 5 517.8, v8 463 742; the repo's own records), so a
   quality gate measured on a dead model's activations measures nothing. The
   base model *is* the model those artifacts quantize, and the attention inputs
   a quality gate must be robust to are the healthy distribution's outliers.
   A capture taken from the served qpk1 artifact would require a capture surface
   inside the Rust engine (the only qpk1 forward); that is a server-contract
   change, not a probe, and it is **not** claimed here.
2. **The PPL clause value recorded by this run** is
   `{eval_set: "wikitext-2-raw-v1 test (8192 tokens, seq 256)", delta:
   −0.268724, bound: 0.05}` → PASS. The bound `0.05` is **not minted here**: it
   is the value this change's own shipped gate tests already use
   (`sage-ppl-ok.json`, `{delta: 0.011, bound: 0.05}`, task 3.3's row), so the
   clause is recorded under the convention of record rather than a fresh
   threshold. **A bound for a real-model eval-set PPL is nonetheless a policy
   threshold and deserves the planner's ruling**; this entry does not claim
   otherwise, and the measured delta passes any sane bound in either direction.
3. **No committed gate record is minted from this capture, and that is a named
   gap rather than an omission.** `gen_attention_gate_record.py` consumes a JSON
   numerics battery; a real-shape case (seq 256 × 16 heads × 128 dim, ×3
   operands) is ~82 KB of JSON *per unit of seq*, so an 84-case real battery is
   orders past the repo's 500 KB large-file gate — a compact (npz / f32-binary)
   capture path for the gate tool is a format change to landed code and belongs
   to its own row. The only real model directory (`artifacts/qwen3-0.6B-r137`)
   is gitignored, and the committed `tests/fixtures/server-model` record belongs
   to the *tiny synthetic* `server-model-fixture` — writing a Qwen3-0.6B PPL
   clause into it would be a provenance lie. The numbers above therefore live in
   this entry, where the change records its verdict, and not in a record file.

## Implementation crystallization (2026-10-25, task 2.1 — feasibility probe)

**The probe ran green, and the working idiom is the deliverable.** Instrument:
throwaway crate `/opt/data/tmp/sage-int8-probe-11` — **NOT repo code**, the
gemm-bench pattern the row names — with the exact revision that produced the
numbers archived under `/opt/data/tmp/sage-int8-probe-11/evidence-2.1/` (`probe-main.rs` sha256
`f84d722b0aaf804e781f4ca71f25e42808ea0f20e2ddb1a9e6a2327720589569`, binary
`2c9b1fab…`, logs `run1…run6`). `cubecl` is pinned `= 0.11.0` — the repo's own
lock version; a caret requirement resolved **0.11.3** first and that is not the
version under probe. Built and run **in this sandbox** (`cargo build --release
--offline`, `BUILD_RC=0`) — record the profile setting, because the first link
was **SIGKILLed by the container's 5 GiB memcg** (`/sys/fs/cgroup/memory.max` =
5368709120; dmesg `Memory cgroup out of memory: Killed process …, task=rustc`)
under the pattern's `lto = "thin"`, and the working build is `lto = false` +
`CARGO_BUILD_JOBS=1`. Host-side LTO does not touch these numbers: the device
kernels are JIT-compiled at runtime by cubecl-llvm into AMD code objects, so
the host profile shapes host code only. Run under the landed recipe:

```
LD_LIBRARY_PATH=/opt/data/tmp/rocm-sdk/lib HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  /opt/data/tmp/gemm-bench/target/release/sage-int8-probe-11     # RUN_RC=0
```

→ `device ok` on the HIP device, then **every dot form the tile path needs is
exact against a host i64 reference** (m=8,k=128,n=8; the reference accumulates
in i64 and is narrowed to the i32 the device accumulates in):
`i8 max|dev| = 0`, `i32-packed manual = 0`, `u32-packed manual = 0`,
`dp4a = 0`. `i8` is a first-class element type (`cubecl-core` `impl_int!(i8,
I8)`, `IntKind::I8`) and the HIP backend this build selects accepts
`Tensor<i8>`, `Tensor<u32>` and `Tensor<i32>` operands alike — the backend is
the LLVM/pliron one, because `cubecl-hip`'s `cpp` feature is off by default and
nothing in the graph enables `hip-cpp` (`HipBackend::default()` is
`if cfg!(feature = "cpp") { Cpp } else { Llvm }`).

**Working idiom (recorded as the row asks).** Host side: `TensorHandle::new(
client.create_from_slice(i8::as_bytes(&v)), vec![len], vec![1],
i8::elem_type_native())`, launch with `CubeCount::Static(cubes, 1, 1)` +
`CubeDim::new_1d(256)` and `handle.into_arg()` / `out.binding()
.into_tensor_arg()`, read back through `client.read_one_unchecked_tensor(
h.handle.clone().copy_descriptor(h.shape().clone(), h.strides().clone(),
size_of::<T>()))` then `i32::from_bytes(&bytes)`. Device side: per-element
`i32::cast_from(t[i])` for `Tensor<i8>`; packed words as `Tensor<i32>`, because
`dp4a`'s frontend contract is `dp4a(i32, i32, i32) -> i32` over four **signed**
byte lanes (`cubecl-core-0.11.0/src/frontend/operation/dp4a.rs:8`); the
`Tensor<u32>` form is the same bytes with `(x ^ 128) - 128` sign extension.
Three traps, all hit, all fixed, all worth having found here rather than inside
a tile kernel: a `#[cube]` helper is not callable as `helper::expand(a, b)` from
a launch body (that form wants a `&Scope`), so hot primitives are inlined; the
packed operand must carry **one word per thread**, not one per four lanes, or
the device reads out of bounds (my v1 caught this only because a host mirror
disagreed); and the host packing buffer must be sized by the *packed* shape
(`rows*kw`), not `rows*cols` — the index-out-of-bounds panic at len 64/index 64
happened before any kernel ran. The 4.1 entry reports the same three defects in
the pre-fix copy it derived from; the two lanes agree on them independently.

**The lowering has no byte-domain dot instruction, and that is a source read of
the pinned crate, not an inference from timings.** `Dp4aOp::should_lower` is
`ctx.target() != Target::Cuda`, and the non-CUDA branch expands
`expand_dp4a_polyfill` in **both** HIP backends
(`cubecl-cpp-0.11.0/src/shared/binary.rs` and
`cubecl-llvm-0.11.0/src/shared/polyfill/math.rs`); the `__dp4a` intrinsic lives
only in `cubecl-cpp/src/cuda/binary.rs`, and `CppSupportedFeatures::dp4a`
defaults false. On gfx1151 `dp4a()` is therefore ~10 integer ops.

**Register-resident ALU isolation — the second, independent reading**
(32 768 threads × 20 000 reps, operands held in registers, all four recurrences
verified **exact** against a host mirror before timing): scalar `i8` 1119,
packed 4-lane manual 2404, `dp4a` 2875 G lane-MAC/s, `f32` scalar 1752. `dp4a`
separates from the hand-written polyfill by only **1.2×** — it *is* the same
code — and beats `f32` by **1.64× per lane-MAC**. A real 4-lane byte dot would
open a far larger gap than that; it does not, which is what makes the source
read a finding rather than a guess. Consequence for the tile path's hot loop:
prefer packed `i32` words over `Tensor<i8>` operands (scalar `i8` measured
**0.50×** `f32` at the S=2048 score shape).

**Shape numbers from this probe are indicative only, and are not the rate of
record.** The box was contended while they ran (two other containers held
`/dev/kfd`: `qk-attention-refbench`, `qk-divergence-gates-tier`), and the spread
across runs is large — `dp4a` at (2048,2048,128) measured 1268, 54 and 546 GOP/s
in three different runs. Run 6's single coherent set: (2048,128,2048) QKᵀ —
`i8` 150, `packed-i32` 439, `packed-u32` 418, `dp4a` 475, `f32` 300 GOP/s;
(4096,128,4096) QKᵀ — 40.5 / 38.6 / 40.7 / 39.0 vs `f32` 43.4 (every INT8 form
≤ 0.94×); (2048,2048,128) P̃Vᵀ — 463 / 410 / 682 / 546 vs `f32` 355. **Task 4.1's
table remains the rate of record**; these numbers are consistent with its
verdict (under the 2× bar, parity at S=4096) and add the `u32`-packed row the
2.1 text names.

**Instrument discipline, recorded because v1 of this instrument lied.** Its
ALU loop was `acc = prod + (acc & 1)` with a loop-invariant product and it
printed **35 950 G lane-MAC/s** — 0.1 ms for 6.5e8 iterations, physically
impossible: the compiler had collapsed the loop and the timer was measuring a
launch. v2 pins the iteration count by construction (an LCG accumulator whose
exact value after `reps` steps is computable on the host, plus a churned operand
so the dot cannot be hoisted) and compares the device output to that mirror
exactly — `0 mismatched threads` for all four forms — with reps-scaling
(20 000 → 80 000, measured 3.2–4.3×) as the independent second witness. Both
artefacts are in `/opt/data/tmp/sage-int8-probe-11/evidence-2.1/` (`run3-alu-v1-bogus.log`,
`run5-alu-v2-verified.log`/`run6-final.log`).

**GO/NO-GO for the INT8 tile path — GO on feasibility, NO-GO on "the byte dot
is the win".** Every form tasks 2.2/2.3 need compiles, runs and is exactly
correct on gfx1151, and the three traps above are named instead of waiting
inside a tile loop: 2.3 proceeds. What does not survive is the speed
*mechanism* — the polyfill finding above plus task 4.1's measured verdict mean
the face must justify itself on halved K/V traffic and the quality gate, and
per this design's own wrong-if clause **no ≥ 2× speed claim may be written from
this stack** until a tiled measurement (task 2.3, if at all) shows the traffic
lever paying. The substrate assumption's fallback branch is also settled in one
direction: since no substrate op provides a hardware INT8 dot on this target,
the tile path will be hand-rolled epilogues on the landed `gpu.rs` forms either
way — `cubek-quant` can still supply quantize/dequantize primitives; it cannot
supply the dot.

## Implementation crystallization (2026-10-25, tasks 2.2–2.3 — the device INT8 tile path)

**Landed:** `GpuContext::attention_sage` + `sage_int8_tile_kernel` in
`crates/quantibbler-kernels/src/gpu_attention.rs` (behind the opt-in `gpu`
feature), with `tests/gpu_attention_sage.rs` (hardware-gated, both states
verified) and the fixture battery `tests/fixtures/attention_sage.json`
regenerated by `tools/gen_attention_sage_fixtures.py`. This is *ours* rather than
the substrate's, which is exactly what the 2.1 entry concluded it would have to
be.

**Shape of the kernel.** One thread per `(batch, head, query token)` row; that
thread owns its row's whole score → softmax → accumulator sequence, so there is
no cross-thread reduction, barrier or atomic anywhere and every scalar is a
per-thread local. The two INT8 dots (`Q̂K̂ᵀ` and `P̂V̂`) accumulate in `i32` on the
device — [SA]'s whole point — and the per-token Q/K scales, per-block γ, the
static P̃ scale, per-channel V, the online-softmax statistics and the
accumulator are the reference scheme stage for stage. Per-row scratch (scores,
P̂ codes, V scales, the block means, Q̂ codes) lives in device buffers indexed by
the owning thread's row: that duplicates per-`(b, kv)` tiles once per query head
and buys the absence of synchronization. Correctness before rate is this
change's standing order, and the 2.1 probe already established there is no ≥ 2×
speed claim to protect.

**Why f32 statistics, and the pre-code measurement that said they were enough.**
The requirement names *f32* statistics/accumulator while the reference computes
in f64, and one flipped INT8 code moves an element by ~1/127 of its
contribution, so the 1e-4 budget is satisfiable only if the device reproduces
the reference's *codes*, not merely its values. That was measured in NumPy
*avant la lettre* (throwaway, NOT repo code,
`/opt/data/tmp/sage-f32-emul/emul2.py`): a full f32 emulation of this loop over
the nine `gen_attention_fixtures` shape families × 12 seeds lands at **worst
infinity-norm relative 2.2e-7 with zero code flips in 72 960 Q codes and
144 384 P̃ codes**. The device then measured **1.201e-6** worst parity band on
the shipped battery — ~80× inside the 1e-4 budget — and its quality metrics
against the exact face (worst Cosine Sim **0.999968778**, worst Relative L1
**7.2183e-3**) are the reference's *own* measured values to eight figures
(0.9999687782 / 7.2183043e-3): the device's extra f32 error does not move the
quality metrics at all. The measured answer to "is f32 enough" is yes, with an
order of magnitude to spare — the requirement's dtype was not quietly overridden
to f64.

**Three `cubecl` 0.11 idioms this path cost, recorded so the next kernel does
not pay them again.**

1. **A constant-initialized `let mut` local becomes a PLAIN Rust scalar, and
   then plain assignment stops compiling.** The `cube` macro const-folds an
   initializer it can evaluate (a negative literal like `-3.0e38f32`, and a
   module-level `const f32` path) into `as_const(...).clone()` instead of
   `NativeExpand::from_lit(...)`, so the local is a bare `f32` and
   `m = <NativeExpand<f32>>` resolves to no `Assign` impl — E0308 "expected
   `f32`, found `NativeExpand<f32>`", reported at the `#[cube(launch)]`
   attribute with no inner span. Isolated with throwaway probe kernels in the
   same crate: a plain literal (`0.0f32`), an expression
   (`f32::cast_from(0)`) and a loop-carried `if v > m { m = v; }` all compile;
   the *negative literal* does not. The working sentinel is therefore an
   expression, `let mut m = f32::cast_from(0) - 3.0e38f32;`.
2. **A `#[cube]` free-function helper is still not callable from a launch
   body** (the 2.1 trap, independently re-hit here): the half-to-even rounding
   and the clamps are written inline at each of their four sites.
3. **The vetted device op surface has no integer `clamp`** (`FloatOps::clamp`
   is float-only), so the codes clamp as two value-form steps — which is also
   the shape `clippy::manual_clamp` accepts, so this needed no `allow`.

A fourth, not `cubecl`: **rustc 1.98.1 on this box has no `f` format trait** —
`{x:.6f}` is a hard error (reproduced standalone in three lines) — so the test
formats floats with `e`.

**Contract decisions, named rather than silent.** The reference masks with
`−inf`; the kernel masks with a finite sentinel (`-3.0e38`), which reproduces
both properties that matter — it never wins a max against a real score, and
`exp(sentinel − m_new)` underflows to exactly 0 — while keeping every
intermediate finite, so `inf − inf` cannot appear. Rounding is half-to-even
(NumPy's `rint`, which is what ψ's definition quantizes with), not
`f32::round`'s half-away-from-zero, because the codes are the contract.
Two edges are typed refusals rather than silent reinterpretations:
`seq_q == seq_kv` is required (γ's shift is taken over the key block in flight,
which the scheme defines for self-attention over one cache) and `block_size`
must be positive. The reference's additive-bias `mask` is **not** implemented:
the exact device face does not take one either, and a bias path that never ran
is not shipped — the gap is named in the method's own docstring.

## Implementation crystallization (2026-10-25, tasks 3.1–3.3 — the `gpu:sage` face, its startup gate, its gate tool)

**Landed:** `AttentionFace::GpuSage` + `gqa_attention_sage` in
`crates/quantibbler-server/src/engine/mod.rs`; the gate and its typed refusal in
`crates/quantibbler-server/src/model/mod.rs` (`attention_gate_check`,
`sage_face_from_env`, `AttentionGateError`) wired into `model::load`;
`crates/quantibbler-server/tests/sage_dispatch.rs` (RED first: `E0425 cannot find
function attention_gate_check in module model`, RC=101); and the committed
producer `tools/gen_attention_gate_record.py` + the shipped record.

**The record, and what the gate actually enforces.** `<model_dir>/attention_gate.json`,
schema `quantibbler.attention_gate/1`: `model_id`, `activations_source`,
`generated_at`, `code_commit`, the `clauses` it was generated under, per-call
([SA] §4.3) metrics, their `worst`, and an optional `ppl` clause. The gate
enforces **its own constants** (`GATE_COSINE_MIN = 0.999`,
`GATE_RELATIVE_L1_MAX = 0.02`) against every per-call metric *and* against the
declared aggregate — a record whose `worst` does not equal the aggregate of its
own calls is refused, and a record that declares a looser clause set than the
gate's is refused rather than trusted. `model_id` binds the record to the served
model, which is what "stale" means here alongside "out of bounds"; the date,
commit and activation source are **logged, never gated on** — they are a record
for a reader, not an identity stamp (the same ruling as the checkpoint
fingerprint's: version control and the CI gates answer "did the code change").

**The one clause that is not required to be present, and why that had to be a
decision rather than a default.** The requirement's letter is a conjunction:
"Cosine Sim ≥ 0.999 and Relative L1 ≤ 0.02 ..., **plus** a quantized-PPL delta
within the bound recorded at the gate run". Nothing in the tree can produce that
third number: it is a model-level eval-set PPL, and the only forward passes in
this repo are the Rust engine's (which has no sage face without hardware) and the
CUDA/NumPy references (attention-only). Its named producer is task 4.2, which is
BLOCKED on an @user GO for the real-model capture. Implementing the letter
literally would therefore refuse **every** record this tree can mint, i.e. the
face could never be enabled and `gate-pass-enables` could never be observed. The
gate instead: enforces the clause on any record that declares one
(`delta <= bound`, both finite, `bound > 0`), and on a record that declares none
emits `attention gate: the record declares no eval-set PPL delta … admitted on
the [SA] §4.3 metrics alone` at startup and records `"ppl": null` in the record
itself. No PPL number is invented — a fabricated measurement would be worse than
the gap. **This is a spec-reading decision offered for a ruling (owner:
@planner): either the PPL clause becomes conditional in the delta, or
`gate-pass-enables` stays unobservable until 4.2's capture lands.** The face's
ship-vs-experimental status is unchanged by this entry: OQ-2 still decides it.

**The shipped record's inputs, named honestly.** The activation fixture this
change ships is the committed device-parity battery
(`crates/quantibbler-kernels/tests/fixtures/attention_sage.json`, task 2.2) — f32
q/k/v fixture inputs, **not** a captured forward pass of a real model. The record
says so in `activations_kind: "fixture-battery"`, and the tool says so in its
docstring. Measured by the tool on that battery: worst Cosine Sim
**0.999968778** and worst Relative L1 **7.218304e-3** over 10 calls — the same
two numbers task 2.3's device run and the reference report (0.9999687782 /
7.2183043e-3), computed here by a third independent path (the tool recomputes
BOTH faces from the fixture inputs and cross-checks each case against the
metrics `gen_attention_sage_fixtures.py` stored, aborting on disagreement).

**Fixture-directory hazard, recorded because it is silent otherwise.**
`tools/gen_server_fixtures.py` removes and rebuilds `tests/fixtures/server-model`
(`shutil.rmtree`), so it takes the shipped record with it. The record is
regenerated by one command
(`uv run python tools/gen_attention_gate_record.py --model-dir tests/fixtures/server-model`)
— and `scenario_gate_record_regenerates_matches_the_shipped_fixture` fails loudly
(missing record ⇒ load refusal) the moment it is dropped, so the loss cannot go
unnoticed.

**The opt-in and the two feature states.** `QUANTIBBLER_ATTN=sage` selects the
quantized face (`=1` remains the exact tiled face, unset the scalar root). The
gate runs in **both** feature states and **before any device is touched**, so a
refused model never gets a device context and the refusal is identical in a
default build (where the face cannot exist at all) — that is what makes
`gate-fail-refuses` a no-hardware scenario, as its task row says. A device-init
failure *after* the gate passes soft-falls-back to the exact scalar face (the
spec's "or scalar", and the proposal's "the EXACT faces stay the default and the
fallback root"); quality is never a fallback trigger.

**The kernel contract forced the query mapping, and the mapping is recorded
rather than hidden.** `GpuContext::attention_sage` requires `seq_q == seq_kv`
(γ's shift is taken over the key block in flight — task 2.3's typed refusal), but
a decode step is one query against `pos + 1` cached keys. The engine therefore
puts the real query in **row 0** of a `pos + 1`-row query buffer padded with
zeros and calls with `causal = false`: the cache already holds exactly this
query's causal window, so row 0 attends to every key present. Only row 0 of each
head is read. The padding rows are computed and discarded, which makes the call
**O(seq_kv²) per decode step** instead of O(seq_kv); that is a property of the
shipped kernel interface, not a claim about the face, and it is named in
`gqa_attention_sage`'s own docstring. Relaxing `seq_q == seq_kv` (a first-class
single-query INT8 path) is the smaller change but belongs to the kernels contract
and its parity gates, not here. `SAGE_BLOCK_SIZE = 64` is the face's γ /
online-softmax granularity (the kernel is one thread per query row, so it is not
a thread-block shape); it is a constant rather than an env knob because nothing
measures it yet, and an unmeasured knob is a way to mislabel a record.

## Implementation crystallization (2026-10-25, task 3.4 — the engine call site, measured on gfx1151)

- **The new scenario found an O(1) defect in this change's OWN call site, and
  the RED/GREEN pair is the measurement.** Nothing before it read a number off
  `gqa_attention_sage`: `crates/quantibbler-kernels/tests/gpu_attention_sage.rs`
  calls the INT8 kernel DIRECTLY with fixtures already in the documented layout,
  and tasks 3.1–3.3 exercise gate admission, refusal, labels and soft fallback.
  The first engine-level run read **Cosine Sim 9.904153659e-1** (gate ≥ 0.999)
  and **Relative L1 1.304462818e-1** (gate ≤ 0.02) on gfx1151 (8 identical
  tokens, 14-key context, RC=101); after the fix, the same device and command
  read **Cosine Sim 9.999999884e-1, Relative L1 1.698875021e-4**. Both metrics
  cross the gate by orders, so this scenario has measured detection power on
  both — unlike the sibling tiled face's token clause (`cross_face_token_identity.rs`),
  whose fixture greedy stream is constant (token `1` for all 8 steps, both
  faces; task 3.3 recorded that the logits clause was the load-bearing one).
  The token clause here is likewise constant and carries almost no power; the
  two metric clauses are what the scenario is worth.
- **The kernel contract is the oracle and the call site was the wrong party —
  the mirror of task 3.3's finding, one face over.** `attention_sage` documents
  the layout it mirrors from `GpuContext::attention` — keys/values in
  `(B, n_kv_heads, S, d)` — and the kernel indexes `kv_tok = (b * n_kv + kvh) *
  seq`, i.e. each kv head's whole `seq × d` plane contiguous (head-major).
  `KvCache`'s raw stack is `(seq_kv, n_kv_heads, head_dim)` (row-major) and the
  two coincide only at `n_kv_heads == 1`, which the served fixture (2 kv heads)
  is not — so the shipped INT8 face computed from a transposed K/V plane for
  every real GQA shape. The fix is `KvCache::keys_head_major` /
  `vals_head_major` at the call site — the same host-side gather task 3.3 put in
  `gqa_attention_tiled`, exactly the "host-side materialization" the substrate
  decision already accepted and already pays for GQA. **`KvCache`'s storage
  order is not touched:** the row-major cache is what the scalar routine, the
  prefill loop and `key`/`val` read, and a storage-layout change is a
  whole-engine change while the gather is O(seq_kv · kv_dim) per attention call.
- **The now-dead accessors were cut, not kept behind `#[allow(dead_code)]`.**
  `KvCache::keys` / `vals` existed only to feed this call site; once it moved to
  the gather they had zero readers (grep-confirmed), so they are gone and their
  layout warning now lives on `keys_head_major`'s doc. The sibling
  `qk-flash-attention` design.md clauses that pointed at "the comment on
  `KvCache::keys`" are reworded to match — a pointer to a symbol that no longer
  exists is prose rot.
- **The scenario is a DELTA requirement, authored before the code because the
  delta had none.** A task row cannot mint a requirement, so `specs/qk-server`
  gains "Serving parity across the quantized attention face" / "Quantized-face
  token identity on the fixture pipeline" before the code changed, and the task
  row cites it. Its budget is the face's OWN gate (Cosine Sim ≥ 0.999, Relative
  L1 ≤ 0.02) with an explicit "NOT the exact face's 1e-5" clause: the INT8 face
  is lossy by design and cannot meet the exact face's reassociation budget.
  Ratification and archiving of this delta are the planner's, as for the rest of
  the change.
- **Gates on this tree, with counts** (interpreter/host: this sandbox, rustc
  1.98.1; the device runs took gfx1151 through the staged ROCm SDK). `cd
  crates/quantibbler-kernels && cargo test --offline --locked` → **24 passed / 0
  failed** (8 targets; the rest are feature-gated to 0), RC=0; kernels
  `clippy --all-targets -- -D warnings` RC=0 both with and without `--features
  gpu`; `cargo clippy -p quantibbler-server --all-targets -- -D warnings` RC=0
  both with and without `--features gpu`; `cargo fmt --all -- --check` RC=0;
  `cargo test -p quantibbler-server` → **179 passed / 0 failed / 10 ignored**
  over 15 targets, RC=0; `cargo test -p quantibbler-server --features gpu`
  (gate unset) → **180 passed / 0 failed / 10 ignored** over 15 targets, RC=0
  (`load.rs` did not flake this window). No kernels source moved, so no `uv sync
  --reinstall-package quantibbler-kernels` was owed.

## Planner rulings (2026-10-25 — task 4.2 evidence, four decisions; card t_fccd1ced)

The coder's measurement (task 4.2, 84 real Qwen3-0.6B attention cases) was
the evidence. The four decisions are:

### Decision 1: Enabling evidence — fixture-battery vs real activations

**Ruled evidence-class distinction.** The spec delta now distinguishes two
classes by `activations_kind`:

- **Fixture-battery** (`"fixture-battery"`): the committed device-parity
  battery, i.i.d. Gaussian inputs. SHALL admit the face as
  `opt-in-experimental` only, with a WARN log naming the unmeasured
  real-activation quality. The shipped record (`tests/fixtures/server-model/
  attention_gate.json`, `activations_kind: "fixture-battery"`) is
  grandfathered for the current experimental status.
- **Real-activation** (a recognized real-activation source): the class
  required for any status beyond experimental. The gate SHALL enforce
  `activations_kind` is a recognized real-activation source before
  considering promotion.

The requirement's letter already said "on captured real activations"; the
gap was that the implementation shipped fixture-battery evidence without
enforcing the source. The ruling closes the gap by distinguishing the
two evidence classes in the spec rather than silently downgrading the
requirement. The ratified evidence hierarchy: fixture-battery for
development and experimental admission; real activations for the enabling
class. The 7.4× over-statement the coder found (fixture record passing at
0.999968778 / 7.2183e-3 vs real activations failing at 0.987266 / 0.148357)
is [SA] Table 3's warning arriving in its own terms, and the ruling makes
that warning structural: a fixture-battery record CANNOT enable the face
beyond experimental, by explicit spec requirement.

### Decision 2: Face status — ratify opt-in-experimental

**Ratified.** The task-4.2 row's own rule fired correctly: 0/84 real-activation
cases pass BOTH [SA] §4.3 metric clauses (worst Cosine Sim 0.987266 vs
≥ 0.999, worst Relative L1 0.148357 vs ≤ 0.02 → 7.4× the gate).
`opt-in-experimental` with the failing numbers recorded. No further
decision needed; the measurement is unambiguous.

### Decision 3: PPL bound — policy threshold 0.05

**Ratified as a policy threshold.** The bound 0.05 was taken from the
change's own shipped gate tests (`sage-ppl-ok.json`, `{delta: 0.011,
bound: 0.05}`, task 3.3) rather than a paper or standard. The measured
real-model delta of −0.268724 (ratio 0.991422396 over 8 192 wikitext-2
test tokens) passes it in the favouring direction — the INT8 face
measures lower PPL than the exact face. The bound is recorded in the spec
delta as a policy threshold with its provenance named. A real-model PPL
bound is a policy threshold with no source in any paper requirement; the
ruling makes that explicit rather than letting a convention harden into
an unexamined constant. The bound may be tightened when a larger eval
set is run.

### Decision 4: Missing record path — compact capture format

**Ruled: a compact capture path is required, deferred to a follow-up card.**
The gate tool (`gen_attention_gate_record.py`) consumes JSON; a real-shape
case (seq 256 × 16 heads × 128 dim × 3 operands) is ~82 KB of JSON per
unit of seq, and an 84-case real battery would be orders past the repo's
500 KB large-file gate. The following are routed to coder card
`t_0ed63061`:

- Add an npz/f32-binary capture path to the gate tool (compact, versioned,
  checksummed)
- Commit a capture script `tools/capture_attention_activations.py`
  producing `tests/fixtures/attention_sage_activations.npz`
- The capture SHALL be from the healthy base Qwen3-0.6B model (not a dead
  qpk1 artifact) — consistent with the measurement already taken
- The large-file gate exemption for the real-activation record is deliberate
  and SHALL be named in the commit message and the tool's docstring
- Capture from the served qpk1 artifact (needing a capture surface in the
  Rust engine) is out of scope for that card and deferred to a separate
  change

## Open Questions

- OQ-1 (owner: orchestrator, probe = task 4.1): INT8 dot throughput on
  gfx1151 vs fp32 — resolves the speed mechanism. **RESOLVED 2026-10-25:
  the ≥ 2× assumption does not hold at the QKᵀ attention shapes
  (measured 1.06–1.64×, two runs); see the crystallization entry below.**
- OQ-2 (owner: orchestrator, probe = task 4.2): quality-gate numbers on
  real r137 activations + PPL delta — resolves ship/experimental status.
  OPEN — not measurable yet: no captured r137 activation record exists
  (`tests/fixtures/attention_sage_activations.npz` absent), the gate tool
  (task 3.3) and the INT8 kernel path (tasks 2.2/2.3) are unlanded, and the
  capture plus the eval-set PPL delta are real-model runs, which the lane
  rule holds for an explicit @user GO.
  **AMENDED 2026-10-25 (tasks 2.2/2.3 and 3.1–3.3 landed): the INT8 kernel path
  and the gate tool now exist — `tools/gen_attention_gate_record.py` mints and
  re-checks a record, and the server refuses/enables the face on it. What still
  blocks this question is unchanged and is exactly the real-model half: no
  captured r137 activation record, no eval-set PPL, both requiring an @user GO.
  The superseded sentence above is kept so the older reading cannot be quoted as
  current.**
  **RESOLVED 2026-10-25 (task 4.2 measured; @user GO given, real-model run
  taken): the face's ship-vs-experimental status is decided — `opt-in-experimental`,
  because the gate FAILS on real activations.** Two runs on captured real
  Qwen3-0.6B attention inputs (84 cases in the full-depth run): worst Cosine Sim
  **0.987266** vs the ≥ 0.999 gate and **0/84** cases meeting Relative L1 ≤ 0.02
  (worst **0.148357**, 7.4× the ≤ 0.02 gate), while the eval-set PPL clause
  passes in the favouring direction (**delta −0.268724**, ratio 0.991422396,
  8 192 wikitext-2 test tokens). The task row's own rule therefore fires:
  "recorded as opt-in-experimental with the failing numbers". The finding that
  outlives the status is that the shipped record's **fixture-battery** evidence
  (0.999968778 / 7.2183e-3) is orders of magnitude more favourable than the real
  activations the face will meet — see the task-4.2 entry. The PPL clause is
  recorded with bound 0.05 (the change's own shipped-test convention) and
  flagged for the planner as a policy threshold.
  **RULED 2026-10-25 (planner card t_fccd1ced): evidence-class distinction
  ratified.** The gate now distinguishes fixture-battery evidence (admits
  experimental only) from real-activation evidence (required for any status
  beyond experimental). The PPL bound 0.05 is ratified as a policy threshold
  with provenance recorded. See "Planner rulings" section above for all four
  decisions. OQ-2 is fully resolved — the measurement decided the status, and
  the ruling closed the spec-implementation gap the measurement uncovered.
