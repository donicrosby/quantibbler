# v7 artifact sanity gate — VERDICT: **NOT-SANE** (as a serviceable serving artifact)

- Date: 2026-10-28 (UTC). Lane: default (sanity-gate child), card `t_6ae320a6`; parent fire `t_1382ce86`.
- Artifact: `artifacts/qwen3-0.6B-r512-v7-bf16-gpu-pertarget`
  — `quantize_report.json` sha256 `7f330b5e6baf8e663091c05216d4eb0ba3bcb6a168f39c55e24e7d5faf9ce306`,
  `manifest.toml` sha256 `4b67f07cab3fd3fabdb381aab6676227760401ffa595634310cd0bf0daf39cf7`,
  `e1.bf16.bin` sha256 `8f29acf519434862d95613b2b4f6b9d14933a5e4d16baebf8ac0b33b410acfb6`
  (all three re-measured here; all three match the fire handoff's pins exactly).
- Verdict standard: identical to v6b's gate
  (`SANITY-VERDICT-v6b-20261027.md`), with one deliberate tightening — every
  generation probe below went through the **LiteLLM gateway**, never directly to
  the model server's port (the serving-leg transport is declared in Leg 2).

## Verdict, per the card's three checks

| # | check | measured | verdict |
|---|---|---|---|
| 1 | **Loads** (both faces; dtype-declared; schema-refusing reader) | 196/196 layers deserialize under the v2 Python face, every one carrying a bias vector; scale dtype tag `float32` on all 196; byte lengths exactly the reader's own size model; sidecar byte models exact (bf16); byte-derived BPW **0.804252** = the report's `effective_bpw`; the Rust directory loader reaches `ready` (`serving_dtype="fp32"`, `load_method="directory"`); a corrupted-magic stream is REFUSED by the reader; the v1 fixture still loads through both CLI spellings | **PASS** |
| 2 | **Serves** — *"non-degenerate greedy output plus clean chat-surface responses"* | binds, answers, deterministic (byte-identical repeats); the chat surface **is available** (`/v1/chat/completions` → 200 — the v6b 400 anti-marker is gone; the `chat_template.jinja` sidecar fix works on a real artifact); **but** greedy output is numeric loops / word-loop soup on every probe, and the chat completion answers *"Name three primary colors."* with `' 2000 2000 2000 2000 …'`; the healthy model's rank-1 token (` Paris`, p=0.6687) is **rank 754** (p≈1.3e-5) in the artifact | **FAIL on the card's own criterion** |
| 3 | **Numeric sanity** vs the recorded baseline | eval PPL from the artifact bytes, in the fire's own (interpreter × device) pair, **364.4655** vs report `ppl_quant_native` **364.5538** (rel. delta **2.42e-4**); selection fingerprint **`29edf9820bb649b3` matched before quoting**; vs the recorded baseline `ppl_baseline_bf16 = 20.2068` → **ratio 18.04** | **fails every a-priori ceiling** (table below) |

The artifact is **well-formed, faithfully packed, faithfully served, and
numerically self-consistent** — the same four-fold shape as v6b. It is not a
serviceable *model*: the served weights have lost basic next-token capability,
and the eval column shows the same damage at a magnitude no declared tolerance
survives. The run's own self-verdict (`status="degraded"`, the 4×–40× band)
and this gate's serviceability bar (NOT-SANE) are consistent readings of the
same number.

## What the artifact is (measured, not quoted)

| field | value |
|---|---|
| wire | qpk1 **v2** (`REPRESENTATION_VERSION = 2`), 196/196 streams, bias present in all 196 |
| scale dtype | `float32` (dtype tag 2), all 196 — as the report declares (`scale_dtype = "float32"`) |
| remainder | `bf16` — `e1.bf16.bin` 311,164,928 B = `vocab·d_model·2` exactly; `norms.bf16.bin` 131,072 B = `((2·d_model+2·head_dim)·n_layers+d_model)·2` exactly; `weight_tying = "lm_head=e1@bf16"` (tied head, matching `tie_word_embeddings=true`) |
| byte-derived BPW | **0.804252** (packed layer bytes 44,274,272 / Σdout·din 440,401,920) — identical to the report's `effective_bpw=0.8042521158854167` |
| bias vectors | **all 196 are exactly zero** (`max|bias| = 0.0`) — the v6b-exonerated legit zero (Qwen3 linears are bias-free); recorded again here because it is the same suspicious-exact-value shape |
| shapes | 5 distinct `(dout,din,r=512)` shapes; 7 roles × 28 blocks |
| report verdict | `status="degraded"`, `paper_faithful=true`, `runtime_deviations=[]`, all fidelity conditions true |

## Dtype / device declarations from the run report (the card requires these named)

- `compute_dtype = "bf16"` — the ONE dtype the pipeline computed in, declared.
- `scale_dtype = "float32"`, `remainder_dtype = "bf16"` — the artifact storage
  dtypes, declared, and confirmed from the bytes (scale tag 2 on all 196
  streams; sidecar byte counts only fit itemsize 2).
- `regime`: baseline_eval = *"bfloat16 (checkpoint native; papers' pedestal)"*;
  solver = *"bf16, per-layer just-in-time (the float64 NumPy backend stays the
  parity oracle, unit-test-only)"*; kd_and_ste = *"model dtype"*; artifact =
  *"qpk1: ±1 bitplanes + float32 scales + bf16 remainder"*.
- Device: all 196 per-target stage events `device=cuda:0`,
  `floor_param_dtype=torch.bfloat16` (fire handoff fidelity block); the fire
  image is `quantibbler-ptq:rocm-official-v7` (digest `cc75bbaa…`, re-verified
  against the local image at gate time), torch `2.14.0+rocm10.1.0`, and this
  gate's leg-3 run observed `cuda.is_available()=True` on the same image.
- Serving face: the Rust server loads the artifact with
  `serving_dtype="fp32"` (engine activations f32 on CPU faces — the long-documented
  serving deviation; the artifact bytes remain bf16/fp32 as declared). This gate
  treated the server's f32 serving face and the torch bf16 rebuild as the two
  execution paths of the cross-face check.
- Baseline declaration: `ppl_baseline_bf16 = 20.20683127270709` on the bf16
  pedestal (the card's "fp32 baseline PPL 20.21" is the same recorded number;
  the report's declared pedestal is bf16).

## Deviation-vs-paper declarations from the run report

- `runtime_deviations = []` — no undeclared runtime deviation.
- Stage-level deviations, all DECLARED in the report's `stage_map`:
  `tune_fp` (per-target-linear cycle placement read from Figure 3; internal
  loss formula an informed assumption), `lb_admm` (the greedy bit-flip polish
  is an extra discrete stage the paper does not describe; round count 0 for
  the untuned-signs ablation), `ste` (one joint pass per block vs Algorithm 1
  line 19's per-linear prose), `tune_scales_kd` (packed binaries frozen during
  scale tuning). `phase1_statistics`, `capture`, `sign_extract_pack` carry
  `deviation=None`.
- Fire-level declared deviations (launch breadcrumbs): local HEAD `21a95f1` =
  spec-sync-only atop the executed tip `2508586`; GPU memory cap via fraction
  0.85 + 32g cgroup (no `--gpu-memory-gb` knob); the fire's server-acquisition
  check was a printed hint only — load+serve is THIS card's job (done above).

## Leg 1 — Loads: **PASS**

Python face (`quantibbler.qpk1`, `/opt/data/quantibbler/src/quantibbler/__init__.py` asserted):

- 196/196 manifest layer files exist, deserialize under **v2**, `version={2: 196}`,
  `dtypes={'float32': 196}`, every layer carries a bias vector (all exactly zero).
- Each stream's byte length equals `qpk1.expected_size(dout, din, r, dtype, with_bias=True)`
  exactly (196/196) — the reader's own size model, re-derived from the header.
- Factors exactly ±1, shapes consistent, 5 distinct shapes at r=512, 7 roles × 28 blocks.
- Sidecar byte models exact (table above); tied-head paths equal.
- Byte-derived BPW `8·44,274,272 / 440,401,920 = 0.804252` — identical to the report.
- **Negative control (new this gate):** a copied first-layer stream with the
  magic corrupted to `XXXX` is REFUSED — `Qpk1Error: bad magic b'XXXX';
  expected b'QPK1'`. The schema-refusing property is demonstrated on real
  bytes, not only asserted by suite fixtures.

Rust face (directory loader, `quantibbler-server` release binary rebuilt at
trunk tip `a536ba6`):

```
INFO quantibbler_server::api: model loaded model_id=Qwen-Qwen3-0.6B vocab_size=151936 d_model=1024
     artifact=/artifact serving_dtype="fp32" load_method="directory"
INFO quantibbler_server::api: ready addr=0.0.0.0:8137 model_id=Qwen-Qwen3-0.6B
/health  -> {"attention_face":"cpu:scalar","available_permits":4,"matvec_face":"cpu",
             "model":"Qwen-Qwen3-0.6B","ready":true}
/v1/models -> {"data":[{"created":…,"dtype":"fp32","id":"Qwen-Qwen3-0.6B",
             "object":"model","owned_by":"quantibbler"}],"object":"list"}
```

Dual-version regression leg: the artifact is v2, and the **v1 (pre-bias)
reader was still exercised** on `tests/fixtures/server-model` (v1 bytes) —
loads through **both** CLI spellings (`--artifact` silent; `--model-dir` emits
the deprecation warning, then the same load line). No v1 *packed artifact*
exists anywhere in `artifacts/` (unchanged since the v6b verdict recorded the
same), so the v1 leg remains fixture-only by construction.

## Leg 2 — Serves: **FAIL** on quality; the v6b blocking ground is FIXED

Transport (declared, per the card's serving discipline): the server ran as a
throwaway container on the `litellm` docker network
(`qkb-sanity-v7-serve`, launch label `qk-v7-sanity-serve`); it was registered
in the LiteLLM gateway **at runtime** (`store_model_in_db`, no restart —
`quantibbler/v7-gate`, model id `0eb85bc5-a5ee-4ad1-8a99-04b144afe2cb`) and
**every** probe below went gateway → server (`localhost:4000` inside the
litellm container), never directly to the server's port. Registration was
deleted and the container/image removed after the leg; zero residue.

**The chat surface works.** `/v1/chat/completions` → **200** with content —
v6b's blocking ground A (template sidecar unread → 400 on every real artifact)
is fixed at trunk and verified on a real fire artifact for the first time.

**Ground — the text is degenerate.** `/v1/completions`, `temperature=0`,
`max_tokens=16`, every body retained:

| probe | prompt | gateway output (verbatim) | healthy bf16 checkpoint |
|---|---|---|---|
| p1 | `The capital of France is` | `' a 2000-12000-120'` | `' Paris. The capital of France is also the capital of the French Republic. The'` |
| p2 | `Q: What is 17 times 23?\nA: ` | `',, as   and may  , "  and\n\n\n the '` | `'391\nA: 391\nA: 39'` (391 = correct) |
| p3 | `def add(a, b):` | `' 2000, 2000, 200'` | — |
| p4 | `ha ha ha ha ha ha ha ha` | `' Express express express express express …'` (16/16) | `' ha ha ha ha ha ha ha ha ha …'` (echo) |
| chat | *"Name three primary colors."* | `' 2000 2000 2000 2000 2000 2000 2'` | — |

Next-token distribution after `The capital of France is`: the artifact puts
` Paris` at **p≈1.3e-5, rank 754**; the healthy bf16 checkpoint puts it at
**p=0.6687, rank 1** (v6b artifact: rank 2545). Determinism: the same request
twice → byte-identical bodies. Looping: numeric-loop attractors on p1/p3/chat
(the `2000/12000` family), word-loop on p4, whitespace/punctuation soup on p2
— the same damage shapes the v6b verdict catalogued, on different surface
tokens.

**The serving path is faithful to the artifact** (cross-face check, second
execution path = torch rebuild of the artifact bytes on `cuda:0` bf16, same
prompts): p1, p3, p4 greedy output **identical** between the f32-CPU server
and the bf16-GPU torch rebuild (p3 and p4 16/16 tokens); p2 diverges exactly
where the degenerate near-flat distribution makes top-2 a near-tie — the
v6b-noted symptom of a flat distribution, not evidence of a server defect.

## Leg 3 — Numeric sanity: the artifact reproduces its own number; the number fails every ceiling

Independently re-measured **from the artifact bytes**, in the fire's own
(interpreter × device) pair — `quantibbler-ptq:rocm-official-v7`
(`cc75bbaa…`), torch `2.14.0+rocm10.1.0`, `cuda:0` — using the **driver's own**
evaluator (`tools/quantize_hf_model.py`: `find_wikitext_shards`,
`load_wikitext_text`, `_tokenize_blocks`, `perplexity_from_logits`) with the
model rebuilt from the artifact alone (196 qpk1 streams → `diag(s1)·Ub·Vbᵀ·diag(s2) + bias`
product in float64 → bf16, bf16 sidecar embeddings/norms read as their own
bytes; `load_state_dict(unexpected=[])`, every parameter explicitly set —
nothing left at random init). Launch `qk-v7-sanity-gpu1`, announced on the
card before firing; `--rm`, read-only repo/artifact mounts, GPU device
discipline (`/dev/kfd`, `/dev/dri`, kfd gid), witness log retained.

| quantity | value |
|---|---|
| eval set | 144 seqs (18 batches × 8 × 2048), `source_tokens=298938`, `full_blocks=145` |
| **selection fingerprint** | **`29edf9820bb649b3` — identical to the report's** `evaluation.selection_fingerprint` (assertion in the script; a drift aborts before any number is quoted) |
| eval PPL (from the artifact) | **364.465502** |
| report's `ppl_quant_native` | 364.553764 |
| relative delta | **2.42e-4** (v6b reproduced to 3.5e-5; v7's slightly larger delta is consistent with bf16 rounding-order differences between the pipeline's live dequant path and the rebuild's f64-product-then-cast — both readings identify the same damaged model) |
| recorded baseline (cited, **not** re-run) | `ppl_baseline_bf16 = 20.20683127270709` |
| **ratio vs the recorded baseline** | **18.037** (report: 18.041) |
| wall | 33 s for all 18 batches |

So: the packed artifact **is** the model the fire evaluated — the export
round-trip is faithful at the measurement's precision. Nothing about the
packing, the sidecars, the wire version, or the eval protocol is in doubt.

**The tolerance, named before the measurement (fail-closed discipline).** The
ceilings below were fixed from the pinned papers, the run's own declared
bands, and the card's wording BEFORE this gate measured its own PPL number:

| ceiling (all stateable a-priori) | value | artifact at 364.47 |
|---|---|---|
| the pinned paper's 0.80-bit row + its 1.15 tolerance (`qk_bar.py`: row 33.79 → 38.8585) | 38.86 | **9.38× over** |
| the paper's 1.00-bit row + tolerance (27.56 → 31.694) — as good as a *higher*-bpw row | 31.69 | **11.50× over** |
| 2× the recorded baseline ("no worse than doubled" fence) | 40.41 | **9.02× over** |
| the run's own `ok` band boundary, 4.0× (the loosest band a SANE verdict could ride; the card's "declared tolerance of the baseline") | 80.83 | **4.51× over** |
| the run's `degraded` band top, 40.0× (quoted only to place the run's self-verdict) | 808.27 | inside (the run says DEGRADED, not FAILED) |

The only tolerance that would admit 364.47 is "≈18× the baseline", which is a
restatement of the measurement, not a serviceability criterion. This is not a
`qk_bar` row verdict (out of scope, and `effective_bpw=0.8043` sits just above
the 0.80-bit budget anyway).

## The dtype-mixing hypothesis verdict (the card's central question)

**v7 NOT-SANE at bf16+GPU with parity-clean cells → the dtype-mixing
hypothesis is FALSIFIED as the sole damage cause; fp32/mixing is exonerated;
the damage question REOPENS.**

- The fire executed the declared regime: bf16 end-to-end, all 196 per-target
  stages on `cuda:0`, zero fp32 leak in the run path (fire handoff fidelity
  block), `runtime_deviations=[]`, all fidelity conditions true.
- This gate verified independently that the damage is IN the artifact and is
  faithfully carried by every execution face around it: the packer round-trips
  (2.4e-4), the loader accepts, the server serves faithfully (cross-face
  token-identical), and the eval column reproduces the fire's number.
- The ratio at bf16+GPU (18.04×) is not better than v6b's fp64-compute run
  (14.14×) — it is worse. Whatever the damage class is, it is *not* the
  solver's fp64 compute mixing with bf16 weights: removing that mixing did not
  repair the artifact. The hunt reopens with fp32/mixing excluded, on a
  damage class that survives BOTH compute regimes and BOTH devices.

## What this verdict does and does not say

- **Does say:** the container is well-formed and faithful end to end (packed
  bytes → server → eval), the chat surface fix is real, the serving-leg
  gateway discipline works end to end on a real artifact, and **this artifact
  must not be promoted into the benchmark program as shipped** — the served
  text is degenerate/looping against a healthy reference on identical prompts,
  and the eval number fails every a-priori ceiling.
- **Does not say:** that the packer is broken, that the loader is at fault,
  that the engine mis-executes the artifact, that the chat-template fix is
  incomplete, or that a bar row was missed. The engine is faithful; the loader
  accepts v1 and v2; the packer round-trips to 2.4e-4.
- **Does not say** what the damage IS. The gate's exclusions aim the reopened
  hunt: it lives in the run's numerics (solver/tune/STE class, active under
  both fp64-compute and bf16-compute regimes — cf. the fire's own solver-probe
  telemetry: continuous STE moves below bf16 resolution while discrete flips
  proceed, per-target cos 0.73–0.86), not in the dtype mix, not in the
  packing, and not in the serving path.

## Routing (nothing implemented here; this card is read-only + verdict)

1. **Damage hunt reopens — owner: planner (+ @user ruling on the next fire's
   suspect class).** The dtype/device axis is exhausted (both regimes damaged,
   bf16+GPU worse). Candidate classes the evidence already points at are the
   solver/STE numerics and the tune stages' interaction with the per-target
   cycle — but choosing the next isolation run is a ruling, not a coder
   choice. The v7 fire's `solver_probe_context` (inert STE rows, cos range)
   is telemetry the hunt should read first.
2. **Bench program stays gated** on a SANE artifact: do not start
   chat-format/comparison cards on this artifact.
3. **Chat-surface fix verified on a real artifact** (v6b Ground A): the
   fixture-only-coverage hole that hid it is closed on the serving side; the
   real-artifact regression legs now exist (this gate's legs), and future
   exporter/loader changes to the template path should reuse them.
4. **Serving-topology precedent (recorded, not ruled):** this gate registered
   the artifact server in LiteLLM at runtime (`store_model_in_db`) and probed
   only through the gateway, per the user's architecture invariant. If the
   bench program wants the same shape, the register/probe/cleanup sequence in
   `serve_probe.py` is the reusable form.

## Evidence index (all under `/opt/data/tmp/t_6ae320a6/`, nothing written to the artifact)

| path | what |
|---|---|
| `load_check.py` + `load_check.out` | leg 1 (Python face 196/196, negative control, sidecar models) |
| `cargo-release.log` | server rebuilt at trunk tip `a536ba6` (RC=0) |
| `v1fix_artifact.log`, `v1fix_model_dir.log` | dual-version regression leg (v1 fixture, both CLI spellings) |
| `serve_probe.py`, `serve_probe.json`, `serve_probe.log` | leg 2 via the LiteLLM gateway; verbatim bodies; determinism; chat 200 |
| `artifact_model.py`, `gpu_leg.py`, `gpu1.log` | cross-face rebuild + leg 3 (fingerprint assert, 18-batch eval, greedy/top-k, Paris rank) |
| `ref_leg.py` + `ref_check.json` | healthy-bf16 reference readings (greedy/top-5 + Paris only — **not** a baseline re-measurement) |
| `ppl_result.json`, `gen_check.json` | leg 3 + cross-face numbers as parsed JSON |
| `GATE-DIGEST.json` | this gate's own digest (the v6b gate's evidence form) |

Launches: `qk-v7-sanity-gpu1-<ts>` (GPU legs; announced on the card before
firing) and `qk-v7-sanity-serve` (serving container on the litellm network);
both `--rm`, no fixed-name reuse, labeled `qkb_launch_card=t_6ae320a6`.

## Not done here (explicit)

- No write to the artifact, its `checkpoints/`, its stage records, or the
  report (read-only mounts; verified).
- No re-run of the baseline PPL (cited from the report as instructed) and no
  CPU baseline eval; the healthy-checkpoint readings are greedy/top-5 + Paris
  rank only.
- No `qk_bar` row verdict and no full-model-quality claim beyond the card's
  serviceability criterion.
- No code or spec change: the reopened damage hunt and the serving-topology
  precedent are reported, not implemented.
- No commit: this verdict doc is left untracked in the shared checkout for the
  orchestrator (the leaf does not land; landing is the orchestrator's).
