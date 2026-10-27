# v6b artifact sanity gate — VERDICT: **NOT-SANE** (as a serviceable serving artifact)

- Date: 2026-10-27 (UTC). Lane: coder, card `t_b6abb772`.
- Artifact: `artifacts/qwen3-0.6B-r512-admm-paperfaithful-v6b-pertarget`
  — `quantize_report.json` sha256 `d776313d1568829ed71fba3164556d30fd24a11420013b5724c3f493d93aa634`
  (matches the fire handoff's claim, re-measured here).
- Doc location note: this card's body named `openspec/changes/qk-v6-fire/`, which does
  not exist in the tree (`git log --all --diff-filter=A` shows no such path; `openspec list`
  has no such change). The per-target cycle's own `tasks.md` says *"The change stays alive as
  the tracking vehicle for this fire"*, so the verdict lands in `qk-pertarget-cycle/`. A bare
  change dir with only a verdict doc was NOT created: it would put an unvalidatable change into
  the tree that every lane's `openspec validate --strict` gate would then see.

## Verdict, per the card's three checks

| # | check | measured | verdict |
|---|---|---|---|
| 1 | **Loads** (both faces; the dual-version reader) | 196/196 layers deserialize under the v2 Python face, every one carrying a bias vector; byte lengths exactly the container's own size model; the Rust directory loader accepts the artifact and reaches `ready`; the v1 (pre-bias) fixture also loads, through both CLI spellings | **PASS** |
| 2 | **Serves** — *"coherent, non-degenerate output (no empty/looping/garbage generations…)"* | binds, answers, deterministic (byte-identical repeats); **but** greedy output is punctuation soup / numeric loops on every probe; the healthy model's rank-1 token (` Paris`, p=0.669) is **rank 2545** in the artifact | **FAIL on the card's own criterion** |
| 3 | **Numeric sanity** vs the recorded FP baseline | eval PPL from the artifact **285.7864** vs report **285.7764** (rel. delta 3.5e-5 — exact reproduction), vs the *recorded* baseline `ppl_baseline_bf16 = 20.2068` → **ratio 14.14** | **fails any justified tolerance** (see below) |

The artifact is **well-formed, faithfully packed, faithfully served, and numerically self-consistent**.
It is not a serviceable *model*: the served weights have lost basic next-token capability, and the
recorded 14.14× PPL ratio is the same damage seen from the eval side.

## What the artifact is (measured, not quoted)

| field | value |
|---|---|
| wire | qpk1 **v2** (`REPRESENTATION_VERSION = 2`), 196/196 streams, bias present in all 196 |
| scale dtype | `float32` (dtype tag 2), all 196 |
| remainder | `fp32` — `e1.f32.bin` 622,329,856 B (`vocab*d_model*4`, tied head), `norms.bin` 262,144 B (`(2*d_model+2*head_dim)*n_layers+d_model)*4`) |
| byte-derived BPW | **0.804252** (packed layer bytes 44,274,272 / Σdout·din 440,401,920) — exactly the report's `effective_bpw` |
| bias vectors | **all 196 are exactly zero** (`max|bias| = 0.0`), so the server's bias term is a no-op for this artifact |
| shapes | 5 distinct `(dout,din,r=512)` shapes; 7 roles × 28 blocks |
| report verdict | `status="degraded"`, `paper_faithful=true`, `runtime_deviations=[]`, 16/16 fidelity conditions true |

## Leg 1 — Loads: **PASS**

Python face (`quantibbler.qpk1`, `/opt/data/quantibbler/src/quantibbler/__init__.py` asserted):

- 196/196 manifest layer files exist, deserialize under **v2**, `version={2: 196}`,
  `dtypes={'float32': 196}`, **every** layer carries a bias vector (and all are exactly zero).
- Each stream's byte length equals `qpk1.expected_size(dout, din, r, dtype, with_bias=True)`
  exactly (196/196) — the reader's own size model, re-derived from the header.
- Factors are exactly ±1 (`set(unique) ⊆ {-1, +1}`), shapes consistent (`s1:(dout,)`,
  `s2:(din,)`, `Ub:(dout,r)`, `Vb:(din,r)`), 5 distinct shapes at r=512, 7 roles × 28 blocks.
- Byte-derived BPW over the packed layers `8·44,274,272 / 440,401,920 = 0.804252` — identical to
  the report's `effective_bpw=0.8042521158854167`.

Rust face (directory loader, `crates/quantibbler-qpk1` via `quantibbler-server`, binary rebuilt at
HEAD `7a5634a` — the pre-existing debug/release binaries predated it):

```
INFO quantibbler_server::api: model loaded model_id=Qwen-Qwen3-0.6B vocab_size=151936 d_model=1024
     artifact=artifacts/qwen3-0.6B-r512-admm-paperfaithful-v6b-pertarget serving_dtype="fp32" load_method="directory"
INFO quantibbler_server::api: ready addr=127.0.0.1:8137 model_id=Qwen-Qwen3-0.6B
$ curl -s .../health  -> {"attention_face":"cpu:scalar","available_permits":4,"matvec_face":"cpu","model":"Qwen-Qwen3-0.6B","ready":true}
$ curl -s .../v1/models -> {"data":[{"created":1793089804,"dtype":"fp32","id":"Qwen-Qwen3-0.6B","object":"model","owned_by":"quantibbler"}],"object":"list"}
```

Dual-version regression leg (the v6 artifact question): this artifact is **v2**, so scope 1's
"predates qk-bias-carry" premise does not apply. The **v1 (pre-bias) reader was still exercised**,
on the only v1 bytes that exist: `tests/fixtures/server-model` (first layer header: magic `QPK1`,
version 1, dtype tag 1) loads through **both** CLI spellings —
`--artifact` (silent) and `--model-dir` (deprecation warning, then the same load line). No v1
*packed artifact* exists anywhere in `artifacts/` (every r137/r512 artifact predates the packer),
so the bench program's "v1 = the v6 artifact" leg can only ever be fixture-based; that premise is
**stale in `t_72781a70`'s body** and is carried as a finding below.

## Leg 2 — Serves: **FAIL** (two independent grounds)

Server: `quantibbler-server` (release, HEAD `7a5634a`), directory artifact, CPU faces
(`matvec_face=cpu`, `attention_face=cpu:scalar`), greedy decoding, fixed probe set,
every response body retained in `serve_probe.json`.

**Ground A — the chat surface is not available on this artifact at all (400).**
`/v1/chat/completions` on the loaded v6b artifact:

```
HTTP 400
{"error":{"code":null,"message":"the loaded model declares no chat template; /v1/chat/completions is unavailable","param":null,"type":"invalid_request_error"}}
```

Measured cause (not inferred): the landed BS-5/S5 chat surface reads the model's Jinja2 template
from the artifact's **`tokenizer.json`** (`crates/quantibbler-server/src/api/chat_template.rs` +
`model/mod.rs:409-454`), and this artifact's `tokenizer.json` carries **no** `chat_template` field
(and neither does its `tokenizer_config.json`). The template that *does* ship with the artifact is
the standalone **`chat_template.jinja`** (4,168 B) that `tokenizer.save_pretrained()` writes under
transformers 5.20.1 — a file the loader never reads. The driver exporter
(`tools/quantize_hf_model.py`) never mentions `chat_template` (zero hits) and never injects the
template into the JSON the loader parses. Every committed server fixture DOES carry it in
`tokenizer.json` (1 hit each in `server-model`, `server-model-bias`, `server-model-bf16`), so
`cargo test -p quantibbler-server -- chat_template` is green while **no real fire artifact can
serve chat**. The template bytes ARE in the artifact — the Python side renders them from the same
directory (`AutoTokenizer.from_pretrained(<artifact>).apply_chat_template([{role:user, content:
"Name three primary colors."}], add_generation_prompt=True)` →
`'<|im_start|>user\nName three primary colors.<|im_end|>\n<|im_start|>assistant\n'`, `gen_check.py`) —
so this is a *file-location* gap, not missing content. Note also that the reviewed delta's wording
scopes itself to the fixture ("the template string from the fixture's `chat_template` field") and
the archived `openspec/specs/` carries no chat-template requirement at all, so nothing in the
requirement set as written would fail on a real artifact: that is the hole, and it is a
spec-and-rails question for the planner. This is the fixture-only-coverage shape, and it is the
*primary* corpus of the
benchmark program (`t_68ff4477` replays Lemonade's chat-format scenarios; `t_72781a70`'s addendum
requires the model's own Qwen3 chat template + prompt-token parity).

**Ground B — text completions are degenerate/looping, measured against the healthy model.**
`/v1/completions`, `temperature=0`, `max_tokens=16`:

| probe | prompt | server output | artifact-rebuilt torch greedy (same prompt) | healthy bf16 checkpoint |
|---|---|---|---|---|
| p1 | `The capital of France is` | `', ,....攸.  .. th. '` | `', ,....攸.  ..攸, '` (first 12 tokens identical) | `' Paris. The capital of France is also the capital of the French Republic. The'` |
| p2 | `Q: What is 17 times 23?\nA: ` | `'23. 23. 23. 23. '` | `'12000, 12000, 12'` | `'391\nA: 391\nA: 39'` (391 = correct) |
| p3 | `def add(a, b):` | `'  constructs in    .. and\n dispute, 20'` | `'  constructs in    ...\n安全,   '` (first 4 identical) | — |
| p4 | `ha ha ha ha ha ha ha ha` | `' Question 2000, 2000, 20'` | `' Question 2000, 2000, 20'` (**16/16 identical**) | `' ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha'` |

Next-token distribution after `The capital of France is`:

| | top-1 | top-2 | top-3 | `Paris` |
|---|---|---|---|---|
| artifact | `','` 0.2381 | `' the'` 0.1854 | `' '` 0.1274 | **rank 2545**, p≈0 |
| healthy bf16 | `' Paris'` **0.6687** | `' located'` 0.0276 | `' the'` 0.0202 | rank 1 |

Determinism: the same request twice → byte-identical bodies (`identical_text=True`). Empty output:
none. Looping: 3 of 4 completion probes lock into a repeated numeric fragment (`23. 23. 23. 23. `,
`2000, 2000, 20`, `12000, 12000, 12`) or into whitespace/punctuation soup (`' '` is top-1 at 0.485
after `def add(a, b):`). Shape checks read the response text, not a vibe.

Two readings that will matter to the reviewer, both measured:

- **The serving path is faithful to the artifact.** On p1 the CPU server and the GPU
  torch-rebuild agree on the first 12 tokens and on p4 on all 16/16; they diverge exactly where the
  artifact's top-2 masses are within ~1e-5 (`'1'` 0.32022 vs `'2'` 0.32022 on p2; `攸` vs `,` on p1).
  A near-tie flip between two numerically different execution paths is itself a symptom of a
  degenerate, near-flat distribution — it is not evidence of a server defect.
- **The damage is in the artifact, not in the server.** Independently: the artifact-rebuilt torch
  model reproduces the fire's own in-pipeline eval number to 3.5e-5 (leg 3).

## Leg 3 — Numeric sanity: the artifact reproduces its own number exactly; the number itself is not serviceable

Independently re-measured, **from the artifact bytes**, in the fire's own (interpreter × device)
pair — `quantibbler-ptq:rocm-official-v6b` (`sha256:dcd9f720…`), torch `2.14.0+rocm10.1.0`,
`cuda.is_available()=True`, device `cuda:0` — using the **driver's own** evaluator
(`tools/quantize_hf_model.py` imported by path: `find_wikitext_shards`, `load_wikitext_text`,
`_tokenize_blocks`, `perplexity_from_logits`) with the model rebuilt from the artifact alone
(196 qpk1 streams → `diag(s1)·Ub·Vbᵀ·diag(s2) + bias` in float64 → bf16, embeddings/norms from the
fp32 sidecars; `load_state_dict(unexpected=[])` and every parameter explicitly set — nothing left
at its random init):

| quantity | value |
|---|---|
| eval set | 144 seqs (18 batches × 8 × 2048), `source_tokens=298938`, `full_blocks=145` |
| **selection fingerprint** | **`29edf9820bb649b3` — identical to the report's** `evaluation.selection_fingerprint` (assertion in the script; a drift aborts before any number is quoted). The artifact's own `tokenizer.json` also reproduces the fire's token stream, since the fingerprint matches |
| eval PPL (from the artifact) | **285.786356** |
| report's `ppl_quant_native` | 285.776444 |
| relative delta | **3.47e-5** (the log/exp round-trip of the single-pass aggregation) |
| recorded FP baseline (cited, **not** re-run) | `ppl_baseline_bf16 = 20.20683127270709` |
| **ratio vs the recorded baseline** | **14.1431** (report: 14.1426) |
| wall | 44.9 s for all 18 batches |

So: the packed artifact **is** the model the fire evaluated — the export round-trip is bit-faithful
at the measurement's precision. Nothing about the packing, the sidecars, the wire version, or the
eval protocol is in doubt.

**The declared tolerance, and why it is missed.** The card asks for a tolerance justified from the
numbers. Three candidate ceilings, all missed:

| ceiling | value | artifact at 285.79 |
|---|---|---|
| the pinned paper's own 0.80-bit row + its 15 % row tolerance (`qk_bar.py`: row 33.79, tol. 38.86) | 38.86 | **7.35× over** |
| the paper's 1.00-bit row + tolerance (nanoquant 27.56 → 31.69) — i.e. "as good as a *higher*-bpw row would have to be" | 31.69 | **9.02× over** |
| 2× the recorded FP baseline (a "no worse than doubled" serviceability fence) | 40.41 | **7.07× over** |
| the loosest fence this verdict is willing to name: the E2E band's own `ok` boundary, 4.0× | 80.83 | **3.54× over** |

Every one of those is a number I can state *before* looking at the artifact. The only tolerance that
would admit 285.79 is "≈14× the FP baseline", which is not a serviceability criterion — it is a
restatement of the measurement. **This is not a bar verdict** (`qk_bar` violation is explicitly out
of scope here, and the artifact's `effective_bpw=0.8043` also sits just above the 0.80-bit row's
budget, so it would not be claimed against that row anyway); it is the observation that the same
damage that makes the text degenerate is visible in the eval column, at a magnitude no a-priori
ceiling survives.

## What this verdict does and does not say

- **Does say:** the container is well-formed and faithful end to end (packed bytes → server →
  eval), and **this artifact must not be promoted into the benchmark program as shipped**, on two
  objective grounds — the chat surface it needs is unavailable (400), and the served text is
  degenerate/looping against a healthy reference on identical prompts.
- **Does not say:** that the packer is broken, that `qk-bias-carry`'s loader is at fault, that the
  serving engine mis-executes the artifact, or that a bar row was missed. The engine is faithful;
  the loader accepts both wire versions; the packer round-trips to 3.5e-5.
- **Conflict of record, named rather than resolved.** The card's serves criterion is explicit
  ("no empty/looping/garbage generations") and is contradicted by the measurement; the same card
  also says "NOT the recreation bar … no full-model-quality claim", which reads as pre-discounting
  quality. I implemented the letter (the explicit criterion) and name the delta: **if the user's
  intent was a throughput-only benchmark, grounds B and C (quality) are survivable and the artifact
  is structurally sufficient — but ground A (no chat template → `/v1/chat/completions` 400) is not
  survivable**, because the comparison's primary corpus and its prompt-token parity assertion are
  chat-format. Fixing ground A is independent of the quality question.

## Routing (nothing implemented here; this card is read-only + verdict)

1. **Ground A — chat template (blocks the bench program as specified; owner: planner/spec).**
   The exporter writes the template to `chat_template.jinja` (the transformers-5.20.1 sidecar) while
   BS-5/S5 reads it from `tokenizer.json`. Two candidate fixes, and which one is right is a ruling,
   not a coder choice: (i) **exporter-side** — inject the `chat_template` string into the exported
   `tokenizer.json` (what BS-5 assumes is already true, and what all three fixtures do); or
   (ii) **loader-side** — fall back to `<model-dir>/chat_template.jinja` when the JSON has no field.
   Either way the regression must be a **real-artifact** leg, not a fixture leg: the fixtures were
   hand-made with the field present, which is exactly why this shipped green.
2. **Ground B/C — artifact quality (owner: @user ruling).** Refire after PTQ damage is reduced, or
   accept the artifact as a throughput probe. The comparison at `t_68ff4477` already captures the
   quantized leg's raw generations for eyeball checks; if it proceeds, the degenerate-output finding
   belongs in the report's methodology block, not in a footnote.
3. **Stale premise in `t_72781a70`** (bench card): "MUST accept both wire versions — v1 = the v6
   artifact, v2 = post-bias-carry". There is **no packed v1 artifact** in the tree and the gated
   artifact is v2; the v1 leg can only be fixture-based (`tests/fixtures/server-model`, version 1).
   Its task text should say so, or the bench's v1 coverage claim will be unbacked.
4. **Do not start** `t_72781a70` / `t_68ff4477` on this artifact until (1) is resolved and (2) is
   ruled on. This card is blocked (`needs_input`) rather than completed, so the kernel does not
   promote the benchmark children.

## Evidence index (all under the shared checkout / scratch, nothing written to the artifact)

| path | what |
|---|---|
| `artifacts/qwen3-0.6B-r512-admm-paperfaithful-v6b-pertarget/` | artifact, read-only; `quantize_report.json` sha256 `d776313d…`, `manifest.toml` sha256 `4317fd51…` |
| `/opt/data/tmp/t_b6abb772/load_check.py` + output | leg 1 (Python face, 196/196) |
| `/opt/data/tmp/t_b6abb772/{cargo-build.log,cargo-release.log}` | server rebuilt at HEAD `7a5634a` (debug + release) |
| `/opt/data/tmp/t_b6abb772/server-release.log` | leg 1b load log (`load_method="directory"`) |
| `/opt/data/tmp/t_b6abb772/serve_probe.py`, `serve_probe.json`, `serve_probe2.log` | leg 2 probe set, verbatim bodies, determinism |
| `/opt/data/tmp/t_b6abb772/v1_fixture_check.py`, `v1_fixture_check.json` | dual-version regression leg (v1 fixture, both CLI spellings) |
| `/opt/data/tmp/t_b6abb772/{artifact_model.py,gen_check.py,gen_check.json,gen_check.log}` | cross-face greedy + top-k from the artifact on `cuda:0` |
| `/opt/data/tmp/t_b6abb772/{ref_check.py,ref_check.json}` | healthy-bf16 reference readings (greedy/top-5 only — **not** a PPL/baseline re-measurement) |
| `/opt/data/tmp/t_b6abb772/ppl_from_artifact.py`, `ppl_result.json`, `ppl_run.log` | leg 3, in the fire's own image/device |

Launches: `qk-v6b-sanity-ppl-20261027T0828Z` (announced on the card before firing),
`qk-v6b-sanity-gen-20261027T0840Z`, `qk-v6b-sanity-fpref-20261027T0845Z`; all `--rm`, read-only
mounts, no fixed container name.

## Not done here (explicit)

- No write to the artifact, its `checkpoints/`, its stage records, or the report (read-only mounts;
  verified: no file under the artifact was modified).
- No re-run of the FP baseline PPL (cited from the report as instructed) and no CPU FP eval; the
  healthy-checkpoint readings are greedy/top-5 only.
- No `qk_bar` row verdict and no full-model-quality claim.
- No code or spec change: the chat-template gap and the stale bench premise are reported, not fixed
  (read-only card; both are rulings for the planner/user).
- No commit: this verdict doc is left untracked in the shared checkout for the orchestrator
  (the leaf does not land; landing is the orchestrator's).
