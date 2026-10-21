# Stage-Trace Verdict — Qwen3-0.6B collapse RCA

Date: 2026-10-21. Owner: planner. Status: **recorded — licenses the ladder fork
(`FORK-SPEC.md`, `/opt/data/tmp/rca-stage-trace-ladder/`).**

## Artifacts and provenance

| Artifact | Path | sha256 |
|---|---|---|
| probe script (trace-producing version) | `/opt/data/tmp/rca-stage-trace/probe_stage_trace.py` | `b6135b098e1e59d32e185c76ea897bd39f3325ad96a5ab1ee935e38f02ea8b35` — recorded by the probe itself in `run.log` and in both JSONs' internal `probe_script_sha256` field |
| attn trace | `/opt/data/tmp/rca-stage-trace-attn/attn.json` | `bc73bb48bba562c3dd5f902c68e2b8bd10a1beb85c2a8e30398f34fb0abf927c` (landed run, verified on disk) |
| mlp trace | `/opt/data/tmp/rca-stage-trace-mlp/mlp.json` | `255e226c475feeebd36810f537f6418a4b9d16ba0e28742eee3727ca6a794ce4` (landed run, verified on disk) |

Probe script lineage (recorded 2026-10-21 after bug-hunter's doc check): the
file was since updated for the ladder extension — on-disk sha now
`3fa1c232142718fb8e94236a6b0ca65abd173dd79c041abb232ed009ae77910c`, whose
predecessor `e54b72a1100012c12ca072aed98350fbbac5fdb3f014f473f07c28063accc2a19`
is recorded in `FIRE-LINE.md`. The two trace rows previously listed
`b6135b09…` for BOTH files — that is the producer probe's sha copied from
the JSONs' internal provenance field, not the files' own hashes; corrected
to the disk hashes above.

`traced_code` identities recorded in both JSONs: block_recon `132b5c8c…`,
checkpoints `92027214…`, driver `85d0d0d8…`. Integrity fields: `void=None`,
`fingerprint_check=passed`, `gate0.passed=true` bit-exact on 4 (attn) / 3 (mlp)
layers, `mapping_assertions.passed=true` with `t1_quotable=true` (call counts
4/4/4 and 3/3/3), every `host_restore_bit_exact=true`. All numbers below were
re-extracted from these JSONs on disk by the planner (2026-10-21), not quoted
from the room.

## Precedence readout (frozen tree, selector owner: sec-eng)

**Precedence 1 — selected T1, and the selection is VOID as a localization.**
The earliest weight jump is T1 on all 7 scoped linears, both traces, trigger
`rel_err>=2x`. Verified on bytes: on all 7 linears `rel_err == |fro_ratio−1|`
to ≤0.01% relative — the trigger is a pure scale-magnitude reading of the
preconditioner (`w_tilde = d_out ⊙ W ⊙ d_in` vs raw `W`) with **zero
directional information**. The T1 output leg (`output_left_exact_zero`,
7.32e11 attn / 1.42e21 mlp) is the same scale-species artifact: preconditioner
magnitude measured with ones-for-scales. The tree was satisfied as written;
as a stage indictment it is empty.

**Precedence 2 — output leg.** Fired at T1 via `output_left_exact_zero` (an
event, not a ratio: PRE/T0 are exactly zero, `denominator_floor=true`). Same
scale-species confound. No Δcos trigger fired anywhere; no over-pre trigger
fired. The falsy-zero fix did its production job: T2's ratio (5.1e-13 attn /
6.8e-22 mlp) was computed against the honest T1 magnitude instead of going
blind.

**Precedence 3 — write-back.** T5 == T4 bit-exact on all 7 linears
(`cos_vs_prev = 1.0000`).

### Final verdict

**No single stage localized; distributed loss with a capacity-exonerated
signature.** Per the selector owner's ruling, the verdict is *not* "T1
selected": the T1 trigger carried no directional information, and the ×/block
lesson stands — a label that misrepresents shape is worse than no label.

## Unconfounded findings (everything T2-forward is same-scale-species as its reference)

| Stage | attn `cos_vs_pristine` | mlp `cos_vs_pristine` | attn out err | mlp out err |
|---|---|---|---|---|
| T0 (tune_fp) | 1.0000 (all) | 1.0000 (all) | 0.0 | 0.0 |
| T1 (svd warm start) | 0.379–0.549 | 0.513–0.608 | 7.32e11 | 1.42e21 |
| T2 (magnitude balance) | 0.393–0.484 | 0.386–0.475 | 0.374 | **0.967** |
| T3 (greedy polish) | 0.743–0.812 | 0.665–0.689 | 0.322 | 0.508 |
| T4 (latent STE) | 0.722–0.803 | 0.657–0.682 | 0.254 | 0.276 |
| T5 (write-back) | 0.722–0.803 | 0.657–0.682 | 0.254 | 0.276 |

1. **Branch 3 (write-back/probe plumbing) exonerated by evidence.** T5 == T4
   to the digit, and T5 `cos_vs_pristine` reproduces the independent checkpoint
   audit to four decimals (attn 0.7219–0.8034, mlp 0.6570–0.6819) from a
   different code path. The "probe measures something else" hypothesis is dead.
2. **TuneFP inert — measured, not inferred.** T0 bit-identical to pristine,
   `rel_err_vs_pristine = 0.0`, output error 0.0, on all 7 scoped linears,
   both traces. Root cause at source (`block_recon.py`): the call site passes
   `y_b = block(x_running)` and `x_b = x_running.clone()` — the teacher is the
   same module's own output on the same input, so the objective is zero at
   init, the gradient is zero, and no step is ever taken. [NQ] Sec 3.2 Step 1
   is never executed. Spec-violating; owed a fix change dir regardless of the
   fork (`qk-fix-tunefp-teacher`).
3. **The headline instrument finding.** Terminal block-local output error is
   **25.4% (attn) / 27.6% (mlp)** after every repair stage; the *failing* mlp
   block hit **96.7% at T2** vs the attn family's 37.4%, while weight metrics
   were already proven indistinguishable between pass and fail. Block-local
   output error on real activations discriminates where weight space cannot.
   Caveat carried with it: these errors are not chained through the prefix —
   the compounding claim stays qualitative until composition is measured.
4. **The warm start already sits at `cos_vs_pristine` 0.38–0.61** — directional
   damage exists before any tuning stage runs. Balance fixes scale without
   moving direction much (`cos_vs_pristine` ~flat through T2); polish is where
   the recovery toward the target happens (T3); STE moves scales, not signs.
5. **STE sign adaptation is a no-op at its nominal target.** Zero sign bits
   flip in 4,096 steps on all 7 linears while latents move 5–33%
   (`latent_rel_change_u` 0.055–0.327). The stage's U/V refinement is inert;
   only s1/s2 adapt. Owed a fix change dir (`qk-fix-ste-sign-adaptation`).
6. **Capacity ordering runs opposite to collapse ordering.** Coder's ladder
   (quoted-not-verified until bug-hunter's independent table lands): attn
   holds the better real rank-512 floor (0.93–0.97) and dies first at depth 2;
   mlp holds worse (0.87–0.89) and survives to depth 3. Terminal block-local
   errors are near-identical (25.4% vs 27.6%). Capacity is largely exonerated
   as the discriminator; **composition is the unmeasured quantity.**

## Ledger rules earned and carried

- Any trigger that selects is reported with its information content; a
  scale-dominated number is never quoted as direction.
- T2's `cos_vs_prev` (0.09–0.21 attn) is NOT quotable as "balance rotates the
  factorization off the warm start": T1 was recorded unscaled (ones for
  s1/s2) while T2 carries preconditioner + balanced scales — a scale-species
  confound. The clean T2 readouts are the output collapse and the ~flat
  `cos_vs_pristine`.
- A cheap fp-prefix replay is NOT available: `rca-rung1-fp/checkpoints/`
  block archives are 22-byte zero-key npz files (verified on disk). A
  continuous-factor replay is a re-run and is budgeted as such — it is not
  part of the ladder fork. The silent fp-resume defect is owed its own change
  dir (`qk-fix-fp-resume-guard`).

## What the trace does not answer

- Truncation loss vs preconditioner direction change at T1 (the SVD runs on
  `w_tilde`; nothing in the JSON cleanly measures rank-512 truncation loss
  against raw `W`).
- Sign price vs scale leak — where T5 sits inside the binary class. → the
  ladder fork, pre-registered in `FORK-SPEC.md`.
- Composition — chaining block-local errors through the prefix. **Pre-declared
  next regardless of which fork branch fires**; spec + pre-registration land
  after the ladder verdict.

## Fork license

This verdict licenses the ladder fork under the frozen spec at
`/opt/data/tmp/rca-stage-trace-ladder/FORK-SPEC.md` (frozen 2026-10-21; nothing
amended after fire). Verdict branches are pre-registered there with the
0.02/0.05 thresholds as ruled by sec-eng.
