# Design: qk-resume-report

## Decision (the ruling this card asks for)

**Option 1 — the restart is declared in the report.** The `resume` record is a
reader-facing statement of the wall-clock discontinuity, recorded from launch
state the driver already has, at no extra solver IO. It is a record, never a
gate: the standing ruling stands — records are not controls, version control
plus the CI gates answer "did the code change", and the `ok`/`degraded`/`failed`
band remains the only per-run verdict.

## Why the value is available at no extra IO

`checkpoints.completed_blocks(ckpt_dir)` is a pure scan of `block-*.npz` stems
and keys. The driver resolves `_ckpt_dir` before `quantize_model` runs
(`_checkpoint_launcher_dir`), so reading `completed_blocks(_ckpt_dir)` at that
same point returns exactly the blocks a prior launch left on disk — the
restored set — before this launch's solve adds new archives. `resume_stage_events`
already calls `completed_blocks` on the same directory for the journal seed, so
the value is being computed regardless; this change surfaces it to the report.

## The record's shape

```json
"resume": {
  "resumed": true,
  "restored_blocks": [1, 2, 3, 4, 5],
  "checkpoint_dir": "/out/checkpoints",
  "image": "quantibbler-ptq:rocm-official-v6b",
  "image_digest": "sha256:dcd9f720…",
  "admitted_by_fingerprint": {
    "fingerprint": "9f32ae64…",
    "model_identity": {
      "digest": "a1b2c3d4…",
      "modules": {…}
    },
    "solver_backend": "torch",
    "compute_dtype": "float64",
    "solver": "admm",
    "rank": 512,
    "factor_mode": "binary",
    "admm_reg": 0.01,
    "rho_start": 1.0,
    "rho_end": 12.0,
    "solver_device": "cuda"
  }
}
```

`image`/`image_digest` are read from the container env the wrapper sets
(`QUANTIBBLER_IMAGE`, `QUANTIBBLER_IMAGE_DIGEST`); absent, they read
`"unknown"` — the exact idiom `_probe.py` already uses for the image digest
("the driver cannot inspect its own image, and a tag is mutable, so the digest
is what makes 'did both runs share one binary' answerable"). A bare
`python tools/quantize_hf_model.py` (CI smoke) has neither, and must not crash.

The `admitted_by_fingerprint` members are copied verbatim from the restored
`fingerprint.json` record: `fingerprint` (the config digest at the on-disk key
`"fingerprint"`), `model_identity` (the model-identity digest at the on-disk
key `"model_identity"`), and the provenance members — the
`_FINGERPRINT_PROVENANCE` tuple
(`solver_backend`, `compute_dtype`, `solver`, `rank`, `factor_mode`,
`admm_reg`, `rho_start`, `rho_end`) plus `solver_device`, which
`write_fingerprint` records separately at checkpoints.py:268. The three
"digest" candidates (config-fingerprint digest, model-identity digest, file
sha) are disambiguated by their on-disk keys — the coder threads the real
record rather than an invented subset.

## The fingerprint non-interaction (SHALL NOT)

`config_fingerprint(cfg)` hashes `dataclasses.asdict(cfg)` + version + `git
describe`. `check_fingerprint` refuses a resume when that digest (or the model
identity) diverges. Two hard rules follow:

1. **No resume marker in `PTQConfig`.** A `resumed` flag on the config would be
   folded by `asdict` into the digest; the pre-resume blocks were written with
   no such flag, so the resumed launch would hash differently and refuse its
   own checkpoint. The resume record is computed at report time from driver
   state, never from config.
2. **No resume marker in `_FINGERPRINT_PROVENANCE`.** Those fields are recorded
   *alongside* the digest, and some are also folded (they sit in the
   `asdict`). A resume marker belongs in neither place: the report carries it,
   the fingerprint does not.

## The "identical bytes" boundary (what the report cannot assert)

RESUME DIRECTIVE rule 4 asked for "blocks 1–5 computed under identical bytes".
That is a *host-side* fact: it was established by
`git diff e757f15 befd2ce -- src/quantibbler/ptq tools/quantize_hf_model.py`
being EMPTY, which only the host (which has `git`) can measure. The container
carries no `git`, so `git describe` folds `"unknown"` and the driver cannot
diff two tips. Two consequences:

- The report's resume record records **the admission**: which blocks were
  restored, from which checkpoint dir, under which fingerprint digest and
  provenance fields. It does **not** emit an `identical_bytes: true` boolean —
  that would be a driver-computed provenance claim the driver cannot compute
  (the same class the coder correctly refused to improvise).
- The code identity across the reboot stays in the launch trail
  (`launch.env` `commit`/`tree_dirty_files`, the launch-attribution surface).
  A reader adjudicates "identical bytes" by comparing the recorded fingerprint
  digest and provenance against the launch trail's commit — evidence, not an
  assertion.

The content-hash-over-src fold for `config_fingerprint` was proposed and ruled out
(orchestrator ruling: the record-only digest conforms; drift-refusal out of
delta). The resume record in the report records the config digest and
model-identity admission as the evidence boundary — the two digests a reader
compares against the launch trail. Content-hash-as-fingerprint is a separate
question whose answer is already settled in the negative for this change.

## Discarded alternatives

1. **Rule the declaration outside the report (option 2).** A sidecar or
   launch-line-only declaration keeps the report schema untouched, but it is a
   manually-authored, machine-unchecked file the reader must find by
   convention, and the resumed artifact stays non-self-describing. Rejected:
   the discontinuity is a property of the run the report already describes.
2. **Fold the resume marker into the fingerprint.** Rejected outright — see
   "The fingerprint non-interaction": a resumed run would refuse itself.
3. **A computed `identical_bytes` boolean in the report.** Rejected — the
   driver cannot compute it (no `git`); it would forge provenance.
4. **Stuff the declaration into `stage_map` or a journal entry.** Rejected —
   a top-level `resume` record is the reader-facing surface; stage_map is a
   per-stage Algorithm-1 map whose semantics are already pinned, and the
   journal is the stage append log.

## Byte-contract / schema-version ruling

- **`SCHEMA_VERSION` — no bump.** `QuantizeReport` stays `extra="allow"`
  (serialization.py:79-81 bumps only when "a model's shape changes in a way
  an older reader could misparse"). Old reports (including v6b) parse under
  new builds; new reports parse under old builds — the extra key is ignored.
- **The `resume` key rides as an extra on `QuantizeReport`**, not an
  enumerated model field. It follows the `QuantizeIdentity` pattern: built
  and threaded alongside identity, present on all report shapes, never
  inside a serialized-model contract.
- **`dump_report`'s minimal failed stub (`band.py:139-144`) carries no
  `resume` key.** It is a pre-existing fourth shape — an error-path stub for
  the `band` module, not a real run — and is out of scope: the resume record
  is wired through the driver's identity path, not the band module. A strict
  consumer that expects the key on every `QuantizeReport` instance should
  handle absence; the three production shapes (completion, probe-abort,
  stage-failure) are the committed contract.

## Wall-clock scope on a resumed run

`wall_seconds` on a resumed run covers the post-resume segment only. The
record's presence is the discontinuity declaration — `resumed: true` plus a
non-empty `restored_blocks` is what tells a reader the wall clock does not
span the full solve. Segment timestamps (the per-block wall time) stay
launch-trail/sidecar material; the report records the single number the
driver measured.

## Open Questions

None. The decision is fully specified above.
