# Proposal: qk-resume-report

## Why

A resumed fire is a recurring production shape. The v6b re-fire (`t_d680fd2c`)
was killed mid-block-6 by a host reboot and resumed from checkpoints: blocks
1–5 were computed pre-reboot at tip `e757f15`, blocks 6–28 plus Phase-3 KD were
computed post-resume at `befd2ce`, from the same image
(`quantibbler-ptq:rocm-official-v6b`, digest `sha256:dcd9f720…`).

Orchestrator RESUME DIRECTIVE rule 4 required the run report to declare that
restart (wall-clock discontinuity, checkpoint source, image tag, identical
bytes for the pre-resume blocks). The driver (`tools/quantize_hf_model.py` /
`qk_report`) emits no resume field: measured on the completed v6b artifact,
`grep -i 'restart|resume|reboot|discontinu'` over `quantize_report.json`
returns zero hits and no such key exists in the report's schema. The coder
correctly refused to hand-edit the generated report (its sha256
`d776313d…` is itself quoted as evidence) and wrote a sidecar
`RESUME-DECLARATION.md` instead. The gap is a missing surface, not a defect:
the report is the machine-generated, provenance-complete, sha256-quoted
artifact, and a resumed run is a discontinuity in the very wall-clock the
report's `wall_seconds` names. A sidecar is a manually-authored file a reader
must know to look for, and nothing machine-checks its presence.

This change rules **option 1**: the restart is declared **in the report**, as a
record for a reader, never a gate or a lock.

## What Changes

- A top-level `resume` record in `quantize_report.json` (ADDED requirement on
  `qk-ptq`), emitted by the real-model driver, carrying:
  - `resumed` — whether the launch restored blocks from a prior launch;
  - `restored_blocks` — the completed blocks present at launch (the
    `checkpoints.completed_blocks` scan), empty for a fresh run;
  - `checkpoint_dir` — the directory the blocks were restored from;
  - `image` / `image_digest` — from `QUANTIBBLER_IMAGE` /
    `QUANTIBBLER_IMAGE_DIGEST` (the wrapper already sets both; `"unknown"`
    when unset, matching the existing `_probe.py` image-digest idiom);
  - the restored checkpoint's `fingerprint.json` admission — its `fingerprint` digest, its `model_identity` digest, and its provenance members (the `_FINGERPRINT_PROVENANCE` tuple plus the separately recorded `solver_device`) — as evidence a reader compares, not an assertion the driver computes.

- The record is informational. It changes no verdict, no exit code, no band
  edge, and no gate. It is present on all three report shapes (completion,
  probe-abort, stage-failure) so the key set stays single-shaped.

## Impact

- **Affected spec**: `qk-ptq`, ADDED requirement + scenarios only. The existing
  resume requirements (fail-closed on empty archives, bind model identity,
  refuse mis-shaped restore) are untouched.
- **Affected code**: `tools/quantize_hf_model.py` (build the record from launch
  state), a torch-free builder in `src/quantibbler/qk_report/` (lean-tested),
  one new test module. No checkpoint-format, qpk1, kernel, or server change.
- **Not invariant-touching**: no storage format, kernel contract, or server
  surface changes.

## Non-goals

- **No fingerprint change.** The resume marker is never folded into
  `config_fingerprint`, never added to `PTQConfig`, and never added to
  `_FINGERPRINT_PROVENANCE`. A folded marker would make a resumed run compute a
  different digest than its own pre-resume blocks and refuse itself.
- **No `identical_bytes` computed assertion.** Cross-reboot source-byte
  identity (the `git diff e757f15 befd2ce -- src/quantibbler/ptq
  tools/quantize_hf_model.py`-is-empty fact) is a host-side measurement, not a
  driver-computable field: the container has no `git`, so it cannot diff two
  tips. The report records the admission evidence; the code identity lives in
  the launch trail (`launch.env` `commit`/`tree_dirty_files`), which is the
  launch-attribution change's surface.
- **No verdict/gate/exit-code behavior change.**
